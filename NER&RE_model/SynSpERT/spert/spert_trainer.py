import argparse
import math
import os
import numpy as np
import csv
from pathlib import Path
from scipy.special import entr
from typing import Optional

import torch
from torch.nn import DataParallel
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim import AdamW
import transformers
from torch.utils.data import DataLoader
from transformers import BertConfig
from transformers import BertTokenizer
from transformers.utils import import_utils

# Allow loading older PyTorch checkpoints despite the torch>=2.6 safety check.
# We only use models we explicitly download, so we opt out of the guard here.
import_utils.check_torch_load_is_safe = lambda: None

from spert import models
from spert import sampling
from spert import util  ##DKS
from spert.entities import Dataset
from spert.evaluator import Evaluator
from spert.input_reader import JsonInputReader, BaseInputReader
from spert.loss import SpERTLoss, Loss
from spert.preference_dataset import PreferenceDataset, preference_collate_fn
from tqdm import tqdm
from spert.trainer import BaseTrainer
import sys

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


class SpERTTrainer(BaseTrainer):
    """ Joint entity and relation extraction training and evaluation """

    def __init__(self, args: argparse.Namespace, config: BertConfig):
        super().__init__(args, config)

        # byte-pair encoding
        #DKS: Commented for now
        print("################ ",args.tokenizer_path)
        self._tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path,
                                                        do_lower_case=args.lowercase,
                                                        cache_dir=args.cache_path)
        tokenizer_vocab_size = getattr(self._tokenizer, "vocab_size", None) or len(self._tokenizer)
        config_vocab_size = getattr(self.config, "vocab_size", None)
        if tokenizer_vocab_size is not None and tokenizer_vocab_size != config_vocab_size:
            if hasattr(self, "_logger"):
                self._logger.warning(
                    "Tokenizer vocab_size (%s) != config vocab_size (%s); overriding config to keep checkpoint loading consistent.",
                    tokenizer_vocab_size,
                    config_vocab_size,
                )
            self.config.vocab_size = tokenizer_vocab_size
        elif tokenizer_vocab_size is not None and config_vocab_size is None:
            self.config.vocab_size = tokenizer_vocab_size
        # path to export predictions to
        self._predictions_path = os.path.join(self._log_path, 'predictions_%s_epoch_%s.json')

        # path to export relation extraction examples to
        self._examples_path = os.path.join(self._log_path, 'examples_%s_%s_epoch_%s.html')

    def _load_pretrained_model(self, input_reader: BaseInputReader, model_path: Optional[str] = None):
        # create model
        model_class = models.get_model(self.args.model_type) 
        #(Above) self.args.model_type = "syn_spert", model_class = 'SpERT'

        # load model
        #config = BertConfig.from_pretrained(self.args.model_path, cache_dir=self.args.cache_path)
        config = self.config   #DKS
        checkpoint_path = model_path or self.args.model_path
        util.check_version(config, model_class, checkpoint_path)

        config.spert_version = model_class.VERSION
        print("**** Calling model_class.from_pretrained(): TYPE: ", self.args.model_type, "****")
        model = model_class.from_pretrained(checkpoint_path,
                                            config=config,
                                            # SpERT model parameters
                                            cls_token=self._tokenizer.convert_tokens_to_ids('[CLS]'),
                                            relation_types=input_reader.relation_type_count - 1,
                                            entity_types=input_reader.entity_type_count,
                                            max_pairs=self.args.max_pairs,
                                            prop_drop=self.args.prop_drop,
                                            size_embedding=self.args.size_embedding,
                                            freeze_transformer=self.args.freeze_transformer,
                                            cache_dir=self.args.cache_path,
                                            use_pos=self.args.use_pos,  
                                            #pos_embedding=self.args.pos_embedding,
                                            use_entity_clf=self.args.use_entity_clf
                                            )
        print("Model type = ", type(model))

        return model


    def train(self, train_path: str, valid_path: str, types_path: str, input_reader_cls: BaseInputReader):
        args = self.args
        #ipconfig = self.config  #DKS
        train_label, valid_label = 'train', 'valid'
        skip_eval = getattr(args, 'skip_eval', False)

        self._logger.info("Datasets: %s, %s" % (train_path, valid_path))
        self._logger.info("Model type: %s" % args.model_type)

        self._ft_mode = getattr(args, "ft_mode", "sft")
        self._dpo_format = getattr(args, "dpo_format", "doc")
        self._dpo_lambda_entity = float(getattr(args, "dpo_lambda_entity", 1.0))
        self._dpo_lambda_relation = float(getattr(args, "dpo_lambda_relation", 1.0))
        if self._ft_mode == "dpo":
            # Freeze candidate sampling knobs for DPO runs
            args.neg_entity_count = 0
            args.neg_relation_count = 0

        # create log csv files
        self._init_train_logging(train_label)
        self._init_eval_logging(valid_label)

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer, args.neg_entity_count,
                                        args.neg_relation_count, args.max_span_size, self._logger)
        input_reader.read({train_label: train_path, valid_label: valid_path})
        self._log_datasets(input_reader)

        train_dataset = input_reader.get_dataset(train_label)
        validation_dataset = input_reader.get_dataset(valid_label)

        preference_loader = None
        preference_dataset = None
        preference_path = getattr(args, "dpo_preferences", None)
        dpo_batch_size = getattr(self.args, "dpo_train_batch_size", None) or self.args.train_batch_size
        if self._ft_mode == "dpo" and preference_path:
            try:
                preference_dataset = PreferenceDataset(
                    label=train_label,
                    preference_path=preference_path,
                    input_reader=input_reader,
                    neg_entity_count=args.neg_entity_count,
                    neg_relation_count=args.neg_relation_count,
                    max_span_size=args.max_span_size,
                    dpo_format=self._dpo_format,
                )
            except Exception as exc:
                self._logger.warning(f"Failed to load DPO preferences '{preference_path}': {exc}")
            else:
                if len(preference_dataset) == 0:
                    self._logger.warning("DPO preference dataset is empty; falling back to standard supervised loss.")
                else:
                    drop_last = len(preference_dataset) >= dpo_batch_size
                    preference_loader = DataLoader(
                        preference_dataset,
                        batch_size=dpo_batch_size,
                        shuffle=True,
                        drop_last=drop_last,
                        num_workers=self.args.sampling_processes,
                        collate_fn=preference_collate_fn,
                    )
                    train_sample_count = len(preference_dataset)
                    if drop_last:
                        updates_epoch = max(train_sample_count // dpo_batch_size, 1)
                    else:
                        updates_epoch = max(math.ceil(train_sample_count / dpo_batch_size), 1)
        if preference_loader is None:
            train_sample_count = train_dataset.document_count
            updates_epoch = max(train_sample_count // args.train_batch_size, 1)

        updates_total = updates_epoch * args.epochs

        self._logger.info("Updates per epoch: %s" % updates_epoch)
        self._logger.info("Updates total: %s" % updates_total)

        # create model
        model = self._load_pretrained_model(input_reader)

        self._ref_model = None
        if self._ft_mode == "dpo":
            ref_path = getattr(self.args, "dpo_reference", None) or self.args.model_path
            self._ref_model = self._load_pretrained_model(input_reader, model_path=ref_path)
            self._ref_model.to(self._device)
            self._ref_model.eval()
            for param in self._ref_model.parameters():
                param.requires_grad_(False)
        # SpERT is currently optimized on a single GPU and not thoroughly tested in a multi GPU setup
        # If you still want to train SpERT on multiple GPUs, uncomment the following lines
        # # parallelize model
        # if self._device.type != 'cpu':
        #     model = torch.nn.DataParallel(model)

        
        model.to(self._device)

        is_dpo_mode = getattr(self, "_ft_mode", "sft") == "dpo"

        # DPO stays strictly comparable with its reference; skip noise / reinit
        if not is_dpo_mode:
            for name, para in model.named_parameters():
                model.state_dict()[name][:] += (
                    (torch.rand(para.size()).to(self._device) - 0.5)
                    * self.args.noise_lambda
                    * torch.std(para).to(self._device)
                )

        def get_layers(model):
            layers = []

            def unfold_layer(model):
                layer_list = list(model.named_children())
                for item in layer_list:
                    module = item[1]
                    sublayer = list(module.named_children())
                    sublayer_num = len(sublayer)

                    if sublayer_num == 0:
                        layers.append(module)
                    elif isinstance(module, nn.Module):
                        unfold_layer(module)


            unfold_layer(model)
            return layers

        if not is_dpo_mode:
            for layer in get_layers(model.bert.encoder.layer[-1]):
                if isinstance(layer, (nn.Linear)):
                    nn.init.xavier_uniform_(layer.weight)

            for layer in get_layers(model.bert.pooler):
                if isinstance(layer, (nn.Linear)):
                    nn.init.xavier_uniform_(layer.weight)

        # create optimizer
        optimizer_params = self._get_optimizer_params(model)
        optimizer = AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay)
        # create scheduler
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=args.lr_warmup * updates_total,
                                                                 num_training_steps=updates_total)
        # create loss function
        rel_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        entity_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        compute_loss = SpERTLoss(
            rel_criterion,
            entity_criterion,
            model,
            optimizer,
            scheduler,
            args.max_grad_norm,
        )
        self._entity_criterion = entity_criterion
        self._rel_criterion = rel_criterion
        self._preference_loader = preference_loader
        self._preference_dataset = preference_dataset

        best_model = None #DKS
        best_rel_f1_micro=0
        best_epoch=0
        model_saved = False

        # eval validation set
        if args.init_eval:
            self._eval(model, validation_dataset, input_reader, 0, updates_epoch)

        # train
        for epoch in range(args.epochs):
            if self._preference_loader is not None:
                pref_label = getattr(self._preference_dataset, "label", train_label)
                self._train_epoch_preference(model, optimizer, scheduler, self._preference_loader, epoch, pref_label)
            else:
                self._train_epoch(model, compute_loss, optimizer, train_dataset, updates_epoch, epoch)

            if skip_eval:
                continue

            if not args.final_eval or (epoch == args.epochs - 1):
                ner_f1_micro, rel_f1_micro = self._eval(model, validation_dataset, input_reader,
                                                        epoch + 1, updates_epoch)

                if rel_f1_micro > best_rel_f1_micro:
                    best_rel_f1_micro = rel_f1_micro
                    best_model = model
                    best_epoch = epoch + 1
                    extra = dict(epoch=args.epochs, updates_epoch=updates_epoch, epoch_iteration=0)
                    self._save_model(self._save_path, best_model, self._tokenizer, 0,
                                     optimizer=optimizer if self.args.save_optimizer else None, extra=extra,
                                     include_iteration=False, name='best_model')
                    model_saved = True
                   
            
        if skip_eval or not model_saved:
            extra = dict(epoch=args.epochs, updates_epoch=updates_epoch, epoch_iteration=0)
            global_iteration = args.epochs * updates_epoch
            self._save_model(self._save_path, model, self._tokenizer, global_iteration,
                             optimizer=optimizer if self.args.save_optimizer else None, extra=extra,
                             include_iteration=False, name='final_model')
        
        self._logger.info("Logged in: %s" % self._log_path)
        self._logger.info("Saved in: %s" % self._save_path)
        self._close_summary_writer()
        

    def eval(self, dataset_path: str, types_path: str, input_reader_cls: BaseInputReader):
        args = self.args
        #ipconfig = self.config  #DKS

        dataset_label = 'test'

        self._logger.info("Dataset: %s" % dataset_path)
        self._logger.info("Model: %s" % args.model_type)

        # create log csv files
        self._init_eval_logging(dataset_label)

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer,
                                        max_span_size=args.max_span_size, logger=self._logger)
        input_reader.read({dataset_label: dataset_path})
        self._log_datasets(input_reader)

        # create model
        model = self._load_pretrained_model(input_reader)

        model.to(self._device)

        # evaluate
        self._eval(model, input_reader.get_dataset(dataset_label), input_reader)

        self._logger.info("Logged in: %s" % self._log_path)
        self._close_summary_writer()

    def _train_epoch(self, model: torch.nn.Module, compute_loss: Loss, optimizer: Optimizer, dataset: Dataset,
                     updates_epoch: int, epoch: int):
        self._logger.info("Train epoch: %s" % epoch)
 
        # create data loader
        dataset.switch_mode(Dataset.TRAIN_MODE)
        data_loader = DataLoader(dataset, batch_size=self.args.train_batch_size, shuffle=True, drop_last=True,
                                 num_workers=self.args.sampling_processes, collate_fn=sampling.collate_fn_padding)

        model.zero_grad()
        
        #print("*************** train_dataset = ", dataset)
        #sys.exit(-1)
        iteration = 0
        total = dataset.document_count // self.args.train_batch_size
        for batch in tqdm(data_loader, total=total, desc='Train epoch %s' % epoch):
            model.train()
            batch = util.to_device(batch, self._device)
            
            # forward step
            
            entity_logits, rel_logits = model(encodings=batch['encodings'], context_masks=batch['context_masks'],
                                              entity_masks=batch['entity_masks'], entity_sizes=batch['entity_sizes'],
                                              relations=batch['rels'], rel_masks=batch['rel_masks'],
                                              dephead= batch['dephead'], deplabel =batch['deplabel'], pos= batch['pos'])

            # compute loss and optimize parameters
            batch_loss = compute_loss.compute(entity_logits=entity_logits, rel_logits=rel_logits,
                                              rel_types=batch['rel_types'], entity_types=batch['entity_types'],
                                              entity_sample_masks=batch['entity_sample_masks'],
                                              rel_sample_masks=batch['rel_sample_masks'])

            # logging
            iteration += 1
            global_iteration = epoch * updates_epoch + iteration

            if global_iteration % self.args.train_log_iter == 0:
                self._log_train(optimizer, batch_loss, epoch, iteration, global_iteration, dataset.label)

    def _train_epoch_preference(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler,
        data_loader: DataLoader,
        epoch: int,
        label: str,
    ):
        if len(data_loader) == 0:
            self._logger.warning("Preference loader returned zero batches; skipping epoch.")
            return

        self._logger.info("Train epoch (DPO preferences): %s", epoch)
        iteration = 0
        total = len(data_loader)

        if getattr(self, "_dpo_format", "doc") != "doc":
            for batch in tqdm(data_loader, total=total, desc=f"DPO epoch {epoch}"):
                model.train()
                blueprint = util.to_device(batch["blueprint"], self._device)
                entity_type_ids = batch["entity_type_ids"].to(self._device)
                relation_type_ids = batch["relation_type_ids"].to(self._device)
                relation_pair_indices = batch["relation_pair_indices"].to(self._device)
                entity_pref_pairs = batch["entity_pref_pairs"].to(self._device)
                relation_pref_pairs = batch["relation_pref_pairs"].to(self._device)

                if entity_pref_pairs.shape[0] == 0 and relation_pref_pairs.shape[0] == 0:
                    continue

                entity_logits, rel_logits = model(
                    encodings=blueprint["encodings"],
                    context_masks=blueprint["context_masks"],
                    entity_masks=blueprint["entity_masks"],
                    entity_sizes=blueprint["entity_sizes"],
                    relations=blueprint["rels"],
                    rel_masks=blueprint["rel_masks"],
                    dephead=blueprint["dephead"],
                    deplabel=blueprint["deplabel"],
                    pos=blueprint["pos"],
                )

                ref_entity_logits = ref_rel_logits = None
                if self._ref_model is not None:
                    with torch.no_grad():
                        ref_entity_logits, ref_rel_logits = self._ref_model(
                            encodings=blueprint["encodings"],
                            context_masks=blueprint["context_masks"],
                            entity_masks=blueprint["entity_masks"],
                            entity_sizes=blueprint["entity_sizes"],
                            relations=blueprint["rels"],
                            rel_masks=blueprint["rel_masks"],
                            dephead=blueprint["dephead"],
                            deplabel=blueprint["deplabel"],
                            pos=blueprint["pos"],
                        )

                entity_loss = self._entity_preference_loss(
                    entity_logits, entity_type_ids, entity_pref_pairs, ref_entity_logits
                )
                relation_loss = self._relation_preference_loss(
                    rel_logits, relation_type_ids, relation_pair_indices, relation_pref_pairs, ref_rel_logits
                )
                total_loss = self.args.dpo_lambda * (
                    self._dpo_lambda_entity * entity_loss + self._dpo_lambda_relation * relation_loss
                )

                if not total_loss.requires_grad:
                    continue

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

                iteration += 1
                global_iteration = epoch * total + iteration
                self._log_train(optimizer, float(total_loss.detach()), epoch, iteration, global_iteration, label)
            return

        for chosen_batch, rejected_batch in tqdm(data_loader, total=total, desc=f"DPO epoch {epoch}"):
            model.train()
            chosen_batch = util.to_device(chosen_batch, self._device)
            rejected_batch = util.to_device(rejected_batch, self._device)

            if iteration < 5:
                for key in ("entity_sample_masks", "rel_sample_masks", "rels"):
                    cb, rb = chosen_batch[key], rejected_batch[key]
                    assert cb.shape == rb.shape and torch.equal(cb, rb), f"DPO misaligned: {key}"

            chosen_entity_logits, chosen_rel_logits = model(
                encodings=chosen_batch["encodings"],
                context_masks=chosen_batch["context_masks"],
                entity_masks=chosen_batch["entity_masks"],
                entity_sizes=chosen_batch["entity_sizes"],
                relations=chosen_batch["rels"],
                rel_masks=chosen_batch["rel_masks"],
                dephead=chosen_batch["dephead"],
                deplabel=chosen_batch["deplabel"],
                pos=chosen_batch["pos"],
            )
            rejected_entity_logits, rejected_rel_logits = model(
                encodings=rejected_batch["encodings"],
                context_masks=rejected_batch["context_masks"],
                entity_masks=rejected_batch["entity_masks"],
                entity_sizes=rejected_batch["entity_sizes"],
                relations=rejected_batch["rels"],
                rel_masks=rejected_batch["rel_masks"],
                dephead=rejected_batch["dephead"],
                deplabel=rejected_batch["deplabel"],
                pos=rejected_batch["pos"],
            )

            chosen_logp = self._preference_log_prob(chosen_entity_logits, chosen_rel_logits, chosen_batch)
            rejected_logp = self._preference_log_prob(rejected_entity_logits, rejected_rel_logits, rejected_batch)

            ref_margin = torch.zeros_like(chosen_logp)
            if self._ref_model is not None:
                with torch.no_grad():
                    ref_chosen_entity_logits, ref_chosen_rel_logits = self._ref_model(
                        encodings=chosen_batch["encodings"],
                        context_masks=chosen_batch["context_masks"],
                        entity_masks=chosen_batch["entity_masks"],
                        entity_sizes=chosen_batch["entity_sizes"],
                        relations=chosen_batch["rels"],
                        rel_masks=chosen_batch["rel_masks"],
                        dephead=chosen_batch["dephead"],
                        deplabel=chosen_batch["deplabel"],
                        pos=chosen_batch["pos"],
                    )
                    ref_rejected_entity_logits, ref_rejected_rel_logits = self._ref_model(
                        encodings=rejected_batch["encodings"],
                        context_masks=rejected_batch["context_masks"],
                        entity_masks=rejected_batch["entity_masks"],
                        entity_sizes=rejected_batch["entity_sizes"],
                        relations=rejected_batch["rels"],
                        rel_masks=rejected_batch["rel_masks"],
                        dephead=rejected_batch["dephead"],
                        deplabel=rejected_batch["deplabel"],
                        pos=rejected_batch["pos"],
                    )
                    ref_chosen_logp = self._preference_log_prob(
                        ref_chosen_entity_logits, ref_chosen_rel_logits, chosen_batch
                    )
                    ref_rejected_logp = self._preference_log_prob(
                        ref_rejected_entity_logits, ref_rejected_rel_logits, rejected_batch
                    )
                    ref_margin = ref_chosen_logp - ref_rejected_logp

            policy_margin = chosen_logp - rejected_logp
            delta = policy_margin - ref_margin
            delta = (delta - delta.mean()) / delta.std(unbiased=False).clamp_min(1e-6)
            d = self.args.dpo_beta * delta
            d = d.clamp(-20.0, 20.0)
            dpo_loss = self.args.dpo_lambda * F.softplus(-d).mean()

            dpo_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            iteration += 1
            global_iteration = epoch * total + iteration
            self._log_train(optimizer, float(dpo_loss.detach()), epoch, iteration, global_iteration, label)

    def _entity_preference_loss(self, logits, type_ids, pref_pairs, ref_logits=None):
        if pref_pairs.shape[0] == 0:
            return torch.tensor(0.0, device=self._device)

        log_probs = F.log_softmax(logits, dim=-1)
        if ref_logits is not None:
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        else:
            ref_log_probs = None

        batch_idx = pref_pairs[:, 0]
        pos_idx = pref_pairs[:, 1]
        neg_idx = pref_pairs[:, 2]
        pos_type_ids = type_ids[batch_idx, pos_idx]
        neg_type_ids = type_ids[batch_idx, neg_idx]

        logp_pos = log_probs[batch_idx, pos_idx, pos_type_ids]
        logp_neg = log_probs[batch_idx, neg_idx, neg_type_ids]
        if ref_log_probs is not None:
            logp_pos_ref = ref_log_probs[batch_idx, pos_idx, pos_type_ids]
            logp_neg_ref = ref_log_probs[batch_idx, neg_idx, neg_type_ids]
        else:
            logp_pos_ref = torch.zeros_like(logp_pos)
            logp_neg_ref = torch.zeros_like(logp_neg)

        diff_policy = logp_pos - logp_neg
        diff_ref = logp_pos_ref - logp_neg_ref
        return -torch.logsigmoid(self.args.dpo_beta * (diff_policy - diff_ref)).mean()

    def _relation_preference_loss(
        self, logits, type_ids, pair_indices, pref_pairs, ref_logits=None
    ):
        if pref_pairs.shape[0] == 0:
            return torch.tensor(0.0, device=self._device)

        log_sigmoid = F.logsigmoid(logits)
        log_none = F.logsigmoid(-logits).sum(dim=-1)

        if ref_logits is not None:
            ref_log_sigmoid = F.logsigmoid(ref_logits)
            ref_log_none = F.logsigmoid(-ref_logits).sum(dim=-1)
        else:
            ref_log_sigmoid = None
            ref_log_none = None

        batch_idx = pref_pairs[:, 0]
        pos_idx = pref_pairs[:, 1]
        neg_idx = pref_pairs[:, 2]

        pos_pair_idx = pair_indices[batch_idx, pos_idx]
        neg_pair_idx = pair_indices[batch_idx, neg_idx]
        pos_type_ids = type_ids[batch_idx, pos_idx]
        neg_type_ids = type_ids[batch_idx, neg_idx]

        logp_pos = self._gather_relation_logp(log_sigmoid, log_none, batch_idx, pos_pair_idx, pos_type_ids)
        logp_neg = self._gather_relation_logp(log_sigmoid, log_none, batch_idx, neg_pair_idx, neg_type_ids)

        if ref_log_sigmoid is not None and ref_log_none is not None:
            logp_pos_ref = self._gather_relation_logp(
                ref_log_sigmoid, ref_log_none, batch_idx, pos_pair_idx, pos_type_ids
            )
            logp_neg_ref = self._gather_relation_logp(
                ref_log_sigmoid, ref_log_none, batch_idx, neg_pair_idx, neg_type_ids
            )
        else:
            logp_pos_ref = torch.zeros_like(logp_pos)
            logp_neg_ref = torch.zeros_like(logp_neg)

        diff_policy = logp_pos - logp_neg
        diff_ref = logp_pos_ref - logp_neg_ref
        return -torch.logsigmoid(self.args.dpo_beta * (diff_policy - diff_ref)).mean()

    @staticmethod
    def _gather_relation_logp(log_sigmoid, log_none, batch_idx, pair_idx, type_ids):
        scores = torch.zeros_like(type_ids, dtype=log_sigmoid.dtype)
        mask = type_ids > 0
        if mask.any():
            rel_ids = type_ids[mask] - 1
            scores[mask] = log_sigmoid[batch_idx[mask], pair_idx[mask], rel_ids]
        if (~mask).any():
            scores[~mask] = log_none[batch_idx[~mask], pair_idx[~mask]]
        return scores

    def _preference_log_prob(self, entity_logits, rel_logits, batch):
        entity_lp = self._entity_log_prob(entity_logits, batch)
        rel_lp = self._relation_log_prob(rel_logits, batch)
        return 1.2 * entity_lp + rel_lp

    def _entity_log_prob(self, entity_logits, batch):
        batch_size = entity_logits.shape[0]
        logits_flat = entity_logits.view(-1, entity_logits.shape[-1])
        entity_types = batch["entity_types"].view(-1)
        entity_losses = self._entity_criterion(logits_flat, entity_types)
        entity_losses = entity_losses.view(batch_size, -1)
        entity_masks = batch["entity_sample_masks"].view(batch_size, -1).float()
        neg_log_likelihood = (entity_losses * entity_masks).sum(dim=1)
        return -neg_log_likelihood

    def _relation_log_prob(self, rel_logits, batch):
        batch_size = batch["rel_sample_masks"].shape[0]
        if rel_logits is None:
            return torch.zeros(batch_size, device=self._device)

        rel_sample_masks = batch["rel_sample_masks"].view(batch_size, -1).float()
        if rel_sample_masks.sum().item() == 0:
            return torch.zeros(batch_size, device=self._device)

        rel_bce = F.binary_cross_entropy_with_logits(rel_logits, batch["rel_types"], reduction="none")
        pos_mask = (batch["rel_types"] > 0.5).float()
        mask = rel_sample_masks.view(batch_size, -1, 1)
        pos_mask = pos_mask * mask
        pos_count = pos_mask.sum(dim=(1, 2)).clamp_min(1.0)
        pos_nll = (rel_bce * pos_mask).sum(dim=(1, 2)) / pos_count
        return -pos_nll

    def _eval(self, model: torch.nn.Module, dataset: Dataset, input_reader: JsonInputReader,
              epoch: int = 0, updates_epoch: int = 0, iteration: int = 0):
        self._logger.info("Evaluate: %s" % dataset.label)
        names = 'agu1'  # Set the name for saving files

        if isinstance(model, DataParallel):
            model = model.module

        evaluator = Evaluator(dataset, input_reader, self._tokenizer,
                              self.args.rel_filter_threshold, self.args.no_overlapping, self._predictions_path,
                              self._examples_path, self.args.example_count, epoch, dataset.label)

        dataset.switch_mode(Dataset.EVAL_MODE)
        data_loader = DataLoader(dataset, batch_size=self.args.eval_batch_size, shuffle=False, drop_last=False,
                                 num_workers=self.args.sampling_processes, collate_fn=sampling.collate_fn_padding)

        dump_dir = getattr(self.args, 'al_dump_dir', None)
        if dump_dir:
            dump_dir = Path(dump_dir)
            dump_dir.mkdir(parents=True, exist_ok=True)
            al_entropy_rel = []
            al_entropy_ent = []
            al_label_counts = []
            al_pooler = []

        with torch.no_grad():
            model.eval()
            total = math.ceil(dataset.document_count / self.args.eval_batch_size)
            for batch in tqdm(data_loader, total=total, desc='Evaluate epoch %s' % epoch):
                batch = util.to_device(batch, self._device)
                entity_clf, rel_clf, rels, pooler_output, rel_sample_masks = model(
                    encodings=batch['encodings'], context_masks=batch['context_masks'],
                    entity_masks=batch['entity_masks'], entity_sizes=batch['entity_sizes'],
                    entity_spans=batch['entity_spans'], entity_sample_masks=batch['entity_sample_masks'],
                    dephead=batch['dephead'], deplabel=batch['deplabel'], pos=batch['pos'], evaluate=True)
                evaluator.eval_batch(entity_clf, rel_clf, rels, batch)

                if dump_dir:
                    # entity entropy (softmax over entity types)
                    entity_probs = torch.softmax(entity_clf, dim=-1)
                    entity_entropy = -(entity_probs * torch.log(entity_probs + 1e-12)).sum(dim=-1)
                    entity_entropy = entity_entropy * batch['entity_sample_masks'].float()
                    entity_entropy = entity_entropy.sum(dim=1).cpu().numpy()

                    # relation entropy (sigmoid over relation logits)
                    rel_probs = torch.sigmoid(rel_clf)
                    rel_entropy = -(rel_probs * torch.log(rel_probs + 1e-12) +
                                    (1 - rel_probs) * torch.log((1 - rel_probs) + 1e-12))
                    rel_entropy = rel_entropy.sum(dim=2)
                    rel_entropy = (rel_entropy * rel_sample_masks.float()).sum(dim=1).cpu().numpy()

                    al_entropy_ent.extend(entity_entropy.tolist())
                    al_entropy_rel.extend(rel_entropy.tolist())

                    rel_pred_mask = (torch.sigmoid(rel_clf) >= self.args.rel_filter_threshold).float()
                    rel_pred_mask = rel_pred_mask * rel_sample_masks.float().unsqueeze(-1)
                    label_counts = rel_pred_mask.sum(dim=1).cpu().numpy()
                    al_label_counts.extend(label_counts.tolist())

                    al_pooler.extend(pooler_output.cpu().numpy().tolist())

        global_iteration = epoch * updates_epoch + iteration
        ner_eval, rel_eval, rel_nec_eval = evaluator.compute_scores()
        self._log_eval(*ner_eval, *rel_eval, *rel_nec_eval,
                       epoch, iteration, global_iteration, dataset.label)

        if self.args.store_predictions and not self.args.no_overlapping:
            evaluator.store_predictions()

        if self.args.store_examples:
            evaluator.store_examples()

        if dump_dir:
            torch.save(torch.tensor(al_entropy_rel, dtype=torch.float32), dump_dir / 'entropy_relation.pt')
            torch.save(torch.tensor(al_entropy_ent, dtype=torch.float32), dump_dir / 'entropy_entities.pt')
            torch.save(torch.tensor(al_label_counts, dtype=torch.float32), dump_dir / 'label_prediction.pt')
            torch.save(torch.tensor(al_pooler, dtype=torch.float32), dump_dir / 'pooler_output.pt')

        return ner_eval[2], rel_eval[2]

    def _get_optimizer_params(self, model):
        param_optimizer = filter(lambda p: p[1].requires_grad, model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_params = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        return optimizer_params

    def _log_train(self, optimizer: Optimizer, loss: float, epoch: int,
                   iteration: int, global_iteration: int, label: str):
        avg_loss = loss / self.args.train_batch_size
        lr = self._get_lr(optimizer)[0]

        self._log_tensorboard(label, 'loss', loss, global_iteration)
        self._log_tensorboard(label, 'loss_avg', avg_loss, global_iteration)
        self._log_tensorboard(label, 'lr', lr, global_iteration)

        self._log_csv(label, 'loss', loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'loss_avg', avg_loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'lr', lr, epoch, iteration, global_iteration)

    def _log_eval(self, ner_prec_micro: float, ner_rec_micro: float, ner_f1_micro: float,
                  ner_prec_macro: float, ner_rec_macro: float, ner_f1_macro: float,
                  rel_prec_micro: float, rel_rec_micro: float, rel_f1_micro: float,
                  rel_prec_macro: float, rel_rec_macro: float, rel_f1_macro: float,
                  rel_nec_prec_micro: float, rel_nec_rec_micro: float, rel_nec_f1_micro: float,
                  rel_nec_prec_macro: float, rel_nec_rec_macro: float, rel_nec_f1_macro: float,
                  epoch: int, iteration: int, global_iteration: int, label: str):

        self._log_tensorboard(label, 'eval/ner_prec_micro', ner_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_micro', ner_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_micro', ner_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_prec_macro', ner_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_macro', ner_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_macro', ner_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/rel_prec_micro', rel_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_recall_micro', rel_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_f1_micro', rel_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_prec_macro', rel_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_recall_macro', rel_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_f1_macro', rel_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/rel_nec_prec_micro', rel_nec_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_recall_micro', rel_nec_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_f1_micro', rel_nec_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_prec_macro', rel_nec_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_recall_macro', rel_nec_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_f1_macro', rel_nec_f1_macro, global_iteration)

        self._log_csv(label, 'eval', ner_prec_micro, ner_rec_micro, ner_f1_micro,
                      ner_prec_macro, ner_rec_macro, ner_f1_macro,
                      rel_prec_micro, rel_rec_micro, rel_f1_micro,
                      rel_prec_macro, rel_rec_macro, rel_f1_macro,
                      rel_nec_prec_micro, rel_nec_rec_micro, rel_nec_f1_micro,
                      rel_nec_prec_macro, rel_nec_rec_macro, rel_nec_f1_macro,
                      epoch, iteration, global_iteration)

    def _log_datasets(self, input_reader):
        self._logger.info("Relation type count: %s" % input_reader.relation_type_count)
        self._logger.info("Entity type count: %s" % input_reader.entity_type_count)

        self._logger.info("Entities:")
        for e in input_reader.entity_types.values():
            self._logger.info(e.verbose_name + '=' + str(e.index))

        self._logger.info("Relations:")
        for r in input_reader.relation_types.values():
            self._logger.info(r.verbose_name + '=' + str(r.index))

        for k, d in input_reader.datasets.items():
            self._logger.info('Dataset: %s' % k)
            self._logger.info("Document count: %s" % d.document_count)
            self._logger.info("Relation count: %s" % d.relation_count)
            self._logger.info("Entity count: %s" % d.entity_count)

        self._logger.info("Context size: %s" % input_reader.context_size)

    def _init_train_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'lr': ['lr', 'epoch', 'iteration', 'global_iteration'],
                                        'loss': ['loss', 'epoch', 'iteration', 'global_iteration'],
                                        'loss_avg': ['loss_avg', 'epoch', 'iteration', 'global_iteration']})

    def _init_eval_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'eval': ['ner_prec_micro', 'ner_rec_micro', 'ner_f1_micro',
                                                 'ner_prec_macro', 'ner_rec_macro', 'ner_f1_macro',
                                                 'rel_prec_micro', 'rel_rec_micro', 'rel_f1_micro',
                                                 'rel_prec_macro', 'rel_rec_macro', 'rel_f1_macro',
                                                 'rel_nec_prec_micro', 'rel_nec_rec_micro', 'rel_nec_f1_micro',
                                                 'rel_nec_prec_macro', 'rel_nec_rec_macro', 'rel_nec_f1_macro',
                                                 'epoch', 'iteration', 'global_iteration']})
