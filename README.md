# SynSpERT.PL Training & Active Learning Guide

This repository contains a lightly modified copy of the MDKG SynSpERT.PL codebase plus support scripts to train on custom data and run an active learning loop.  
The notes below focus on two workflows:

1. Train / evaluate SynSpERT on your dataset.
2. Produce uncertainty features and drive the adaptive sampling script.

All paths are relative to the repository root.

---

## 1. Environment

```bash
# create (once)
conda create -n mdkgspert python=3.10 pytorch torchvision torchaudio cpuonly -c pytorch -c conda-forge
conda install -n mdkgspert scikit-learn transformers tqdm numpy scipy more-itertools faiss-cpu -c conda-forge

# activate (every session)
conda activate mdkgspert
```

The commands below assume the environment is active. On macOS you may also want to pin OpenMP vars to avoid shared-memory errors:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export KMP_AFFINITY=disabled KMP_INIT_AT_FORK=FALSE KMP_DUPLICATE_LIB_OK=TRUE
```

---

## 2. Data preparation

Place your annotated documents in the root (e.g. `spert_training_data_10.json`) and run:

```bash
python NER&RE_model/SynSpERT/generate_augmented_input.py \
  --input spert_training_data_10.json \
  --output_dir NER&RE_model/InputsAndOutputs/data/datasets \
  --prefix diabetes --seed 13
```

This creates `diabetes_{train,valid,test,all}.json` and augments each document with POS / dependency features.  
If SciSpaCy is installed it is used, otherwise a rule-based fallback kicks in.

---

## 3. Train the model

```bash
python NER&RE_model/SynSpERT/main.py --mode train
```

Key notes:

- Uses `bert-base-uncased` by default.
- Training data: `InputsAndOutputs/data/datasets/diabetes_{train,valid}.json`.
- Checkpoints land in `InputsAndOutputs/data/save/diabetes_small_run/<timestamp>/final_model/`.
- Logs (loss, args) appear in `InputsAndOutputs/data/log/diabetes_small_run/<timestamp>/`.
- Validation is skipped by default (`--skip_eval` in `main.py`) to sidestep OpenMP issues; remove the flag if you need per-epoch dev metrics.

---

## 4. Evaluate / prediction dump

```bash
python NER&RE_model/SynSpERT/main.py \
  --mode eval \
  --model_dir NER&RE_model/InputsAndOutputs/data/save/diabetes_small_run/<timestamp>/final_model
```

Outputs:

- Metrics (`eval_test.csv`) and HTML examples at `InputsAndOutputs/data/log/diabetes_small_run_eval/<timestamp>/`.
- `predictions_test_epoch_0.json` includes tokens, entities, relations, and softmax scores.

You can override `--dataset_path` to score any JSON file that follows the SynSpERT schema.

---

## 5. Active learning workflow

1. **Prepare an unlabeled pool**  
   Build a JSON `diabetes_unlabeled.json` with the same schema as the SynSpERT training data (`relations` can be empty).

2. **Run “uncertainty inference”**  
   ```bash
   python NER&RE_model/SynSpERT/main.py \
     --mode eval \
     --dataset_path diabetes_unlabeled.json \
     --model_dir NER&RE_model/InputsAndOutputs/data/save/diabetes_small_run/<timestamp>/final_model \
     --al_dump_dir Outputs/al_round_01 \
     --label diabetes_al_round01 \
     --store_predictions
   ```
   The flag `--al_dump_dir` triggers dumps of:
   - `entropy_entities.pt`
   - `entropy_relation.pt`
   - `label_prediction.pt`
   - `pooler_output.pt`

3. **Select samples**  
   ```bash
   python Active_learning.py \
     --dump_dir Outputs/al_round_01 \
     --unlabeled_json diabetes_unlabeled.json \
     --output_prefix round01 \
     --top_k 20 --sample_per_group 10 --beta 0.1 --gamma 0.1 --ncentroids 20 --use_weights
   ```
   This produces:
   - `sampling_json_round01.json` – documents to send for annotation.
   - `sampling_text_round01.txt` – detokenized sentences for quick review.
   - `selected_indices_round01.pt` – indices of chosen items.
   - Optional analytic files (`weighted_embedding_*.pt`, `class_number_*.pt`).

4. **Annotate & refresh training data**  
   1. Apply any manual fixes in `sampling_json_round01.json`.
   2. Merge the revised documents back into `spert_training_data_10.json`.
   3. Regenerate train/valid/test splits:
      ```bash
      python NER&RE_model/SynSpERT/generate_augmented_input.py \
        --input spert_training_data_10.json \
        --output_dir NER&RE_model/InputsAndOutputs/data/datasets \
        --prefix diabetes \
        --seed 13
      ```
   4. Retrain:
      ```bash
      python NER&RE_model/SynSpERT/main.py --mode train
      ```

Repeat the cycle until budget is exhausted or performance converges.

---

## 6. File reference

| Path / Script | Purpose |
|---------------|---------|
| `NER&RE_model/SynSpERT/generate_augmented_input.py` | Token-level augmentation & train/valid/test split |
| `NER&RE_model/SynSpERT/main.py` | Unified CLI for training (`--mode train`) and evaluation (`--mode eval`) |
| `NER&RE_model/SynSpERT/spert/spert_trainer.py` | Core training loop; `_eval` emits AL features when `--al_dump_dir` is set |
| `Active_learning.py` | Entropy + diversity sampler that produces JSON for manual labeling |
| `NER&RE_model/InputsAndOutputs/data/log/*` | Training / evaluation logs, predictions, metrics |
| `NER&RE_model/InputsAndOutputs/data/save/*` | Saved transformers models (config + weights) |

---

Feel free to adapt the hyper-parameters or sampling strategy. The current defaults prioritise getting a CPU-only pipeline running end-to-end with minimal manual tweaks.  
If you add new AL rounds, consider versioning the output directories (`al_round_02`, `al_round_03`, …) for easy tracking.

