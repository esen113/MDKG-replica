# SynSpERT.PL Training & Active Learning Guide

This repository packages a lightly modified copy of the MDKG SynSpERT.PL codebase together with tooling to:

1. Prepare MDIEC- / MDERC-style data for SynSpERT.
2. Train / evaluate the joint NER & RE model on top of arbitrary BERT-family backbones.
3. Run the entropy/diversity-based active learning loop.

All commands below assume the repository root (`/Users/<user>/MDKG-replica-3`) as the working directory unless stated otherwise.

---

## 1. Environment (once)

```bash
conda create -n mdkgspert python=3.10 -y
conda run -n mdkgspert pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
conda run -n mdkgspert pip install scikit-learn transformers tqdm numpy scipy \
  more-itertools faiss-cpu spacy scispacy huggingface-hub nltk
conda run -n mdkgspert python -m spacy download en_core_web_sm
conda run -n mdkgspert python -m spacy download en_core_sci_sm
conda run -n mdkgspert python -m nltk.downloader punkt punkt_tab
huggingface-cli login    # required for auto-downloading Hugging Face models
```

Activate the environment for every new shell:

```bash
conda activate mdkgspert
```

Optional (macOS CPU) to avoid OpenMP oversubscription:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export KMP_AFFINITY=disabled KMP_INIT_AT_FORK=FALSE KMP_DUPLICATE_LIB_OK=TRUE
```

---

## 2. Dataset preparation

### 2.1 Fetch official MDIEC annotations

```bash
python scripts/fetch_official_data.py --overwrite
```

The script pulls `data/MDIEC.zip` from `YF0808/MDKG_data` on Hugging Face and extracts into `NER&RE_model/InputsAndOutputs/data/dataset/MDIEC/`.

> Already downloaded data? Place your `.ann/.txt` pairs directly under `NER&RE_model/InputsAndOutputs/data/dataset/<folder>/`.

### 2.2 Convert Brat annotations to SynSpERT JSON

```bash
python 'NER&RE_model/SynSpERT/generate_input.py' \
  --input_dir 'NER&RE_model/InputsAndOutputs/data/dataset/MDIEC' \
  --output_json 'NER&RE_model/InputsAndOutputs/data/dataset/MDIEC.json'
```

This aggregates all documents into a single JSON file with SynSpERT’s expected schema (tokens, entities, relations, `orig_id`, …).

### 2.3 Generate augmented train/valid/test splits

```bash
python 'NER&RE_model/SynSpERT/generate_augmented_input.py' \
  --input 'NER&RE_model/InputsAndOutputs/data/dataset/MDIEC.json' \
  --output_dir 'NER&RE_model/InputsAndOutputs/data/datasets' \
  --prefix diabetes \
  --seed 13
```

Outputs:
- `NER&RE_model/InputsAndOutputs/data/datasets/diabetes_{train,valid,test,all}.json`
- Each document augmented with POS, dependency labels, heads, and verb indicators (SciSpaCy when available, rule-based fallback otherwise).

> `nutrition_diabetes_types.json` now mirrors MDIEC’s ontology (9 entity types, 8 relation types). Update this file if you introduce other datasets.

---

## 3. Training

The training entrypoint supports both local checkpoints and Hugging Face model IDs. Key CLI flags:

- `--bert_model`: model name or local path (default `bert-base-uncased`)
- `--config_override`: optional SynSpERT config JSON (otherwise the model directory/config is reused)
- `--download_dir`: cache location for downloaded Hugging Face models (defaults to `NER&RE_model/InputsAndOutputs/models`)
- `--run_seed`: overrides the run seed used in logs/checkpoint names

### Example: CODER++

```bash
python 'NER&RE_model/SynSpERT/main.py' --mode train \
  --bert_model GanjinZero/coder_eng_pp \
  --config_override 'NER&RE_model/SynSpERT/configs/config-coder.json' \
  --download_dir 'NER&RE_model/InputsAndOutputs/models' \
  --run_seed 11
```

### Other backbone examples

```bash
# PubMedBERT (uncased)
python 'NER&RE_model/SynSpERT/main.py' --mode train \
  --bert_model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
  --config_override 'NER&RE_model/SynSpERT/configs/config.json' \
  --download_dir 'NER&RE_model/InputsAndOutputs/models' \
  --run_seed 13

# BioBERT (cased)
python 'NER&RE_model/SynSpERT/main.py' --mode train \
  --bert_model dmis-lab/biobert-base-cased-v1.1 \
  --config_override 'NER&RE_model/SynSpERT/configs/config-biobert.json' \
  --download_dir 'NER&RE_model/InputsAndOutputs/models' \
  --run_seed 13
```

Outputs (per run label, default `diabetes_small_run`):

- Checkpoints: `NER&RE_model/InputsAndOutputs/data/save/diabetes_small_run/<timestamp>/final_model/`
- Logs & predictions: `NER&RE_model/InputsAndOutputs/data/log/diabetes_small_run/<timestamp>/`
- Validation is skipped by default to avoid OpenMP contention (`--skip_eval` baked into default args). Remove the flag inside `main.py` if per-epoch validation is needed.

---

## 4. Evaluation & prediction dumps

```bash
python 'NER&RE_model/SynSpERT/main.py' --mode eval \
  --model_dir 'NER&RE_model/InputsAndOutputs/data/save/diabetes_small_run/<timestamp>/final_model' \
  --bert_model GanjinZero/coder_eng_pp \
  --config_override 'NER&RE_model/SynSpERT/configs/config-coder.json' \
  --download_dir 'NER&RE_model/InputsAndOutputs/models' \
  --dataset_path 'NER&RE_model/InputsAndOutputs/data/datasets/diabetes_test.json' \
  --label coderpp_eval \
  --store_predictions
```

Evaluation artifacts:

- `eval_test.csv`, HTML entity/relation tables under `.../data/log/<label>/<timestamp>/`
- `predictions_test_epoch_0.json` with tokens/entities/relations/scores
- Optional active-learning tensors when `--al_dump_dir` is provided (see next section)

`--dataset_path` can point to any compatible JSON (e.g., unlabeled pool).

---

## 5. Active learning workflow

1. **Prepare unlabeled pool** – JSON with SynSpERT schema (relations may be empty).
2. **Uncertainty inference**:
   ```bash
   python 'NER&RE_model/SynSpERT/main.py' --mode eval \
     --dataset_path diabetes_unlabeled.json \
     --model_dir 'NER&RE_model/InputsAndOutputs/data/save/diabetes_small_run/<timestamp>/final_model' \
     --al_dump_dir Outputs/al_round_01 \
     --label diabetes_al_round01 \
     --bert_model GanjinZero/coder_eng_pp \
     --config_override 'NER&RE_model/SynSpERT/configs/config-coder.json' \
     --download_dir 'NER&RE_model/InputsAndOutputs/models' \
     --store_predictions
   ```
   Generates entropy tensors (`entropy_entities.pt`, `entropy_relation.pt`, `label_prediction.pt`, `pooler_output.pt`) under `Outputs/al_round_01/`.

3. **Sample selection**:
   ```bash
   python Active_learning.py \
     --dump_dir Outputs/al_round_01 \
     --unlabeled_json diabetes_unlabeled.json \
     --output_prefix round01 \
     --top_k 20 --sample_per_group 10 --beta 0.1 --gamma 0.1 \
     --ncentroids 20 --use_weights
   ```
   Produces:
   - `sampling_json_round01.json` (documents to annotate)
   - `sampling_text_round01.txt` (detokenised sentences)
   - `selected_indices_round01.pt` (+ optional analytics)

4. **Annotate & refresh** – merge annotated items back into the training JSON, rerun §2.3 and §3, repeat until convergence/budget exhaustion.

---

## 6. File reference

| Path / Script | Purpose |
|---------------|---------|
| `scripts/fetch_official_data.py` | Pull MDIEC zip from Hugging Face and extract locally |
| `NER&RE_model/SynSpERT/generate_input.py` | Convert `.ann/.txt` directories to SynSpERT JSON |
| `NER&RE_model/SynSpERT/generate_augmented_input.py` | Add linguistic features & produce train/valid/test splits |
| `NER&RE_model/SynSpERT/main.py` | Unified CLI for training (`--mode train`) and evaluation (`--mode eval`) with backbone selection |
| `NER&RE_model/SynSpERT/spert/spert_trainer.py` | Training loop; emits AL tensors when `--al_dump_dir` is set |
| `Active_learning.py` | Entropy + diversity sampler used during annotation rounds |
| `NER&RE_model/InputsAndOutputs/data/log/*` | Training/evaluation logs, metrics, HTML previews |
| `NER&RE_model/InputsAndOutputs/data/save/*` | Saved checkpoints (`config.json`, `pytorch_model.bin`, `tokenizer.*`, etc.) |

---

Feel free to tweak hyperparameters, backbone choices, or sampling heuristics. The current defaults prioritise a CPU-friendly end-to-end reproduction path using MDIEC + CODER++, while keeping the pipeline extensible for future datasets. Continuous active-learning rounds are recommended to be versioned (`al_round_01`, `al_round_02`, …) for clarity.***
