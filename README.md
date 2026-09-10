# Customer Sentiment Analysis using BERT

An end-to-end NLP project that turns Yelp customer reviews into **Negative**, **Neutral**, or **Positive** sentiment. It includes reproducible data preparation, exploratory analysis, a TF-IDF + Logistic Regression benchmark, BERT fine-tuning, rigorous held-out evaluation, error exports, and a FastAPI prediction service.


## Business problem and ML objective

Companies receive far more reviews than people can read manually. Sentiment classification summarizes product and service feedback, helps support teams triage negative feedback, and lets product teams monitor customer experience.

The model input is one review string. The output is one of `Negative`, `Neutral`, or `Positive`, plus a model confidence score. This is a supervised three-class text-classification problem. It is **not** a measure of customer satisfaction beyond the text presented, and confidence is not a guarantee of correctness.

## Dataset

The pipeline downloads [Yelp Review Full](https://huggingface.co/datasets/Yelp/yelp_review_full), a public dataset supplied through Hugging Face Datasets. It contains 700,000 Yelp reviews: 650,000 in the original training split and 50,000 in the original test split. Its relevant columns are `text` (review body) and `label` (a zero-indexed 1–5 star rating).

For this business framing, labels 1–2 become Negative, 3 becomes Neutral, and 4–5 become Positive. The resulting class counts, missing values, duplicate count, and final train/validation/test distribution are calculated after cleaning by `src.prepare_data`; do not copy unverified counts into a report.

## Architecture

```text
Yelp reviews
    ↓
Light cleaning + remove empty/duplicate reviews
    ↓
Stratified train (80%) / validation (10%) / test (10%) split
    ├── TF-IDF → Logistic Regression baseline
    └── BERT tokenizer → BERT fine-tuning
                         ↓
          held-out test metrics, report, confusion matrix, errors
                         ↓
                    saved model/tokenizer
                         ↓
                      FastAPI → sentiment prediction
```

## Project layout

```text
├── api/                 # FastAPI service
├── data/                # download instructions; generated data is ignored by Git
├── models/              # generated baseline and BERT artifacts
├── results/             # generated metrics, plots, reports, and errors
├── scripts/             # optional controlled tuning trials
├── src/                 # preparation, EDA, training, evaluation modules
├── requirements.txt
└── README.md
```

## Installation and a practical run

Python 3.10+ is recommended (the local Python 3.9 launcher may work but current PyTorch releases increasingly require newer Python). Create a virtual environment, activate it, and install dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start with a 12,000-review stratified sample. It validates the full workflow on a typical laptop; BERT is substantially faster with a GPU. Remove `--limit` only when you have the storage/time/compute for the full dataset.

```powershell
python -m src.prepare_data --limit 12000
python -m src.eda
python -m src.train_baseline
python -m src.evaluate --model baseline
python -m src.train_bert --epochs 3 --batch-size 16 --max-length 256
python -m src.evaluate --model bert
```

For Google Colab, select **Runtime → Change runtime type → T4 GPU**, then run:

```python
!git clone <YOUR-GITHUB-REPOSITORY-URL>
%cd customer-sentiment-bert
!pip install -q -r requirements.txt
!python -m src.prepare_data --limit 12000
!python -m src.eda
!python -m src.train_baseline
!python -m src.evaluate --model baseline
!python -m src.train_bert --epochs 3 --batch-size 16 --max-length 256
!python -m src.evaluate --model bert
```

`bert-base-uncased` is downloaded on the first BERT run. CPU fine-tuning is technically supported but normally slow; use the small sample first or Colab GPU. The preparation script fixes random seed 42 and uses stratification, so every sentiment class remains represented in all splits.

## EDA and preprocessing

`python -m src.eda` records missing values, duplicates, class counts, and character/word-length summaries in `results/eda_summary.json`. It also produces class-distribution and review-length charts.

Cleaning is intentionally minimal:

```python
frame["review"] = frame["review"].map(clean_text)  # whitespace normalization only
frame = frame.dropna(subset=["review"]).drop_duplicates(subset=["review"])
frame = frame.loc[frame["review"].str.len().gt(0)]
```

Aggressively removing punctuation, stop words, or negations would discard useful context such as “not good”. `star_to_sentiment` provides explicit label encoding, while `make_splits` performs the stratified 80/10/10 split.

## Baseline versus BERT

TF-IDF + Logistic Regression treats a review as weighted word/phrase features. It is fast, interpretable, and a necessary benchmark, but it has limited context. An LSTM learns ordered word sequences, but typically requires more task-specific training and still processes text sequentially. BERT is pretrained bidirectionally on large text corpora and its self-attention can use words on both sides of a term. Fine-tuning adapts those pretrained representations to these three labels with comparatively little task-specific architecture.

BERT does not receive raw words directly. Its tokenizer creates subword `input_ids` and an `attention_mask`. `[CLS]` begins the sequence and contributes the classification representation; `[SEP]` marks its end (and separates paired sequences). Batches use padding so tensors have equal dimensions; the attention mask prevents padded positions from affecting attention. Truncation caps unusually long reviews at `max_length` (256 by default), balancing retained context and memory/time.

The implementation is in `src.train_bert`:

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
tokenizer(batch["text"], truncation=True, max_length=256)
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
)
```

## Evaluation, tuning, and error analysis

Both models are selected using validation data and evaluated once on the untouched test split. The scripts save accuracy, macro precision, macro recall, and macro F1. Macro scores average per-class performance, preventing large positive/negative classes from concealing poor neutral performance. They also save a classification report and confusion matrix.

Use this table **after** running both evaluation commands:

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | Read `results/baseline_test_metrics.json` | | | |
| BERT | Read `results/bert_test_metrics.json` | | | |

Do not fill the table until the JSON files exist. `results/*_errors.csv` contains the 100 highest-confidence wrong predictions for manual review. During review, tag patterns such as ambiguous wording, sarcasm, negation, very short reviews, and long mixed-opinion reviews; explain them as observed cases, not generic claims.

### Verified local smoke test — do not use as a portfolio result

To validate the complete pipeline on this CPU-only machine, a balanced 600-review public-data sample was split into 480/60/60 reviews. The baseline used the project defaults. The BERT run used `prajjwal1/bert-tiny`, 2 epochs, batch size 32, and maximum length 128 solely because a full `bert-base-uncased` run is not practical on CPU. Both were evaluated on the same untouched 60-review test split:

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.6333 | 0.4747 | 0.5278 | 0.4948 |
| BERT-tiny smoke test | 0.2500 | 0.1682 | 0.2083 | 0.1853 |

This run confirms that the code executes; it does **not** demonstrate BERT improvement. The result is expected to be unstable with only 480 training examples and a compact model trained for two epochs. A genuine portfolio comparison requires the documented larger GPU run (at least the 12,000-review stratified sample) and then an honest replacement of this table using its generated test JSON.

`scripts/run_hyperparameter_trials.py` explores learning rate, epochs, maximum length, and weight decay. Run it only on a GPU and a constrained sample, then record each validation macro F1 and choose the simplest configuration that generalizes. Batch size is limited by GPU memory. The default configuration is a defensible starting point, not a claim that it is optimal.

## API

Train BERT first, then start the service:

```powershell
uvicorn api.main:app --reload
```

Call it from a second terminal:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/predict `
  -ContentType 'application/json' `
  -Body '{"text":"The product quality is excellent and delivery was fast."}'
```

Response shape:

```json
{"sentiment":"Positive","confidence":0.0}
```

The numeric confidence shown above is a schema example, not a model result. `GET /health` returns `model_not_found` until a model exists; `POST /predict` returns HTTP 503 in that case and validates empty/oversized input.

## Interview-ready summary

“I built a three-class customer-review classifier using Yelp Review Full. I mapped star ratings to negative, neutral, and positive, applied minimal text cleaning, and used a stratified 80/10/10 split. I benchmarked TF-IDF with Logistic Regression before fine-tuning `bert-base-uncased`. I chose BERT because its bidirectional pretrained representations capture context and negation better than bag-of-words features. I selected configurations on validation macro F1, evaluated once on a held-out test set, exported confusion matrices and errors, and served the saved tokenizer/model through FastAPI.”

Quick answers:

1. **What is fine-tuning?** Continuing training of pretrained BERT on labelled reviews so its weights adapt to this task.
2. **Why not LSTM?** It is a useful sequential baseline, but BERT has stronger bidirectional pretrained language knowledge and parallel attention.
3. **What are attention masks?** Ones identify real tokens; zeros prevent padding from influencing attention.
4. **How was imbalance handled?** Stratified splits preserve proportions; the baseline uses balanced class weights; macro F1 evaluates all classes equally.
5. **Why F1?** It combines precision and recall and is more informative than accuracy when class sizes differ.
6. **How was overfitting controlled?** Validation-based best-checkpoint selection, weight decay, limited epochs, and final evaluation on data never used for tuning.
7. **How does FastAPI communicate with the model?** The endpoint tokenizes request text, runs a no-gradient forward pass, applies softmax, and returns the highest-probability label.
8. **What would you improve?** Calibrate confidence, test domain transfer, add human error tags, run systematic tuning, and add monitoring after deployment.

For the remaining interview questions: describe the actual choices and actual generated metrics; never state an accuracy that you have not reproduced.

## Resume bullets (fill after running)

- Built an end-to-end three-class customer-review sentiment pipeline using Hugging Face BERT and PyTorch, from public-data preparation through FastAPI inference.
- Benchmarked fine-tuned BERT against TF-IDF + Logistic Regression using a stratified held-out test set and macro precision, recall, and F1.
- Add your verified test result only after running: “Achieved **[actual macro F1 / accuracy]** on **[actual evaluation split and dataset sample]**.”

## Completion checklist

- [x] Problem definition, data source, mapping, reproducible cleaning and splits
- [x] EDA, baseline, BERT fine-tuning, tuning runner, evaluation and error-export code
- [x] FastAPI service, installation/run instructions, architecture, interview and resume guidance
- [x] Install dependencies, download public data, and complete a 600-review CPU smoke run
- [x] Run baseline/BERT training and held-out test evaluation on that smoke run
- [ ] Run the documented larger GPU experiment, review its actual errors, and replace the resume placeholder with those verified metrics
- [ ] Optionally commit generated figures only if the dataset licence/policy permits; never commit raw review data

