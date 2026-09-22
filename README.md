# DURel on English: transferring Diachronic Usage Relatedness to TempoWiC

**Raja Kishore Reddy Talakola — Matriculation Number 1866163**

A full replication/extension of **Schlechtweg, Schulte im Walde & Eckmann (2018),
*DURel: A Framework for the Annotation of Lexical Semantic Change*** (arXiv:1804.06517),
carried out on the **English TempoWiC** benchmark instead of the German DTA corpus.

The original study annotated 1,320 German use pairs for 22 target words with five
human annotators. Here, English Twitter data replaces German historical prose and
relatedness models replace the annotators, while the framework itself — use pairs,
the EARLIER / LATER / COMPARE groups, the 4-point relatedness scale, and the
ΔLATER / COMPARE measures — is kept intact.

## What is in this project

```
data/          TempoWiC (from github.com/cardiffnlp/TempoWiC), incl. released gold test labels
src/           the full pipeline (see below)
results/       every generated result file (JSON/CSV/NPZ)
figures/       all figures (PDF + PNG)
report/        LaTeX source, generated tables, and the compiled report (main.pdf)
```

## Pipeline

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `src/data_prep.py` | Loads TempoWiC, treats every instance as a COMPARE pair, samples 100 EARLIER and 100 LATER pairs per target word → `results/pairs.jsonl` |
| 2 | `src/extract_encoder.py` | Extracts a RoBERTa-base encoder + byte-level BPE tokenizer from the spaCy `en_core_web_trf` package into HF format (needed because the run environment has no model-hub access) |
| 3 | `src/encode.py` | Contextual vector of the target word in every unique use, at all 13 encoder layers |
| 4 | `src/zeroshot.py` | Unsupervised relatedness: per-layer target cosine, sentence cosine, Jaccard, random; thresholds tuned on validation |
| 5 | `src/finetune.py` | Fine-tunes a RoBERTa cross-encoder on the TempoWiC training split (multiple seeds) |
| 6 | `src/score_all.py` | Scores all 10,097 use pairs with the fine-tuned model |
| 7 | `src/durel.py` | Mean relatedness per word and group, ΔLATER / COMPARE / ΔCOMPARE, agreement matrix, correlation with the human change rate |
| 8 | `src/simlr.py` | Logistic regression over the similarity features (the shared task's strongest baseline family) |
| 9 | `src/figures.py`, `src/make_tables.py` | All figures and all LaTeX tables + the macros for every number quoted in the report |
| 10 | `src/verify.py` | Independent re-computation of the headline numbers as a check (42 checks) |

Run in that order:

```bash
python3 src/data_prep.py
python3 src/extract_encoder.py
python3 src/encode.py
python3 src/zeroshot.py
python3 src/finetune.py --seed 42
python3 src/score_all.py
python3 src/durel.py
python3 src/figures.py && python3 src/make_tables.py
cd report && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Design in one paragraph

Every TempoWiC instance pairs a tweet from year *Y* with a tweet from year *Y*+1
containing the same target word, labelled *same meaning* / *different meaning*.
That is exactly DURel's **COMPARE** group. The **EARLIER** and **LATER** groups do
not exist in TempoWiC, so they are re-sampled from the same pool of tweets, 100
pairs per word per period, as DURel prescribes. Relatedness scores are mapped onto
DURel's 1–4 scale (`rel = 1 + 3·p(same meaning)`), and the word-level measures are
computed exactly as in the paper.

## Headline findings

**TempoWiC test set (macro-F1, official metric)**

| System | macro-F1 |
|---|---|
| Random / majority | 50.6 / 38.8 |
| Lexical overlap (Jaccard) | 53.5 |
| Zero-shot target-word cosine, layer picked on dev (L5) | 63.6 |
| Zero-shot target-word cosine, last layer (L12) | 68.0 |
| Zero-shot target-word cosine, oracle layer (L8) | 70.3 |
| Logistic regression over similarity features | 65.4 |
| Fine-tuned cross-encoder (3 seeds) | 61.9 ± 1.4 |

Supervision does **not** help: train and test splits share no target words, so
supervised models fit word-specific cues that don't transfer. The official
shared-task baselines show the same ordering (similarity LR 70.3 vs fine-tuned
RoBERTa-large 59.1; winning system 77.1).

**DURel measures vs. the human change rate (34 words, Spearman)**

| Measure | ρ |
|---|---|
| COMPARE (dev-selected model) | **−0.61** (p = 0.0001) |
| COMPARE (best model, similarity LR) | **−0.68** (p < 0.001) |
| ΔCOMPARE | −0.23 (n.s.) |
| ΔLATER | +0.34 |

* **COMPARE transfers.** Low COMPARE = strong change, exactly as the framework predicts,
  now validated numerically against human judgements — something the German study could not do.
* **ΔLATER does not.** 31 of 34 words come out "reductive", because in 2020–2021 words like
  *mask*, *virus*, *delta*, *epicenter* became topically concentrated. DURel reads increased
  within-period homogeneity as meaning loss; what actually happened is a shift in topic prevalence.
* **Models are weaker annotators than people.** Unsupervised models agree with each other up to
  ρ = 0.84 but only ρ ≤ 0.36 with the gold labels, against ρ = 0.66 between the five human
  annotators in the original study.

See `report/main.pdf` for the full write-up, tables and figures.

## Reproducibility note

No number in the report is typed by hand: `make_tables.py` writes every table and
every in-text macro straight from the files in `results/`.
