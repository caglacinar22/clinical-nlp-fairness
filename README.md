# Equity-Constrained Active Learning for Clinical NLP

A clinical NLP system that classifies medical transcriptions by specialty,
with an equity-constrained active learning pipeline that ensures fair
annotation budget distribution across medical specialties.

## Motivation
Standard active learning picks the most uncertain samples to annotate.
This sounds optimal, but it systematically ignores underrepresented
specialties — the model becomes confident on common cases and blind
on rare ones. This project adds an equity constraint to the annotation
selection process.

## Approach
1. **EDA** — explore medical transcription dataset, specialty distributions
2. **Preprocessing** — clean notes, encode labels, train/test split
3. **Baseline Model** — fine-tune DistilBERT on clinical notes
4. **Active Learning** — standard uncertainty sampling vs equity-constrained sampling
5. **Fairness Audit** — per-specialty accuracy comparison

## Key Results
| Metric | Standard AL | Equitable AL |
|--------|-------------|--------------|
| Overall Accuracy | 0.220 | 0.140 |
| Variance across specialties | 0.090 | 0.054 |

> Equitable AL reduced variance across specialties by 40%, meaning fairer
> performance distribution — with a trade-off in overall accuracy due to
> the equity constraint forcing coverage of underrepresented specialties.

## Dataset
MTSamples — 5,000 real medical transcriptions across 40+ specialties

## Tech Stack
- PyTorch
- HuggingFace Transformers (DistilBERT)
- FairLearn
- scikit-learn

## Project Structure
```
├── data/
│   ├── raw/               # original mtsamples.csv
│   └── processed/         # cleaned, split data
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_active_learning.ipynb
│   └── 05_fairness_audit.ipynb
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── fairness.py
│   └── train.py
└── results/
    └── figures/
```

## Author
Cagla CINAR — BSc Computer Engineering, Politecnico di Torino