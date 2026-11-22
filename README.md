# CamoAug with TAPL: Multi-Class Fraud Detection under Camouflage Drift

Code for the paper:

> **A Novel Camouflage Augmentation with Trust-Aware Pseudo-Labeling Method for Enhancing Model Robustness in Multi-Class Fraud Detection under Data Drift**
> Izra Noor Zahara Aliya, Riyanarto Sarno, Kelly Rossa Sungkono, Agus Tri Haryono, Ratih Nur Esti Anggraini, Sholiq.
> *Intelligent Automation & Soft Computing* (submitted).

The repository contains one Jupyter notebook that reproduces every table and figure of the paper, from loading the
data to the statistical tests.

## Problem

Fraudsters change their behaviour so that one fraud class gradually resembles another class (*camouflage drift*).
A model trained on historical data then misclassifies these transactions with high confidence, while the labels of
recent transactions are not yet available for retraining. Confidence-based self-training reinforces these errors.

## Method

The method has two stages:

1. **CamoAug (camouflage augmentation).** Before deployment, synthetic training samples are created by masking the
   features that best distinguish each class (Kolmogorov–Smirnov statistic on TRAIN), so that the model can still
   recognise a class after its markers disappear. It uses KS-guided masking, worst-of-n donors, fragility-based
   budget allocation and an evidence filter.
2. **TAPL (trust-aware pseudo-labeling).** During deployment, unlabeled transactions arrive in batches. Pseudo-labels
   are filtered with per-class confidence thresholds and a veto, wrong predictions are corrected with a
   Neyman–Pearson test (NP-Recovery, false positive rate 5 × 10⁻⁴), and a budgeted logit bias (BLB) is applied.
   Retraining uses an anchor blend, a per-class cap and a VALIDATION gate.

No label of the test stream is used by any method, neither for adaptation, calibration nor hyper-parameter
selection. Stream labels are used only for evaluation and, in the simulated scenarios, by the simulator to select the
rows that it modifies.

## Dataset

Public synthetic dataset on Kaggle:
**https://www.kaggle.com/datasets/mohamedadeldesak/enterprise-fraud-dataset**

- 200,000 transactions, 44 columns, six classes: `legit`, `identity_theft`, `account_takeover`, `card_testing`,
  `fraud_ring`, `synthetic_identity`.
- Chronological split: TRAIN 1 Jan – 20 Mar 2025, VALIDATION 21 – 31 Mar 2025, test stream 1 – 29 Apr 2025
  (10 batches of 5,000 transactions).
- The dataset is not included in this repository; the notebook loads it from Kaggle.

## How to run

### On Kaggle (recommended; used for the paper)

1. Create a new Kaggle notebook and upload `camoaug_tapl_fraud_detection.ipynb`.
2. Add the dataset with **+ Add Input** (search `enterprise-fraud-dataset` by mohamedadeldesak), or turn on
   **Settings → Internet** so that the notebook downloads it with `kagglehub`.
3. Select **Accelerator → GPU T4**.
4. **Save & Run All**. A full run takes about 3.5 hours on a Tesla T4.

### Locally

```bash
pip install numpy==2.1.3 pandas==2.3.3 xgboost==3.4.1 scikit-learn==1.6.1 scipy==1.16.3 matplotlib==3.10.0 kagglehub
```

Download the dataset from Kaggle and set `DATA_PATH` in the dataset cell to the CSV file or its folder, then run all
cells. Without a CUDA GPU, XGBoost runs on the CPU, which is much slower.

`QUICK_TEST = True` in the configuration cell runs a fast smoke test of the code; its numbers are not meant for
reporting.

## Reproducibility

- Seeds 42–51 (10 seeds) for the main experiment; the seeds of every experiment are listed in the configuration cell.
- All hyper-parameters are fixed in the configuration cell before the experiments.
- The environment used for the paper:

  | Component | Version |
  |---|---|
  | Python | 3.13.16 |
  | numpy | 2.1.3 |
  | pandas | 2.3.3 |
  | xgboost | 3.4.1 |
  | scikit-learn | 1.6.1 |
  | scipy | 1.16.3 |
  | matplotlib | 3.10.0 |
  | GPU | NVIDIA Tesla T4 |

- The notebook prints a warning if the installed versions differ. Results are bit-identical only with the same
  library versions and GPU type; on other hardware they can differ in the third or fourth decimal.
- Two independent runs on Kaggle produced identical results.

## Outputs

The notebook writes:

| Path | Content |
|---|---|
| `tables/` | every table of the paper and the supporting results (CSV) |
| `figures/` | every figure of the paper (PNG at 300 dpi and PDF) |
| `env.json`, `config.json` | library versions, GPU and the full configuration |
| `tables/check_manuscript_vs_this_run.csv` | the main numbers of the paper next to the values of the current run |
| `camoaug_tapl_journal_outputs.zip` | all of the above in one archive |

## Main results

Test stream (April 2025), mean over 10 seeds:

| Method | Macro-F1 | F1 `identity_theft` |
|---|---|---|
| A1 XGBoost without augmentation | 0.911 | 0.582 |
| R0 CamoAug (static model) | 0.946 | 0.784 |
| Best competing retraining method (R3 FreeMatch-style) | 0.931 | 0.697 |
| **R6 CamoAug with TAPL** | **0.954** | **0.856** |

The notebook also contains the statistical tests (Friedman–Nemenyi, Wilcoxon with Holm correction, bootstrap
confidence intervals), the mechanism analysis, the ablations, the robustness and simulated drift scenarios, the
sensitivity analysis and a leakage audit.

## Citation

If you use this code, please cite the paper (details will be added after publication).

## Acknowledgements

This research was supported by the RIIM BRIN and LPDP Grants (grant number 2402/PKS/ITS/2026) and by the Asian
Development Bank together with the Directorate General of Higher Education, Ministry of Higher Education, Science,
and Technology of the Republic of Indonesia through the Frontiers Scheme 2026 (Contract No. 0057/01.PKS/PPK-HETI/ITS/2026).
The authors thank Institut Teknologi Sepuluh Nopember.

## License

To be added.
