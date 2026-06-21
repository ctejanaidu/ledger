# Examples

- **`sample_lending_report.html`** — a real executive report Ledger generated for the
  lending demo. Download and open it in a browser; use **Print → Save as PDF** for the
  PDF version (charts render correctly that way).
- **`sample_cli_output.txt`** — a trimmed transcript of `python run.py` on the lending
  dataset, showing the findings, diagnostician, projections, validator guardrails, BLUF
  summary, and grounded Q&A.

## Datasets (not committed)

`data/*.csv` is gitignored to keep the repo lean. To reproduce:

- **Lending demo** (synthetic, regenerable): `python -m data.generate_lending`
- **Credit-card fraud** (the imbalanced showcase): download `creditcard.csv` from the
  Kaggle "Credit Card Fraud Detection" dataset and drop it in `data/`, then
  `python run.py data/creditcard.csv "what is driving fraud?"`
