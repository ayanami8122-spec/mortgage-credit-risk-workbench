# GitHub Release Checklist / GitHub 发布检查清单

## Included in the repository / 仓库应包含

- Source code under `src/`, the Streamlit entry point `app.py`, tests, `README.md`, `environment.yml` and `requirements.txt`.
- Documentation under `docs/`, including the model card, data-source notes and model-selection record.
- Four compact, non-loan-level figures in `assets/` so GitHub renders the project without generated reports.

## Deliberately excluded / 刻意排除

- `data/raw/`: Fannie Mae downloads are large and subject to source terms.
- `data/processed/`: derived loan-level snapshots, scores and monitoring tables are reproducible from raw files and should not be published by default.
- `reports/`: generated reports and full figure set are reproducible; selected non-sensitive previews live in `assets/`.
- Local logs, virtual environments, caches and secrets.

## Before the first push / 首次推送前

1. Confirm that no file under `data/` is staged: `git status --short`.
2. Review the staged content: `git diff --cached --stat` and `git diff --cached`.
3. Run `pytest -q` and the Streamlit smoke test documented in the project notes.
4. Create an empty GitHub repository, add its remote URL, then push the default branch.
5. In the GitHub repository description, use: `Out-of-time mortgage PD modelling, decision strategy simulation and portfolio monitoring with Fannie Mae performance data.`

## Suggested repository topics / 建议标签

`credit-risk`, `risk-analytics`, `xgboost`, `lightgbm`, `streamlit`, `model-monitoring`, `data-quality`, `fintech`

## Suggested first release title / 建议首个发布标题

`v1.0 - OOT PD Modelling and Risk Monitoring Workbench`

The release description should state that the repository contains code and selected figures only; users must independently obtain the Fannie Mae source files and accept its terms. 发布说明应明确：仓库只包含代码和精选图表，使用者需自行获取 Fannie Mae 源文件并接受其使用条款。
