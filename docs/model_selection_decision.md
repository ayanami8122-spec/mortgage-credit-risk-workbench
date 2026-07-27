# Model Selection Decision / 模型选择决策记录

## Decision / 当前决策

**LightGBM is included as a reproducible third challenger. XGBoost remains the current policy-model candidate because it has the strongest OOT ranking metrics; LightGBM remains valuable evidence of a stable alternative.**

**LightGBM 已作为可复现的第三个挑战者纳入。由于 XGBoost 的时间外排序指标仍略高，策略层暂使用 XGBoost 作为候选模型；LightGBM 仍提供了有价值的稳定替代方案证据。**

## Comparable OOT evidence / 可比较的时间外证据

All three models use the same point-in-time snapshot, bad definition, `2024-04-01` OOT cutoff, 750,000-row stratified training cap and 131,334-loan OOT test population.

三个模型使用相同的时点快照、坏样本定义、`2024-04-01` OOT 切分点、75 万条分层训练上限和 131,334 笔贷款的 OOT 测试集。

| Model / 模型 | Role / 角色 | OOT AUC | OOT KS | PR-AUC | Brier score / Brier 分数 | Train-to-OOT PSI |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Logistic regression / 逻辑回归 | Transparent baseline / 可解释基准 | 0.841 | 0.522 | 0.252 | 0.00699 | N/A |
| XGBoost / XGBoost | Current policy-model candidate / 当前策略候选 | 0.857 | 0.540 | 0.264 | 0.00691 | 0.267 |
| LightGBM / LightGBM | Third challenger / 第三个挑战者 | 0.853 | 0.535 | 0.254 | 0.00698 | 0.187 |

LightGBM is close to XGBoost but does not exceed it on AUC, KS, PR-AUC or Brier score in this run. Its score PSI is lower, which is a useful stability signal but does not by itself justify replacing XGBoost. The current evidence supports retaining both in the comparison while keeping XGBoost scores in the strategy simulation.

本次运行中，LightGBM 接近 XGBoost，但在 AUC、KS、PR-AUC 与 Brier 分数上均未超过它。LightGBM 的分数 PSI 更低，这是有价值的稳定性信号，但不足以单独构成替换 XGBoost 的理由。因此保留两者的比较，同时在策略模拟中继续使用 XGBoost 分数。

## Why LightGBM strengthens the project / 为什么加入它能增强项目

| Dimension / 维度 | What it adds / 它增加了什么 |
| --- | --- |
| Model governance / 模型治理 | The comparison is now baseline vs two independently implemented gradient-boosting challengers, not a one-model claim / 现在是基准模型与两个独立树提升挑战者的比较，而不是单模型结论 |
| Reproducibility / 可复现性 | It runs in a pinned Python 3.10 environment documented in `environment.yml` / 在 `environment.yml` 记录的固定 Python 3.10 环境中运行 |
| Decision discipline / 决策纪律 | The selected candidate is based on a common OOT protocol, rather than model popularity / 候选模型基于统一 OOT 协议选择，而非算法流行度 |
| Stability discussion / 稳定性讨论 | It creates a concrete trade-off: XGBoost ranks slightly better; LightGBM has lower score PSI / 形成具体取舍：XGBoost 排序略强，LightGBM 分数 PSI 更低 |

## Environment record / 环境记录

The original base Python 3.11 environment used PyPI LightGBM `4.7.0`; it imported successfully but failed during `fit()` with a Windows native access violation. The working challenger environment is `fin_risk`, using Python 3.10 and LightGBM `4.6.0`. This isolation avoids changing the Streamlit base environment and makes the workaround explicit.

原 base Python 3.11 环境使用 PyPI LightGBM `4.7.0`，可导入但在 `fit()` 时触发 Windows 原生访问错误。当前可工作的挑战者环境是 `fin_risk`，使用 Python 3.10 与 LightGBM `4.6.0`。隔离环境避免改动 Streamlit 的 base 环境，也使处理方式透明可复现。

## How to reproduce / 如何复现

```powershell
conda env create -f environment.yml
conda run -n fin_risk python src/lightgbm_oot_challenger.py
```

The script writes its own metrics, OOT scores, PSI, feature importance, bilingual figures and a LightGBM challenger report under `reports/`. Generated reports are intentionally excluded from GitHub; the comparable metrics and the final selection rationale are retained in this document and in the [model card](model_card.md).

脚本会在 `reports/` 下输出自己的指标、OOT 分数、PSI、特征重要性、中英双语图表和 LightGBM 挑战者报告。生成报告刻意不上传 GitHub；可比较指标与最终选择理由保留在本文档和[模型卡](model_card.md)中。

## Interview-ready explanation / 面试可用说法

> 我用逻辑回归做可解释基准，并在完全相同的时间外测试协议下比较 XGBoost 和 LightGBM。XGBoost 的 AUC 0.857、KS 0.540，略高于 LightGBM 的 0.853 和 0.535，因此当前策略模拟使用 XGBoost；但 LightGBM 的 score PSI 为 0.187，低于 XGBoost 的 0.267，说明模型选择不能只看 AUC。我保留 LightGBM 作为可复现挑战者，并持续比较性能、校准与稳定性。
