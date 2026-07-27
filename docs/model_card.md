# Model Card: Mortgage 90+ DPD PD Model / 模型卡：按揭贷款 90+ DPD PD 模型

## Model identity / 模型身份

| Item / 项目 | Detail / 说明 |
| --- | --- |
| Project / 项目 | Credit Risk Workbench / 信贷风险评分与策略监控平台 |
| Current strategy candidate / 当前策略候选 | XGBoost probability-of-default model / XGBoost 违约概率模型 |
| Alternatives / 对照模型 | Logistic regression baseline; LightGBM challenger / 逻辑回归基准；LightGBM 挑战者 |
| Status / 状态 | Portfolio demonstration, not production deployment / 作品集演示，非生产部署 |
| Prediction target / 预测目标 | 90+ DPD during months 7-18 after origination / 起源后第 7-18 月内是否发生 90 天以上逾期 |

## Intended use and exclusions / 预期用途与排除项

The model is intended to rank mortgage-loan risk and support a hypothetical pass/manual-review/reject simulation. It is **not** a complete underwriting system, a legal lending decision, a fairness assessment, a collection model or a loss forecast.

该模型用于按揭贷款风险排序，并支持假设性的通过/人工审核/拒绝策略模拟。它**不是**完整授信审批系统、真实贷款法律决策、公平性评估、催收模型或损失预测模型。

## Data design / 数据设计

- **Source / 来源:** Fannie Mae Single-Family Loan Performance Data, four Q1 combined monthly files (2021-2024).
- **Eligibility / 入选条件:** Each loan snapshot needs a complete six-month feature window and a complete twelve-month outcome window.
- **Feature window / 特征窗:** Months-on-book 1-6. Example inputs include borrower credit score, original LTV/DTI/UPB, loan purpose, occupancy, property type/state, early maximum DPD, 30+ DPD months and unpaid-balance ratio.
- **Outcome window / 标签窗:** Months-on-book 7-18.
- **Bad definition / 坏样本定义:** at least one monthly `current_delinquency_status >= 3`, namely 90+ days past due.
- **Snapshot count / 快照量:** 2,015,703 eligible loan snapshots.

No future performance field is used as a feature. The observation date fixes what was known when the score would have been produced. 不使用任何未来表现字段作为特征；观察日期固定了评分当时实际可得的信息。

## Evaluation protocol / 验证协议

| Population / 样本 | Rule / 规则 | Purpose / 用途 |
| --- | --- | --- |
| Main training / 主训练 | Observation date before 2024-04-01; stratified cap 750,000 | Fit competing models / 拟合候选模型 |
| Final OOT / 最终时间外 | 2024Q2-2024Q3, 131,334 observations | Honest chronological evaluation / 诚实的时间外评估 |
| Calibration study / 校准研究 | Train before 2023Q3; use 2023Q3 only to fit mapping; evaluate later OOT | Check probability quality without label leakage / 无标签泄漏地检查概率质量 |

Chronological OOT evaluation is used instead of a random split because a real credit model scores a later population than the one it learned from. 使用时间外验证而非随机切分，因为真实信贷模型面对的是晚于训练期的新客群。

## Performance evidence / 性能证据

| Model / 模型 | OOT AUC | OOT KS | PR-AUC | Brier | Score PSI (Train to OOT) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic regression / 逻辑回归 | 0.841 | 0.522 | 0.252 | 0.00699 | N/A |
| XGBoost / XGBoost | **0.857** | **0.540** | **0.264** | **0.00691** | 0.267 |
| LightGBM / LightGBM | 0.853 | 0.535 | 0.254 | 0.00698 | **0.187** |

**Selection rationale / 选择理由:** XGBoost is currently used in strategy simulation because its OOT discrimination and Brier are the best in this experiment. LightGBM remains a challenger because it is close on performance and more stable by score PSI. The difference is evidence for ongoing governance, not a blanket claim that one model is universally superior.

**选择理由：** 本实验中 XGBoost 的 OOT 区分度和 Brier 最好，所以策略模拟当前使用它。LightGBM 因性能接近且 score PSI 更低而作为挑战者保留。这支持持续模型治理，而不是宣称某一模型在任何情况下都绝对更优。

## Calibration result / 校准结果

On the final OOT population, the raw XGBoost score had Brier `0.00693` and ECE `0.0007`; the isolated Isotonic calibration version had Brier `0.00694` and ECE `0.0010`. It was therefore **not adopted**. The raw probability is already well aligned at this evaluation scale, and the historical mapping did not transfer better to the later cohort.

在最终 OOT 人群上，原始 XGBoost 的 Brier 为 `0.00693`、ECE 为 `0.0007`；隔离 Isotonic 校准版本的 Brier 为 `0.00694`、ECE 为 `0.0010`，因此**没有采用**。在本次评估尺度下原始概率已较贴近实际，历史校准映射并未更好地迁移到后续客群。

## Policy linkage / 策略连接

The Streamlit simulator applies PD thresholds to the unchanged OOT population:

- **Pass / 通过:** PD at or below the lower threshold.
- **Manual review / 人工审核:** PD between lower and upper thresholds.
- **Reject / 拒绝:** PD above the upper threshold.

It reports non-reject rate, observed bad rate, exposure, booked expected loss and rejected expected loss avoided. `Expected loss = PD × exposure × assumed LGD`; LGD is a scenario parameter rather than a realized recovery observation.

## Monitoring and triggers / 监控与触发条件

| Monitor / 监控项 | Current convention / 当前约定 | Response / 响应 |
| --- | --- | --- |
| Score PSI / 分数 PSI | 0.10 review; 0.25 alert | Review score distribution and policy cutoffs / 复核分数分布与阈值 |
| Feature PSI / 特征 PSI | Same thresholds on key numeric and categorical features | Locate the changed customer or product mix / 定位变化的客群或产品结构 |
| Data quality / 数据质量 | Missing >2% or invalid >1% is alert | Stop/review data feed and field mapping / 停止或复核数据供给与字段映射 |
| Outcome performance / 结果表现 | Track bad rate, AUC, KS, Brier by later cohort | Investigate discrimination and calibration deterioration / 调查区分度和校准退化 |

Current train-to-OOT feature PSI flags unpaid-balance ratio (`1.878`) and loan purpose (`1.324`) as material movement, with LTV (`0.246`), DTI (`0.127`) and state (`0.106`) needing review. The 2024Q2-to-Q3 comparison is low, and configured quality rules have no alert. This is an investigation signal, not a causal conclusion.

当前训练集到 OOT 的特征 PSI 中，未偿余额比（`1.878`）和贷款用途（`1.324`）出现明显变化，LTV（`0.246`）、DTI（`0.127`）和州别（`0.106`）需要复核；2024Q2 到 Q3 的同口径变化较低，配置的数据质量规则也没有告警。这是调查信号，不是因果结论。

## Limitations and next improvements / 局限与下一步改进

1. The sample is a public mortgage-performance dataset, not the target lender's application population. 公共按揭表现数据不等同于目标机构的申请客群。
2. Q1 source-file selection creates observation-date gaps; it is not a full monthly production feed. 只选择 Q1 文件会产生观察日期间隔，不是完整的生产月度流水。
3. The label measures severe delinquency rather than realized economic loss; LGD is assumed. 标签是严重逾期而非实际经济损失，LGD 为假设。
4. No fairness, adverse-action or regulatory validation has been performed. 尚未进行公平性、拒绝原因或监管验证。
5. A stronger next stage would add quarterly source files, vintage-specific calibration, reject-inference considerations and formal threshold optimization against capacity and risk appetite. 下一阶段可增加季度源文件、vintage 校准、拒绝推断，以及面向审核产能与风险偏好的正式阈值优化。
