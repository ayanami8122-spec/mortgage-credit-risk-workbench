# Data Dictionary and Analysis Scope / 数据字典与分析范围

## Data source and grain / 数据来源与粒度

- Source / 来源: UCI Machine Learning Repository, *Default of Credit Card Clients*.
- Grain / 粒度: one credit-card customer record at the observation point; `ID` is the customer identifier. 每条记录对应观察时点的一名信用卡客户，`ID` 为客户标识。
- Target / 目标变量: `default_payment_next_month`; 1 means default in the following month and 0 means non-default. 取值 1 表示下月违约，0 表示未违约。
- Time limitation / 时间限制: the source provides six months of historical repayment, bill, and payment information but no calendar date or application date. It supports cross-sectional analysis, not a real production time-series monitoring conclusion. 数据支持横截面分析，不支持真实生产时序监控结论。

## Field groups / 字段分组

| Group / 分组 | Fields / 字段 | Business meaning / 业务含义 | Quality and analysis treatment / 质量与分析处理 |
| --- | --- | --- | --- |
| Identifier / 标识 | `ID` | Customer record identifier / 客户记录标识 | Check uniqueness; exclude from analysis and future models. 核查唯一性；从分析与未来模型中排除。 |
| Credit limit / 授信额度 | `LIMIT_BAL` | Approved credit limit / 批准授信额度 | Positive monetary field; use quartiles for segmentation. 正值金额字段，按四分位做分层。 |
| Customer profile / 客户属性 | `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` | Demographic and background attributes / 人口与背景属性 | Recode undocumented education/marriage values as other/unknown. Use carefully in a real lending context due to fairness and regulatory requirements. 未文档化类别归为其他/未知；真实信贷场景须审查公平性与合规性。 |
| Repayment status / 还款状态 | `PAY_0`, `PAY_2`-`PAY_6` | Monthly repayment status from most recent to older months / 从最近到较早月份的还款状态 | Codes range from -2 to 8. Values >=1 are treated as delinquency indicators in this project. 取值范围为 -2 至 8，本项目将 >=1 视为逾期。 |
| Billing history / 账单历史 | `BILL_AMT1`-`BILL_AMT6` | Monthly statement balance / 月度账单余额 | Negative amounts can be a credit balance, not necessarily an error. Profile but do not remove automatically. 负数可能为贷方余额，需画像但不自动删除。 |
| Payment history / 还款历史 | `PAY_AMT1`-`PAY_AMT6` | Monthly payment amount / 月度还款金额 | Zeros may reflect no payment or no amount due; interpret together with bill and repayment status. 零值需结合账单与还款状态解释。 |
| Outcome / 结果变量 | `default_payment_next_month` | Next-month default flag / 下月违约标识 | Use only as the outcome; never include in input features. 仅作为结果变量，不得作为输入特征。 |

## Derived analysis fields / 衍生分析字段

| Derived field / 衍生字段 | Definition / 定义 | Purpose / 目的 |
| --- | --- | --- |
| `age_segment` | <=25, 26-35, 36-45, 46-55, 56+ / 年龄区间 | Compare risk by age band / 比较年龄层风险。 |
| `limit_segment` | Quartiles of `LIMIT_BAL` / 授信额度四分位 | Compare risk by credit-limit band / 比较额度层风险。 |
| `max_delinquency_status` | Maximum of the six `PAY_*` fields / 六期 `PAY_*` 的最大值 | Summarize the worst observed repayment status / 汇总最严重还款状态。 |
| `delinquent_month_count` | Number of `PAY_*` fields >=1 / `PAY_*` 中 >=1 的次数 | Measure persistence of delinquency / 衡量逾期持续性。 |
| `recent_delinquency_status` | `PAY_0` / 最近一期还款状态 | Retain the most recent repayment signal / 保留近期还款信号。 |

## Boundaries / 使用边界

This document records analysis conventions for a public dataset. It is not a production credit policy, and its demographic segmentation is descriptive only. Any real lending use would require legal, fairness, privacy, sample-period, and validation review.

本文档记录公开数据集的分析约定，不构成生产信贷政策；人口属性分层仅用于描述。任何真实信贷应用都需要经过法律、公平性、隐私、样本时间范围和模型验证审查。
