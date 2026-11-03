# ACE Formula 复现实验

本目录保存 FTEC5660 复现实验的准备代码和协议。上游方法、提示词、预处理及判分文件保持原样。
基线固定为 `82709de050e1db6e6ef2f07bcb0393560b94992a`；新增代码仅在运行时替换模型接口，
补上 DeepSeek 兼容参数、有限重试、费用日志和停止线。

## 当前阶段

- 已完成环境安装、原始数据结构和重复检查、样本顺序清单，以及本地模拟训练。
- 模拟结果只验证流程，不是模型效果、论文复现结果或正式试验。
- 两项改进已实现并通过离线检查；第二阶段六次学习与验证已完成，使用冻结的 60/30 数据。
- 第三阶段已封存并启动：五组 × 三个种子 × 100 个固定测试 ID，全部预测完成前不评分。
- 真实试跑已完成：空手册与 ACE 验证均为 6/10，未观察到提升。详情见 `artifacts/STAGE1.md`。
- 已按成本锁定正式规模 60/30/100；不根据测试成绩更换样本、模型或参数。

## 环境与离线验证

在仓库根目录运行：

```bash
UV_CACHE_DIR="$PWD/../.uv-cache" uv sync --python 3.12 --locked
.venv/bin/python -m repro.prepare
.venv/bin/python -m unittest discover -s repro/tests -v
.venv/bin/python -m repro.pilot --check
.venv/bin/python -m repro.pilot --mock
```

第一次加载 tokenizer 需要下载公开的 `cl100k_base` 词表；已下载后离线复用。
默认将词表缓存在工作区的 `.runtime-cache/tiktoken`，不会调用 embedding 模型。

## 配置真实试跑

本地 `.env` 已创建为空白模板，权限为 0600，且已被 Git 忽略。只在该文件中填入：

```dotenv
DEEPSEEK_API_KEY=你的实际密钥
```

不要把密钥发到聊天中、放入命令行参数、日志或提交记录。检查命令仅显示是否已配置，不显示内容。

```bash
.venv/bin/python -m repro.pilot --check
.venv/bin/python -m repro.pilot --live
```

真实试跑使用 10 条训练和 10 条验证，最多三轮错误反思，在训练第 5、10 步验证，
不加载测试记录。提示词、答案解析及数值匹配沿用官方实现，保持 `temperature=0`、
`max_tokens=4096`、思考模式关闭、并发为 1。模型默认 `deepseek-flash`，服务端返回的模型名另行记录。

每次 pilot 的费用记账停止线为 20 元，整个项目为 180 元。价格采用 2026-11-03 官方峰时美元价，
以 8 元/美元作为保守记账系数（并非实时汇率）；正式报告须区分估算和账单实付。
请求前记录输入预估与最大输出的费用预留；响应后按 API usage 更新，超时导致的未知费用继续保留。
本地分词估计不能严格保证服务商账单上限，停止线保留了距总预算 200 元的余量。
每次请求最多尝试三次，认证错误直接停止；空响应和缺失 usage 停止试跑。
真实诊断后，对非空截断输出改为保留上游处理行为，同时保存原文和 `finish_reason=length`。
截断不能当作完整响应，正式结果需单列截断率及其对解析、经验更新的影响。

输出保存在 Git 忽略的 `results/repro/`。`mock-*` 仅含合成响应；`pilot-*` 才是真实模型试跑。
重复执行 `--live` 会新建一次试跑，可能再次收费；不会自动启动正式实验。

## 数据边界与已知上游行为

- 数据：原始 train 500、validation 300、test 200，SHA-256 见 `artifacts/data_audit.json`。
- 固定样本顺序见 `artifacts/sample_manifest.json`；pilot 的 10+10 条与正式候选样本分开。
- 测试集仅做结构、哈希、重复检查，不显示内容或标签，不进入试跑。
- 主结果保留原始标签，疑似标签问题只作单独说明，不依模型输出改标签。
- 官方 Formula 解析器会移除题目前的公式说明；本次检查的全部 500 条训练和 300 条验证均走此分支。
  这是保留的上游预处理行为，不是我们新加的改进。
- 正确样本也会反思；`curator_frequency` 间隔更新不等于汇总之前的反思。
- 原始评估代码可能从分母排除异常项。本阶段 transport 的致命错误会直接停止，
  并检查无经验基线分母；后续正式评估入口必须固定测试分母并记录失败项。

## 后续协议

正式协议保存在 `configs/protocol.json`。试跑后根据实际调用、token 与手册增长，
加入 30% 余量，在三个预设规模中选择能完成三次重复和全部五组评估的最大一档。
目前已锁定 60/30/100，含 30% 余量的预测总记账约 95 元。测试前冻结所有方法与手册。
不得根据测试效果换样本、模型或调参数。

试跑汇总与费用复算（不会调用模型）：

```bash
.venv/bin/python -m repro.analyze_pilot
```

不要在预算或结果变化后反复调用 `--lock-size` 改动已冻结的样本规模。

参考：[ACE 论文](https://arxiv.org/abs/2510.04618v3)、
[DeepSeek 定价](https://api-docs.deepseek.com/quick_start/pricing/)、
[思考模式开关](https://api-docs.deepseek.com/guides/thinking_mode/)。


## 第二阶段：正式学习与方法实现

选择性反思只作用于有标签的训练阶段：初次答错保留全部原流程；初次答对且训练步数不是 5 的倍数，
跳过 Reflector、Curator 和重复答题。关闭开关时直接调用上游实现，离线对照验证调用顺序与手册一致。

检索只用于冻结后的答题输入。保存完整手册，用本地 TF-IDF 按问题选择最多 12 条，
标题、ID 和完整条目一同计入 1,500 tokens 上限。小手册在同时满足条目数和 token 上限时原样使用。
本阶段只检查检索的输入压缩，未测量它的真实答题质量。

```bash
# 只读进度，不会发起模型调用
.venv/bin/python -m repro.progress
# 当前批次结束后才可运行；已完成的学习任务会跳过，未完成任务按已保存响应回放
.venv/bin/python -m repro.batch_train --workers 3
# 六个任务全部完成后汇总并冻结手册，不调用模型
.venv/bin/python -m repro.summarize_training
```

三个训练顺序种子分别串行完成原始 ACE 与选择性反思，不在同一个学习序列内并行改写手册。
种子之间并行，费用预留使用跨进程文件锁。请求指纹包含逻辑位置、提示词、模型和参数；
已完成请求恢复时直接回放，结果不明且无响应记录的请求会停止，禁止盲目重新付费。

每次学习在第 30、60 步验证，分母固定为 30，按验证表现选择手册，平分保留较早检查点。
因此第二阶段准确率属于选模验证结果，第三阶段才能给出独立测试结论。
正式学习的单任务记账线为 20 元、最多 650 个逻辑调用，整个项目仍为 180 元。

下一阶段的固定测试方案见 `artifacts/STAGE3_PLAN.md`。


第二阶段结果见 `artifacts/STAGE2.md`，选定的六份手册原样保存在 `artifacts/books/`。
中途发生过三个明确的 HTTP 402 拒绝；余额只读查询显示可用后，使用 `repro.resume_balance`
恢复，509 个已完成调用直接回放。详情、初次失败状态和最终完成状态均保留。

`repro.balance` 只做账户只读查询，结果保存在被 Git 忽略的本地结果目录，不显示 key。
不要把“账户余额可用”与“所有 key 的额度状态都正常”混为一谈；本次余额读数与 402 不一致的根因未确认。

## 第三阶段：五组独立测试

测试输入只保留原始预处理后的问题与上下文，真实标签在预处理前替换为空字符串。
原始 ACE 与检索组使用同一份原始训练手册；选择性反思与组合组使用同一份选择性手册。
各组在种子 42/43/44 的队列中轮换执行顺序，条件内串行、跨队列并发上限为 3。
所有预测先落盘，十五个条件全部完成后才允许加载标签统一判分。

```bash
# 首次测试前封存；已有契约不得更换代码、输入或手册
.venv/bin/python -m repro.holdout_freeze
# 已完成条件直接跳过；中断恢复回放已保存的响应，不从头重跑
.venv/bin/python -m repro.holdout_batch --workers 3
# 盲态查看完成数和费用，不读取输出或成绩
.venv/bin/python -m repro.holdout_progress
# 以下命令必须等十五个真实条件全部完成
.venv/bin/python -m repro.holdout_analysis
.venv/bin/python -m repro.verify_stage3
.venv/bin/python -m repro.report_stage3
```

图表单独使用 `.plot-venv`，不修改原始 `uv.lock` 或主实验依赖。精确绘图版本写入
`artifacts/stage3_plot_environment.json`；绘图命令为 `.plot-venv/bin/python -m repro.plot_stage3`。
原始 API 响应和恢复缓存留在忽略的 `results/repro/`；开封后的逐题输出、统计与审计保存在
`artifacts/stage3_*.json*`，图表保存为 PNG 和 SVG。

主要比较预先指定为组合方案减原始 ACE，按题 ID 聚类 bootstrap（每题保留三个种子），
10,000 次、随机种子 2026。成本同时报告含缓存核算与统一无缓存价格；准确率和成本均不能
仅凭点估计解释为正式非劣效结论。三次种子控制的是训练样本顺序，不控制服务端随机性。
