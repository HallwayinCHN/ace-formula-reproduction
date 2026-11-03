# ACE Formula 复现：第一阶段

本目录保存 FTEC5660 复现实验的准备代码和协议。上游方法、提示词、预处理及判分文件保持原样。
基线固定为 `82709de050e1db6e6ef2f07bcb0393560b94992a`；新增代码仅在运行时替换模型接口，
补上 DeepSeek 兼容参数、有限重试、费用日志和停止线。

## 当前阶段

- 已完成环境安装、原始数据结构和重复检查、样本顺序清单，以及本地模拟训练。
- 模拟结果只验证流程，不是模型效果、论文复现结果或正式试验。
- 正式五组实验尚未实现/运行；第一阶段只运行原始 ACE 的小规模试跑。
- 真实试跑已完成：空手册与 ACE 验证均为 6/10，未观察到提升。详情见 `artifacts/STAGE1.md`。
- 已按成本锁定正式规模 60/30/100；两项改进及正式实验尚未开始。

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
