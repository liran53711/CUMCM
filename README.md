# CUMCM

A project for China Undergraduate Mathematical Contest in Modeling.

## Directory Layout

- `Coding/`: 软件工程同学主编辑。存放 Python 代码、Notebook、数据清洗脚本、模型求解脚本、图表生成脚本和可复现实验记录。
- `Model/`: 集成电路同学主编辑。存放建模思路、变量定义、公式推导、目标函数、约束条件、算法流程、模型检验和敏感性分析。
- `Writing/`: 数字经济同学主编辑。存放论文正文、摘要、问题重述、模型假设、指标体系、结果解释、政策/管理建议和参考文献。
- `Share/`: 全员共同编辑。存放赛题分析、选题策略、公共资料、会议记录、任务清单、模板和可复用知识库。

## Team Roles

- 软件工程：主程序、数据分析、可视化、支撑材料复现。主要编辑 `Coding/`，同时把关键结果同步给 `Model/` 和 `Writing/`。
- 集成电路：主建模、数学推导、优化/仿真。主要编辑 `Model/`，并给 `Coding/` 提供可实现的算法流程。
- 数字经济：主论文、指标体系、现实解释。主要编辑 `Writing/`，并把论文需要的图表、公式、结果及时反馈给另外两位。

## Shared Knowledge

- [赛题结构与题型分析](Share/contest_problem_analysis.md)
- [三人组赛题选择策略](Share/team_problem_selection.md)

## AI Assistance

本库提供 AI 协作规约：

- `AGENTS.md`: 给 Codex/AI 助手读取的完整项目规则。
- `anent.md`: AI 协作入口说明，指向 `AGENTS.md`。

拉取本仓库后，建议先阅读 `README.md`、`AGENTS.md` 和 `Share/` 下的两份赛题分析文件，再开始训练或比赛协作。
