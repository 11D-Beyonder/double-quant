---
name: 3rd-test
description: Create, revise, package, or review qFinanceCode third-party acceptance tests and Chinese验收材料. Use when Codex is asked to produce or polish tests/scripts, tests/docs deliverables, results.md, test_report.md, technical_report.md, 测试用例.docx, annotated UI screenshots, terminal screenshots, Chinese benchmark charts, Chinese Matplotlib plot style, chart validation scripts, or tests/docs/3rd-testtable.xlsx entries for qFinanceCode feature acceptance, third-party testing, or performance validation.
---

# qFinanceCode 第三方测试

## 先读资料

1. 进入仓库根目录，先读 `AGENTS.md` 和 `README.md`，遵守项目的 A100、构建、测试和交付约定。
2. 打开 `3rd-testtable.xlsx`，以其中的“序号、性质、测试项目、研究成果、交付物”为准。报告、图表、测试用例中的功能名称必须和表格一致。
3. 按任务类型读取一个参考示例：
   - UI 操作型：`examples/tests/docs/151-operations-interface/`
   - 终端运行或性能基准型：`examples/tests/docs/76-data-codec-optimization/`
4. 如仓库缺少验收表模板，可参考 `3rd-testtable.xlsx`。

## 语言和版式

- 文档、表格、图表、截图标注尽量全部使用中文；只有命令、接口路径、代码标识符、文件名等必要内容保留英文。
- 中文名词以 `3rd-testtable.xlsx` 为唯一口径，不要自造同义词。
- 生成图片前设置中文字体；生成后必须检查中文是否正常显示，不接受乱码、缺字、方框字。如果中文出现乱码或者显示错误，一律拒绝。然后修改代码重新生成，直到中文显示正常。
- UI 截图必须有清晰的红色框选或箭头标注，说明当前测试关注的菜单、按钮、输入框或结果区域。
- 终端截图优先使用 `scripts/terminal_renderer.py` 生成，保证命令和输出能在报告中直接引用。

## 中文图表

- 绘制报告图表时优先导入 `scripts/chinese_plot_style.py`，不要在每个测试脚本里重复手写字体、字号、画布和保存参数。
- 图宽使用 `SINGLE_COLUMN_MM`、`MEDIUM_WIDTH_MM` 或 `DOUBLE_COLUMN_MM`；验收报告内常用 `MEDIUM_WIDTH_MM`。
- 图表只使用两种字号：标题用 `TITLE_FONT_SIZE`，坐标轴、刻度、图例、标注、注释统一用 `SMALL_FONT_SIZE`。
- 中文字体优先使用微软雅黑、仿宋、黑体等系统字体；没有系统字体时使用 skill 内置 `STHeiti`、`Songti` 或 `Arial Unicode` 字体。
- 保存图表后运行 `scripts/validate_chinese_plot.py`。校验必须通过后才能把图片写入报告。

示例：

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("skills/3rd-test/scripts").resolve()))
from chinese_plot_style import apply_chinese_style, save_figure, style_axes
```

校验命令：

```bash
python skills/3rd-test/scripts/validate_chinese_plot.py \
  tests/docs/{序号}-{英文短横线名称}/images/{图名}.png
```

## 必交产物

为每个功能或性能验收项创建独立产物：

- 测试脚本：`tests/scripts/{序号}-{功能英文名}.py`
- 文档目录：`tests/docs/{序号}-{英文短横线名称}/`
- 结果文档：`tests/docs/{序号}-{英文短横线名称}/results.md`
- 测试报告：`tests/docs/{序号}-{英文短横线名称}/test_report.md`
- 技术报告：`tests/docs/{序号}-{英文短横线名称}/technical_report.md`
- 测试用例：`tests/docs/{序号}-{英文短横线名称}/测试用例.docx`
- 图片目录：`tests/docs/{序号}-{英文短横线名称}/images/`，按需包含 `raw/` 原图

完成后填写 `tests/docs/3rd-testtable.xlsx` 对应行最后四列：测试命令、结果文档位置、技术文档位置、测试报告位置。使用 `openpyxl` 修改表格，尽量保留原有样式和列宽。

## 测试脚本

- 每个验收项使用一个独立 `.py` 文件，不要把多个功能混在同一脚本里。
- 脚本输出使用中文，包含“开始、访问地址或测试命令、逐项检查结果、最终结果”等关键信息。
- UI 测试要覆盖页面可访问性、关键交互、登录或权限前置条件、接口返回结构。
- 性能基准测试要固定随机种子和输入规模，输出可复核的指标，并生成图表。
- 脚本中不要写入密码、token、SSH 配置等敏感信息；默认账号只按项目公开约定使用。

## results.md

`results.md` 只记录事实输出，参考以下结构：

```markdown
# {序号} {测试项目} {func-序号或者perf-序号} 测试结果

## 测试对象
## 测试命令
## 程序输出
## 图片输出
## 关键结果
## 输出说明
```

终端输出放入 fenced code block。图片使用相对路径引用，例如 `![接口巡检工作区](images/02-ops-interface-overview.png)`。

## test_report.md

`test_report.md` 写结论和分析，参考以下结构：

```markdown
# {序号} {测试项目} {func-序号或者perf-序号} 测试报告

## 测试目标
## 测试范围
## 测试方法
## 通过标准
## 测试结果分析
## 实际验证记录
## 风险与限制
## 测试结论
```

测试结论要明确说明是否通过。未运行的验证、环境限制、A100 部署限制、Docker 或数据库限制必须写清楚。

## technical_report.md

`technical_report.md` 写实现细节，参考以下结构：

```markdown
# {序号} {测试项目} {func-序号或者perf-序号} 技术报告

## 技术目标
## 实现位置
## 实现概述
## 关键技术点
## 验收脚本设计
## 验证方法
## 技术结论
```

如果涉及前端、后端、数据库、Python 算法或部署，分别列出修改文件、接口、数据表、配置和验证方式。

## 测试用例.docx

使用示例中的中文`测试用例.docx`作为模板，并填写：
- 测试项目： 与表格 `3rd-testtable.xlsx` 中的“测试项目”一致
- 测试目的： 与表格里面的指标要求一致
- 测试环境： 详细说明运行环境
- 研究成果:
- 交付物: 多个交付物，禁止出现不同交付内容写在同一个文档里面的情况。
- 必选/可选
- 前置条件
- 测试流程
- 预期结果

最后两项“测试结果”和“测试结论”保留为空，交给最终验收填写。

终端运行型测试：在测试流程中写清执行命令和核对步骤，插入终端截图；如有图表，把图表作为预期结果证据。

UI 操作型测试：在测试流程中写清打开页面、登录、点击、输入、查看结果等步骤，并插入带红框或箭头标注的截图。

## 终端截图

从 `results.md` 生成终端截图：

```bash
python skills/3rd-test/scripts/terminal_renderer.py \
  --results-md tests/docs/{序号}-{英文短横线名称}/results.md \
  --output tests/docs/{序号}-{英文短横线名称}/images/terminal_run.png \
  --command "{实际测试命令}"
```

生成后检查 `images/terminal_run.png`：中文正常、命令完整、关键输出可读。如果有markdown表格，一定要渲染成表格形式，而不是markdown文本。以及如果有代码块，一定要渲染成代码块形式，而不是markdown文本。如果中文出现乱码或者显示错误，一律拒绝。然后修改代码重新生成，直到中文显示正常。

## 完成检查

交付前逐项检查：

- `tests/docs/3rd-testtable.xlsx` 中功能名称和报告名称一致。
- `results.md`、`test_report.md`、`technical_report.md` 均为中文主导，且没有乱码。
- `测试用例.docx` 已生成，最后两项为空。
- UI 截图有红框或箭头标注；终端截图和图表中文正常。
- 所有相对图片链接有效。每个图片都能在 `images/` 目录下找到。并且没有任何乱码、缺字、方框字。
- 已按 `AGENTS.md` 运行必要验证；未验证项已在报告中说明。

最终回复用户时，列出测试脚本、三份 Markdown、`测试用例.docx`、图片目录和表格的可点击路径，并明确请用户检查确认 `测试用例.docx`。在用户确认前，不要声称已经完成最终人工验收。一定要等到用户有反馈，收到用户的反馈之后，才能够进行下一步任务。
