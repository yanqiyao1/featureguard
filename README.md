# FeatureGuard：训练与推理特征顺序检查器

一份加工报价，正确答案是 **320 元**，程序却报了 **680 元**，没有任何异常。模型训练时第一列是人工工时，上线时第一列变成了设备工时；输入依旧是两列浮点数。`.to_numpy()` 丢掉了列名，但源码里还留着两次选列的顺序。

FeatureGuard 顺着源码追踪这个顺序，在**同一模型对象**的 `fit` 与 `predict` 之间检查输入契约。它只读 Python 源码，**不运行目标代码、不读取 CSV、不反序列化模型，也不需要安装 pandas 或 scikit-learn**。

**课程提交入口：** [项目报告（PDF）](docs/report.pdf) · [报告 Markdown 与源码链接](docs/report.md) · [真实数据实验](docs/case-study.md)

报告以 Markdown 为唯一正文来源。需要重新排版时，在装有 Pandoc、XeLaTeX、Fontconfig 和中文字体的环境中运行 [tools/render-report.sh](tools/render-report.sh)；运行检查器不需要排版工具。

## 两分钟演示

在仓库根目录使用 **Python 3.10+**，无需安装：

```sh
python3 -m featureguard examples/reordered/quote.py
python3 -m featureguard examples/shared/quote.py
```

第一条返回 `1`。在 `quote.py:12:9` 的 `predict` 处报告 `FG001`：需要 `["labor_hours", "machine_hours"]`，实际为 `["machine_hours", "labor_hours"]`。报告同时给出训练点、矩阵来源和本地函数调用位置。即使训练和推理调用的是**同一个**选列函数，也能区分它在两处收到的参数。

第二条返回 `0`：`1 prediction site, 1 verified, 0 diagnostics`。这里两种列序都可能出现，但每条分支上的训练与预测使用同一份配置，所以不应报警。用 `--max-paths 1` 复查这个文件可以看到：当分析无法保留这种关联时，它会明确说“无法验证”，而非猜一个结果。

修复第一种问题时，应让训练和服务使用同一份**有序**特征契约，或在接入处按已保存的训练顺序取列；不要只检查列数，也不要分别排序后比较集合。

## 检查自己的项目

```sh
python3 -m featureguard path/to/entry.py --root path/to
python3 -m featureguard path/to/entry.py --root path/to --format json --require-prediction
```

选择真正发起训练和推理调用的**入口文件**。默认根目录为入口所在目录；如果项目按包组织，用 `--root` 指定包搜索根。报告位置相对于该根目录，行、列均从 1 开始。可在虚拟环境中执行 `python3 -m pip install .` 安装命令 `featureguard`。

| 规则 | 含义 | 建议 |
| --- | --- | --- |
| `FG001` | 到达预测点的已建模状态均为同名特征的顺序错位 | 对照首个不同位置和两端来源 |
| `FG002` | 已建模状态均不匹配，且不全是纯排列差异 | 检查缺列、多列、改名和重复列次数 |
| `FG004` | 部分路径可见错位，其他路径匹配或未知 | 查看报告给出的分支或调用见证 |
| `FG003` | 该预测点证据不足，不能验证通过 | 检查动态选列、未见训练或不透明变换 |
| `FG900` | 遇到不支持的控制流、可能修改已知对象的操作，或分析预算上限 | 先看具体位置；它不是训练/推理错误的定论 |
| `FG901` | 启用 `--require-prediction` 后未识别到预测点 | 核对入口和估计器来源，不让 CI 因零覆盖而通过 |

退出码：`0` 无诊断；`1` 有诊断（包括未知或覆盖不足）；`2` 命令或输入错误。JSON 的 `schema_version` 为 1，`stats` 中的 `predictions` 是**预测调用源码位置数**，不是路径数；`verified` 是所有已建模到达状态均匹配的位置数。`diagnostics` 提供规则、位置、有序列名和关联证据；`errors` 单列输入错误。没有启用严格选项时，`0 prediction sites` 不是验证成功。

## 课程技术与范围

- **前向数据流分析**：在抽象域中区分“列名列表”“带标签的投影”“已去掉标签的矩阵”和“未知”，为赋值、选列、转数组、按列拼接编写转移函数。
- **过程间分析与指向关系**：在本地普通函数调用处分配参数、匹配返回；对模型对象记录最后一次训练。`alias.fit(...)` 会更新原模型的记录，而两次工厂调用不会仅因共用同一行构造代码就合并。
- **分支精度**：在有限预算内区分 `if` 和条件表达式的路径，既检出只在某一分支触发的错误，也避免把训练分支 A 和预测分支 B 强行配对。

首版支持显式字符串列表取列、`df.loc[:, cols]`、`.values`、`.to_numpy()`、已知矩阵的 `np.asarray`、`np.hstack` 和 `np.concatenate(..., axis=1)`。识别标准 pandas 数据帧构造入口和[白名单中的 scikit-learn 估计器](featureguard/domain.py)。检查 `predict`、`predict_proba` 和 `decision_function` 的输入顺序，不代替库的类型/API 合法性检查。

不承诺支持动态生成的特征名、任意插件、循环不动点、任意 sklearn Pipeline、列值单位或独立进程间的模型溯源。本地导入只解析源码与受限声明，不模拟模块初始化。默认路径预算 32、本地助手调用深度最多 16；整次分析超过 50,000 次语句/表达式转移时停止，不发布部分“已验证”结果。[报告第 6 节](docs/report.md)说明这些边界怎样产生漏报或不确定结论。

## 验证与复现

```sh
python3 -W error -m unittest discover -s tests -v
python3 -m tools.experiment
python3 -m featureguard examples/diabetes/study.py
```

排列与边界对照 **43/43** 符合预期。最后一条在真实模型实验脚本上应返回 `1`：两个预测点，一个验证通过，一个顺序错位。要复现数值后果而非仅检查源码，可在装有科学计算库的环境中主动执行 `python3 examples/diabetes/study.py`。相同的 111 行留出样本，交换两列后 MAE 从 **42.156** 变为 **395.705**；[实验记录](docs/case-study.md)提供固定版本、随机种子及解读限制。
