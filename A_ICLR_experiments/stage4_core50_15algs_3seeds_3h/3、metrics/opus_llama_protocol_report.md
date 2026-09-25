# Core50：Opus5指标计算与Llama训练报告

核验日期：2026-09-26。正文说明当前实现；附录保留完整固定提示词、动态字段及真实运行的spec。历史阶段依据当时保存的参数和结果区分，未保存的历史API报文不作逐字复原声明。

## 1. Opus5怎样参与指标计算

当前新增后处理在iaaccn22通过API调用`claude-opus-5`，独立单轮请求，`thinking=adaptive`、`effort=xhigh`、`max_tokens=65536`。历史已验收裁决可以复用，不能将全部结果都计为本轮新增API调用。当前范围仅处理最终公式。

1. **化简**：Ground Truth与预测公式分别请求化简。输入包含原式、变量、允许函数、定义域、protected算子语义及确定性证据。小数按其书写值精确保留；微小非零常数不能删除。程序检查JSON格式、表达式有效性与化简前后等价性，保存输入及响应哈希。
2. **等价性**：把已冻结的预测公式与Ground Truth配对，提供定义域、符号证据和数值检验。Opus返回`equivalent / not_equivalent / undetermined`。数值误差小不自动获得等价判定。
3. **结构一致性**：同一algorithm×task×condition的三个seed，两两比较，共三对。Opus判断数学等价、常数抽象后的相同结构、不同结构或无法确定。
4. **本地评分**：Python从化简结果生成AST、变量集合、算子集合与节点数，结合上述判定计算指标。Opus返回判定和理由，不直接提供六轴分数。

### 六轴中的具体作用

以下分数均为0至1，表格均值乘100。

- **ID/OOD**：`q = 1 - (clip(log10(max(NMSE, 1e-12)), -12, 2) + 12)/14`。由原生公式及数值评估计算；无效数值为0，Opus不参与。
- **SYM**：等价时为1；有效但未判定等价时为`0.5 × (T × F_var × F_op)^(1/3)`。`T = 1 - TED/(n_pred+n_gt)`；TED为规范化AST之间的有序树编辑距离，分母使用两棵树节点数之和。变量和算子分别使用集合F1：`2|P∩G|/(|P|+|G|)`，两个集合均为空时为1。Opus的化简及等价性判定影响该轴。
- **MIN**：`min(1, n_gt/n_pred)`，节点数来自化简后的AST。Opus通过化简影响复杂度；MIN本身不要求预测与Ground Truth等价。
- **EFF**：180个分钟点上令`q_t=(ID_t+OOD_t)/2`，计算`mean(q_t/max(q_t))`；全部为0时EFF=0。Opus不参与。
- **STAB**：`(N×V×C)^(1/3)`。`N=1-mean_seed_pairs((|ID_a-ID_b|+|OOD_a-OOD_b|)/2)`，`V=有效seed数/3`，`C=结构一致的seed对数/3`。Opus的`mathematically_equivalent`和`same_canonical_structure`计入C，其余判定不计入。最终STAB按50个task求均值。

算法本身无有效公式时，SYM/MIN按0记录；缺少必要评估证据需要单独记录，不能自动当成算法失败。当前JAXSR噪声EFF仍有4份轨迹缺失，均值分母另有记录。

### 真实例子：PySR / Nguyen-9 / clean / seed520

Ground Truth：`sin(x1) + sin(x2**2)`。

预测公式：`sin(x1) + sin(x2**2) + 1.1685188e-8`。

Opus保留微小常数，化简结果为`unchanged`；等价性为`not_equivalent`，理由为两个函数在定义域内始终相差该非零常数。ID NMSE=`4.540324608516168e-15`，OOD NMSE=`8.28263308044139e-17`，两个数值轴均达到100分。

AST节点数为预测8、参考7，`T=14/15`、两个集合F1均为1，所以SYM=`0.48863240295941257`，展示为**48.86**；MIN=`7/8`，展示为**87.50**。这是数值恢复与严格符号恢复分别评分的具体表现。

证据：`terminal_run_metrics.jsonl.gz`中g0644/pysr/clean/520；等价性裁决`opus/core50_comparisons/execution/equivalence__clean/frozen/56164e34e3b207d7d9cb333b94747f19d619ba45ca34626089540e177be66a3f.json`。

## 2. LLM-SR与DrSR怎样使用Llama

当前正式配置使用DeepInfra的`meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo`；iaaccn22两份运行配置核验值均为`temperature=0.6, top_p=0.3, max_tokens=1024`。这里的1024是训练模型输出配置；Opus后处理配置为65536。两者均不更新Llama权重。

Core50每项预算10800秒、seed520/521/522、clean/noise001/noise005；`niterations=100000`、每轮4个候选、`max_params=10`。共享spec至少预留`max(10, 特征数+1)`个参数位置。候选按训练目标选择，ID/OOD用于评估。

### LLM-SR

1. 提示词包括任务背景、变量描述、代码前言及历史候选函数；让Llama续写下一个`equation_vN`。
2. 多个island保存候选，每次按训练评分抽取至多两个历史函数，以评分顺序放入提示词。候选函数不断更新，固定说明文字保持不变。
3. 每轮请求4个公式结构。每个结构在本地用BFGS优化`params`，初值全部为1，目标为训练MSE；评分取负MSE，越高越好。
4. 优质候选进入历史池，持续生成新公式，预算内保留原生训练目标最好的候选。

实际发给API的是一个user消息。运行spec中的`evaluate()`由本地执行；生成函数提示词时只保留代码前言、历史`equation_vN`和待续写函数头。附录同时提供保存的完整spec与按实际代码重新渲染的完整初始消息，明确区分两者。

### DrSR

使用同样的公式生成与候选池机制，增加两类Llama请求：

- **经验总结**：逐候选依据训练评分改善、下降或执行失败，发送Good/Bad/None模板，生成经验供下一轮使用。
- **残差分析**：出现满足触发条件的更好候选后，把训练输入、目标、残差及候选公式提交给Llama，形成分析结果供后续生成使用。残差矩阵为`[X, y, y-y_pred]`，数值保留3位小数；传入提示词时使用NumPy数组文本表示，大数组可能自动省略部分行。

DrSR实际系数评估由`evaluate_on_problems.py`执行：5组`Uniform(-1,1)`初值进行BFGS，`maxiter=200, gtol=1e-10, eps=1e-12`，保留训练MSE最小的结果。保存的spec中`evaluate()`是接口模板，运行时采用上述独立评估器。

后续生成消息顺序为：任务头、可选残差分析、可选经验、生成指令、候选函数代码。失败经验最多3条，Good/Bad各最多2条且各有0.5概率选用；单条经验取前500字符，残差分析取前2000字符，超出时追加省略符号。残差区块在已有记录等条件满足时以0.5概率注入。生成、经验总结、残差分析均调用同一个Llama客户端。

## 3. 早期探针与Core50的背景信息

- **Stage1，664×2×1seed×1小时**：两个探针为PySR和LLM-SR。LLM-SR显式设置`inject_prompt_semantics=False`，背景统一为附录B.1的通用文字，不注入数据集变量/目标描述。目的为控制不同数据集的语义提示差异与答案信息。该阶段没有DrSR；模型计划为Llama-3.1-8B-Instruct普通版。
- **Stage2，200任务候选算法比较**：早期冻结参数给LLM-SR和DrSR使用同一通用背景，其中LLM-SR显式关闭语义注入；DrSR该历史参数文件未明确保存这个开关，不能仅凭背景文字认定其他字段完全没有语义信息。后续单独进行了语义补充实验：两算法各200项，seed1314、1小时，实际400行结果全部记录`inject_prompt_semantics=True`；每个算法普通版与Turbo版各100项。
- **Stage3，664×4×3seed**：最终四个construction probes为DSO、PyOperon、iMCTS、uDSR。LLM-SR和DrSR属于前面的候选/补充评估过程。
- **当前Core50**：开启`inject_prompt_semantics=True`与`canonical_prompt_variables=True`。提示词使用x0、x1等统一变量名及目标y；从metadata.description、feature descriptions、target description构造背景。未提供description时使用附录B.2的通用生成规则。Nguyen类任务的背景可以仍然是抽象回归描述；有物理说明的数据集会提供相应知识。

对含干扰变量的SRSD任务，只提供无序的语义种类、数量及干扰变量数量，各变量统一描述为`candidate variable; semantic role hidden`，不提供变量编号与语义的对应关系。完整补充句见附录B.3。模板构造路径不直接读取Ground Truth公式；metadata中的物理描述本身仍可能提示已知定律，因此这里属于允许语义先验的训练设置。

历史Stage1/Stage2逐次API报文没有在本次核验的保存材料中找到。附录准确提供已保存背景、当前完整模板及当前真实spec；不能据此宣称早期所有固定文字与当前版本逐字相同。

当前结果存在一份用户指定的选式例外：LLM-SR/g0436/noise005/seed522使用此前可处理的原生候选sample196，证据在`formula_override_g0436_s522/`；该记录原始结果仍保留。

## 4. 完整模板附录

固定文字从源码直接提取，保留原始拼写；花括号为随任务替换的数据字段。Nguyen-6完整消息通过当前真实渲染器和该次保存的spec生成，属于可复现重建，未声称是历史HTTP抓取记录。其余轮次使用实际历史函数、经验、残差替换相应字段，不存在一份永远不变的完整报文。

### A.1 Opus共用system消息

```text
You are a strict mathematical benchmark evaluator.
All domain assumptions, declared variables/functions, instantiated constants, and output-schema rules supplied by the user are hard constraints.
Never add assumptions from physical interpretation, typical parameter signs, or observed probe ranges.
Judge preservation in the declared number system, not in a broader complex domain.
If an expression has no valid value in the declared real-valued domain, or relies on undeclared symbolic entities, follow the task's unable/undetermined rule rather than treating textual identity as sufficient.
Return only the requested JSON object, with no additional properties or prose.
```

API的user消息为对应模板替换`{{REQUEST_JSON}}`后，再追加`\n\nOUTPUT_JSON_SCHEMA_DRAFT_07:\n`和该任务的完整JSON Schema。REQUEST_JSON包含当前公式、变量、定义域、证据与来源绑定；固定文字和完整Schema如下。

### A.2 simplify user模板

```text
You are the mandatory single-turn symbolic simplification judge for SymbolicArena.

Simplify exactly the expression in request.expression, preserving its exact mathematical value and its supplied real-domain and protected-operator semantics.

Rules:
1. Use only the declared variables and functions. Keep variable names unchanged.
2. Every supplied decimal literal denotes its exact written value. Keep numeric products, quotients, sums, and differences as exact expressions unless you can compute an exactly equal rational value. Never replace them with rounded floating-point values.
3. Preserve constant sin, cos, exp, log, powers, and roots symbolically. A tiny nonzero term must remain present. For example, retain 30*(-0.000056683)*cos(-0.000056683) instead of approximating or dropping it.
4. Preserve all coefficients and exponents exactly. Numerical agreement to machine precision does not establish exact equivalence.
5. request.expression is authoritative. Canonical strings and numerical probes are supporting evidence and may contain floating-point constant folding; do not substitute their rounded constants for the authoritative expression.
6. Apply only justified algebraic cancellations and compact factorizations. Preserve domains, absolute values, branch conditions, protected division, and roots. Do not expand an expression only for presentation.
7. When no valid reduction is justified, return outcome "unchanged" with the original request.expression copied exactly. Use this outcome only after evaluating the available reductions. Do not claim a simplification you cannot establish.
8. When preservation cannot be established, return outcome "unable", simplified_expression null, and equivalence_assessment "undetermined". For "simplified" or "unchanged", equivalence_assessment must be "preserved".
9. Return one JSON object containing exactly these six keys: outcome, simplified_expression, equivalence_assessment, assumptions, confidence, brief_reason. No extra keys, markdown fences, comments, trailing explanation, or second object. Use a brief reason, at most eight assumptions, and confidence between 0 and 1.
10. Complete the judgment in this response. Do not expose private reasoning or simulate tool execution.

REQUEST_JSON:
{{REQUEST_JSON}}
```

### A.2.1 完整输出Schema

```json
{"$id":"symbolicarena.simplify.v1","$schema":"http://json-schema.org/draft-07/schema#","additionalProperties":false,"properties":{"assumptions":{"items":{"type":"string"},"maxItems":8,"type":"array"},"brief_reason":{"maxLength":1000,"minLength":1,"type":"string"},"confidence":{"maximum":1,"minimum":0,"type":"number"},"equivalence_assessment":{"enum":["preserved","not_preserved","undetermined"],"type":"string"},"outcome":{"enum":["simplified","unchanged","unable"],"type":"string"},"simplified_expression":{"type":["string","null"]}},"required":["outcome","simplified_expression","equivalence_assessment","assumptions","confidence","brief_reason"],"type":"object"}
```

### A.3 equivalence user模板

```text
You are the mandatory single-turn mathematical equivalence judge for SymbolicArena.

Judge exactly one predicted expression against its frozen Ground Truth reference using the supplied domain assumptions and deterministic evidence.

Rules:
1. Judge mathematical meaning, not superficial syntax.
2. Constants are already instantiated; do not treat numeric coefficients as freely adjustable.
3. Variable identities are fixed and may not be permuted unless the request explicitly defines a mapping.
4. Respect protected-operator and domain assumptions.
5. A deterministic counterexample is evidence of non-equivalence.
6. If the evidence is insufficient for a reliable yes/no result, return "undetermined".
7. A decision of "equivalent" or "not_equivalent" must not use evidence_basis "insufficient".
8. Return only the JSON object required by the supplied schema.
9. Give only a brief reason; do not provide hidden reasoning or chain-of-thought.

REQUEST_JSON:
{{REQUEST_JSON}}
```

### A.3.1 完整输出Schema

```json
{"$id":"symbolicarena.equivalence.v1","$schema":"http://json-schema.org/draft-07/schema#","additionalProperties":false,"properties":{"assumptions":{"items":{"type":"string"},"maxItems":8,"type":"array"},"brief_reason":{"maxLength":1000,"minLength":1,"type":"string"},"confidence":{"maximum":1,"minimum":0,"type":"number"},"decision":{"enum":["equivalent","not_equivalent","undetermined"],"type":"string"},"evidence_basis":{"enum":["symbolic_proof","numerical_support","structural_analysis","mixed","insufficient"],"type":"string"}},"required":["decision","evidence_basis","assumptions","confidence","brief_reason"],"type":"object"}
```

### A.4 structure user模板

```text
You are the mandatory single-turn structural consistency judge for SymbolicArena STAB.

Compare exactly one pair of frozen, simplified clean-run expressions.

Decision meanings:
- mathematically_equivalent: the two expressions represent the same mathematical function.
- same_canonical_structure: not established as equivalent, but they share the same canonical symbolic structure after constants are abstracted consistently.
- different_structure: neither condition holds.
- undetermined: the available evidence is insufficient.

Rules:
1. Variable identities are fixed.
2. Respect supplied domain and protected-operator assumptions.
3. Use deterministic equivalence and canonical-tree evidence, but make the required judgment yourself.
4. Return only the JSON object required by the supplied schema.
5. Give only a brief reason; do not provide hidden reasoning or chain-of-thought.

REQUEST_JSON:
{{REQUEST_JSON}}

```

### A.4.1 完整输出Schema

```json
{"$id":"symbolicarena.structure.v1","$schema":"http://json-schema.org/draft-07/schema#","additionalProperties":false,"properties":{"brief_reason":{"maxLength":1000,"minLength":1,"type":"string"},"confidence":{"maximum":1,"minimum":0,"type":"number"},"decision":{"enum":["mathematically_equivalent","same_canonical_structure","different_structure","undetermined"],"type":"string"}},"required":["decision","confidence","brief_reason"],"type":"object"}
```

structure模板保留原文`clean-run`；本轮执行器也对noise001/noise005建立独立条件的结构任务，实际请求中的condition/noise_tag及对应公式决定比较范围。已有裁决使用各自保存的prompt哈希，化简精确模板描述当前补充调用规则。

### B.1 早期探针通用background完整文字

```text
This is a symbolic regression task. Find a compact mathematical equation that predicts the target from the observed variables.
```

### B.2 metadata未提供description时的background完整模板

```text
Find the mathematical function skeleton that represents {target_desc}, given data on {feature_text}.
```

### B.3 SRSD干扰变量background补充句完整模板

```text
There are {n_total} candidate variables in an unknown order. The unordered semantic-role multiset is: {role_text}. The mapping from semantic roles to variable names is intentionally hidden; do not assume which x_i corresponds to which semantic role.
```

`role_text`按语义名称排序，条目以`; `连接；语义条目为`{count} variable(s) with semantic role "{role}"`，干扰条目为`{count} distractor/meaningless variable(s)`；数量为1时使用variable，其他数量使用variables。

### C.1 LLM-SR固定生成指令

```text
You are a helpful assistant tasked with discovering mathematical function structures for scientific systems.                              Complete the 'equation' function below, considering the physical meaning and relationships of inputs.

```

### C.2 LLM-SR真实完整运行spec：Nguyen-6/clean/520

```python
"""
Find the mathematical function skeleton that fits the data.

Background:
This is an abstract symbolic regression problem.

Variables:
- Independents:
  - x0: Independent variables (anonymized)
- Dependent:
  - y: Dependent variable (target)
"""

import numpy as np
from scipy.optimize import minimize

# Initialize parameters
MAX_NPARAMS = 10
params = [1.0]*MAX_NPARAMS
# 全局变量用于在沙箱中读取 BFGS 结果参数
BFGS_PARAMS = None

@evaluate.run
def evaluate(data: dict) -> float:
    """ Evaluate the equation on data observations. """
    inputs, outputs = data['inputs'], data['outputs']
    X = inputs

    def loss(params):
        y_pred = equation(*X.T, params)
        return np.mean((y_pred - outputs) ** 2)

    result = minimize(loss, [1.0]*MAX_NPARAMS, method='BFGS')
    global BFGS_PARAMS
    try:
        BFGS_PARAMS = result.x
    except Exception:
        BFGS_PARAMS = None
    loss_val = result.fun
    if np.isnan(loss_val) or np.isinf(loss_val):
        return None
    return -loss_val

@equation.evolve
def equation(x0: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Equation to be evolved.

    Background:
    This is an abstract symbolic regression problem.

    Variables:
    - Independents:
  - x0: Independent variables (anonymized)
- Dependent:
  - y: Dependent variable (target)

    Parameters:
    - params (np.ndarray): Trainable coefficients used by the equation skeleton.

    Output requirement:
    - Write the final formula as a single-line return statement.
    - Do not split the final formula across multiple lines.
    - Example: return params[0] + params[1] * x0
    """
    return params[0] + params[1] * x0
```

### C.3 LLM-SR完整初始user消息：Nguyen-6

```text
You are a helpful assistant tasked with discovering mathematical function structures for scientific systems.                              Complete the 'equation' function below, considering the physical meaning and relationships of inputs.


"""
Find the mathematical function skeleton that fits the data.

Background:
This is an abstract symbolic regression problem.

Variables:
- Independents:
  - x0: Independent variables (anonymized)
- Dependent:
  - y: Dependent variable (target)
"""

import numpy as np
from scipy.optimize import minimize

# Initialize parameters
MAX_NPARAMS = 10
params = [1.0]*MAX_NPARAMS
# 全局变量用于在沙箱中读取 BFGS 结果参数
BFGS_PARAMS = None

def equation_v0(x0: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Equation to be evolved.

    Background:
    This is an abstract symbolic regression problem.

    Variables:
    - Independents:
  - x0: Independent variables (anonymized)
- Dependent:
  - y: Dependent variable (target)

    Parameters:
    - params (np.ndarray): Trainable coefficients used by the equation skeleton.

    Output requirement:
    - Write the final formula as a single-line return statement.
    - Do not split the final formula across multiple lines.
    - Example: return params[0] + params[1] * x0
    """
    return params[0] + params[1] * x0


def equation_v1(x0: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Improved version of `equation_v0`."""

```

后续轮次仍使用同一指令和代码前言，历史函数按`equation_v0`、`equation_v1`编号；第二个及后续函数的docstring为`Improved version of `加反引号包围的前一个函数名及句号；最后追加同签名、空函数体的下一版本。历史函数体来自该次训练保存的候选。

### D.1 DrSR任务头模板

```text
Find the mathematical function skeleton that represents {dependent}, given data on {independent}. 
```

### D.2 DrSR固定生成指令

```text
You are a helpful assistant tasked with discovering mathematical function structures for scientific systems. Complete the 'equation' function below, considering the physical meaning and relationships of inputs. Write the final formula as a single-line return statement only. Example: return params[0] + params[1] * x0 + params[2] * x1

```

生成指令末尾再拼接`Variables:\n{variables_block}\nBackground: {background_text}\n`；任务头中的dependent包含目标描述，independent包含各变量描述。下方完整消息展示实际替换结果。

### D.3 DrSR真实完整运行spec：Nguyen-6/clean/520

```python
"""
Find the mathematical function skeleton that fits the data.

Background:
This is an abstract symbolic regression problem.

Variables:
- Independents:
  - x0: Independent variables (anonymized)
- Dependent:
  - y: Dependent variable (target)
"""

import numpy as np

# Initialize parameters
MAX_NPARAMS = 10
params = [1.0]*MAX_NPARAMS

@evaluate.run
def evaluate(data: dict) -> float:
    """ Evaluate the equation on data observations. """
    inputs, outputs = data['inputs'], data['outputs']
    cols = [inputs[:, i] for i in range(inputs.shape[1])]
    try:
        y_pred = equation(*cols, params)
        loss = np.mean((y_pred - outputs) ** 2)
        if np.isnan(loss) or np.isinf(loss):
            return None
        return -loss
    except Exception:
        return None

@equation.evolve
def equation(x0: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Equation to be evolved.

    Background:
    This is an abstract symbolic regression problem.

    Variables:
    - Independents:
  - x0: Independent variables (anonymized)
- Dependent:
  - y: Dependent variable (target)

    Parameters:
    - params (np.ndarray): Trainable coefficients used by the equation skeleton.

    Output requirement:
    - Write the final formula as a single-line return statement.
    - Do not split the final formula across multiple lines.
    - Example: return params[0] + params[1] * x0
    """
    return params[0] + params[1] * x0
```

### D.4 DrSR完整初始user消息：Nguyen-6

```text
Find the mathematical function skeleton that represents y (Dependent variable (target)), given data on x0 (Independent variables (anonymized)). 

You are a helpful assistant tasked with discovering mathematical function structures for scientific systems. Complete the 'equation' function below, considering the physical meaning and relationships of inputs. Write the final formula as a single-line return statement only. Example: return params[0] + params[1] * x0 + params[2] * x1

Variables:
- Independents:
  - x0: Independent variables (anonymized)
- Dependent:
  - y: Dependent variable (target)
Background: This is an abstract symbolic regression problem.

"""
Find the mathematical function skeleton that fits the data.

Background:
This is an abstract symbolic regression problem.

Variables:
- Independents:
  - x0: Independent variables (anonymized)
- Dependent:
  - y: Dependent variable (target)
"""

import numpy as np

# Initialize parameters
MAX_NPARAMS = 10
params = [1.0]*MAX_NPARAMS

def equation_v0(x0: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Equation to be evolved.

    Background:
    This is an abstract symbolic regression problem.

    Variables:
    - Independents:
  - x0: Independent variables (anonymized)
- Dependent:
  - y: Dependent variable (target)

    Parameters:
    - params (np.ndarray): Trainable coefficients used by the equation skeleton.

    Output requirement:
    - Write the final formula as a single-line return statement.
    - Do not split the final formula across multiple lines.
    - Example: return params[0] + params[1] * x0
    """
    return params[0] + params[1] * x0


def equation_v1(x0: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Improved version of `equation_v0`."""

```

### D.5 DrSR经验总结完整会话模板

```text
Here's our previous conversation:

user: {prompt}

assistant: {sample}

user: {question}
```

`prompt`取buffer生成的`prompt.code`，包含代码前言和候选函数；此处不包含生成请求额外添加的任务头、经验区块等外层文字。`sample`为模型回答，`question`使用下方对应模板，整体作为一个user消息发送。以下使用实际目标名y和参数预算10渲染，`{error}`保留为实际错误文字字段。

### D.6.1 Good完整追问模板

```text
The optimized function skeleton you just answered scored higher. Please summarize useful experience.
STRICTLY follow these rules:
1. Use the exact phrasing "when seeking for the mathematical function skeleton that represents y, I can ..."
2. Summarize ONLY the key success factors
3. You need to make your answer as concise as possible
```

### D.6.2 Bad完整追问模板

```text
The optimized function skeleton you just answered scored lower. What lessons can you draw from it?
STRICTLY follow these rules: 
1. Use the exact phrasing "when seeking for the mathematical function skeleton that represents y, I can ..."
2. Identify ONE crucial improvement point
3. You need to make your answer as concise as possible
```

### D.6.3 None完整追问模板

```text
The optimized function skeleton you just answered failed with error: {error}. What lessons can you draw from it?
The current evaluator passes exactly 10 trainable parameters, indexed from params[0] to params[9]. Treat this failure as a negative example rather than a requirement to satisfy. If the error is about parameter length or indexing, do not solve it by asking for more parameters. Instead, rewrite the equation so it stays within params[0]..params[9] and avoid any explicit minimum-length checks above 10.
STRICTLY follow these rules:
1. Use the exact phrasing "when seeking for the mathematical function skeleton that represents y, I need ..."
2. Address the SPECIFIC error: {error}
3. Treat this failed sample as a negative example to avoid, not as a target requirement to satisfy
4. Identify ONE concrete change that would prevent the next sample from repeating this failure
5. You need to make your answer as concise as possible
```

### D.7 DrSR残差分析完整user模板：Nguyen-6变量配置

```text
You are a data analysis expert.
Background: This is an abstract symbolic regression problem.
previous conclusions:{last_analysis}
dataset:{residual}
The equation corresponding to the residuals:{sample}

The independent variables are:
- x0: Independent variables (anonymized)

The dependent variable is y (Dependent variable (target)).
The forth column contains residuals (observed - predicted).

Task Requirements:

1. Analyze and summarize how changes of each independent variable influence the dependent variable, and the possible intrinsic relationships among independent variables.

Your response should follow the structure below; no need to show the reasoning process.

2.##Output Format##:
STRICTLY deliver results in the following structured format:

  "output_format": {
    "analysis": {
      "independent_to_dependent_relationships": {
        "x0 ": [
          "Hint: analyze the functional relationship between x0 and y in different intervals"
        ],
      },
      "inter_relationships_between_independents": {
        "": []
      }
    }
  }
```

`last_analysis`为既有分析，`residual`为当前训练残差矩阵的文本，`sample`为对应候选代码。多变量时按全部变量生成independent_to_dependent_relationships，并为每个变量对生成inter_relationships_between_independents。原模板的`forth column`为固定文字；实际残差位于矩阵最后一列，列数为特征数加2，这条文字仅在两个特征时符合列号。

### D.8 DrSR经验注入区块完整模板

```text


### The following are ideas summarized based on past experiences in solving such problems. ###

idea{index}：
{analysis_text}
---

```

### D.9 DrSR残差注入区块完整模板

```text


### The following is the analysis result of the existing data, which will assist you in answering the question. ###

{last_analysis}

```

D.8按选中经验重复生成条目，index从1递增；可选区块按照正文说明加入生成消息。D.5至D.9中的占位字段都来自当前训练记录。

## 5. 核验与来源

- 已核验当前两算法共900份result.json：各450份，均开启语义注入和统一变量命名。两份完整spec来自实际保存文件；初始消息使用各算法原生buffer渲染，未请求API。
- Stage2语义补充实验400条结果的模型分组和语义开关已按CSV重新计数。当前API生成参数于2026-09-26在iaaccn22读取两份配置的非敏感字段核验；历史每个HTTP请求的参数未逐条恢复。

源码与模板输入SHA256如下，路径相对于项目根目录。重新生成脚本：`1、preflight/build_llm_protocol_report.py`。

- `AAAI_experiments/stage5_metric_calculation_0831/config/prompts/equivalence.v1.txt`：`375e4fe63c802922de6393902f0162e3194d41b096c7d8a33fe23c2c4a4913e9`
- `AAAI_experiments/stage5_metric_calculation_0831/config/prompts/simplify_core50_exact.v1.txt`：`09fa8d4daa1a9ed69d6b02e3f2a5677fd132d5c81f4882a9038122b9c4584303`
- `AAAI_experiments/stage5_metric_calculation_0831/config/prompts/structure.v1.txt`：`64940c7ff791772bef05f5e9641ee8de54ef91b8e203b5860ef07982ec13d8c9`
- `AAAI_experiments/stage5_metric_calculation_0831/config/schemas/equivalence.v1.json`：`64e2dea56b4551ca3a19c3887998271b2ddb80bbf111214f55415f2ba0a3a5b8`
- `AAAI_experiments/stage5_metric_calculation_0831/config/schemas/simplify.v1.json`：`eb0957a945d11292bb16c8d00a2f47d347ae74269ddf02f7d8f962e4797ff5ba`
- `AAAI_experiments/stage5_metric_calculation_0831/config/schemas/structure.v1.json`：`f38e475728bb4be46c61ce1e92c65e8032ef37a337f72ae083ce7f54ba841347`
- `AAAI_experiments/stage5_metric_calculation_0831/pipeline/anthropic_api_runner.py`：`17d1aa4639a758057ea3b48a74f5a593a27796636d245422dff807ebfeecb835`
- `AAAI_experiments/stage5_metric_calculation_0831/pipeline/metrics.py`：`093f17abd1ed589e8f6be18b0ccf4a12744ab465100224f53d1d3c166b23a630`
- `AAAI_experiments/stage5_metric_calculation_0831/pipeline/symbolic_evidence.py`：`4abac09e0ff98214a15d439da9449c8d2137014cb352f0a8b914140c71f2a3cd`
- `A_ICLR_experiments/stage1_664dats_2probes_1seed_1h/03_launch_materials/launch_llmsr_probe.py`：`89921b8a5294d660d6765bae8515286b417d6a44028df35398182ec188eee5da`
- `A_ICLR_experiments/stage1_664dats_2probes_1seed_1h/03_launch_materials/probe_parameter_plan.md`：`e53700585544a0f479db4cffcefdb324c46769148f2c2c18de30d56fd5e344b0`
- `A_ICLR_experiments/stage2_200dats_12algs_1seed_1h/04_semantic_llm_override/semantic_results_raw_400.csv`：`feb2be35b73169f3fcf81ed222f950b6dafa32c00598fdcc8eeea97cf7519799`
- `A_ICLR_experiments/stage2_200dats_12algs_1seed_1h/07_stage2_run_provenance/hyperparams_snapshot_20260423/drsr.json`：`b874d66ee0f626fe7a86cecc150a4440d2a28fdcfeaca574c5d20589f5c02007`
- `A_ICLR_experiments/stage2_200dats_12algs_1seed_1h/07_stage2_run_provenance/hyperparams_snapshot_20260423/llmsr.json`：`90f05f23082ff06ec2f51f62413915a2bd6ec3c7e73600f3e2045ee138c51e84`
- `A_ICLR_experiments/stage4_core80_15algs_3seeds_3h/2、experiments/clean/drsr/Nguyen-6/520/experiments/g0641_Nguyen-6_drsr_seed520_20260924-094647/specs/generated_spec.txt`：`afc14604b535354ed77a71d2e656982080bb43aa1db5f2dab2dfec18a8f70498`
- `A_ICLR_experiments/stage4_core80_15algs_3seeds_3h/2、experiments/clean/drsr/Nguyen-6/520/result.json`：`28336277758f01f91707cf159ee3efc2956dde388f5a93e96d581f373a154048`
- `A_ICLR_experiments/stage4_core80_15algs_3seeds_3h/2、experiments/clean/llmsr/Nguyen-6/520/experiments/g0641_Nguyen-6_llmsr_seed520_20260924-094646/spec_dynamic.txt`：`f74f19c692fcce7909cf453348a1fd62a2df0c61e230eb2a5cfc2f462084646d`
- `A_ICLR_experiments/stage4_core80_15algs_3seeds_3h/2、experiments/clean/llmsr/Nguyen-6/520/result.json`：`fd96ff3caefc6d35fd05c1701a952567e18e80b44364a644fa47b27e3b0f9a75`
- `scientific_intelligent_modelling/algorithms/drsr_wrapper/drsr/drsr_420/buffer.py`：`94251912b8cfa3ef498f4d538f8bb4e004285eae3bae5aa08e299c888009c75e`
- `scientific_intelligent_modelling/algorithms/drsr_wrapper/drsr/drsr_420/code_manipulation.py`：`31bbc99b3f2d02d30c661b60580aa79891d8c3556f4899c2d6f8e3e0548283d0`
- `scientific_intelligent_modelling/algorithms/drsr_wrapper/drsr/drsr_420/evaluate_on_problems.py`：`dc19fd77760a6c9cd5d601ac5207f11eede8c3df7ffa0e21a9f8140063f0c448`
- `scientific_intelligent_modelling/algorithms/drsr_wrapper/drsr/drsr_420/prompt_config.py`：`575d5713862394dd5f11c3cf6ca8da9b712c048cd25c6d74fe96b43f598e70d3`
- `scientific_intelligent_modelling/algorithms/drsr_wrapper/drsr/drsr_420/sampler.py`：`79b379f4b92e6d7aa82fe3ebada12f95b938d6884c56064957cfecd0a653e676`
- `scientific_intelligent_modelling/algorithms/llmsr_wrapper/llmsr/llmsr/buffer.py`：`cde81e905076561900468378264db188d10e8377267684796e34d0270fe3fe4d`
- `scientific_intelligent_modelling/algorithms/llmsr_wrapper/llmsr/llmsr/code_manipulation.py`：`cbdb20df389b8a6b31dfdd60f564bf628a5adc9afc30e5d7305f1fd7732aa533`
- `scientific_intelligent_modelling/algorithms/llmsr_wrapper/llmsr/llmsr/sampler.py`：`034b9b3248a42e160891cad2a996ca22fae87812b6472a3c8cb4948e106cf277`
- `scientific_intelligent_modelling/benchmarks/runner.py`：`16cb510d91ecccae11b0f53bbcac12682c34b75778045c65db88720e87090c86`
- `scientific_intelligent_modelling/srkit/spec_builder.py`：`3c7d6728499d66a5147c399daf1ac93f515da1ad5ddd60bfce81dd124e828465`
