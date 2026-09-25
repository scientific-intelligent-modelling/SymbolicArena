import ast
import csv
import hashlib
import importlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
STAGE = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "AAAI_experiments/stage5_metric_calculation_0831"
ALGORITHMS = ROOT / "scientific_intelligent_modelling/algorithms"
OUTPUT = STAGE / "3、metrics/opus_llama_protocol_report.md"
SOURCES = {}
BLOCKS = []

MAIN = r'''# Core50：Opus5指标计算与Llama训练报告

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
'''


def source_text(path):
    path = path.resolve()
    data = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return data.decode("utf-8")


def literal_assignment(path, name):
    tree = ast.parse(source_text(path))
    values = [ast.literal_eval(node.value) for node in ast.walk(tree)
              if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == name
                      for target in node.targets)]
    assert len(values) == 1, (path, name, len(values))
    return values[0]


def section(title, body, language="text"):
    BLOCKS.append(body)
    return f"\n### {title}\n\n```{language}\n{body}" + ("" if body.endswith("\n") else "\n") + "```\n"


def load_module(name, path):
    source_text(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_initial_message(algorithm):
    package = "llmsr" if algorithm == "llmsr" else "drsr_420"
    package_root = ALGORITHMS / f"{algorithm}_wrapper" / algorithm
    sys.path.insert(0, str(package_root))
    manipulation = importlib.import_module(f"{package}.code_manipulation")
    buffer = importlib.import_module(f"{package}.buffer")
    source_text(Path(buffer.__file__))
    source_text(Path(manipulation.__file__))
    result_root = STAGE / f"2.2 core50 experiments/clean/{algorithm}/Nguyen-6/520"
    pattern = "experiments/*/spec_dynamic.txt" if algorithm == "llmsr" else "experiments/*/specs/generated_spec.txt"
    paths = list(result_root.glob(pattern))
    assert len(paths) == 1, paths
    specification = source_text(paths[0])
    params = json.loads(source_text(result_root / "result.json"))["params"]
    template = manipulation.text_to_program(specification)
    island = buffer.Island(template, "equation", 2, 0.1, 30000)
    program = island._generate_prompt([template.get_function("equation")])
    return specification, params, program


def main():
    report = MAIN
    report += section("A.1 Opus共用system消息", literal_assignment(
        PIPELINE / "pipeline/anthropic_api_runner.py", "STRICT_EVALUATOR_SYSTEM_PROMPT"))
    report += "\nAPI的user消息为对应模板替换`{{REQUEST_JSON}}`后，再追加`\\n\\nOUTPUT_JSON_SCHEMA_DRAFT_07:\\n`和该任务的完整JSON Schema。REQUEST_JSON包含当前公式、变量、定义域、证据与来源绑定；固定文字和完整Schema如下。\n"
    for index, kind in enumerate(("simplify", "equivalence", "structure"), 2):
        filename = "simplify_core50_exact.v1.txt" if kind == "simplify" else f"{kind}.v1.txt"
        report += section(f"A.{index} {kind} user模板", source_text(PIPELINE / "config/prompts" / filename))
        schema = json.loads(source_text(PIPELINE / f"config/schemas/{kind}.v1.json"))
        report += section(f"A.{index}.1 完整输出Schema", json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")), "json")
    report += "\nstructure模板保留原文`clean-run`；本轮执行器也对noise001/noise005建立独立条件的结构任务，实际请求中的condition/noise_tag及对应公式决定比较范围。已有裁决使用各自保存的prompt哈希，化简精确模板描述当前补充调用规则。\n"
    neutral = "This is a symbolic regression task. Find a compact mathematical equation that predicts the target from the observed variables."
    report += section("B.1 早期探针通用background完整文字", neutral)
    report += section("B.2 metadata未提供description时的background完整模板", "Find the mathematical function skeleton that represents {target_desc}, given data on {feature_text}.")
    report += section("B.3 SRSD干扰变量background补充句完整模板", 'There are {n_total} candidate variables in an unknown order. The unordered semantic-role multiset is: {role_text}. The mapping from semantic roles to variable names is intentionally hidden; do not assume which x_i corresponds to which semantic role.')
    report += '\n`role_text`按语义名称排序，条目以`; `连接；语义条目为`{count} variable(s) with semantic role "{role}"`，干扰条目为`{count} distractor/meaningless variable(s)`；数量为1时使用variable，其他数量使用variables。\n'
    instruction = literal_assignment(ALGORITHMS / "llmsr_wrapper/llmsr/llmsr/sampler.py", "instruction_prompt")
    llm_spec, _, llm_program = build_initial_message("llmsr")
    report += section("C.1 LLM-SR固定生成指令", instruction)
    report += section("C.2 LLM-SR真实完整运行spec：Nguyen-6/clean/520", llm_spec, "python")
    report += section("C.3 LLM-SR完整初始user消息：Nguyen-6", "\n".join((instruction, llm_program)))
    report += "\n后续轮次仍使用同一指令和代码前言，历史函数按`equation_v0`、`equation_v1`编号；第二个及后续函数的docstring为`Improved version of `加反引号包围的前一个函数名及句号；最后追加同签名、空函数体的下一版本。历史函数体来自该次训练保存的候选。\n"
    pc_path = ALGORITHMS / "drsr_wrapper/drsr/drsr_420/prompt_config.py"
    pc = load_module("report_prompt_config", pc_path)
    dr_spec, params, dr_program = build_initial_message("drsr")
    ctx = pc.PromptContext(
        n_features=params["n_features"], feature_names=params["prompt_feature_names"],
        dependent_name=params["prompt_target_name"], background=params["background"],
        feature_descriptions=params["feature_descriptions"],
        target_description=params["target_description"], max_params=params["max_params"],
    )
    report += section("D.1 DrSR任务头模板", pc.head_template)
    report += section("D.2 DrSR固定生成指令", pc.instruction_prompt)
    report += "\n生成指令末尾再拼接`Variables:\\n{variables_block}\\nBackground: {background_text}\\n`；任务头中的dependent包含目标描述，independent包含各变量描述。下方完整消息展示实际替换结果。\n"
    report += section("D.3 DrSR真实完整运行spec：Nguyen-6/clean/520", dr_spec, "python")
    report += section("D.4 DrSR完整初始user消息：Nguyen-6", ctx.render_head() + "\n" + "\n".join((ctx.render_instruction(), dr_program)))
    report += section("D.5 DrSR经验总结完整会话模板", pc.analysis_conversation_template)
    report += "\n`prompt`取buffer生成的`prompt.code`，包含代码前言和候选函数；此处不包含生成请求额外添加的任务头、经验区块等外层文字。`sample`为模型回答，`question`使用下方对应模板，整体作为一个user消息发送。以下使用实际目标名y和参数预算10渲染，`{error}`保留为实际错误文字字段。\n"
    for index, quality in enumerate(("Good", "Bad", "None"), 1):
        report += section(f"D.6.{index} {quality}完整追问模板", ctx.render_analysis_question(quality, error="{error}"))
    report += section("D.7 DrSR残差分析完整user模板：Nguyen-6变量配置", ctx.render_residual_analysis_prompt("{last_analysis}", "{residual}", "{sample}"))
    report += "\n`last_analysis`为既有分析，`residual`为当前训练残差矩阵的文本，`sample`为对应候选代码。多变量时按全部变量生成independent_to_dependent_relationships，并为每个变量对生成inter_relationships_between_independents。原模板的`forth column`为固定文字；实际残差位于矩阵最后一列，列数为特征数加2，这条文字仅在两个特征时符合列号。\n"
    report += section("D.8 DrSR经验注入区块完整模板", pc.ideas_block_title + pc.idea_item_prefix + "{analysis_text}\n---\n\n")
    report += section("D.9 DrSR残差注入区块完整模板", ctx.render_residual_block_title() + "{last_analysis}\n\n")
    report += "\nD.8按选中经验重复生成条目，index从1递增；可选区块按照正文说明加入生成消息。D.5至D.9中的占位字段都来自当前训练记录。\n"
    report += "\n## 5. 核验与来源\n\n"
    counts = Counter()
    for condition in ("clean", "noise001", "noise005"):
        for algorithm in ("llmsr", "drsr"):
            paths = list((STAGE / f"2.2 core50 experiments/{condition}/{algorithm}").glob("*/*/result.json"))
            assert len(paths) == 150, (condition, algorithm, len(paths))
            for path in paths:
                parameters = json.loads(path.read_text())["params"]
                counts[(algorithm, parameters.get("inject_prompt_semantics"), parameters.get("canonical_prompt_variables"))] += 1
    assert counts == Counter({("llmsr", True, True): 450, ("drsr", True, True): 450}), counts
    report += "- 已核验当前两算法共900份result.json：各450份，均开启语义注入和统一变量命名。两份完整spec来自实际保存文件；初始消息使用各算法原生buffer渲染，未请求API。\n"
    stage2 = ROOT / "A_ICLR_experiments/stage2_200dats_12algs_1seed_1h/04_semantic_llm_override/semantic_results_raw_400.csv"
    with stage2.open(newline="") as handle:
        records = list(csv.DictReader(handle))
    assignments = Counter((row["algorithm"], row["inject_prompt_semantics"], row["llm_model_assignment"]) for row in records)
    assert len(records) == 400 and len(assignments) == 4 and set(assignments.values()) == {100}
    source_text(stage2)
    report += "- Stage2语义补充实验400条结果的模型分组和语义开关已按CSV重新计数。当前API生成参数于2026-09-26在iaaccn22读取两份配置的非敏感字段核验；历史每个HTTP请求的参数未逐条恢复。\n"
    additional_sources = [
        ROOT / "scientific_intelligent_modelling/benchmarks/runner.py",
        ROOT / "scientific_intelligent_modelling/srkit/spec_builder.py",
        ALGORITHMS / "drsr_wrapper/drsr/drsr_420/evaluate_on_problems.py",
        ALGORITHMS / "drsr_wrapper/drsr/drsr_420/sampler.py",
        PIPELINE / "pipeline/metrics.py",
        PIPELINE / "pipeline/symbolic_evidence.py",
        ROOT / "A_ICLR_experiments/stage1_664dats_2probes_1seed_1h/03_launch_materials/launch_llmsr_probe.py",
        ROOT / "A_ICLR_experiments/stage1_664dats_2probes_1seed_1h/03_launch_materials/probe_parameter_plan.md",
        ROOT / "A_ICLR_experiments/stage2_200dats_12algs_1seed_1h/07_stage2_run_provenance/hyperparams_snapshot_20260423/llmsr.json",
        ROOT / "A_ICLR_experiments/stage2_200dats_12algs_1seed_1h/07_stage2_run_provenance/hyperparams_snapshot_20260423/drsr.json",
    ]
    for path in additional_sources:
        source_text(path)
    report += "\n源码与模板输入SHA256如下，路径相对于项目根目录。重新生成脚本：`1、preflight/build_llm_protocol_report.py`。\n\n"
    for path, digest in sorted(SOURCES.items()):
        report += f"- `{path}`：`{digest}`\n"
    assert all(block in report for block in BLOCKS)
    assert report.count("```") % 2 == 0
    for marker in ("olmcp_", "Bearer ", '"api_key":'):
        assert marker not in report, marker
    if OUTPUT.exists():
        assert OUTPUT.read_text() == report, "报告内容已变化，请核验后明确更新文件"
    else:
        OUTPUT.write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(OUTPUT), "template_blocks": len(BLOCKS), "verified_runs": 900,
                      "source_files": len(SOURCES), "report_sha256": hashlib.sha256(report.encode()).hexdigest()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
