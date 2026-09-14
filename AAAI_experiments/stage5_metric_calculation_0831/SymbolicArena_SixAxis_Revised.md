# SymbolicArena Revised Six-Axis Evaluation

## 1. Evaluation Scope

The revised formal six-axis evaluation uses:

- **15 symbolic regression algorithms**
- **50 SSR-50 benchmark tasks**
- **3 random seeds:** 520, 521, and 522
- **3-hour wall-clock budget per run**
- **180 minute-level checkpoints** for efficiency analysis
- **Clean runs** for the formal six-axis leaderboard

The formal clean evaluation therefore contains:

\[
15 \times 50 \times 3 = 2250
\]

algorithm-task-seed runs.

Two additional noisy-training conditions are retained only as supplementary robustness diagnostics:

- `noise001`: 1% training-label noise
- `noise005`: 5% training-label noise

Across clean, noise001, and noise005, the complete experiment contains:

\[
15 \times 50 \times 3 \times 3 = 6750
\]

runs.

The revised formal six axes are:

\[
\boxed{\mathrm{ID,\ OOD,\ SYM,\ MIN,\ EFF,\ STAB}}
\]

They are organized into three groups:

- **Numerical Quality:** ID, OOD
- **Symbolic Quality:** SYM, MIN
- **Search Behavior:** EFF, STAB

ROBU is removed from the formal six-axis leaderboard and retained as a supplementary noise-robustness diagnostic.

---

## 2. Common Notation and Numerical Quality Mapping

All metrics are defined for one algorithm at a time, so the algorithm index is omitted for clarity.

Local quantities are defined for the current task or task-seed pair unless otherwise specified.

Throughout this section,

\[
\mathbb{E}[\cdot]
\]

denotes the empirical average over the evaluation units corresponding to the current metric. Task-seed-level quantities are averaged over all task-seed pairs, while task-level quantities are averaged over all benchmark tasks.

Let:

- \(D\): the set of SSR-50 benchmark tasks, with \(|D|=50\);
- \(S\): the random-seed set \(\{520,521,522\}\), with \(|S|=3\);
- \(T=180\): the search horizon in minutes;
- \(NMSE^{ID}\): in-distribution normalized mean squared error;
- \(NMSE^{OOD}\): out-of-distribution normalized mean squared error.

The common numerical quality mapping is

\[
\phi(x)
=
1-
\frac{
\operatorname{clip}
\left(
\log_{10}(\max(x,10^{-12})),
-12,2
\right)+12
}{14}.
\]

The corresponding ID and OOD quality values are

\[
q^{ID}
=
\phi\left(NMSE^{ID}\right),
\]

\[
q^{OOD}
=
\phi\left(NMSE^{OOD}\right).
\]

Here:

- \(x\) denotes an NMSE value;
- \(\log_{10}\) is the base-10 logarithm;
- \(10^{-12}\) prevents taking the logarithm of zero;
- \(\operatorname{clip}(z,-12,2)\) restricts log-NMSE to the fixed range \([-12,2]\);
- \(\phi(x)\in[0,1]\);
- larger values indicate better numerical prediction quality.

Invalid, timed-out, unparsable, NaN/Inf, or otherwise unevaluable outputs receive numerical quality 0.

Unlike the previous implementation, the revised formal scores do not first take the median of raw NMSE values across seeds. Each seed is evaluated independently, mapped to a bounded quality score, and then included in the empirical average.

---

## 2.1 Mandatory Direct Opus5 API Single-Turn Contract

All LLM-assisted operations in the complete symbolic-processing pipeline are mandatory direct Opus5 API evaluations. The LLM is not restricted to difficult cases and is not used merely as an optional fallback.

The mandatory coverage is:

1. each of the 50 original Ground Truth expressions receives an independent single-turn Opus5 API simplification;
2. every run across `clean`, `noise001`, and `noise005` with a non-empty final expression receives an independent single-turn Opus5 API simplification, covering up to 6750 final expressions;
3. every simplified experiment final expression receives an independent single-turn Opus5 API equivalence adjudication against its frozen simplified Ground Truth reference;
4. every valid clean seed pair used by the formal STAB axis receives an independent single-turn Opus5 API structural-consistency adjudication.

The same frozen simplified prediction is reused by SYM, MIN, and STAB. Reusing this artifact avoids contradictory simplifications without reducing LLM coverage.

Every API invocation must:

- request `claude-opus-5` with `output_config.effort=xhigh`;
- use `stream=false` and remain stateless;
- use the Anthropic Messages API directly, without Claude Code or tools;
- require a versioned JSON Schema response;
- receive only the explicitly prepared expression payload and deterministic evidence;
- record the input, prompt version, input hash, parsed text output, response hash, model metadata, token usage, latency, retry history, and terminal status. Internal thinking content is redacted from persistent artifacts.

The required invocation pattern is equivalent to:

```json
{
  "model": "claude-opus-5",
  "stream": false,
  "max_tokens": 6144,
  "thinking": {"type": "adaptive"},
  "output_config": {"effort": "xhigh"},
  "messages": [{"role": "user", "content": "<versioned-prompt-and-payload>"}]
}
```

SymPy checks, independent numerical probes, canonical-tree parsing, and other deterministic procedures remain mandatory evidence sources. They do not replace the Opus5 API call. When a request must be retried because of transport failure or schema-invalid output, every attempt is retained and the first schema-valid terminal result is frozen according to the versioned retry policy.

With complete valid coverage, the symbolic pipeline requires:

\[
50
+
6750
+
6750
+
15\times50\times\binom{3}{2}
=
15800
\]

independent single-turn Opus5 API evaluations. The formal six-axis scores still consume only clean runs, while the LLM artifacts for `noise001` and `noise005` are retained for supplementary diagnostics. Runs without a final expression and seed pairs containing such a run are recorded explicitly as missing or inconsistent rather than silently omitted.

---

# 3. ID: In-Distribution Quality

## Objective

ID measures:

> The final numerical prediction quality of an algorithm on in-distribution test data.

For one task-seed run,

\[
q^{ID}
=
\phi\left(NMSE^{ID}\right).
\]

The final ID score is

\[
\boxed{
ID
=
100\,\mathbb{E}\left[q^{ID}\right]
}
\]

where:

- \(q^{ID}\): ID quality of the current task-seed run;
- \(NMSE^{ID}\): ID NMSE of the current task-seed run;
- \(\mathbb{E}[\cdot]\): empirical average across all clean task-seed runs.

A higher ID score indicates better in-distribution predictive accuracy.

---

# 4. OOD: Out-of-Distribution Quality

## Objective

OOD measures:

> The final absolute numerical prediction quality of an algorithm on out-of-distribution test data.

For one task-seed run,

\[
q^{OOD}
=
\phi\left(NMSE^{OOD}\right).
\]

The final OOD score is

\[
\boxed{
OOD
=
100\,\mathbb{E}\left[q^{OOD}\right]
}
\]

where:

- \(q^{OOD}\): OOD quality of the current task-seed run;
- \(NMSE^{OOD}\): OOD NMSE of the current task-seed run;
- \(\mathbb{E}[\cdot]\): empirical average across all clean task-seed runs.

A higher OOD score indicates better absolute predictive quality under distribution shift.

The previous ID-to-OOD retention term is no longer included in the formal OOD score. OOD degradation or retention can be reported separately as a supplementary diagnostic.

---

# 5. SYM: Symbolic Fidelity

## Objective

SYM measures:

> How faithfully the recovered expression matches the underlying symbolic relation.

SYM uses a simplified and fixed Ground Truth reference expression.

For the original Ground Truth \(f^{gt}\),

\[
f^{ref}
=
\mathcal{S}\left(f^{gt}\right),
\]

where:

- \(f^{gt}\): original Ground Truth expression;
- \(\mathcal{S}(\cdot)\): unified expression-simplification procedure that requires an independent single-turn Opus5 API simplification;
- \(f^{ref}\): simplified Ground Truth reference expression.

Each of the 50 Ground Truth simplifications is executed once under the versioned single-turn API contract, reviewed against deterministic symbolic evidence, and frozen before any predicted expression is evaluated. All algorithms and all seeds use the same frozen reference for a given task.

For the predicted expression \(\hat f\),

\[
\tilde f
=
\mathcal{S}\left(\hat f\right),
\]

where:

- \(\hat f\): final expression returned by the SR algorithm;
- \(\tilde f\): simplified predicted expression.

Every non-empty final expression from all 6750 experiment runs is processed by an independent Opus5 API single-turn simplification. This requirement applies even when deterministic parsing or simplification already succeeds. Formal SYM uses the 2250 clean final results, while noisy-run symbolic artifacts remain supplementary. In addition, the mandatory dynamic telemetry extension in Section 11 applies the same frozen simplification contract to every distinct minute-level selected expression. Minute-level scoring is performed after the timed search and does not affect the algorithm wall-clock budget.

---

## 5.1 Mathematical Equivalence

Define

\[
Eq
=
\mathbf{1}
\left[
\tilde f
\equiv
f^{ref}
\right].
\]

Here:

- \(Eq=1\): the simplified prediction and simplified Ground Truth reference are judged mathematically equivalent;
- \(Eq=0\): exact mathematical equivalence is not established.

The equivalence procedure requires all of the following evidence:

- symbolic algebraic checking;
- independent numerical equivalence checking;
- an independent single-turn Opus5 API equivalence judgment.

Opus5 adjudicates every available final expression and every distinct minute-level selected expression across `clean`, `noise001`, and `noise005` against the corresponding frozen Ground Truth reference, not only difficult cases. Formal SYM uses only clean final-expression adjudications; minute-level adjudications form the dynamic telemetry artifact. Symbolic and numerical checks are included in every adjudication record as evidence and do not eliminate the mandatory API call.

---

## 5.2 Partial Symbolic Recovery

When exact equivalence is not established, SYM measures partial symbolic recovery.

Tree similarity is

\[
T
=
1-
NED
\left(
\tilde f,
f^{ref}
\right),
\]

where:

- \(NED\): normalized expression-tree distance;
- \(T\in[0,1]\);
- larger \(T\) indicates greater structural similarity.

Variable recovery is

\[
V
=
F1^{var},
\]

where:

- \(V\in[0,1]\);
- \(F1^{var}\) is the F1 score between the variable sets of \(\tilde f\) and \(f^{ref}\).

Operator recovery is

\[
O
=
F1^{op},
\]

where:

- \(O\in[0,1]\);
- \(F1^{op}\) is the F1 score between the operator sets of \(\tilde f\) and \(f^{ref}\).

The task-seed symbolic score is

\[
\boxed{
m^{SYM}
=
\begin{cases}
0,
&
\text{if no valid and parsable final expression is produced},
\\[4pt]
1,
&
Eq=1,
\\[4pt]
0.5
\left(
TVO
\right)^{1/3},
&
Eq=0.
\end{cases}
}
\]

The final SYM score is

\[
\boxed{
SYM
=
100\,\mathbb{E}\left[m^{SYM}\right]
}
\]

where:

- \(m^{SYM}\): symbolic-fidelity score of one final task-seed expression;
- \(Eq\): exact-equivalence indicator;
- \(T\): tree similarity;
- \(V\): variable recovery;
- \(O\): operator recovery.

The design gives full credit to mathematically equivalent expressions. If exact
equivalence is undetermined for an otherwise valid and parsable expression, it
is handled by the \(Eq=0\) branch and receives only evidence-backed partial
credit. Invalid, missing, or unparsable final expressions receive zero. Other
non-equivalent expressions receive partial credit according to tree, variable,
and operator recovery and can receive at most 0.5.

---

# 6. MIN: Expression Minimality

## Objective

MIN measures:

> How minimal and concise the final recovered expression is relative to the simplified Ground Truth reference.

MIN and SYM share the same simplified Ground Truth reference

\[
f^{ref}
=
\mathcal{S}\left(f^{gt}\right).
\]

The predicted expression is also simplified:

\[
\tilde f
=
\mathcal{S}\left(\hat f\right).
\]

The simplification procedure requires Opus5 to identify a concise mathematical form for every non-empty final expression and every distinct minute-level selected expression across all three conditions. Formal MIN uses the 2250 clean final results. MIN and its dynamic telemetry reuse exactly the frozen Opus5-simplified Ground Truth and prediction produced by the common symbolic pipeline; they do not run a competing simplification of the same expression.

The formal leaderboard MIN score is evaluated from final expressions. Minute-level selected expressions are evaluated separately under the dynamic telemetry contract in Section 11.

---

## 6.1 Expression Complexity

Define expression complexity as

\[
C(f)
=
\text{number of nodes in the canonical expression tree of } f.
\]

The Ground Truth reference complexity is

\[
C^{ref}
=
C\left(f^{ref}\right).
\]

The predicted-expression complexity is

\[
C^{pred}
=
C\left(\tilde f\right).
\]

Here:

- \(C(f)\): canonical expression-tree size;
- \(C^{ref}\): complexity of the simplified Ground Truth reference;
- \(C^{pred}\): complexity of the simplified predicted expression.

All algorithms use the same expression representation and complexity-counting rule.

---

## 6.2 Minimality Score

The task-seed MIN score is

\[
\boxed{
m^{MIN}
=
\min
\left(
1,
\frac{
C^{ref}
}{
C^{pred}
}
\right)
}
\]

If no valid and parsable final expression is produced,

\[
m^{MIN}=0.
\]

The final MIN score is

\[
\boxed{
MIN
=
100\,\mathbb{E}\left[m^{MIN}\right]
}
\]

Interpretation:

- if \(C^{pred}=C^{ref}\), then \(m^{MIN}=1\);
- if \(C^{pred}=2C^{ref}\), then \(m^{MIN}=0.5\);
- if \(C^{pred}=4C^{ref}\), then \(m^{MIN}=0.25\);
- if \(C^{pred}<C^{ref}\), the score is capped at 1.

A higher MIN score indicates a more concise recovered expression.

Under the current 15-algorithm, 50-task, 3-seed setting, the formal SYM and MIN
leaderboard aggregates

\[
15\times50\times3=2250
\]

clean final expressions. The symbolic pipeline still processes every available
final expression from all 6750 clean and noisy runs. The 4500 noisy-run symbolic
artifacts are retained only for supplementary diagnostics.

---

# 7. EFF: Efficiency

## Objective

EFF measures:

> How quickly an algorithm's native search or model-selection rule reaches the internally preferred solutions that it discovers within the fixed budget.

EFF is the only axis whose formal leaderboard definition aggregates the complete minute-level trajectory. The dynamic telemetry artifact nevertheless records all six axes at every minute.

For the current experiment,

\[
T=180.
\]

The formal trajectory uses **algorithm-native internal best-so-far**. At minute
\(t\), the incumbent is selected only from candidates discovered by that time,
using the algorithm's own loss, reward, score, or native model-selection rule.
A minimization objective keeps the lowest loss; a maximization objective keeps
the highest reward or score. Exact ties retain the earliest incumbent unless
the algorithm declares a different native tie-break.

The selected expression is then sent to the common evaluator for ID and OOD
scoring. ID/OOD test results never select a candidate and never affect search,
model selection, or parameter updates. Consequently \(q(t)\) is allowed to
decrease even while the native internal objective improves. LLM-assisted
simplification, SYM, MIN, and structural STAB adjudication are computed after
the timed run so that evaluation overhead cannot change search behavior.

Before the first auditable incumbent, quality is zero. Between updates and
after an early algorithm termination, the latest native incumbent is carried
forward through minute 180 with its discovery minute recorded. A future
snapshot, final formula, or test quality must not be used to fill an earlier
minute. A minute lacking auditable candidate-selection evidence is marked
`unavailable`; it is not silently filled with zero and blocks formal EFF for
that run. An algorithm is formal-ready only when all 150 clean runs are
auditable under this rule.

---

## 7.1 Minute-Level Numerical Quality

At minute \(t\),

\[
q^{ID}(t)
=
\phi\left(NMSE^{ID}(t)\right),
\]

\[
q^{OOD}(t)
=
\phi\left(NMSE^{OOD}(t)\right).
\]

The combined search quality is

\[
q(t)
=
\frac{
q^{ID}(t)
+
q^{OOD}(t)
}{2}.
\]

where:

- \(t\in\{1,\ldots,180\}\);
- \(NMSE^{ID}(t)\): ID NMSE of the native incumbent expression at minute \(t\);
- \(NMSE^{OOD}(t)\): OOD NMSE of the same expression;
- \(q^{ID}(t)\): ID quality at minute \(t\);
- \(q^{OOD}(t)\): OOD quality at minute \(t\);
- \(q(t)\in[0,1]\): combined numerical search quality.

After the complete trajectory has been frozen and evaluated, define

\[
q^{*}
=
\max_{1\le t\le T}q(t).
\]

where:

- \(q^*\): largest post-hoc numerical quality of the native-incumbent
  trajectory. It is a normalization constant only and never participates in
  incumbent selection.

---

## 7.2 Relative Search Progress

If \(q^*>0\),

\[
\boxed{
m^{EFF}
=
\frac{1}{T}
\sum_{t=1}^{T}
\frac{
q(t)
}{
q^{*}
}
}
\]

If the run has a complete auditable trajectory but \(q^*=0\),

\[
m^{EFF}=0.
\]

The final EFF score is

\[
\boxed{
EFF
=
100\,\mathbb{E}\left[m^{EFF}\right]
}
\]

where:

- \(m^{EFF}\): efficiency score of one clean task-seed run;
- \(T=180\): total search horizon in minutes;
- \(q(t)\): combined numerical quality at minute \(t\);
- \(q^*\): best numerical quality reached within the complete budget.

EFF measures the evaluated progress of the algorithm-native incumbent relative
to the best evaluated point on that same frozen trajectory. It does not turn
the trajectory into numerical best-so-far: \(q(t)\) can decrease. Absolute
final prediction quality is separately measured by ID and OOD.

---

# 8. STAB: Stability

## Objective

STAB measures:

> How consistently an algorithm behaves across repeated random-seed runs on the same task.

The current experiment uses three seeds:

\[
S=\{520,521,522\}.
\]

Therefore each algorithm-task pair contains

\[
\binom{3}{2}=3
\]

seed pairs.

STAB contains three components:

1. Numerical Consistency;
2. Validity;
3. Structural Consistency.

The previous performance correction based on ID, OOD, and SYM is removed.

---

## 8.1 Numerical Consistency

For two seeds \(i\) and \(j\), define

\[
\delta^{num}_{ij}
=
\frac{
\left|
q^{ID}_{i}
-
q^{ID}_{j}
\right|
+
\left|
q^{OOD}_{i}
-
q^{OOD}_{j}
\right|
}{2}.
\]

where:

- \(q^{ID}_i\), \(q^{ID}_j\): final ID qualities produced by seeds \(i\) and \(j\);
- \(q^{OOD}_i\), \(q^{OOD}_j\): final OOD qualities produced by seeds \(i\) and \(j\);
- \(\delta^{num}_{ij}\in[0,1]\): numerical disagreement between the two runs.

For one benchmark task,

\[
\boxed{
N
=
1-
\frac{1}{\binom{|S|}{2}}
\sum_{i<j}
\delta^{num}_{ij}
}
\]

where:

- \(N\in[0,1]\): Numerical Consistency;
- larger \(N\) indicates more consistent numerical results across seeds.

With three seeds, the average is computed over the pairs:

- (520, 521)
- (520, 522)
- (521, 522)

---

## 8.2 Validity

Define

\[
\boxed{
V
=
\frac{
\#\{\text{valid seeds}\}
}{
|S|
}
}
\]

where:

- \(V\in[0,1]\): Validity;
- \(V=1\): all three seeds produce valid final expressions;
- smaller \(V\): the algorithm more frequently produces invalid outputs, failures, or timeouts.

---

## 8.3 Structural Consistency

For a pair of seeds \(i,j\), define

\[
I^{struct}_{ij}
=
\begin{cases}
1,
&
\text{if the two simplified final expressions are structurally consistent},
\\
0,
&
\text{otherwise}.
\end{cases}
\]

Formal structural consistency is assessed from the frozen Opus5-simplified final expressions shared with the SYM/MIN symbolic-processing pipeline. Dynamic structural consistency applies the same rule to the three same-minute selected expressions. Every distinct valid expression pair receives an independent single-turn Opus5 API structural-consistency adjudication; an identical repeated pair may reuse its frozen result.

A pair is considered structurally consistent when Opus5 judges that the final expressions are mathematically equivalent or share the same canonical symbolic structure, using deterministic equivalence and canonical-tree evidence supplied with the request.

If either seed does not produce a valid final expression, the pair is treated as inconsistent.

For one benchmark task,

\[
\boxed{
C
=
\frac{1}{\binom{|S|}{2}}
\sum_{i<j}
I^{struct}_{ij}
}
\]

where:

- \(C\in[0,1]\): Structural Consistency;
- larger \(C\) indicates more consistent recovered symbolic structures across seeds.

---

## 8.4 Final Stability Score

For one benchmark task,

\[
\boxed{
m^{STAB}
=
\left(
NVC
\right)^{1/3}
}
\]

The final STAB score is

\[
\boxed{
STAB
=
100\,\mathbb{E}\left[m^{STAB}\right]
}
\]

where:

- \(N\): Numerical Consistency;
- \(V\): Validity;
- \(C\): Structural Consistency;
- \(m^{STAB}\): task-level stability score.

The geometric mean requires numerical, output-validity, and symbolic-structure consistency to be jointly strong.

---

# 9. Evaluation Artifacts Used by Each Axis

| Evaluation Artifact | ID | OOD | SYM | MIN | EFF | STAB |
|---|---:|---:|---:|---:|---:|---:|
| Clean final expression | ✓ | ✓ | ✓ | ✓ |  | ✓ |
| Minute-level selected/best-so-far trajectory | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Mandatory Opus5 API single-turn processing |  |  | Every distinct expression | Reuses SYM simplification |  | Every distinct valid seed pair |
| noise001 / noise005 runs | Supplementary | Supplementary | Supplementary | Supplementary | Supplementary | Supplementary |

The formal leaderboard keeps the computational separation:

\[
\text{Minute-level trajectory}
\rightarrow
\text{ID/OOD numerical evaluation}
\rightarrow
EFF
\]

and

\[
\text{Final expression}
\rightarrow
ID,\ OOD,\ SYM,\ MIN,\ STAB.
\]

The dynamic telemetry extension additionally applies mandatory Opus5 API processing to each distinct minute-level selected expression, its Ground Truth equivalence decision, and each distinct valid same-minute seed pair. Exact duplicate expressions and identical adjudication inputs may reuse a content-addressed frozen result, but no distinct input may bypass the API call.

---

# 10. Supplementary Noise Robustness

Noise robustness is no longer one of the six formal leaderboard axes.

The current experiment retains two noisy-training conditions:

\[
\sigma\in\{0.01,0.05\}.
\]

These correspond to:

- 1% training-label noise;
- 5% training-label noise.

They are reported separately as supplementary robustness diagnostics. Their final and minute-level expressions receive the same mandatory single-turn Opus5 API simplification and equivalence adjudication as clean runs, but those symbolic outputs do not enter the formal clean six-axis leaderboard.

The complete experiment therefore contains:

\[
2250\text{ clean}
+
2250\text{ noise001}
+
2250\text{ noise005}
=
6750\text{ runs}.
\]

---

# 11. Mandatory Minute-Level Six-Axis Telemetry

This telemetry contract applies to historical backfilling and to every future
data-collection run. It extends observability without changing which artifacts
define the formal clean leaderboard.

For every algorithm-task-seed run and every minute

\[
t\in\{1,\ldots,180\},
\]

the retained run-minute record contains the selected or best-so-far expression,
its source evidence, validity, \(q^{ID}(t)\), \(q^{OOD}(t)\),
\(m^{SYM}(t)\), \(m^{MIN}(t)\), and cumulative efficiency

\[
m^{EFF}(t)
=
\frac{1}{t}
\sum_{u=1}^{t}
\frac{q(u)}{q^*},
\qquad
q^*=\max_{1\le u\le180}q(u).
\]

If no new candidate is emitted at minute \(t\), the record explicitly carries
forward the preceding selected expression and identifies the source minute.
Before the first valid expression, the minute is recorded as invalid with
numerical and symbolic scores equal to zero. A missing expression must never be
replaced by the final expression.

For every algorithm-task-minute, dynamic stability aligns the three seeds at
the same minute and computes

\[
m^{STAB}(t)
=
\left(N(t)V(t)C(t)\right)^{1/3},
\]

using the same numerical-consistency, validity, and structural-consistency
definitions as the final STAB axis. Structural decisions operate on the frozen
Opus5-simplified same-minute expressions.

The complete dynamic artifact contains, per condition,

\[
15\times50\times3\times180=405000
\]

run-minute rows and

\[
15\times50\times180=135000
\]

task-minute STAB rows. Every row preserves the condition, algorithm, dataset,
seed or seed pair, minute, logical identity, expression hash, source path and
hash, evaluation basis, LLM contract version, and frozen adjudication
identifiers.

All LLM work is executed after the timed algorithm run through a resumable,
content-addressed plan. Each distinct expression and each distinct adjudication
input receives the mandatory single-turn Opus5 evaluation. Exact duplicates may
reuse a frozen response. API failures are retried under the declared retry
budget; unresolved inputs remain explicit and block a complete dynamic release.
