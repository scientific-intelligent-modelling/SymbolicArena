# OpenReview Fields

## Title

```text
SymbolicArena: A Unified Platform for Dynamic and Multi-Dimensional Evaluation of Symbolic Regression
```

## Keywords

```text
Symbolic Regression, Benchmarking, Knowledge Discovery, Scientific Discovery, Evaluation Protocols, Reproducibility
```

## TL;DR

```text
SymbolicArena is a unified benchmark distillation and evaluation platform for symbolic regression, providing a validated Core-50 benchmark, 12-algorithm evaluation, standardized artifacts, and multi-dimensional scoring.
```

## Abstract

```text
Symbolic regression (SR) is rapidly diversifying across evolutionary, reinforcement-learning, tree-search, neural-guided, transformer-based, and LLM-assisted paradigms, while evaluation remains fragmented across heterogeneous benchmarks and protocols. SymbolicArena introduces a unified infrastructure for benchmark distillation and dynamic SR evaluation. Starting from nearly one thousand candidate tasks, SymbolicArena constructs a 664-task quality-checked ground-truth reservoir, integrates 12 representative algorithms, and distills a fixed Core-50 benchmark through cascaded probe-based selection. The selection preserves task-structure coverage, algorithmic response diversity, method discriminability, redundancy control, difficulty balance, and seed-level stability at substantially lower evaluation cost. SymbolicArena further evaluates algorithms under a unified one-hour protocol with standardized artifacts, failure semantics, expression canonicalization, and minute-level best-so-far logging. A fixed absolute 0--100 hexagonal protocol covers in-distribution numerical quality, out-of-distribution generalization, symbolic fidelity, efficiency, stability, and noisy-training robustness. Supported by nearly 30,000 runs, SymbolicArena provides a reproducible and extensible evaluation substrate for modern SR research.
```

## Review Mode

```text
Double-blind
```

Use single-blind only if the hosted dataset/code cannot be anonymized. The current package is prepared for double-blind review.

## Dataset Submission

```text
This submission includes a dataset: yes
```

## Dataset URL

```text
TODO_REPLACE_WITH_ANONYMOUS_DATASET_URL
```

Use the anonymous URL where `SymbolicArena_NeurIPS26_ED_full_dataset_payload.tar.zst` is hosted.

## Dataset Large URL

```text
Leave blank unless OpenReview requires it.
```

The compressed full dataset payload is approximately 1.1GB, which is below the 4GB large-dataset threshold shown in the OpenReview form. If OpenReview still requires a sample URL, use the same anonymous dataset-hosting page or a Core-50 sample page from the same host.

## Code URL

```text
TODO_REPLACE_WITH_ANONYMOUS_CODE_URL
```

Use the anonymous repository created from `upload-ready/anonymous-code-url/` or the cleaned full runnable repository.

## Code Submission Justification

Use this only if a Code URL is not provided. The recommended path for this submission is to provide a Code URL.

```text
Not applicable. This submission provides an anonymized Code URL for the benchmark infrastructure and evaluation scripts.
```

## Supplementary Material

```text
SymbolicArena_NeurIPS26_ED_supplementary_material.zip
```

## License

```text
CC BY 4.0
```

## LLM Usage

Select all applicable confidential checkboxes in the OpenReview form. At minimum, editing was used. If available in the form, also select categories covering code assistance, experiment orchestration, data analysis, and artifact preparation.
