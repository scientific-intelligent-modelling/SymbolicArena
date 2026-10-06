# Agent-driven algorithm integration

The sr-tool-onboarder skill takes an author repository and integrates its real implementation into SymbolicArena. The agent reads the training code, edits the adapter with file editing tools, registers an isolated environment, and runs executable acceptance gates. The sim-onboard CLI provides source inspection, environment setup, and evidence-producing validation.

An integration is defined by tools/sr_onboarder/manifests/&lt;tool&gt;.json. The manifest fixes the upstream commit, required source file hashes, wrapper class, environment, native search defaults, acceptance settings, and Core50 selection. Its JSON Schema is maintained in scientific_intelligent_modelling/onboarding/manifest.py.

## Agent workflow

Give the agent the source URL and ask it to use sr-tool-onboarder. It uses the prepare command to create a pinned checkout, source-review.json, and agent_task.md. The agent then implements the actual training and prediction adapters and executes:

    sim-onboard validate --manifest tools/sr_onboarder/manifests/dgp.json --output-root .agent/work/dgp-integration/structure
    sim-onboard setup --manifest tools/sr_onboarder/manifests/dgp.json --output-root .agent/work/dgp-integration/environment
    sim-onboard accept --manifest tools/sr_onboarder/manifests/dgp.json --stage smoke --dataset-dir examples/BPG0 --output-root .agent/work/dgp-integration/smoke
    sim-onboard accept --manifest tools/sr_onboarder/manifests/dgp.json --stage budget --dataset-dir examples/BPG0 --output-root .agent/work/dgp-integration/budget
    sim-onboard accept --manifest tools/sr_onboarder/manifests/dgp.json --stage core50 --datasets-root /path/to/sim-datasets-data --workers 4 --output-root .agent/work/dgp-integration/core50

Install the CLI with python -m pip install -e . Before it is installed, the same commands can use python -m scientific_intelligent_modelling.onboarding.cli.

## Acceptance criteria

| Gate | Required evidence |
| --- | --- |
| Source and structure | Fixed author commit, file checksums, complete wrapper API, matching environment and algorithm registration, frozen Core50 checksum |
| Native API | A real fit, sample-length finite predictions, numerical replay matching the native operator semantics, serialization with identical restored predictions |
| Subprocess and runner | Real training through SymbolicRegressor, nonempty native equation, canonical artifact, ID/OOD metrics in the shared result.json |
| Budget and minutes | Budget-limited search with a recoverable native candidate, actual minute snapshots, native training objective, immutable candidate files and SHA256 provenance |
| Core50 | Prevalidated metadata and split hashes, all 50 tasks for every configured seed, one actual standard result record for each run |
| Formal six-axis evaluation | Complete minute-level symbolic scoring and progress evidence, plus cross-seed stability evidence |

The reports contain their input versions, parameters, source files, prediction evidence, and result paths. A stage writes passed=false until all of its required work completes. Acceptance configurations are retained separately from native search defaults. Missing symbolic or cross-seed evidence is explicit in unresolved_axes; formal_six_axis_ready remains false until that evaluation is completed.

## DGP source and adapter

DGP uses [the author's repository](https://github.com/songxt3/DGP) at commit dc8a8634363399270291576ec575c379dff1f9bf. Set SIM_DGP_SOURCE to that external checkout. The source does not provide a license; its code is referenced externally. The integration's source changes are retained in tools/sr_onboarder/patches/dgp.apply_patch and applied by the agent's file editor. The manifest verifies the resulting source hashes.

The source adjustments support one to four input features, keep gradients through the native log/exp operators, use the computed protected denominator in native division, and expose worker, budget, and candidate callbacks. Search continues to use the author's differentiable tree optimizer and DEAP diversification. The wrapper preserves the training-R² selection policy and exports the actual selected DEAP tree. Protected division, log, and exp are reproduced explicitly in the canonical artifact; variable indices remain unchanged.

The adapter writes each best candidate to an immutable file, then references it from the current-best record and history. The shared runner reads these records through the algorithm's registered progress filenames and normalizer. Additional algorithms can implement this same interface without adding algorithm-specific branches to the runner.

Continuous optimization uses float64 tensors and constants to support the raw numerical ranges in Core50. Finite training candidates are admitted across the full training-R² range; non-finite training candidates receive an explicit invalid objective. Minute records retain their logical task key, original numerical errors, clipped log-NMSE quality scores, expression and complexity evidence, native objective, metadata checksum, and pending symbolic/progress/stability judgments.

Remote acceptance uses an isolated checkout of the integration branch and an explicit dataset root. The --tmux option gives each dataset and seed a separate session and task log while limiting concurrent sessions with --workers. Sockets are stored inside the task directory. Source and data hashes are checked on the executing host. The deployment helper is tools/sr_onboarder/deploy_remote.py; it creates a new checkout, verifies its commit, transfers the pinned native source and dedicated environment, and launches Core50 acceptance. Native source, dependencies, result records, and process logs must be retained with the acceptance report.
