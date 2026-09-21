# Shared KIT foundation and heliaEDGE direction

Status: architecture proposal, 2026-09-20. This extends the SleepKit refactor to
heartKIT, sleepKIT and compressionKIT. It does not claim the shared APIs, backend
support or performance improvements below are implemented. heliaEDGE is the intended
home for proven domain-neutral capabilities; KITs remain owners of domain recipes.

## Progressive adoption

A user should be able to import one block, compose an experiment in ordinary Python,
make it reproducible, and promote its artifacts to a supported release. Each step
adds optional capabilities without requiring a new base class or universal runner.
Golden automation invokes the same public functions as a custom script. Configuration
selects a reviewed recipe and its parameters; Python defines the operations.

| Boundary | heliaEDGE responsibility | KIT responsibility |
| --- | --- | --- |
| Data preparation | Optional cache/shard utilities, fingerprints, sample manifests, loader adapters | Dataset access, channel meaning/units, annotation policy, subject splits, physiological transforms |
| Modeling | Portable Keras layers/models/losses/metrics; explicit backend-specific components when needed | Task architectures, targets, objectives and training choices |
| Execution | Optional loader/training helpers, profiling and run-evidence utilities | Readable recipes, experiment parameters and quality criteria |
| Artifacts | File/tensor contracts, hashes, conformance vectors, conversion adapters, validation and HF transport | Task/codec semantics, attribution, model licensing, model cards, golden IDs and publishing destinations |

The data contract describes identity, shape/dtype, units/time alignment where relevant,
validity and provenance. It does not require a universal in-memory tensor wrapper or
one dataset file format. Native arrays, tensor structures and loaders remain usable.
Adapters translate contracts at boundaries instead of making every component depend
on SleepKit records or compressionKIT codec specifications.

An initial reusable block should have two real consumers before its public API is
promoted as shared. Migrate incrementally through compatibility adapters; preserve
existing released bundles and baseline identities. Avoid a flag-day rename or moving
detection-specific helpers wholesale into heliaEDGE.

## Backend support means tested workflows

Support both TensorFlow and PyTorch as first-class execution choices:

- Keras with either backend, using `keras.ops`, layers and random APIs for portable
  components where feasible.
- Native TensorFlow loops using appropriate components and ordinary TF tensors.
- Native PyTorch loops using appropriate components, optimizers and DataLoaders.
- Native user models can use shared data, evidence and release utilities without
  being rewritten as Keras models.

[Keras supports native training-loop integration](https://keras.io/guides/writing_a_custom_training_loop_in_torch/),
and its [portable layer guidance](https://keras.io/guides/making_new_layers_and_models_via_subclassing/)
distinguishes Keras operations from backend-specific code. Reuse a portable Keras
implementation when its native integration is suitable. Provide a native implementation
or adapter where measured performance or interoperability requires it; do not duplicate
every block speculatively or hide unsupported behavior behind a nominal backend switch.

Separate optional backend dependencies and lazy imports. Importing artifact utilities
must not initialize a training framework. Installing and using Torch support must not
require TensorFlow; a TF-only environment must not require Torch. Keras backend choice
is established before imports, with backend tests in separate processes/environments.
Exact package extra names are an implementation decision, not a fixed API in this plan.

For each supported component, test forward results and gradients against declared
tolerances, parameter registration, train/eval behavior, randomness, save/reload and
optimizer updates. Test custom-loop interoperability as well as `fit()`. Record
mixed-precision, compilation, distribution and export support separately. Matching
architecture does not promise identical training trajectories or optimizer checkpoints
across backends. A Torch-training claim is not automatically a LiteRT export claim.

Convenience trainers can own a default loop, but objectives, models, data preparation,
evaluation and export must remain independently callable. Backend-specific gradient
application belongs in small explicit adapters. Do not introduce a universal trainer
with hooks for every possible model family.

## Inspection findings and their limits

These are source observations, not measured performance results:

| Project and inspected revision | Observation | Consequence to investigate |
| --- | --- | --- |
| heliaEDGE `99a12249acc1ab7fb04364a1e223c44051783086` | TensorFlow is required in `pyproject.toml`; `helia_edge/__init__.py` eagerly imports modules; `trainers/mask_autoencoder.py` and `trainers/contrastive.py` reject Torch training/testing | Packaging/import isolation and real trainer implementation are needed before first-class Torch claims |
| sleepKIT `edd4ec8` | `recipes/detection/recipe.py` uses `Dataset.from_generator`, a private thread pool of one and prefetch of one; the recipe also owns fixed detection shapes | Correctness-first reference path, not a throughput-optimized shared loader/trainer |
| heartKIT `64cd51b3e1d82c4a325ae2097fe245fc839dea36` | `datasets/dataloader.py` wraps Python patient generators in `tf.data`; task pipelines add parallel maps/prefetch afterward | Parallel downstream transforms do not establish parallel source decoding |
| compressionKIT `7fb36636498af6b053e1451a87286a2e699d3874` | `datasets/ecg.py` interleaves Python generators; `datasets/ppg.py` uses a serial `py_function` read path with an EDF thread-safety comment; trainer transforms also use NumPy callbacks | Measure read/decode/transform costs and use process isolation or prepared data where appropriate |

heliaEDGE's `utils/preprocessing.py:create_interleaved_dataset_from_generator`
currently names interleaved generators workers; this is not a process pool. Its
floor-division ID partitioning also leaves a remainder unassigned when the ID count
is not divisible by worker count. Address partition coverage and empty-input behavior
with focused tests before treating it as a foundation for parallel loading.

TensorFlow documents the Python/same-process constraints of
[`from_generator`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator).
[h5py serializes HDF5 API calls across threads](https://docs.h5py.org/en/stable/threads.html),
even across different files. These are plausible bottlenecks; adding threads or
`AUTOTUNE` alone cannot be assumed to remove them. No TF-versus-Torch speed ranking
or target speedup is established by this audit.

## Data path and performance work

Keep preprocessing inside the recipe, with materialization as an optional execution
choice. Cache deterministic decoding/transforms where beneficial; fingerprint their
source and semantics. Keep training-fitted state bound to the training partition and
apply stochastic augmentation after reusable deterministic preparation. Never cache
one epoch's random augmentation as if it were the general training distribution.

Separate source decoding, prepared examples and backend batching. Evaluate chunked
arrays/memory mapping or other indexed shards according to access patterns and memory
cost. The shared identity/provenance contract must not require TFRecord or a Torch
serialization format. Backend-specific materializations can be optional adapters.

Benchmark these candidate execution paths before selecting defaults:

- TF-native readers and vectorized batch transforms, parallel reads/maps and bounded
  prefetch. Python generator/callback paths remain useful compatibility paths with
  their limitations visible. Follow [TensorFlow's input performance guidance](https://www.tensorflow.org/guide/data_performance).
- PyTorch Dataset/DataLoader adapters with measured worker counts, persistent workers,
  bounded prefetch and pinned host memory when GPU transfers benefit. See
  [PyTorch's loader documentation](https://docs.pytorch.org/docs/2.14/data.html).
- Process-based decoding or batch-oriented Keras PyDataset when Python/EDF/HDF5 work
  dominates. Workers open their own file handles; do not inherit live reader state.
- Cross-framework loader use where useful: Keras accepts multiple loader types, but
  conversion/copy overhead must be measured rather than assumed away.

More workers can increase memory, IPC cost, storage contention and thread
oversubscription. Preserve bounded resource use. Multi-GPU model distribution is a
separate benchmark from CPU loading parallelism.

Measure the stages separately: cold preparation, warm loading without a model,
model-only steps with resident batches, and end-to-end training. Record examples/s,
batch-wait latency, device utilization, host/device memory, bytes read, transfer time,
startup/compile cost and steady-state step time. Synchronize device timings where
needed. Pin hardware, dataset/split/preprocessing fingerprints, model size, batch,
precision, optimizer, warmup, repetitions and software environment.

Publish TF/Torch comparisons using the same logical examples and training workload.
Distinguish controlled loader/backend comparisons from each backend's best optimized
configuration. CI enforces correctness and import isolation; repeatable hardware jobs
track performance regressions. Establish improvement thresholds after baseline timing,
then freeze them before evaluating candidate optimizations.

## Golden datasets, runs and release traceability

A golden dataset definition contains source identities/versions, access instructions,
content hashes, selection/exclusion policy, splits and prepared-data fingerprints.
It need not redistribute restricted recordings. Synthetic fixtures make the contracts
and complete release mechanics publicly testable.

A golden run binds recipe code and external/custom components, resolved parameters,
dataset and split definitions, fitted preprocessing, seed/augmentation policy, backend,
loader settings, environment, model/checkpoint selection rule and expected checks.
Resume support must explicitly cover optimizer, RNG and sampler state if claimed.
The sampling plan explicitly declares finite versus repeated streams, sampling with
or without replacement, subject/window/class weighting, samples or steps per epoch,
and rank/worker assignment. Coverage alone does not preserve the training distribution:
interleaving equally weighted repeated worker streams with unequal ID partitions can
over-sample IDs in smaller partitions. Fix remainder handling together with explicit
sampling weights or a global logical sample schedule; do not silently change the
objective while accelerating the loader. Verify realized counts for fixed schedules
and the declared probability law for stochastic samplers.

Exact sample order and sample coverage are separate guarantees: declare both, and
verify unintended duplicates, missing samples, dropped tails and distributed padding.
Derive per-example random decisions from logical example/epoch identities when worker-
count-independent behavior is required, rather than scheduling order or shared RNG.

Reproduction levels are explicit: artifact byte identity, inference conformance within
tolerance, and training-quality reproduction across declared runs/environments. Do not
promise bitwise training identity across TF and Torch. Quality and performance thresholds
are set using validation/benchmark protocols; do not repeatedly select against an
already inspected test set.

The traceable chain is:

```text
source data + split/policy → prepared data + fitted state → training run/checkpoint
→ evaluation + conversion/calibration evidence → immutable release → HF commit
```

Release utilities bind evidence to exact files regardless of the originating framework.
Each KIT supplies license/attribution and task-specific quality requirements. Public
bundles retain appropriate hashes and aggregate evidence, with private data-access
records and subject-level evidence kept separately. A golden release is promoted by
passing these contracts, not by inheriting a runner class.

## Delivery sequence and acceptance

1. **Baseline and correct the data path.** Capture stage timings for representative
   SleepKit, HeartKit and compressionKIT workloads. Fix demonstrated partition/coverage
   errors separately. Publish reproducible benchmark scripts and evidence; no unsupported
   speedup claim.
2. **heliaEDGE backend isolation.** Introduce optional dependencies/lazy imports and
   TF-only/Torch-only CI environments. Port one existing TF-only trainer through shared
   objective computation and explicit gradient adapters. Test both Keras and native use.
3. **Two-consumer data proof.** Build indexed preparation and loader capabilities used
   by SleepKit plus a different task in HeartKit or compressionKIT. Verify sample/label/
   timing equivalence, sampling-distribution preservation, train-only state and
   worker-count behavior, then measure throughput.
4. **Shared evidence and release proof.** Extract mature generic artifact/provenance
   pieces into heliaEDGE with compatibility tests for existing KIT bundles. Produce
   golden presets for the two consumers and exercise HF staging/download/replay.
5. **Broaden after evidence.** Expand model/trainer/backend/export coverage and migrate
   remaining KIT recipes. Keep unsupported combinations explicit until tested.

The next foundation milestone requires two real consumers, both training backends,
native-loop examples, independently usable blocks, documented reproduction tolerances,
and measured input-pipeline improvements. SleepKit's existing verified detection path
is a reference consumer and regression baseline throughout this work.
