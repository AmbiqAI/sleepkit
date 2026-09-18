# SleepKit next-major design proposal

Status: implementation started; dependency targets checked against PyPI on 2026-09-17.
The first alpha implements the integrated synthetic flow described in the
[v1 guide](guides/v1-experiments.md). The remaining sections describe the full target;
they are not a claim that every migration milestone is complete.
Target: a new major release (proposed 1.0, following the current 0.11.x line).

## Goal

Make SleepKit a collection of capabilities that an experiment can adopt incrementally.
A new experiment should start as an ordinary Python script: load data, build a model,
call `model.fit`, evaluate predictions, and optionally export. Existing model artifacts
remain historical baselines. New evaluations and exports receive separate identities.

Follow compressionKIT's [experiment architecture](https://github.com/AmbiqAI/compressionkit/blob/main/docs/experiment-architecture.md):
reusable components, ready-made experiments, then optional release automation.
Avoid mandatory task inheritance, global registration, and a universal training configuration.

## Redesign the public flow; reuse proven implementation

The next major version should introduce a new public API, developed incrementally. Retain useful
signal processing, dataset decoding, model builders, and metric implementations after checking
their behavior. Replace the task/factory orchestration and disconnected feature-file handoff.
Preserve the 0.x release line for historical workflows rather than making the new API satisfy
every old configuration. Provide targeted baseline import/migration tools where they are useful.

The major-version milestone is an integrated raw-record-to-prediction recipe, reproducible
evaluation, validated deployment packaging, and documented extension points. A dependency bump
alone does not complete that milestone.

## Preprocessing belongs to the experiment flow

An experiment selects its data source, preprocessing, model, and evaluation explicitly:

```text
dataset adapter -> signal records -> preprocessing -> model inputs -> model
                         |                 |
                     annotations ------> aligned targets and validity masks
```

Preprocessing can clean/resample signals, compute features, normalize, and assemble model
windows. A raw-waveform experiment can omit feature extraction. The trainer receives validated
model inputs produced by the selected preprocessing, rather than independently guessing the
meaning of an HDF5 directory. Keep annotation alignment separate from input transforms: inference
must work without labels, and input features must never depend on ground-truth annotations.

The flow remains ordinary Python composition. A small optional preparation helper can execute
and cache a callable preprocessing recipe; it must not introduce a workflow engine, global
registry, mandatory base class, or configuration language for selecting every operation.

### Dataset adapters describe observations

Use a small record representation at the ingestion boundary with arrays and metadata:

- Dataset-qualified subject and recording identities, including session and grouping metadata.
- Named signals with units, individual sampling rates/time coordinates, sensor location,
  acquisition modality, and validity/quality information.
- Separate timestamped annotations with source label vocabulary and provenance.
- Source revision or fingerprints for reproducibility and cache validation.

Adapters handle file formats, channel discovery, unit conversion, and source annotation decoding.
Preprocessing declares required signals and acceptable semantics; task policies map source
annotations into target classes. Signal alignment and label-window aggregation have explicit
policies. Do not force all channels onto a shared sampling grid in the reader, discard temporal
gaps, or assume annotations and signals share an origin without checking.

Common names must not hide physiological differences. For example, the current MESA feature
code derives movement from `Leg`, YSYW derives a movement proxy from `ABD`, and CMIDSS supplies
ENMO/ZANGLE. Those cannot silently become interchangeable wrist accelerometry. Preserve the
source sensor/proxy identity and require an explicit recipe policy for using a substitute.

A new dataset should require an adapter and any necessary annotation mapping, while existing
feature functions continue to work when the observations satisfy their requirements. If required
signals are absent or incompatible, fail early with an actionable message or apply a declared
missing-channel policy. Do not silently zero-fill or silently drop recordings.

### Preprocessing has an explicit output contract

Describe ordered feature/channel names, units, dtype, shape, feature cadence, window/stride,
time alignment, validity masks, normalization policy, fitted state, and causal/look-ahead behavior.
Use separate meanings for the raw sampling rate, feature extraction window, feature cadence,
and model context length. Validate the model input against this contract before training or
inference. These descriptors can accompany ordinary arrays; they do not require a custom tensor type.

Stateless transforms are ordinary functions. Transforms that learn statistics expose a small
fit/transform interface, fit only on training subjects, and save their fitted state. Streaming
transforms expose reset/update state when needed and reset at recording boundaries. Keep
training-only stochastic augmentation outside the deterministic evaluation/inference path.

### Caching is an execution option

The same preprocessing definition can run on demand or materialize reusable intermediate
results. Materialization is useful for costly physiological features and should remain available
as a standalone operation, but training recipes can request it automatically.

Cache identity includes source identity/revision, channel selection, transform implementation
version and parameters, temporal policies, and output schema. Fitted transforms also include
the training-split fingerprint and fitted-state hash. Cache deterministic per-record transforms
before fitted normalization where practical, allowing model and split experiments to reuse work.
Do not cache random augmentations as deterministic outputs. Record failed/skipped subjects and
publish completed cache entries atomically so interrupted work cannot appear complete.

Changing an HRV window or feature order invalidates the relevant cache. Changing the neural
network alone should reuse compatible prepared inputs. Memory and disk execution must agree
on features, targets, masks, and timestamps.

### Deployment carries the preprocessing contract

Export includes the preprocessing specification, fitted statistics, required runtime capabilities,
model input/output specifications, and conformance vectors. Offer two explicit artifact scopes:
feature-input inference and raw-signal inference with the required preprocessing implementation.
Never describe a feature-only bundle as a complete raw-sensor application.

Use the same transform definition for training, evaluation, and host inference. Embedded DSP
may need a separate C implementation, verified against the Python reference. Packaging a Python
callable or a JSON description does not make it deployable on an MCU. Track unsupported transforms,
look-ahead, state memory, startup behavior, and processing cost alongside neural-network cost.

The first vertical slice must start with raw recordings and demonstrate preparation, training,
evaluation, export, and inference. A second dataset should exercise the same feature transforms
through an adapter where signal semantics permit it; sensor substitutions require explicit policy.
Include synthetic records so the complete plumbing can be tested without restricted datasets.

## Dependency upgrade candidates

| Component | Pre-migration lock | Proposed first target |
| --- | --- | --- |
| Python | Project allows 3.12–3.13 | Keep 3.12 as the reference environment; validate 3.13 separately |
| TensorFlow | 2.20.0 | 2.21.0, initially constrain the supported minor series |
| Keras | 3.10.0, transitive dependency | Explicit dependency; test 3.15.1 with TF 2.21 |
| HeliaEdge | 0.4.1 | 0.6.2, behind optional model/export integrations |
| PhysioKit | 0.10.1 | 0.11.0, with feature regression checks |
| NumPy / SciPy | 2.1.3 / 1.15.3 | Resolve supported current 2.x / 1.x versions and validate numerics |
| pandas | 2.3.0, transitive dependency | Declare directly where used; retain 2.3.x initially, test 3.x separately |
| h5py | 3.13.0 | Compatible 3.14.x; TF 2.21 requires >=3.11,<3.15 |
| LiteRT | Absent | Evaluate ai-edge-litert 2.2.0 for standalone inference |
| Hugging Face Hub | Absent | Optional integration; evaluate current 1.x API |

The latest package is not automatically the compatible package. Validate the resolved lock,
Keras archive loading, a small training step, conversion, and runtime parity together.
TF 2.21 no longer installs TensorBoard transitively: declare it in a training/diagnostics
extra. Remove the old profiler upper bound only after testing the replacement. Keep notebooks
and profiling tools in development groups. Modernize the Parquet engine for CMIDSS independently
of training; test existing files before replacing the old fastparquet pin with PyArrow.

Separate platform support: Linux CPU CI first, Linux GPU smoke checks in the development
container, macOS Metal only after an explicit compatibility check. HeliaEdge currently
requires tensorflow-metal on macOS, so that behavior cannot be removed merely by editing
SleepKit's direct dependencies.

Sources: [TF release](https://github.com/tensorflow/tensorflow/releases/tag/v2.21.0),
[TF package metadata](https://pypi.org/pypi/tensorflow/2.21.0/json),
[Keras](https://pypi.org/project/keras/), [HeliaEdge](https://pypi.org/project/helia-edge/),
[PhysioKit](https://pypi.org/project/physiokit/),
[LiteRT migration](https://developers.google.com/edge/litert/migration).

## Dependency and import boundaries

Target installations:

- Core: lightweight types, label definitions, manifests, and NumPy-based utilities.
- `sleepkit[train]`: Keras/TensorFlow and the dependencies used by reference recipes.
- `sleepkit[features]`: physiological signal processing and feature extraction.
- `sleepkit[datasets]`: HDF5/EDF/Parquet readers; download integrations imported only when used.
- `sleepkit[runtime]`: LiteRT and inference utilities, without TensorFlow or HeliaEdge.
- `sleepkit[hf]`: Hub download/publishing helpers.
- `sleepkit[reports]`, `sleepkit[wandb]`, `sleepkit[evb]`: optional visualization, tracking, hardware.

These are proposed boundaries, not promises about today's install. HeliaEdge brings TensorFlow,
plotting, and cloud dependencies transitively; keep it out of core/runtime. Stop eager package
imports and logger initialization in `sleepkit/__init__.py`. Test that importing core and
runtime does not import training, plotting, serial, or dataset download integrations.

## Components and recipes

Evolve existing modules as capabilities are extracted; do not scaffold empty abstraction layers.

| Area | Responsibility |
| --- | --- |
| `datasets/` | Dataset-specific readers and ingestion; produce records usable outside SleepKit |
| `data/` | Explicit subject splits, windows, masks, cache metadata; optional tf.data adapter |
| `preprocessing/` | Composable signal transforms, fitted state, input specifications, and optional materialization |
| `features/` | Reusable feature transforms consumed by preprocessing recipes |
| `models/` | Ordinary model builders returning Keras models; HeliaEdge builders are reusable choices |
| `training/` | Small callbacks, schedules, losses, and optional helpers around standard Keras training |
| `evaluation/` | Metrics on arrays, subject/event aggregation, scorecards; accept predictions from any runtime |
| `export/` | Conversion, quantization, manifests, reference vectors, validation, publication |
| `runtime/` | Local/Hub bundle loading and feature-input inference |
| `recipes/` | Readable staging, detection, and apnea examples that compose the above |

Use small typed settings where validation helps: split specification, feature specification,
evaluation policy, export settings, and deployment manifest. Put component selection in Python;
configuration contains experiment values. A recipe may accept arrays, a standard Keras dataset,
or a TensorFlow dataset as appropriate; do not require a universal SleepKit batch wrapper.

The first recipe should build the current TCN directly, call standard `compile`/`fit`, evaluate
on a fixed held-out split, and export explicitly. It must also accept a user-created Keras model
without registration or subclassing. Keep TensorFlow-specific data and conversion code at
integration boundaries. Supporting arbitrary training backends is not part of the first migration.

Extract helpers when behavior repeats. A general experiment runner, callback bus, plugin
registry, or trainer base class is not required. The CLI should invoke the same public functions
that scripts use. Keep legacy CLI/config workflows on the 0.x line; add narrowly scoped import
tools for preserved baselines instead of requiring full compatibility in the new major version.

## Correctness work before model comparisons

- Persist dataset-qualified subject IDs and train/validation/test membership. The existing
  loader uses unsorted filesystem order, and keys subjects by filename stem, risking collisions
  across datasets. Apnea training splits all subjects while evaluation selects a test subset.
- Recover historical split membership where possible. If it cannot be recovered, label the
  old score's provenance as incomplete. A new random split cannot establish that an existing
  checkpoint has never seen those subjects.
- Compute class weights from training labels. Current training routines use validation labels.
- Define missing-data behavior, valid-label masks, sampling cadence, window stride, and tail
  handling. Fix exact-length and too-short window cases in the current sampler.
- Separate full-record normalization used by historical baselines from causal preprocessing
  for streaming experiments. Version feature definitions and normalization policy in artifacts.
- Make logits/probabilities explicit across training, evaluation, and runtime. Fix apnea export
  validation, which currently passes argmax labels to categorical metrics.
- Use training-only representative data for quantization, preserve a separate evaluation set,
  and fail validation on unacceptable degradation. Stage export currently checks a different
  metric key from the one it validates, and export threshold failures only produce warnings.
- Investigate the four-stage baseline's unusually small TFLite output scale before promotion.
- Compare macro/per-class F1, balanced accuracy, and kappa for staging; event and subject-level
  measures for apnea/detection; report compute, size, and hardware cost separately. Record
  evaluation protocol and metric definitions with every score.

## Release boundary

Any experiment can become publishable by producing a validated bundle. No trainer inheritance
is required. Adapt compressionKIT's staging/model-card/upload workflow, not its codec schemas.

A bundle contains model artifacts, a versioned input/output and preprocessing specification,
label map, sanitized resolved configuration, evaluation provenance, dependency/source versions,
checksums, synthetic conformance vectors, a model card, and license. Publishing requires no
training dataset. Reference vectors test conformance; they do not establish model accuracy.

Historical artifacts remain immutable. A repaired export is a new artifact revision; a retrained
model is a new experiment. Preserve the six documented baseline identities and distinguish their
original metrics from results produced by the updated evaluation protocol.

## Incremental delivery

1. **Baseline inventory and environment:** record historical artifacts and provenance gaps;
   build an isolated TF 2.21 environment, update the lock, and test loading/conversion of baselines.
   Keep compatibility findings separate from changes to training semantics.
2. **One complete recipe:** implement the record/preprocessing/input contract and migrate
   two-stage staging from raw recordings through cached or on-demand preprocessing, training,
   evaluation, and export. Prove that an ordinary script can use a custom Keras model without
   `TaskParams` or registration; include an unlabeled inference path.
3. **Prove reuse:** add a second dataset adapter and migrate detection/apnea. Verify feature reuse
   where signal semantics match, fix metric/export behavior, and remove duplicate orchestration.
   Demonstrate feature-cache invalidation and task-appropriate reports.
4. **Independent runtime and Hub releases:** implement runtime imports/extras, validate bundles,
   add synthetic examples and a dry-run publisher, then promote validated baselines.
5. **New experiments:** use the established protocol to compare feature subsets, causal
   normalization, window/context lengths, compact architectures, and INT8 versus INT16x8.

Each increment should leave an executable example. CI should cover core-only imports,
split disjointness, masks/windows, a synthetic training smoke run, Keras serialization,
quantized inference conformance, and offline bundle validation. Hardware profiling remains
a separate check; desktop LiteRT success is insufficient to establish MCU compatibility.
Also test multi-rate/time alignment, missing signals, fitted-transform leakage, cached/on-demand
equivalence, cache invalidation, unlabeled inference, and state reset between recordings.

Use HeliaEdge for broadly reusable edge-model and conversion capabilities. Keep sleep-specific
features, labels, dataset policies, scorecards, and publication identities in SleepKit. Consider
extracting generic utilities upstream only after their API is demonstrated by a working recipe.
