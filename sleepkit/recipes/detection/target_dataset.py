"""Read-only annotated-period targets bound to a frozen local experiment protocol.

Alignment and coverage reports are trusted local evidence, not attestations. The
adapter reads original sensor arrays, reconstructs their verified UTC clock, and
builds targets in memory. Historical ``sleep_stages`` are never read.
"""

from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np

from sleepkit.artifacts.package import sha256
from .audit import load_events
from .candidate_audit import _evidence, context_counts
from .candidate_labels import POLICY, build_candidates
from .data import Recording, read_recording
from .split import TARGET, _read_json, assign


class AnnotatedDataset:
    """Small source adapter; callers retain control of training and orchestration."""

    def __init__(self, root, events_path, alignment_path, coverage_path, frozen_dir):
        self.root = Path(root)
        frozen_dir = Path(frozen_dir)
        self.split_path = frozen_dir / "split.json"
        paths = {
            "events": Path(events_path), "alignment": Path(alignment_path),
            "coverage": Path(coverage_path), "protocol": frozen_dir / "protocol.json",
            "split": self.split_path, "groups": frozen_dir / "groups.json",
            "sources": frozen_dir / "sources.json",
        }
        self._evidence_hashes = {path: sha256(path) for path in paths.values()}
        hashes = {key: self._evidence_hashes[path] for key, path in paths.items()}
        protocol = _read_json(paths["protocol"])
        coverage = _read_json(paths["coverage"])
        # Reject duplicate keys in alignment evidence before its existing validator.
        _read_json(paths["alignment"])
        self._alignment = _evidence(paths["alignment"], hashes["events"])
        self._split = _read_json(paths["split"])
        groups = _read_json(paths["groups"])
        self._source_hashes = _read_json(paths["sources"])
        self._coverage = coverage
        context_policy = protocol.get("context_policy", {})
        context = context_policy.get("features_per_context")
        if (
            protocol.get("schema") != "sleepkit.frozen_split/v1"
            or protocol.get("target") != TARGET or protocol.get("policy") != POLICY
            or protocol.get("coverage_sha256") != hashes["coverage"]
            or protocol.get("files") != {f"{key}.json": hashes[key] for key in ("split", "groups", "sources")}
            or type(context) is not int or context < 1
            or context_policy != {
                "features_per_context": context, "window_samples": 12, "stride_samples": 6,
                "target": "last source sample in each window",
                "grouping": "nonoverlapping native contexts; drop incomplete tail; never stitch",
                "exclusion": "any nonfinite sensor sample or unknown target; reasons counted independently and jointly",
            }
        ):
            raise ValueError("Invalid frozen target protocol, file hashes, or context policy")
        self.context = context
        self._target = deepcopy(TARGET)
        files = {path.stem: path for path in self.root.glob("*.h5")}
        details = coverage.get("subjects", {})
        if (
            not files or not isinstance(groups, dict) or not isinstance(self._source_hashes, dict)
            or set(files) != set(groups) or set(files) != set(self._source_hashes)
            or set(files) != set(self._alignment["subjects"]) or set(files) != set(details)
            or coverage.get("schema") != "sleepkit.candidate_label_audit/v1"
            or coverage.get("policy") != POLICY or coverage.get("context_policy") != context_policy
            or coverage.get("source_subjects") != len(files)
            or coverage.get("events_sha256") != hashes["events"]
            or coverage.get("alignment_sha256") != hashes["alignment"]
            or coverage.get("parquet_sha256") != self._alignment["parquet_sha256"]
        ):
            raise ValueError("Frozen cohort and source/coverage/alignment evidence must match exactly")
        if self._split != assign(groups, seed=protocol.get("seed")):
            raise ValueError("Frozen split must reproduce the group-disjoint seeded assignment")
        if protocol.get("grouping") not in ("series_id_only", "supplied_group_mapping_unverified"):
            raise ValueError("Unsupported frozen grouping")
        if protocol["grouping"] == "series_id_only" and groups != {s: s for s in files}:
            raise ValueError("Series-only grouping must use original series identifiers")
        summaries = {}
        for partition, subjects in self._split.items():
            counts = Counter()
            for subject in subjects:
                detail, verified = details[subject], self._alignment["subjects"][subject]
                if (
                    detail.get("source_h5_sha256") != self._source_hashes[subject]
                    or verified["h5_sha256"] != self._source_hashes[subject]
                    or detail.get("first_utc_seconds") != verified["first_utc_seconds"]
                    or detail.get("counts", {}).get("samples") != verified["h5_samples"]
                ):
                    raise ValueError("Frozen source hash, sample count, or clock differs from verified evidence")
                subject_counts = detail.get("counts", {})
                if any(type(value) is not int or value < 0 for value in subject_counts.values()):
                    raise ValueError("Invalid candidate coverage counts")
                counts.update(subject_counts)
            summaries[partition] = {
                "subjects": len(subjects), "groups": len({groups[s] for s in subjects}), "counts": dict(counts),
                "both_candidate_classes_retained": counts["retained_sleep_targets"] > 0
                and counts["retained_wake_targets"] > 0,
            }
        if protocol.get("partitions") != summaries:
            raise ValueError("Frozen partition summaries differ from candidate coverage")
        self._events = load_events(paths["events"])
        if set(self._events) - set(files):
            raise ValueError("Events contain subjects outside the frozen cohort")
        self._provenance = {
            "kind": "sleepkit.annotated_dataset/v1", "target": deepcopy(TARGET), "policy": deepcopy(POLICY),
            "context_policy": deepcopy(context_policy), "source_subjects": len(files),
            "grouping": protocol["grouping"], "seed": protocol["seed"],
            **{f"{key}_sha256": value for key, value in hashes.items()},
            "parquet_sha256": self._alignment["parquet_sha256"],
            "sample_clock": {"units": "unix_seconds", "dtype": "int64", "period_seconds": 5},
        }
        self.verify_unchanged()

    @property
    def target(self):
        return deepcopy(self._target)

    @property
    def split(self):
        return deepcopy(self._split)

    @property
    def source_hashes(self):
        return dict(self._source_hashes)

    @property
    def provenance(self):
        """Public artifact metadata: hashes and aggregate policy, without subject IDs."""
        return deepcopy(self._provenance)

    def _verify_evidence(self):
        if any(sha256(path) != digest for path, digest in self._evidence_hashes.items()):
            raise ValueError("Frozen protocol or evidence changed after dataset construction")

    def verify_unchanged(self):
        self._verify_evidence()
        files = {path.stem: path for path in self.root.glob("*.h5")}
        if set(files) != set(self._source_hashes):
            raise ValueError("Source cohort changed after dataset construction")
        if any(sha256(files[subject]) != digest for subject, digest in self._source_hashes.items()):
            raise ValueError("Source HDF5 changed after dataset construction")

    def read(self, subject, *, labels=True):
        if not isinstance(subject, str) or subject not in self._source_hashes:
            raise ValueError("Subject is outside the frozen cohort")
        self._verify_evidence()
        path, digest = self.root / f"{subject}.h5", self._source_hashes[subject]
        if sha256(path) != digest:
            raise ValueError("Source HDF5 changed after dataset construction")
        recording = read_recording(self.root, subject, labels=False)
        detail = self._alignment["subjects"][subject]
        count, start = detail["h5_samples"], detail["first_utc_seconds"]
        if recording.data.shape != (3, count):
            raise ValueError("Source sample count differs from verified evidence")
        clock = np.arange(count, dtype=np.int64) * 5 + np.int64(start)
        if recording.sample_time is not None and not np.array_equal(recording.sample_time, clock):
            raise ValueError("Stored sample clock differs from verified source clock")
        target = None
        if labels:
            candidate = build_candidates(self._events.get(subject, {}), count, first_utc_seconds=start)
            target = candidate.labels
            expected = self._coverage["subjects"][subject]
            counts = {
                "samples": count, "sleep_candidate_samples": int((target == 1).sum()),
                "wake_candidate_samples": int((target == 0).sum()), "unknown_samples": int((target == -1).sum()),
                "sleep_intervals": len(candidate.sleep_intervals), "wake_intervals": len(candidate.wake_intervals),
                **context_counts(recording.data, target, self.context),
            }
            if (counts != expected.get("counts") or candidate.issues != expected.get("issues")
                    or list(candidate.valid_nights) != expected.get("valid_nights")):
                raise ValueError("Rebuilt targets differ from frozen candidate coverage")
        if sha256(path) != digest:
            raise ValueError("Source HDF5 changed during read")
        self._verify_evidence()
        return Recording(recording.data, target, clock)
