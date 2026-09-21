# Saved-feature sleep staging baseline

`sleepkit.recipes.staging` prepares ordered, per-subject FS-W-PA-14 feature records
for the historical SS-3-TCN-SM input contract. It provides ordinary Python
functions for reading, preprocessing, and windowing. It does not train a model,
run inference, or establish model accuracy.

```python
from sleepkit.recipes.staging import read_subject, prepare_subject, window_subject

record = read_subject("/path/to/subject.h5")
prepared = prepare_subject(record)
windows = window_subject(prepared)
# windows.features: float32 [window_count, 240, 14]
# windows.targets: integer [window_count, 240], unknown targets remain -1
# windows.scoring_mask: quality-valid epochs with known targets
print(windows.coverage)
```

## Feature and label contract

The HDF5 file must contain `features` float32 `[time,14]`, `stage_labels` integer
`[time]`, and `mask` binary integer or boolean `[time]`. Required datasets must be
self-contained; indirect links, virtual datasets, and external storage are
rejected. Other datasets, such as apnea labels, are ignored.

Feature columns are ordered:

1. `hr_bpm`
2. `hrv_td_mean_nn`
3. `hrv_td_sd_nn`
4. `hrv_td_median_nn`
5. `hrv_fd_lfhf_ratio`
6. `spo2_mu`
7. `spo2_std`
8. `spo2_med`
9. `mov_mu`
10. `mov_std`
11. `mov_med`
12. `rsp_bpm`
13. `spo2_qos`
14. `hrv_qos`

The historical configuration specifies 60-second feature windows sampled every
30 seconds. A 240-epoch model window contains two hours of nominal epoch time;
overlapping feature support extends over 120.5 minutes. These timing and feature
semantics are caller requirements; the reader cannot establish them from an
unannotated HDF5 array.

| Source stage | Model target |
| --- | --- |
| 0: Wake | 0: WAKE |
| 1–4: N1, N2, N3, N4 | 1: NREM |
| 5: REM | 2: REM |
| 6: noise/unscored | -1: excluded target |

Other stage codes are rejected. The historical configuration called the NREM
class `SLEEP`; this adapter uses `NREM` to make its meaning explicit. Unknown
labels retain their temporal positions. Before computing loss or classification
metrics, select `windows.scoring_mask` from both targets and per-epoch logits.
Never pass unknown `-1` targets directly to a classification accumulator.

## Offline normalization and coverage

`prepare_subject` validates caller-constructed records as well as reader output.
All rows where `mask == 1` must have finite features, and at least one such row
must exist. These rows define per-column median, mean, and population variance.
Every invalid row is replaced with that median, including any NaN or infinity
in invalid rows. The transform is `(features - mean) / sqrt(variance + 1e-6)`.
Statistics and arithmetic intentionally use float32 to preserve the historical
calculation on supported inputs. Overflow or any nonfinite result is rejected.
The returned record includes the statistics; input arrays are not mutated.

This normalization is **offline and transductive**: each subject's entire record,
including later epochs and any eventual incomplete tail, contributes to that
subject's statistics. A quality-valid unknown-stage row also contributes because
normalization does not use labels. These are not training-population statistics,
and the procedure is not causal streaming or independent per-window normalization.

`window_subject` forms complete, nonoverlapping 240-epoch windows at offsets
0, 240, 480, and so on. Invalid and unknown epochs remain inside each window as
context. Only known targets with `mask == 1` can be scored. The final incomplete
window is omitted; records shorter than 240 epochs return no windows. Windows
never combine different records or stitch across invalid positions.

Coverage reports input epochs, window count, windowed epochs, remainder epochs,
eligible epochs over the entire record, scored epochs within retained windows,
and all unscored epochs. Unscored epochs include both the remainder and excluded
positions within complete windows. A duration derived from epoch counts uses
30 seconds per epoch; a sleep fraction over scored positions does not establish
whole-night sleep efficiency when coverage is incomplete.

## Historical limitations

This adapter starts from saved features. Historical MESA extraction used PPG,
SpO2, signal quality, and leg movement as a proxy for wrist motion. The configured
raw MESA path was unavailable during the baseline audit, so raw feature generation
and equivalence have not been revalidated. Existing stores do not identify their
extraction version or feature order in HDF5 metadata.

Historical extraction assigned each feature window the numeric median of raw
stage codes, cast to an integer. At transitions this can differ from categorical
majority voting. The adapter preserves stored labels; it does not regenerate or
correct them. Its strict finite-input checks also reject cases the old loader
could handle inconsistently.

The historical loader divided filesystem discovery order into an 80% train and
validation pool and 20% evaluation pool, then randomly divided the former.
Historical training and validation membership is not established by a complete
manifest, and saved configurations contain differing seeds. A newly trained
baseline needs explicit, disjoint subject membership before windowing. Availability
of a historical checkpoint and evaluation subject list does not establish fresh
held-out performance. This preprocessing module deliberately makes no split or
model-quality claim.
