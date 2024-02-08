---
hide:
  - toc
---

# Model Zoo

A number of pre-trained models are available for download to use in your own project. These models are trained on the datasets listed below and are available in TensorFlow flatbuffer formats.

## <span class="sk-h2-span">Sleep Detection</span>

The following table provides the latest performance and accuracy results for sleep detection models. Additional result details can be found in [Tasks → Detect → Results](./detect/results.md).

| # Classes | Model | Dataset | Fs     | Params | FLOPs    | Accuracy  | Config       | Download    |
| --------- | ----- | ------- | ------ | ------ | -------- | --------- | ------------ | ----------- |
| 2         | TCN   | CMIDSS  | 64 Hz  | 6K     | 425K/hr  | 92.5% AP  | [config](./) | [model](./) |

---

## <span class="sk-h2-span">Sleep Staging</span>

The following table provides the latest performance and accuracy results for sleep staging models. Additional result details can be found in [Tasks → Staging → Results](./stages/results.md).

| # Classes | Model | Dataset | Fs     | Params | FLOPs    | Accuracy  | Config       | Download    |
| --------- | ----- | ------- | ------ | ------ | -------- | --------- | ------------ | ----------- |
| 2         | TCN   | MESA    | 64 Hz  | 10K    | 1.7M/hr  | 88.8% F1  | [config](./) | [model](./) |
| 3         | TCN   | MESA    | 64 Hz  | 14K    | 2.2M/hr  | 84.2% F1  | [config](./) | [model](./) |
| 4         | TCN   | MESA    | 64 Hz  | 14K    | 2.3M/hr  | 76.4% F1  | [config](./) | [model](./) |
| 5         | TCN   | MESA    | 64 Hz  | 17K    | 2.8M/hr  | 70.2% F1  | [config](./) | [model](./) |

---

## <span class="sk-h2-span">Sleep Apnea</span>

__Coming soon...__

---

## <span class="sk-h2-span">Sleep Arousal</span>

__Coming soon...__
