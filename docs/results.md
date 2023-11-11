# Results

## <span class="sk-h2-span">Overview</span>

The following table provides performance and accuracy results of all models when running on Apollo4 Plus EVB.

| Task           | Params   | FLOPS   | Metric     | Cycles/Inf | Time/Inf   |
| -------------- | -------- | ------- | ---------- | ---------- | ---------- |
| 2-Stage Sleep  | 10K      | 1.7M/hr | 88.8% F1   |  88M       | 459ms      |
| 3-Stage Sleep  | 14K      | 2.2M/hr | 84.2% F1   | 123M       | 639ms      |
| 4-Stage Sleep  | 14K      | 2.3M/hr | 76.4% F1   | 123M       | 642ms      |
| 5-Stage Sleep  | 17K      | 2.8M/hr | 70.2% F1   | 140M       | 728ms      |
| Sleep Apnea    | --K      | --M     | ----% F1   | ---M       | ---ms      |
| Sleep Arousal  | --K      | --M     | ----% F1   | ---M       | ---ms      |

---

## <span class="sk-h2-span">Sleep Stage Results</span>

| Task           | Params   | FLOPS    | Accuracy  | F1       | AP        |
| -------------- | -------- | -------- | --------- | -------- | --------- |
| 2-Stage Sleep  | 10K      | 1.7M/hr  | 88.8%     | 88.8%    | 96.2%     |
| 3-Stage Sleep  | 14K      | 2.2M/hr  | 83.9%     | 84.2%    | 91.5%     |
| 4-Stage Sleep  | 14K      | 2.3M/hr  | 75.8%     | 76.4%    | 83.1%     |
| 5-Stage Sleep  | 17K      | 2.8M/hr  | 70.4%     | 70.2%    | 76.4%     |

---


## <span class="sk-h2-span">Sleep Apnea Results</span>


__Coming soon...__

---

## <span class="sk-h2-span">Sleep Arousal Results</span>

__Coming soon...__
