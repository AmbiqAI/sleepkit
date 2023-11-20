# Results

## <span class="sk-h2-span">Overview</span>

The following table provides performance and accuracy results of all models when running on Apollo4 Plus EVB.

| Task           | Params   | FLOPS   | Metric     | Cycles/Inf | Time/Inf   | Arena Size |
| -------------- | -------- | ------- | ---------- | ---------- | ---------- | ---------- |
| 2-Stage Sleep  | 10K      | 1.7M/hr | 88.8% F1   |  11M/hr    | 58ms/hr    |  35KB      |
| 3-Stage Sleep  | 14K      | 2.2M/hr | 84.2% F1   |  16M/hr    | 80ms/hr    |  38KB      |
| 4-Stage Sleep  | 14K      | 2.3M/hr | 76.4% F1   |  16M/hr    | 80ms/hr    |  38KB      |
| 5-Stage Sleep  | 17K      | 2.8M/hr | 70.2% F1   |  18M/hr    | 91ms/hr    |  43KB      |
| Sleep Apnea    | --K      | --M     | ----% F1   | ---M       | ---ms      | --KB       |
| Sleep Arousal  | --K      | --M     | ----% F1   | ---M       | ---ms      | --KB       |


---

## <span class="sk-h2-span">Sleep Stage Results</span>

The following table provides the latest performance and accuracy results for sleep stage classification. Additional result details can be found in [Staging Results Section](./stages/results.md).

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
