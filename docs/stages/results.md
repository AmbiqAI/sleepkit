# Results

| Task           | Params   | FLOPS    | Accuracy  | F1       | AP        |
| -------------- | -------- | -------- | --------- | -------- | --------- |
| 2-Stage Sleep  | 10K      | 1.7M/hr  | 88.8%     | 88.8%    | 96.2%     |
| 3-Stage Sleep  | 14K      | 2.2M/hr  | 83.9%     | 84.2%    | 91.5%     |
| 4-Stage Sleep  | 14K      | 2.3M/hr  | 75.8%     | 76.4%    | 83.1%     |
| 5-Stage Sleep  | 17K      | 2.8M/hr  | 70.4%     | 70.2%    | 76.4%     |

## <span class="sk-h2-span">Confusion Matrices</span>

### 2-Stage Sleep Classification

<figure markdown>
  ![2-Stage Sleep Stage Confusion Matrix](../assets/sleep-stage-2-cm.png){ width="540" }
  <figcaption>2-Stage Sleep Classification CM</figcaption>
</figure>

### 3-Stage Sleep Classification

<figure markdown>
  ![3-Stage Sleep Stage Confusion Matrix](../assets/sleep-stage-3-cm.png){ width="540" }
  <figcaption>3-Stage Sleep Classification CM</figcaption>
</figure>

### 4-Stage Sleep Classification

<figure markdown>
  ![4-Stage Sleep Stage Confusion Matrix](../assets/sleep-stage-4-cm.png){ width="540" }
  <figcaption>4-Stage Sleep Classification CM</figcaption>
</figure>

### 5-Stage Sleep Classification

<figure markdown>
  ![5-Stage Sleep Stage Confusion Matrix](../assets/sleep-stage-5-cm.png){ width="540" }
  <figcaption>5-Stage Sleep Classification CM</figcaption>
</figure>

## <span class="sk-h2-span">Sleep Efficiency Plot</span>

<div class="sk-plotly-graph-div">
--8<-- "assets/stage-2-eff.html"
</div>

## <span class="sk-h2-span">Total Sleep Time (TST) Plot</span>

<div class="sk-plotly-graph-div">
--8<-- "assets/stage-2-tst.html"
</div>

---

## <span class="sk-h2-span">EVB Performance</span>

The following table provides the latest performance and accuracy results of all models when running on Apollo4 Plus EVB. These results are obtained using neuralSPOTs [Autodeploy tool](https://ambiqai.github.io/neuralSPOT/docs/From%20TF%20to%20EVB%20-%20testing%2C%20profiling%2C%20and%20deploying%20AI%20models.html). From neuralSPOT repo, the following command can be used to capture EVB results via Autodeploy:

``` console
python -m ns_autodeploy \
--tflite-filename sleep-stage-4-model.tflite \
--model-name sleepstage4 \
--cpu-mode 192 \
--arena-size-scratch-buffer-padding 0 \
--max-arena-size 80 \

```

| Task           | Params   | FLOPS   | Metric     | Cycles/Inf | Time/Inf   | Arena | NVM   | RAM   |
| -------------- | -------- | ------- | ---------- | ---------- | ---------- | ----- | ----- | ----- |
| 2-Stage Sleep  | 10K      | 1.7M/hr | 88.8% F1   |  88M       | 459ms      |  36KB | 195KB |  53KB |
| 3-Stage Sleep  | 14K      | 2.2M/hr | 84.2% F1   | 123M       | 639ms      |  40KB | 210KB |  58KB |
| 4-Stage Sleep  | 14K      | 2.3M/hr | 76.4% F1   | 123M       | 642ms      |  40KB | 211KB |  58KB |
| 5-Stage Sleep  | 17K      | 2.8M/hr | 70.2% F1   | 140M       | 728ms      |  43KB | 216KB |  61KB |


In addition, we can capture statistics from each layer. The following bar plot provides the latency of each block in the 4-stage sleep classification TCN model. For example, `ENC` refers to initial encoder 1-d seperable convulional layer, `B1.1` refers to all the layers in block 1, depth 1, `B1.2` refers to block 1, depth 2, and so on. We can see that as we go deeper into the network we see an increase in latency due to the increasing number of channels. The final `DEC` layer refers to the decoder layer which is a 1-d convolutional layer with 3 output channels (4 classes).

<div class="sk-plotly-graph-div">
--8<-- "assets/block-latency.html"
</div>

---

## <span class="sk-h2-span">Comparison</span>

We compare our 3-stage and 4-stage model to the SLAMSS model from [Song et al., 2023](https://doi.org/10.1371/journal.pone.0285703). Their model was also trained on MESA dataset using only motor and cardiac physiological signals. In particular, they extract activity count, heart rate, and heart rate standard deviation in 30 second epochs. They fed 12 epochs (6 minutes) of the 3 features (12x6) as input to the network. The newtork consists of 3 1-D CNN layers, 2 LSTM layers, and 1 attention layer. The underlying design of the attention block is unclear but using only the 3 CNN and 2 LSTM layers the network requires roughly 8.8 MFLOPS per epoch. This equates to roughly __450X__ more computation (__1,056 MFLOPS/hr__) compared to our 4-stage sleep classification model (__2.3 MFLOPS/hr__).

<!-- SLAMSS FLOPS:
K=9, Cin=3, Cout=64, T=12, P=4, S=1
CNN1: 3*64*9*12 = 20,736
CNN2: 64*64*9*12 = 442,368
CNN3: 64*64*9*12 = 442,368
LSTM1: 2*((64+256+1)*4*256+256)*12 = 7,895,040
LSTM2: 2*((64+256+1)*4*256+256)*12 = 75,816
TOTAL: 20736+442368+442368+7895040+75816 -->

### 3-Stage Sleep Classification (MESA)

| Reference         | Acc       | F1        | WAKE      | NREM      | REM       |
| ----------------- | --------- | --------- | --------- | --------- | --------- |
| [Song et al., 2023](https://doi.org/10.1371/journal.pone.0285703) | 79.1      | 80.0      | 78.0      | 81.8      | 70.9      |
| **SleepKit**      | **83.9**  | **84.2**  | **80.3**  | **86.6**  | **83.5**  |


### 4-Stage Sleep Classification (MESA)

| Reference         | Acc       | F1        | WAKE      | CORE      | DEEP      | REM       |
| ----------------- | --------- | --------- | --------- | --------- | --------- | --------- |
| [Song et al., 2023](https://doi.org/10.1371/journal.pone.0285703) | 70.0      | 72.0      | 78.7      | 66.3      | 55.9      | 63.0      |
| SleepKit          | **75.8**  | **76.4**  | **80.6**  | **73.9**  | **52.2**  | **81.7**  |
