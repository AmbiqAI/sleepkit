# Sleep Stage Models

## Model Overview

The following table provides the latest pre-trained models for sleep detection. Below we also provide additional details including training configuration, accuracy metrics, and hardware performance results for the models.

--8<-- "assets/zoo/stage/stage-model-zoo-table.md"

## Model Details

=== "SS-2-TCN-SM"

    The __SS-2-TCN-SM__ model is a 2-stage sleep stage model that uses a Temporal Convolutional Network (TCN) architecture. The model is trained on PPG and IMU data collected from the wrist and is able to classify sleep and wake stages. The model requires only 10K parameters and achieves an accuracy of 88.8% and an average AP score of 96.2%.

    - **Location**: Wrist
    - **Classes**: Awake, Sleep
    - **Frame Size**: 2 hours
    - **Datasets**: [MESA](../datasets/mesa.md)
    - **Features**: [FS-W-PA-14](../features/fs_w_pa_14.md)

    | Base Class              | Target Class | Label     |
    | ----------------------- | ------------ | --------- |
    | 0-WAKE                  | 0            | WAKE      |
    | 1-N1, 2-N2, 3-N3, 5-REM | 1            | SLEEP     |

=== "SS-3-TCN-SM"

    The __SS-3-TCN-SM__ model is a 3-stage sleep stage model that uses a Temporal Convolutional Network (TCN) architecture. The model is trained on PPG and IMU data collected from the wrist and is able to classify awake, sleep and rem stages. The model requires only 14K parameters and achieves an accuracy of 84% and an average AP score of 91.5%.

    - **Location**: Wrist
    - **Classes**: Awake, NREM, REM
    - **Frame Size**: 2 hours
    - **Datasets**: [MESA](../datasets/mesa.md)
    - **Features**: [FS-W-PA-14](../features/fs_w_pa_14.md)

    | Base Class              | Target Class | Label     |
    | ----------------------- | ------------ | --------- |
    | 0-WAKE                  | 0            | WAKE      |
    | 1-N1, 2-N2, 3-N3        | 1            | NREM      |
    | 5-REM                   | 2            | REM       |

=== "SS-4-TCN-SM"

    The __SS-4-TCN-SM__ model is a 4-stage sleep stage model that uses a Temporal Convolutional Network (TCN) architecture. The model is trained on PPG and IMU data collected from the wrist and is able to classify awake, core, deep, and rem stages. The model requires only 18K parameters and achieves an accuracy of 75.8% and an average AP score of 83.1%.

    - **Location**: Wrist
    - **Classes**: Awake, Core, Deep, REM
    - **Frame Size**: 2 hours
    - **Datasets**: [MESA](../datasets/mesa.md)
    - **Features**: [FS-W-PA-14](../features/fs_w_pa_14.md)

    | Base Class              | Target Class | Label     |
    | ----------------------- | ------------ | --------- |
    | 0-WAKE                  | 0            | WAKE      |
    | 1-N1, 2-N2              | 1            | CORE      |
    | 3-N3                    | 2            | DEEP      |
    | 5-REM                   | 3            | REM       |


=== "SS-5-TCN-SM"

    The __SS-5-TCN-SM__ model is a 5-stage sleep stage model that uses a Temporal Convolutional Network (TCN) architecture. The model is trained on PPG and IMU data collected from the wrist and is able to classify awake, n1, n2, n3, and rem stages. The model requires only 17K parameters and achieves an accuracy of 70.4% and an average AP score of 76.4%.

    - **Location**: Wrist
    - **Classes**: Awake, N1, N2, N3, REM
    - **Frame Size**: 2 hours
    - **Datasets**: [MESA](../datasets/mesa.md)
    - **Features**: [FS-W-PA-14](../features/fs_w_pa_14.md)

    | Base Class              | Target Class | Label     |
    | ----------------------- | ------------ | --------- |
    | 0-WAKE                  | 0            | WAKE      |
    | 1-N1                    | 1            | N1        |
    | 2-N2                    | 2            | N2        |
    | 3-N3                    | 3            | N3        |
    | 5-REM                   | 4            | REM       |

---

## Model Performance

=== "SS-2-TCN-SM"

    ### Confusion Matrix

    <div class="sk-plotly-graph-div">
    --8<-- "assets/zoo/stage/ss-2-tcn-sm-cm.html"
    </div>

    ### Sleep Efficiency Plot

    <div class="sk-plotly-graph-div">
    --8<-- "assets/zoo/stage/ss-2-tcn-sm-eff.html"
    </div>

    ### Total Sleep Time (TST) Plot

    <div class="sk-plotly-graph-div">
    --8<-- "assets/zoo/stage/ss-2-tcn-sm-tst.html"
    </div>

=== "SS-3-TCN-SM"

    ### Confusion Matrix

    <div class="sk-plotly-graph-div">
    --8<-- "assets/zoo/stage/ss-3-tcn-sm-cm.html"
    </div>


=== "SS-4-TCN-SM"

    ### Confusion Matrix

    <div class="sk-plotly-graph-div">
    --8<-- "assets/zoo/stage/ss-4-tcn-sm-cm.html"
    </div>

=== "SS-5-TCN-SM"

    ### Confusion Matrix

    <div class="sk-plotly-graph-div">
    --8<-- "assets/zoo/stage/ss-5-tcn-sm-cm.html"
    </div>

---

## EVB Performance

The following table provides the latest performance and accuracy results of all models when running on Apollo4 Plus EVB. These results are obtained using neuralSPOTs [Autodeploy tool](https://ambiqai.github.io/neuralSPOT/docs/From%20TF%20to%20EVB%20-%20testing%2C%20profiling%2C%20and%20deploying%20AI%20models.html). From neuralSPOT repo, the following command can be used to capture EVB results via Autodeploy:

``` console
python -m ns_autodeploy \
--tflite-filename model.tflite \
--model-name model \
--cpu-mode 192 \
--arena-size-scratch-buffer-padding 0 \
--max-arena-size 80 \

```

--8<-- "assets/zoo/stage/stage-model-hw-table.md"

In addition, we can capture statistics from each layer. The following bar plot provides the latency of each block in the 4-stage sleep classification TCN model. For example, `ENC` refers to initial encoder 1-d seperable convulional layer, `B1.1` refers to all the layers in block 1, depth 1, `B1.2` refers to block 1, depth 2, and so on. We can see that as we go deeper into the network we see an increase in latency due to the increasing number of channels. The final `DEC` layer refers to the decoder layer which is a 1-d convolutional layer with 3 output channels (4 classes).

<div class="sk-plotly-graph-div">
--8<-- "assets/zoo/stage/block-latency.html"
</div>

---

## Comparison

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
| **sleepKIT**      | **83.9**  | **84.2**  | **80.3**  | **86.6**  | **83.5**  |


### 4-Stage Sleep Classification (MESA)

| Reference         | Acc       | F1        | WAKE      | CORE      | DEEP      | REM       |
| ----------------- | --------- | --------- | --------- | --------- | --------- | --------- |
| [Song et al., 2023](https://doi.org/10.1371/journal.pone.0285703) | 70.0      | 72.0      | 78.7      | 66.3      | **55.9**      | 63.0      |
| sleepKIT          | **75.8**  | **76.4**  | **80.6**  | **73.9**  | 52.2  | **81.7**  |


---

## Downloads

=== "SS-2-TCN-SM"

    | Asset                                                                | Description                   |
    | -------------------------------------------------------------------- | ----------------------------- |
    | [configuration.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/sleepkit/stage/ss-2-tcn-sm/latest/configuration.json)   | Configuration file            |
    | [model.keras](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/sleepkit/stage/ss-2-tcn-sm/latest/model.keras)            | Keras Model file              |
    | [model.tflite](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/sleepkit/stage/ss-2-tcn-sm/latest/model.tflite)       | TFLite Model file             |
    | [metrics.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/sleepkit/stage/ss-2-tcn-sm/latest/metrics.json)       | Metrics file                  |

=== "SS-3-TCN-SM"

    | Asset                                                                | Description                   |
    | -------------------------------------------------------------------- | ----------------------------- |
    | [configuration.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/sleepkit/stage/ss-3-tcn-sm/latest/configuration.json)   | Configuration file            |
    | [model.keras](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/sleepkit/stage/ss-3-tcn-sm/latest/model.keras)            | Keras Model file              |
    | [model.tflite](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/sleepkit/stage/ss-3-tcn-sm/latest/model.tflite)       | TFLite Model file             |
    | [metrics.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/sleepkit/stage/ss-3-tcn-sm/latest/metrics.json)       | Metrics file                  |

=== "SS-4-TCN-SM"

    | Asset                                                                | Description                   |
    | -------------------------------------------------------------------- | ----------------------------- |
    | [configuration.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/sleepkit/stage/ss-4-tcn-sm/latest/configuration.json)   | Configuration file            |
    | [model.keras](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/sleepkit/stage/ss-4-tcn-sm/latest/model.keras)            | Keras Model file              |
    | [model.tflite](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/sleepkit/stage/ss-4-tcn-sm/latest/model.tflite)       | TFLite Model file             |
    | [metrics.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/sleepkit/stage/ss-4-tcn-sm/latest/metrics.json)       | Metrics file                  |

=== "SS-5-TCN-SM"

    | Asset                                                                | Description                   |
    | -------------------------------------------------------------------- | ----------------------------- |
    | [configuration.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/sleepkit/stage/ss-5-tcn-sm/latest/configuration.json)   | Configuration file            |
    | [model.keras](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/sleepkit/stage/ss-5-tcn-sm/latest/model.keras)            | Keras Model file              |
    | [model.tflite](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/sleepkit/stage/ss-5-tcn-sm/latest/model.tflite)       | TFLite Model file             |
    | [metrics.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/sleepkit/stage/ss-5-tcn-sm/latest/metrics.json)       | Metrics file                  |

---
