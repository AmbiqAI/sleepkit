# Sleep Apnea Detection

## <span class="sk-h2-span">Model Overview</span>

The following table provides the latest pre-trained models for sleep apnea. Below we also provide additional details including training configuration, accuracy metrics, and hardware performance results for the models.

--8<-- "assets/zoo/apnea/apnea-model-zoo-table.md"

## <span class="sk-h2-span">Model Details</span>

=== "SA-2-TCN-SM"

    The __SA-2-TCN-SM__ model is a 2-stage sleep apnea detection model that uses a Temporal convolutional network (TCN). The model is trained on PPG sensor data collected from the wrist and is able to locate apnea/hypopnea events.

    - **Location**: Wrist
    - **Classes**: Normal, Apnea
    - **Frame Size**: 10 minutes
    - **Datasets**: [MESA](../datasets/mesa.md)
    - **Features**: [FS-W-P-5](../features/fs_w_p_5.md)

## <span class="sk-h2-span">Model Performance</span>

=== "SA-2-TCN-SM"

    The following plots show the model's performance in detecting apnea/hypopnea events. The first plot shows the confusion matrix for apnea detection.


    <div class="sk-plotly-graph-div">
    --8<-- "assets/zoo/apnea/sa-2-tcn-sm-cm.html"
    </div>

    The following plot shows the model's ability to detect AHI (Apnea-Hypopnea Index) compared to the ground truth AHI values. The x-axis represents the true AHI values, while the y-axis represents the predicted AHI values. The plot shows a strong correlation between the true and predicted AHI values.

    <div class="sk-plotly-graph-div">
    --8<-- "assets/zoo/apnea/sa-2-tcn-sm-ahi-scatter.html"
    </div>

    The following table provides the corresponding confusion matrix for AHI.

    <div class="sk-plotly-graph-div">
    --8<-- "assets/zoo/apnea/sa-2-tcn-sm-ahi-cm.html"
    </div>

---

## <span class="sk-h2-span">EVB Performance</span>


The following table provides the latest performance results when running on Apollo4 Plus EVB. These results are obtained using neuralSPOTs [Autodeploy tool](https://ambiqai.github.io/neuralSPOT/docs/From%20TF%20to%20EVB%20-%20testing%2C%20profiling%2C%20and%20deploying%20AI%20models.html). From neuralSPOT repo, the following command can be used to capture EVB results via Autodeploy:

```console
python -m ns_autodeploy \
--tflite-filename model.tflite \
--model-name model \
--cpu-mode 192 \
--arena-size-scratch-buffer-padding 0 \
--max-arena-size 60 \
```

--8<-- "assets/zoo/apnea/apnea-model-hw-table.md"

## <span class="sk-h2-span">Downloads</span>

=== "SA-2-TCN-SM"

    | Asset                                                                | Description                   |
    | -------------------------------------------------------------------- | ----------------------------- |
    | [configuration.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/sleepkit/apnea/sa-2-tcn-sm/latest/configuration.json)   | Configuration file            |
    | [model.keras](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/sleepkit/apnea/sa-2-tcn-sm/latest/model.keras)            | Keras Model file              |
    | [model.tflite](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/sleepkit/apnea/sa-2-tcn-sm/latest/model.tflite)       | TFLite Model file             |
    | [metrics.json](https://ambiqai-model-zoo.s3.us-west-2.amazonaws.com/sleepkit/apnea/sa-2-tcn-sm/latest/metrics.json)       | Metrics file                  |
