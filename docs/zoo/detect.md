# Sleep Detection Model

## <span class="sk-h2-span">Model Overview</span>

The following table provides the latest pre-trained models for sleep detection. Below we also provide additional details including training configuration, accuracy metrics, and hardware performance results for the models.

--8<-- "assets/zoo/detect/detect-model-zoo-table.md"

---

## <span class="sk-h2-span">Model Details</span>

=== "SD-2-TCN-SM"

    The __SD-2-TCN-SM__ model is a 2-stage sleep detection model that uses a Temporal Convolutional Network (TCN) architecture. The model is trained on accelerometer sensor data collected from the wrist and is able to classify sleep and wake stages.

    - **Location**: Wrist
    - **Classes**: Awake, Sleep
    - **Frame Size**: 2 hours
    - **Datasets**: [MESA](../datasets/mesa.md)
    - **Feature Generator**: [FS-W-A-5](../features/featset03.md)
    - **Feature Set**: [FS-W-A-5-60]()

    ??? note "Configuration"

        --8<-- "assets/zoo/detect/sleep-detect-2-config.md"

## <span class="sk-h2-span">Model Performance</span>

=== "SD-2-TCN-SM"

    ### Confusion Matrix

    The confusion matrix provides a detailed breakdown of the model's performance on the test set. The rows represent the true labels, while the columns represent the predicted labels. The diagonal elements represent the number of correct predictions for each class, while the off-diagonal elements represent the number of incorrect predictions.

    <div class="sk-plotly-graph-div">
    --8<-- "assets/zoo/detect/detect-2-cm.html"
    </div>

    ### Sleep Efficiency Plot

    The following plot shows the model's performance in detecting sleep efficieny- percentage of time spent asleep while in bed. The x-axis represents the true sleep efficiency, while the y-axis represents the predicted sleep efficiency.

    <div class="sk-plotly-graph-div">
    --8<-- "assets/zoo/detect/detect-2-eff.html"
    </div>

    ### Total Sleep Time (TST) Plot

    The following plot shows the model's performance in detecting total sleep time (TST). The x-axis represents the true TST, while the y-axis represents the predicted TST. The plot shows a strong correlation between the true and predicted TST values.

    <div class="sk-plotly-graph-div">
    --8<-- "assets/zoo/detect/detect-2-tst.html"
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

--8<-- "assets/zoo/detect/detect-model-hw-table.md"
