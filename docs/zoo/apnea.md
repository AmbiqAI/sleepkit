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
    - **Feature Generator**: [FS-W-P-5](../features/featset01.md)
    - **Feature Set**: [FS-W-P-5-1]()

## <span class="sk-h2-span">Model Performance</span>

=== "SA-2-TCN-SM"

    The following plots show the model's performance in detecting apnea/hypopnea events. The first plot shows the confusion matrix for apnea detection.


    <div class="sk-plotly-graph-div">
    --8<-- "assets/zoo/apnea/SA-2-TCN-SM-cm.html"
    </div>

    The following plot shows the model's ability to detect AHI (Apnea-Hypopnea Index) compared to the ground truth AHI values. The x-axis represents the true AHI values, while the y-axis represents the predicted AHI values. The plot shows a strong correlation between the true and predicted AHI values.

    <div class="sk-plotly-graph-div">
    --8<-- "assets/zoo/apnea/sleep-apnea-ahi-scatter.html"
    </div>

    The following table provides the corresponding confusion matrix for AHI.

    <div class="sk-plotly-graph-div">
    --8<-- "assets/zoo/apnea/sleep-apnea-ahi-cm.html"
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
