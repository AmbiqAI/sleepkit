# Model Exporting

## <span class="sk-h2-span">Introduction </span>

Export mode is used to convert the trained TensorFlow model into a format that can be used for deployment onto Ambiq's family of SoCs. Currently, the command will convert the TensorFlow model into both TensorFlow Lite (TFL) and TensorFlow Lite for micro-controller (TFLM) variants. The command will also verify the models' outputs match. The activations and weights can be quantized by configuring the `quantization` section in the configuration file or by setting the `quantization` parameter in the code.

## <span class="sk-h2-span">Usage</span>

!!! Example

    The following command will export the sleep detection model to TF Lite and TFLM:

    === "CLI"

        ```bash
        sleepkit --mode export --task stage --config ./configs/sleep-stage-2.json
        ```

    === "Python"

        ```python
        from pathlib import Path
        import sleepkit as sk

        task = sk.TaskFactory.get("stage")
        task.export(sk.SKExportParams(
            ...
        ))
        ```


## <span class="sk-h2-span">Arguments </span>

The following table lists the parameters that can be used to configure the export mode. Once converted, the TFLM header file will be copied to location specified by `tflm_file`. The `threshold` flag can be used to set the model's output threshold.  The `use_logits` flag can be used to set the model's output to use logits or softmax.

--8<-- "assets/modes/export-params.md"
