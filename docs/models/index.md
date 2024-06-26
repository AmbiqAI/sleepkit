# :factory: Model Factory

SleepKit provides a model factory that allows you to easily create and train customized models. The model factory is a wrapper around the [TensorFlow Keras API](https://www.tensorflow.org/api_docs/python/tf/keras) that allows you to create functional-based models using high-level parameters. Most of the models are based on state-of-the-art architectures that have been modified to allow for more fine-grain customization. We also provide 1D variants to allow for training on time-series data. The included models are well suited for efficient, real-time edge applications.

<!-- * Seperable (depthwise + pointwise) Convolutions
* MBConv Blocks w/ Squeeze & Excitation
* Over-Parameterized Convolutional Branches
* Dilated Convolutions
* Quantization Aware Training (QAT) and Post-Training Quantization (PTQ) -->

---

## <span class="sk-h2-span">Available Models</span>

- **[TCN](./tcn.md)**: A CNN leveraging dilated convolutions
- **[U-Net](./unet.md)**: A CNN with encoder-decoder architecture for segmentation tasks
- **[U-NeXt](./unext.md)**: A U-Net variant leveraging MBConv blocks
- **[EfficientNetV2](./efficientnet.md)**: A CNN leveraging MBConv blocks
- **[MobileOne](./mobileone.md)**: A CNN aimed at sub-1ms inference
- **[ResNet](./resnet.md)**: A popular CNN often used for vision tasks

---

## <span class="sk-h2-span">Usage</span>

The model factory can be invoked either via CLI or within the `sleepkit` python package. At a high level, the model factory performs the following actions based on the provided configuration parameters:

!!! Example

    === "JSON"

        ```json
        {
            "name": "tcn",
            "params": {
                "input_kernel": [1, 3],
                "input_norm": "batch",
                "blocks": [
                    {"depth": 1, "branch": 1, "filters": 12, "kernel": [1, 3], "dilation": [1, 1], "dropout": 0.10, "ex_ratio": 1, "se_ratio": 0, "norm": "batch"},
                    {"depth": 1, "branch": 1, "filters": 20, "kernel": [1, 3], "dilation": [1, 1], "dropout": 0.10, "ex_ratio": 1, "se_ratio": 2, "norm": "batch"},
                    {"depth": 1, "branch": 1, "filters": 28, "kernel": [1, 3], "dilation": [1, 2], "dropout": 0.10, "ex_ratio": 1, "se_ratio": 2, "norm": "batch"},
                    {"depth": 1, "branch": 1, "filters": 36, "kernel": [1, 3], "dilation": [1, 4], "dropout": 0.10, "ex_ratio": 1, "se_ratio": 2, "norm": "batch"},
                    {"depth": 1, "branch": 1, "filters": 40, "kernel": [1, 3], "dilation": [1, 8], "dropout": 0.10, "ex_ratio": 1, "se_ratio": 2, "norm": "batch"}
                ],
                "output_kernel": [1, 3],
                "include_top": true,
                "use_logits": true,
                "model_name": "tcn"
            }
        }
        ```

    === "Python"

        ```python
        import keras
        from sleepkit.models import Tcn, TcnParams, TcnBlockParams

        inputs = keras.Input(shape=(800, 1))
        num_classes = 5

        model = Tcn(
            x=inputs,
            params=TcnParams(
                input_kernel=(1, 3),
                input_norm="batch",
                blocks=[
                    TcnBlockParams(filters=8, kernel=(1, 3), dilation=(1, 1), dropout=0.1, ex_ratio=1, se_ratio=0, norm="batch"),
                    TcnBlockParams(filters=16, kernel=(1, 3), dilation=(1, 2), dropout=0.1, ex_ratio=1, se_ratio=0, norm="batch"),
                    TcnBlockParams(filters=24, kernel=(1, 3), dilation=(1, 4), dropout=0.1, ex_ratio=1, se_ratio=4, norm="batch"),
                    TcnBlockParams(filters=32, kernel=(1, 3), dilation=(1, 8), dropout=0.1, ex_ratio=1, se_ratio=4, norm="batch"),
                ],
                output_kernel=(1, 3),
                include_top=True,
                use_logits=True,
                model_name="tcn",
            ),
            num_classes=num_classes,
        )
        ```

---
