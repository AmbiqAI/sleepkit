import keras

# import numpy as np
import neuralspot_edge as nse

from ..defines import NamedParams
# from .nstdb import NstdbNoise


def create_augmentation_layer(augmentation: NamedParams) -> keras.Layer:
    """Create an augmentation layer from a configuration

    Args:
        augmentation (NamedParams): Augmentation configuration

    Returns:
        keras.Layer: Augmentation layer

    Example:

    ```python
    import sleepkit as sk
    x = keras.random.normal
    layer = sk.datasets.augmentation.create_augmentation_layer(
        sk.NamedParams(name="random_noise", params={"factor": 0.01}),
    )
    y = layer(x)
    ```
    """
    match augmentation.name:
        case "amplitude_warp":
            return nse.layers.preprocessing.AmplitudeWarp(**augmentation.params)
        case "augmentation_pipeline":
            return create_augmentation_pipeline(augmentation.params)
        case "random_augmentation":
            return nse.layers.preprocessing.RandomAugmentation1DPipeline(
                layers=[
                    create_augmentation_layer(augmentation)
                    for augmentation in [NamedParams(**p) for p in augmentation.params["layers"]]
                ],
                augmentations_per_sample=augmentation.params.get("augmentations_per_sample", 3),
                rate=augmentation.params.get("rate", 1.0),
                batchwise=True,
            )
        case "random_background_noise":
            pass
            # nstdb = NstdbNoise(target_rate=sampling_rate)
            # noises = np.hstack(
            #     (nstdb.get_noise(noise_type="bw"), nstdb.get_noise(noise_type="ma"), nstdb.get_noise(noise_type="em"))
            # )
            # noises = noises.astype(np.float32)
            # return nse.layers.preprocessing.RandomBackgroundNoises1D(noises=noises, **augmentation.params)
        case "random_sine_wave":
            return nse.layers.preprocessing.RandomSineWave(**augmentation.params)
        case "random_cutout":
            return nse.layers.preprocessing.RandomCutout1D(**augmentation.params)
        case "random_noise":
            return nse.layers.preprocessing.RandomGaussianNoise1D(**augmentation.params)
        case "random_noise_distortion":
            return nse.layers.preprocessing.RandomNoiseDistortion1D(**augmentation.params)
        case "resizing":
            return nse.layers.preprocessing.Resizing1D(**augmentation.params)
        case "sine_wave":
            return nse.layers.preprocessing.AddSineWave(**augmentation.params)
        case "filter":
            return nse.layers.preprocessing.CascadedBiquadFilter(**augmentation.params)
        case "layer_norm":
            return nse.layers.preprocessing.LayerNormalization1D(**augmentation.params)
        case _:
            raise ValueError(f"Unknown augmentation '{augmentation.name}'")
    # END MATCH


def create_augmentation_pipeline(augmentations: list[NamedParams]) -> nse.layers.preprocessing.AugmentationPipeline:
    """Create an augmentation pipeline from a list of augmentation configurations.

    This is useful when running from a configuration file to hydrate the pipeline.

    Args:
        augmentations (list[NamedParams]): List of augmentation configurations

    Returns:
        nse.layers.preprocessing.AugmentationPipeline: Augmentation pipeline

    Example:

    ```python
    import sleepkit as sk
    x = keras.random.normal(shape=(256, 1), dtype="float32")

    augmenter = sk.datasets.create_augmentation_pipeline([
        sk.NamedParams(name="random_noise", params={"factor": 0.01}),
        sk.NamedParams(name="random_cutout", params={"factor": 0.01, "cutouts": 2}),
    ])

    y = augmenter(x)
    """
    if not augmentations:
        return keras.layers.Lambda(lambda x: x)
    aug = nse.layers.preprocessing.AugmentationPipeline(
        layers=[create_augmentation_layer(augmentation) for augmentation in augmentations]
    )
    return aug
