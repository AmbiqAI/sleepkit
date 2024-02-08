# Model Factory

SleepKit provides a model factory that allows you to easily create and train custom models. The model factory is a wrapper around the [TensorFlow Keras API](https://www.tensorflow.org/api_docs/python/tf/keras) that allows you to create functional-based models using high-level parameters. Most of the models are based on state-of-the-art architectures that have been modified to allow for more fine-grain customization. We also provide 1D variants to allow for training on time-series data.

---

## <span class="sk-h2-span">Temporal Convolutional Network (TCN)</span>

### Overview

Temporal convolutional network (TCN) is a type of convolutional neural network (CNN) that is commonly used for sequence modeling tasks such as speech recognition, text generation, and video classification. TCN is a fully convolutional network that consists of a series of dilated causal convolutional layers. The dilated convolutional layers allow TCN to have a large receptive field while maintaining a small number of parameters. TCN is also fully parallelizable, which allows for faster training and inference times.

For more info, refer to original paper [Temporal Convolutional Networks: A Unified Approach to Action Segmentation](https://doi.org/10.48550/arXiv.1608.08242)

### Additions

The TCN architecture has been modified to allow the following:

* Convolutional pairs can factorized into depthwise separable convolutions.
* Squeeze and excitation (SE) blocks can be added between convolutional pairs.
* Normalization can be set between batch normalization and layer normalization.
* ReLU is replaced with the approximated ReLU6.

---

## <span class="sk-h2-span">U-Net </span>

### Overview

U-Net is a type of convolutional neural network (CNN) that is commonly used for segmentation tasks. U-Net is a fully convolutional network that consists of a series of convolutional layers and pooling layers. The pooling layers are used to downsample the input while the convolutional layers are used to upsample the input. The skip connections between the pooling layers and convolutional layers allow U-Net to preserve spatial/temporal information while also allowing for faster training and inference times.

For more info, refer to original paper [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://doi.org/10.1007/978-3-319-24574-4_28)

### Additions

The U-Net architecture has been modified to allow the following:

* Enable 1D and 2D variants.
* Convolutional pairs can factorized into depthwise separable convolutions.
* Specifiy the number of convolutional layers per block both downstream and upstream.
* Normalization can be set between batch normalization and layer normalization.
* ReLU is replaced with the approximated ReLU6.

---

## <span class="sk-h2-span">U-NeXt </span>

### Overview

U-NeXt is a modification of U-Net that utilizes techniques from ResNeXt and EfficientNetV2. During the encoding phase, mbconv blocks are used to efficiently process the input.

### Additions

The U-NeXt architecture has been modified to allow the following:

* MBConv blocks used in the encoding phase.
* Squueze and excitation (SE) blocks added within blocks.

---

## <span class="sk-h2-span">EfficientNetV2 </span>

### Overview

EfficientNetV2 is an improvement to EfficientNet that incorporates additional optimizations reduce both computation and memory. In particular, the architecture leverages both fused and non-fused MBConv blocks, non-uniform layer scaling, and training-aware NAS.

For more info, refer to original paper [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)

### Additions

The EfficientNetV2 architecture has been modified to allow the following:

* Enable 1D and 2D variants.

---

## <span class="sk-h2-span">MobileOne </span>

### Overview

MobileOne is a fully convolutional neural network designed to have minimal latency when running in mobile/edge devices. The architecture consists of a series of depthwise separable convolutions and squeeze and excitation (SE) blocks. The network also uses standard batch normalization and ReLU activations that can be easily fused into the convolutional layers. Lastly, the network uses over-parameterized branches to improve training, yet can be merged into a single branch during inference.

For more info, refer to original paper [MobileOne: An Improved One millisecond Mobile Backbone](https://doi.org/10.48550/arXiv.2206.04040)

### Additions

The MobileOne architecture has been modified to allow the following:

* Enable 1D and 2D variants.
* Enable dilated convolutions.

---

## <span class="sk-h2-span">ResNet </span>

### Overview

ResNet is a type of convolutional neural network (CNN) that is commonly used for image classification tasks. ResNet is a fully convolutional network that consists of a series of convolutional layers and pooling layers. The pooling layers are used to downsample the input while the convolutional layers are used to upsample the input. The skip connections between the pooling layers and convolutional layers allow ResNet to preserve spatial/temporal information while also allowing for faster training and inference times.

For more info, refer to original paper [Deep Residual Learning for Image Recognition](https://doi.org/10.1109/CVPR.2016.90)

### Additions

* Enable 1D and 2D variants.
