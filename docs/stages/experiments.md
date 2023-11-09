# Experiments

<!-- ## Architecture Exploration
TCN, RNN, Unet
Quantization
fp32 int16x8 int8x8 -->


## Ablation Studies

In the following, we perform ablation studies to investigate the impact of different design choices on the performance of the model. All ablation studies are performed using the 4-stage sleep classification model trained on the MESA dataset. Unless otherwise noted, all experiments are carried out with identical training procedures.

### Temporal Context

We first evaluate the impact of the temporal context- the number of time steps that are considered when making a prediction. For example, a temporal context of 1 means that only the current time step is considered when making a prediction. A temporal context of 2 means that the current time step and the previous time step are considered when making a prediction. The following plot shows the impact of the temporal context on the validation loss. As we can see, increasing the temporal context provides a roughly proportional decrease in loss up to around 2 hours after which the loss increases.

A typical night sleep involves 4 or 5 sleep cycles each of which lasts around 90-120 minutes. By providing the model with nearly an entire cycle allows the model to learn the patterns associated with each stage. However, providing the model with more than one cycle does not provide additional benefit. This is likely due to the fact that the model is already able to learn the patterns associated with each stage within a single cycle.

<div class="sk-plotly-graph-div">
--8<-- "assets/ablation-temporal.html"
</div>


### Model Width

Another important design choice is the width (aka. # channels) of the network. To reduce the number of hyper-parameters, we provide a width multiplier to adjust number of channels in each layer relative to the baseline model. For example, a width multiplier of 0.5 means that the number of channels in each layer is half of the baseline model. The following plot shows the impact of the width multiplier on the validation loss. Increasing the width to 1.25 provides a slight 1.2% reduction in loss but requires 47% more FLOPS. As we continue to increase the width, the loss largely plateaus while the FLOPS and memory requirements increase dramatically. On the other hand, decreasing the width below 1 causes a stark increase in loss.

<div class="sk-plotly-graph-div">
--8<-- "assets/ablation-width.html"
</div>


### Kernel Size

We further investigate the impact on kernel size on the validation loss. Traditionally, computer vision tasks leveraged 3x3 filter sizes. However, recent works have shown that larger filter sizes can provide better performance especially in 1D time-series by providing larger receptive fields. The plot below shows the impact of the kernel size on the validation loss. Using a filter length of 5 provides a significant reduction in loss while requiring 10% increase in FLOPS and 1K additional parameters. Beyond a filter length of 5, the loss plateaus while the FLOPS and memory requirements increase.

<div class="sk-plotly-graph-div">
--8<-- "assets/ablation-kernelsize.html"
</div>


### Squeeze and Excitation Ratio

In the following ablation, we evaluate the impact of the squeeze-and-excitation (SE) ratio. As background, the SE block "adaptively recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels." The SE reduction ratio is a hyperparameter that controls the amount channel compression within the SE block. When the SE block is utilized, the reduction ratio has negligible impact on the computational and memory cost. The following plot shows the impact on validation loss for different SE reduction ratios. NOTE: `ratio=0` corresponds to no SE block. We see that using an SE ratio of 4 results in the lowest validation loss. This provides a 16% reduction in loss while requiring only 3K additional parameters versus no SE blocks.

<div class="sk-plotly-graph-div">
--8<-- "assets/ablation-se-ratio.html"
</div>


### Dilated Convolution

Taking inspiration from WaveNet, we evaluate the impact of dilated convolutions on the validation loss. By utilizing increasing dilation rates, we are able to increase the temporal context without increasing the number of parameters nor computation. For example, using a dilation rate of 4 with a kernel length of 5 gives a temporal resolution of 20 time steps compared to 5 time steps for a standard convolution. The following bar chart shows the impact of using dilation convolutions on the validation loss. In the case with dilation, dilation rates of 1, 2, 4, and 8 are used. By adding dilation the model loss decreases by 23% without any increase on memory or computation footprint.


<div class="sk-plotly-graph-div">
--8<-- "assets/ablation-dilation.html"
</div>
