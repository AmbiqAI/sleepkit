# Model Architecture

For sleep stage classification, we leverage a modified Temporal Convolutional Network (TCN) architecture. During preliminary exploration we also tried various 1-D CNN, RNN (LSTM and GRU), and U-NET based architectures but found them to either require too much memory, computation, or suffer significant accuracy degredation when quantizing to 8-bit. Please refer to [Experiments](./experiments.md) for more details on architecture exploration.

The standard TCN architecture consists of several TCN blocks with each block consisting of 2 sequential dilated 1-D convolutional layers followed by a residual connection. Each convolutional layer is followed by weight normalization and ReLU layers. In our implementation, we replace the first convolutional layer with a separable convolutional layer and the second convolutional layer with a depthwise convolutional layer (w/o dilation). We also added a squeeze and excitation (SE) block between the convolutional layers to emphasize specific channels. In place of weight normalization we use standard batch normalization to enable fusing them after training. ReLU is also replaced with the approximated ReLU6.

Before the TCN, the inputs are encoded using a 1-D seperable convolutional layer. The outputs of the TCN are passed through a 1-D convolutional layer to reduce the number of channels to the number of classes instead of using a fully connected layer. The final output is passed through a softmax layer to produce the final class probabilities.

The below diagram shows the full model architecture for the sleep stage classification.

<!-- d: dilation rate
k: kernel size
s: stride -->
