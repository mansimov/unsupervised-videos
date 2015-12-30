## Unsupervised Learning of Video Representations using LSTMs

Code for paper [Unsupervised Learning of Video Representations using LSTMs](http://arxiv.org/abs/1502.04681)

Note that the code at [this link](http://www.cs.toronto.edu/~nitish/unsupervised_video/) is deprecated.

### Getting Started

To compile cudamat library you need to modify CUDA_ROOT in the cudamat/Makefile to the relevant cuda root path.

Other libraries you need to install are:

* h5py
* google.protobuf

Next compile .proto file by calling
`protoc -I=./ --python_out=./ config.proto`

### Bouncing (Moving) MNIST dataset
python lstm_combo.py models/lstm_combo_2layer_mnist.pbtxt datasets/bouncing_mnist.pbtxt datasets/bouncing_mnist.pbtxt 1