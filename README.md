## Unsupervised Learning of Video Representations using LSTMs

We use Long Short Term Memory (LSTM) networks to learn representations of video sequences. Our model uses an encoder LSTM to map an input sequence into a fixed length representation. This representation is decoded using single or multiple decoder LSTMs to perform different tasks, such as reconstructing the input sequence, or predicting the future sequence. We experiment with two kinds of input sequences – patches of image pixels and high-level representations (``percepts") of video frames extracted using a pretrained convolutional net. We explore different design choices such as whether the decoder LSTMs should condition on the generated output. We analyze the outputs of the model qualitatively to see how well the model can extrapolate the learned video representation into the future and into the past. We further evaluate the representations by finetuning them for a supervised learning problem – human action recognition on the UCF-101 and HMDB-51 datasets. We show that the representations help improve classification accuracy, especially when there are only few training examples. Even models pretrained on unrelated datasets (300 hours of YouTube videos) can help action recognition performance.

## Getting Started

Compile cudamat library.
Modify CUDA_ROOT in the Makefile

Compile .proto file
protoc -I=./ --python_out=./ config.proto

## Bouncing MNIST
python lstm_combo.py models/lstm_combo_2layer_mnist.pbtxt datasets/bouncing_mnist.pbtxt datasets/bouncing_mnist.pbtxt 1