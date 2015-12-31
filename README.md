## Unsupervised Learning of Video Representations using LSTMs

Code for paper [Unsupervised Learning of Video Representations using LSTMs](http://arxiv.org/abs/1502.04681) by Nitish Srivastava, Elman Mansimov, Ruslan Salakhutdinov; ICML 2015.

Note that the code at [this link](http://www.cs.toronto.edu/~nitish/unsupervised_video/) is deprecated.

### Getting Started

To compile cudamat library you need to modify CUDA_ROOT in the cudamat/Makefile to the relevant cuda root path.

Other libraries you need to install are:

* h5py
* google.protobuf

Next compile .proto file by calling
```
protoc -I=./ --python_out=./ config.proto
```

You will also need to download the dataset files. These can be obtained by running

```
wget http://www.cs.toronto.edu/~emansim/datasets/mnist.h5
wget http://www.cs.toronto.edu/~emansim/datasets/bouncing_mnist_test.npy
wget http://www.cs.toronto.edu/~emansim/datasets/ucf101_train_patches.npy
wget http://www.cs.toronto.edu/~emansim/datasets/ucf101_valid_patches.npy
```

**Note to Toronto users:** You don't need to download any files, as they are available in my gobi3 repository and correct path is already set up.

### Bouncing (Moving) MNIST dataset

To train the sample model on a dataset of moving MNIST digits you need to set correct path to datasets in datasets/bouncing_mnist.pbtxt and datasets/bouncing_mnist_valid.pbtxt and then run (you may need to change the board id of gpu)

```
python lstm_combo.py models/lstm_combo_1layer_mnist.pbtxt datasets/bouncing_mnist.pbtxt datasets/bouncing_mnist.pbtxt 1
```

To visualize the sample reconstruction and future prediction results of the pretrained model run

```
python display_results.py models/lstm_combo_1layer_mnist_pretrained.pbtxt 1
```

### UCF-101 patches