# CycleGAN Implementation for Voice Conversion By Mxnet-Gluon

This project is the **GLUON** implementation of CycleGAN. It is used to train VC(Voice Conversion) task, which convert a voice of source speaker to target speaker with same content.

* If you want to run this project on your own enviroment, I recommend you prepare your own data. And you should prepare '.scp' files which include data paths of trainset or testset, such like demo files I post in this project

* Each data in scp file should be 2 dimension matrix with shape (frames, feature_dimension), I use 24+1 dim mgc(mel-generalized cepstrum) here. You can also use features you like. You need to extract faetures by your own. **Be careful:** Because I use CNN, so if you use your own feature with different dimensions(for example, mel-sepstrum with 80 dims), you should change output channels of Generator network, and change stride of Discriminator to ensure output height is 1.

* There are also one thing you should do first is to normalize your feature for each dimension. I compute and store demo mean and std in norm directory, and normalize feature in data_iter.py. 

* I tried this project with VCC2016 dataset and get a not bad result, but not as good as paper shown. So there are still space to improve.


# Enviroment

* python 3.6.2

* mxnet 1.1.0

* numpy 1.13.3

# Reference Paper: [arXiv:1711.11293, 2017](https://arxiv.org/abs/1711.11293)
