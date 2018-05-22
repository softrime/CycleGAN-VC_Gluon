# CycleGAN Implementation for Voice Conversion By Mxnet-Gluon
This project is the **GLUON** implementation of CycleGAN. It is used to train VC(Voice Conversion) task, which convert a voice of source speaker to target speaker with same content.
* If you want to run this project on your own enviroment, you should prepare your own '.scp' file which include data path of trainset or testset, such like demo files I post in this project
* Each data in scp file should be 2 dimension matrix with shape (frames, feature_dimension), I use mgc(mel-generalized cepstrum) here. You can also use features you like. You should extract faetures by your own.
* There are also one thing you should do first is to normalize your feature for each dimension. I compute and store my mean and std in norm directory, and normalize feature in data_iter.py. 
* I tried this project with VCC2016 dataset and get a not bad result(but not as good as paper shown). So there still are space to improve.

# Reference Paper: [arXiv:1711.11293, 2017](https://arxiv.org/abs/1711.11293)

