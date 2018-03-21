import mxnet as mx
from mxnet import gluon, nd, init
from model import Generator, Discriminator

##### super parameters #####


##### make data #####
g_data = nd.ones((1, 24, 1, 1000))
d_data = nd.ones((1, 1, 24, 128))

##### create model #####
G_A = Generator()
G_B = Generator()
D_A = Discriminator()
D_B = Discriminator()
G_A.collect_params().initialize(ctx=mx.cpu())
G_B.collect_params().initialize(ctx=mx.cpu())
D_A.collect_params().initialize(ctx=mx.cpu())
D_B.collect_params().initialize(ctx=mx.cpu())
##### forward #####
#g_output = G_A(g_data)
d_output = D_A(d_data)

##### check outputs #####
print(d_output.shape)