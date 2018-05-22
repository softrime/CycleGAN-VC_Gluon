import mxnet as mx
from mxnet import gluon
from mxnet import nd

class InstanceNorm(gluon.nn.Block):
  def __init__(self, channels, **kwargs):
    super(InstanceNorm, self).__init__(**kwargs)
    with self.name_scope():
      self.gamma = self.params.get('gamma', shape=(channels, ))
      self.beta = self.params.get('beta', shape=(channels, ))

  def forward(self, x):
    outputs = nd.InstanceNorm(x, gamma=self.gamma.data(), beta=self.beta.data())
    return outputs


class GLU(gluon.nn.Block):
  def __init__(self, **kwargs):
    super(GLU, self).__init__( **kwargs)

  def forward(self, x):
    a, b = nd.split(x, 2, 1)
    outputs = a * nd.sigmoid(b)
    return outputs


class PixelShuffler(gluon.nn.Block):
  def __init__(self, **kwargs):
    super(PixelShuffler, self).__init__(**kwargs)

  def forward(self, x, size=(1, 2)):
    # data.shape = 'NCHW'
    n, c, h, w = x.shape
    #print(x.shape)
    rh, rw = size
    oh, ow = h * rh, w * rw
    oc = c // (rh * rw)
    outputs = nd.reshape(data=x, shape=(n, oc, rh, rw, h, w))
    outputs = nd.transpose(data=outputs, axes=(0, 1, 4, 2, 5, 3))
    outputs = nd.reshape(data=outputs, shape=(n, oc, oh, ow))
    return outputs


class Generator(gluon.nn.Block):

  def __init__(self, **kwargs):
    super(Generator, self).__init__(**kwargs)
    with self.name_scope():
      self.Conv_in = gluon.nn.Conv2D(channels=128, kernel_size=(1, 15), strides=(1, 1), padding=(0, 7))
      self.GLU_in = GLU()

      self.Conv_d1 = gluon.nn.Conv2D(channels=256, kernel_size=(1, 5), strides=(1, 2), padding=(0, 2))
      self.IN_d1 = InstanceNorm(channels=256)
      self.GLU_d1 = GLU()
      self.Conv_d2 = gluon.nn.Conv2D(channels=1024, kernel_size=(1, 5), strides=(1, 2), padding=(0, 2)) # warning: This channel in paper is 512. I do this because GLU would half the channel.
      self.IN_d2 = InstanceNorm(channels=1024)
      self.GLU_d2 = GLU()

      self.Conv_r1_1 = gluon.nn.Conv2D(channels=1024, kernel_size=(1, 3), strides=(1, 1), padding=(0, 1))
      self.IN_r1_1 = InstanceNorm(channels=1024)
      self.GLU_r1_1 = GLU()
      self.Conv_r1_2 = gluon.nn.Conv2D(channels=512, kernel_size=(1, 3), strides=(1, 1), padding=(0, 1))
      self.IN_r1_2 = InstanceNorm(channels=512)
      self.Conv_r2_1 = gluon.nn.Conv2D(channels=1024, kernel_size=(1, 3), strides=(1, 1), padding=(0, 1))
      self.IN_r2_1 = InstanceNorm(channels=1024)
      self.GLU_r2_1 = GLU()
      self.Conv_r2_2 = gluon.nn.Conv2D(channels=512, kernel_size=(1, 3), strides=(1, 1), padding=(0, 1))
      self.IN_r2_2 = InstanceNorm(channels=512)
      self.Conv_r3_1 = gluon.nn.Conv2D(channels=1024, kernel_size=(1, 3), strides=(1, 1), padding=(0, 1))
      self.IN_r3_1 = InstanceNorm(channels=1024)
      self.GLU_r3_1 = GLU()
      self.Conv_r3_2 = gluon.nn.Conv2D(channels=512, kernel_size=(1, 3), strides=(1, 1), padding=(0, 1))
      self.IN_r3_2 = InstanceNorm(channels=512)
      self.Conv_r4_1 = gluon.nn.Conv2D(channels=1024, kernel_size=(1, 3), strides=(1, 1), padding=(0, 1))
      self.IN_r4_1 = InstanceNorm(channels=1024)
      self.GLU_r4_1 = GLU()
      self.Conv_r4_2 = gluon.nn.Conv2D(channels=512, kernel_size=(1, 3), strides=(1, 1), padding=(0, 1))
      self.IN_r4_2 = InstanceNorm(channels=512)
      self.Conv_r5_1 = gluon.nn.Conv2D(channels=1024, kernel_size=(1, 3), strides=(1, 1), padding=(0, 1))
      self.IN_r5_1 = InstanceNorm(channels=1024)
      self.GLU_r5_1 = GLU()
      self.Conv_r5_2 = gluon.nn.Conv2D(channels=512, kernel_size=(1, 3), strides=(1, 1), padding=(0, 1))
      self.IN_r5_2 = InstanceNorm(channels=512)
      self.Conv_r6_1 = gluon.nn.Conv2D(channels=1024, kernel_size=(1, 3), strides=(1, 1), padding=(0, 1))
      self.IN_r6_1 = InstanceNorm(channels=1024)
      self.GLU_r6_1 = GLU()
      self.Conv_r6_2 = gluon.nn.Conv2D(channels=512, kernel_size=(1, 3), strides=(1, 1), padding=(0, 1))
      self.IN_r6_2 = InstanceNorm(channels=512)

      self.Conv_u1 = gluon.nn.Conv2D(channels=1024, kernel_size=(1, 5), strides=(1, 1), padding=(0, 2))
      self.ps_u1 = PixelShuffler()
      self.IN_u1 = InstanceNorm(channels=512)
      self.ps_u1_g = PixelShuffler()
      self.GLU_u1 = GLU()
      self.Conv_u2 = gluon.nn.Conv2D(channels=512, kernel_size=(1, 5), strides=(1, 1), padding=(0, 2))
      self.ps_u2 = PixelShuffler()
      self.IN_u2 = InstanceNorm(channels=256)
      self.ps_u2_g = PixelShuffler()
      self.GLU_u2 = GLU()

      self.Conv_out = gluon.nn.Conv2D(channels=24, kernel_size=(1, 15), strides=(1, 1), padding=(0, 7))


  def forward(self, inputs):
    # data.shape = 'NCHW' h1-wT-c72
    # Preprocess
    inputs = self.Conv_in(inputs)
    inputs = self.GLU_in(inputs)

    # Downsample
    downsample1 = self.Conv_d1(inputs)
    downsample1 = self.IN_d1(downsample1)
    downsample1 = self.GLU_d1(downsample1)
    downsample2 = self.Conv_d2(downsample1)
    downsample2 = self.IN_d2(downsample2)
    downsample2 = self.GLU_d2(downsample2)

    # 6 residual blocks
    residual_input = downsample2
    # residual block 1
    residual_1 = self.Conv_r1_1(residual_input)
    residual_1 = self.IN_r1_1(residual_1)
    residual_1 = self.GLU_r1_1(residual_1)
    residual_1 = self.Conv_r1_2(residual_1)
    residual_1 = self.IN_r1_2(residual_1)
    residual_1 = residual_input + residual_1
    # residual block 2
    residual_2 = self.Conv_r2_1(residual_1)
    residual_2 = self.IN_r2_1(residual_2)
    residual_2 = self.GLU_r2_1(residual_2)
    residual_2 = self.Conv_r2_2(residual_2)
    residual_2 = self.IN_r2_2(residual_2)
    residual_2 = residual_1 + residual_2
    # residual block 3
    residual_3 = self.Conv_r3_1(residual_2)
    residual_3 = self.IN_r3_1(residual_3)
    residual_3 = self.GLU_r3_1(residual_3)
    residual_3 = self.Conv_r3_2(residual_3)
    residual_3 = self.IN_r3_2(residual_3)
    residual_3 = residual_2 + residual_3
    # residual block 4
    residual_4 = self.Conv_r4_1(residual_3)
    residual_4 = self.IN_r4_1(residual_4)
    residual_4 = self.GLU_r4_1(residual_4)
    residual_4 = self.Conv_r4_2(residual_4)
    residual_4 = self.IN_r4_2(residual_4)
    residual_4 = residual_3 + residual_4
    # residual block 5
    residual_5 = self.Conv_r5_1(residual_4)
    residual_5 = self.IN_r5_1(residual_5)
    residual_5 = self.GLU_r5_1(residual_5)
    residual_5 = self.Conv_r5_2(residual_5)
    residual_5 = self.IN_r5_2(residual_5)
    residual_5 = residual_4 + residual_5
    # residual block 6
    residual_6 = self.Conv_r6_1(residual_5)
    residual_6 = self.IN_r6_1(residual_6)
    residual_6 = self.GLU_r6_1(residual_6)
    residual_6 = self.Conv_r6_2(residual_6)
    residual_6 = self.IN_r6_2(residual_6)
    residual_6 = residual_5 + residual_6
    residual_out = residual_6
    # Upsample
    upsample1 = self.Conv_u1(residual_out)
    upsample1 = self.ps_u1(upsample1) # no problem
    upsample1 = self.IN_u1(upsample1)
    upsample1 = self.GLU_u1(upsample1)
    upsample2 = self.Conv_u2(upsample1)
    upsample2 = self.ps_u2(upsample2)
    upsample2 = self.IN_u2(upsample2)
    upsample2 = self.GLU_u2(upsample2)

    # Postprocess
    outputs = self.Conv_out(upsample2)
    
    # output.shape = (1, T, 72)
    return outputs






class Discriminator(gluon.nn.Block):

  def __init__(self, **kwargs):
    super(Discriminator, self).__init__(**kwargs)
    with self.name_scope():
      self.Conv_1 = gluon.nn.Conv2D(channels=128, kernel_size=(3, 3), strides=(1, 2), padding=(1, 1))
      self.GLU_1 = GLU()
      self.Conv_d1 = gluon.nn.Conv2D(channels=256, kernel_size=(3, 3), strides=(2, 2), padding=(1, 1))
      self.IN_d1 = InstanceNorm(channels=256)
      self.GLU_d1 = GLU()
      self.Conv_d2 = gluon.nn.Conv2D(channels=512, kernel_size=(3, 3), strides=(2, 2), padding=(1, 1))
      self.IN_d2 = InstanceNorm(channels=512)
      self.GLU_d2 = GLU()
      self.Conv_d3 = gluon.nn.Conv2D(channels=1024, kernel_size=(6, 3), strides=(1, 2), padding=(0, 1))
      self.IN_d3 = InstanceNorm(channels=1024)
      self.GLU_d3 = GLU()
      self.fc = gluon.nn.Dense(units=1, activation='sigmoid')

  def forward(self, inputs):
    # data.shape = 'NCHW' h24-wT-c1
    # pre
    inputs = self.Conv_1(inputs)
    inputs = self.GLU_1(inputs)

    # Downsample
    downsample_1 = self.Conv_d1(inputs)
    downsample_1 = self.IN_d1(downsample_1)
    downsample_1 = self.GLU_d1(downsample_1)
    downsample_2 = self.Conv_d2(downsample_1)
    downsample_2 = self.IN_d2(downsample_2)
    downsample_2 = self.GLU_d2(downsample_2)
    downsample_3 = self.Conv_d3(downsample_2)
    downsample_3 = self.IN_d3(downsample_3)
    downsample_3 = self.GLU_d3(downsample_3)

    # post
    outputs = self.fc(downsample_3)
    return outputs
