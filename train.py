import os, sys

import mxnet as mx
from mxnet import gluon
from mxnet import nd

from config_util import parse_args, get_checkpoint_path, parse_contexts
from model import Generator, Discriminator
from data_iter import SentenceIter

def do_train():
  pass


if __name__=="__main__":

  ##############################################################################
  # Get Parameters                                                                     
  ##############################################################################

  args = parse_args('default.cfg')
  contexts = mx.cpu()


  ##############################################################################
  # Log Configure                                                                     
  ##############################################################################



  ##############################################################################
  # Load Data                                                                     
  ##############################################################################

  dataA_iter = SentenceIter()
  dataB_iter = SentenceIter()
  testA_iter = SentenceIter()
  testB_iter = SentenceIter()


  ##############################################################################
  # Create Models                                                                     
  ##############################################################################

  G_A = Generator()
  G_B = Generator()
  D_A = Discriminator()
  D_B = Discriminator()

  G_A.collect_params().initialize(ctx=contexts)
  G_B.collect_params().initialize(ctx=contexts)
  D_A.collect_params().initialize(ctx=contexts)
  D_B.collect_params().initialize(ctx=contexts)


  ##############################################################################
  # Training                                                                     
  ##############################################################################