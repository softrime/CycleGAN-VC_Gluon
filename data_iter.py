import os, sys
import codecs
import random

sys.path.append('/mnt/lustre/sjtu/users/szw73/work/VC/CycleGAN/SF1-TF2')

import numpy as np
import mxnet as mx
from mxnet import nd
from config_util import parse_args, get_checkpoint_path, parse_contexts

class SentenceIter(mx.io.DataIter):
  def __init__(self, contexts, scp, feat_dim, gv_file, is_train=True, segment_length=128):
    self.feat_dim = feat_dim
    self.contexts = contexts[0]
    self.load_data(scp, feat_dim, gv_file)
    self.is_train = is_train
    self.segment_length = segment_length
  def load_data(self, scp, feat_dim, gv_file):
    F = codecs.open(scp)
    files = F.readlines()
    F.close()
    self.data_sets = []
    gv = np.loadtxt(gv_file, dtype=np.float32)
    self.mean = gv[0][:]
    self.std  = gv[1][:]
    for _, f in enumerate(files):
      path = f.strip().split('\n')[0]
      print('loading:%s' % path)
      data = np.fromfile(path, dtype='<f', count=-1, sep='').reshape(-1, 25)
      data = data + self.mean
      data = data * self.std
      data = data[:, 1:25]
      data = nd.array(data, ctx=self.contexts)
      assert(data.shape[1]==self.feat_dim)
      self.data_sets.append(data[:, :feat_dim])
    self.num_sentences = len(self.data_sets)

  def __iter__(self):
    return self

  def reset(self):
    random.shuffle(self.data_sets)
    self.cur_sentence = 0

  def __next__(self):
    return self.next()

  def next(self):
    if self.is_train:
      sentence_idx = random.randint(0, self.num_sentences-1)
      sentence = self.data_sets[sentence_idx]
      
      segment_idx = random.randint(0, sentence.shape[0]-self.segment_length)
      data = sentence[segment_idx:segment_idx+self.segment_length, :]
      data = nd.swapaxes(nd.array(data), 1, 0)
      data = nd.reshape(data, (1, self.feat_dim, 1, self.segment_length))
      return data
    else:
      if self.cur_sentence < self.num_sentences:
        data = self.data_sets[self.cur_sentence]
        data = nd.swapaxes(nd.array(data), 1, 0).reshape((1, self.feat_dim, 1, -1))
        self.cur_sentence += 1
        return data
      else:
        raise StopIteration
