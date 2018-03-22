import os, sys
import codecs
import random

import numpy as np
import mxnet as mx
from mxnet import nd


class SentenceIter(mx.io.DataIter):
  def __init__(self, scp, feat_dim, is_train=True, segment_length=128):
    self.feat_dim = feat_dim
    self.load_data(scp, feat_dim)
    self.is_train = is_train
    self.segment_length = segment_length
  
  def load_data(self, scp, feat_dim):
    F = codecs.open(scp)
    files = F.readlines()
    F.close()
    self.data_sets = []
    for _, f in enumerate(files):
      path = f.strip().split('\n')[0]
      data = np.fromfile(path, dtype='<f', count=-1, sep='').reshape(-1, 25)
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