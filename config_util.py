#!/usr/bin/python

import re
import sys
import os, os.path
import argparse
import mxnet as mx
import numpy as np

if sys.version_info >= (3, 0):
  import configparser
else:
  import ConfigParser as configparser

def parse_args(cfgfile):
  default_cfg = configparser.ConfigParser()
  default_cfg.read(os.path.join(os.path.dirname(__file__), cfgfile))
  parser = argparse.ArgumentParser()
  parser.add_argument("--configure", help = "config file for training parameters")

  for sec in default_cfg.sections():
    for name, _ in default_cfg.items(sec):
      arg_name = '--%s_%s' %(sec, name)
      doc = 'Overwrite %s in section [%s] of config file' %(name, sec)
      parser.add_argument(arg_name, help = doc)
  
  args = parser.parse_args()
  args.config = default_cfg
  return args

  for sec in default_cfg.sections():
    for name, _ in default_cfg.items(sec):
      arg_name = ('%s_%s'% (sec, name)).replace('-', '_')
      if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
        sys.stderr.write('!! CMDLine overwriting %s.%s:\n' % (sec, name))
        sys.stderr.write("    '%s' => '%s'\n" % (default_cfg.get(sec, name),getattr(args, arg_name)))
        default_cfg.set(sec, name, getattr(args, arg_name))
  sys.stderr.write("="*80+"\n")
  args.config = default_cfg
  return args

def get_checkpoint_path(args):
  prefix = args.config.get('train', 'prefix')
  if os.path.isabs(prefix):
    return prefix
  return os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints', prefix))

def parse_contexts(args):
  contexts = re.split(r'\W+', args.config.get('train', 'context'))
  for i, ctx in enumerate(contexts):
    if ctx[:3] == 'gpu':
      contexts[i] = mx.context.gpu(int(ctx[3:]))
    else:
      contexts[i] = mx.context.cpu(int(ctx[3:]))
  return contexts
