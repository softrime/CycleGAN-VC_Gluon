#!/mnt/lustre/sjtu/users/szw73/.miniconda/envs/gluon/bin/python
import os, sys
import codecs

sys.path.append('/mnt/lustre/sjtu/users/szw73/work/VC/CycleGAN/SF1-TF2')

import numpy as np
import mxnet as mx
from mxnet import gluon, nd

from config_util import parse_args, get_checkpoint_path, parse_contexts
from model import Generator, Discriminator
from data_iter import SentenceIter
import ttspacker
packer = ttspacker.ttspacker()
if __name__=='__main__':
  args = parse_args('default.cfg')
  source_scp = args.config.get('test', 'test_source_scp')
  target_scp = args.config.get('test', 'test_target_scp')
  feat_dim   = args.config.getint('data', 'feat_dim')
  segment_length   = args.config.getint('train', 'segment_length')
  num_iteration    = args.config.getint('train', 'num_iteration')
  G_learning_rate  = args.config.getfloat('train', 'G_learning_rate')
  D_learning_rate  = args.config.getfloat('train', 'D_learning_rate')
  momentum         = args.config.getfloat('train', 'momentum')
  source_speaker   = args.config.get('data', 'source_speaker')
  target_speaker   = args.config.get('data', 'target_speaker')
  lambda_cyc       = args.config.getfloat('train', 'lambda_cyc')
  lambda_id        = args.config.getfloat('train', 'lambda_id')
  input_gv         = args.config.get('data', 'source_gv')
  output_gv        = args.config.get('data', 'target_gv')
  G_A_check_iter   = args.config.getint('test', 'G_A_check_iter')
  G_B_check_iter   = args.config.getint('test', 'G_B_check_iter')

  source_gv = np.loadtxt(input_gv)
  target_gv = np.loadtxt(output_gv)
  source_mean = source_gv[0]
  source_std  = source_gv[1]
  target_mean = target_gv[0]
  target_std  = target_gv[1]
  
  if G_A_check_iter >= 10000:
    lambda_id = 0

  G_A = Generator()
  G_B = Generator()

  G_A.collect_params().load('checkpoints/G_A/'+'G_A_'+source_speaker+'-'+target_speaker+'_mgc-'+str(feat_dim)+'_iteration-'+str(G_A_check_iter)+'_seglen-'+str(segment_length)+'_lambda-'+str(lambda_cyc)+'-'+str(lambda_id)+'_lr-'+str(G_learning_rate)+'-'+str(D_learning_rate)+'.params', ctx=mx.cpu())
  G_B.collect_params().load('checkpoints/G_B/'+'G_B_'+source_speaker+'-'+target_speaker+'_mgc-'+str(feat_dim)+'_iteration-'+str(G_B_check_iter)+'_seglen-'+str(segment_length)+'_lambda-'+str(lambda_cyc)+'-'+str(lambda_id)+'_lr-'+str(G_learning_rate)+'-'+str(D_learning_rate)+'.params', ctx=mx.cpu())
  

  s2t_outpath = 'output/'+source_speaker + '-' + target_speaker + '_' + str(G_A_check_iter)
  t2s_outpath = 'output/'+target_speaker + '-' + source_speaker + '_' + str(G_B_check_iter)
  os.system('mkdir -p %s'% s2t_outpath)
  os.system('mkdir -p %s'% t2s_outpath)

  writer = ttspacker.ttspacker('HTK', 'HTK')
  files = codecs.open(source_scp).readlines()
  for _, f in enumerate(files):
    path = f.strip().split('\n')[0]
    print('processing source:' + path)
    basename = path.split('/')[-1].split('.')[0]
    savepath = s2t_outpath + '/' + basename + '.htk'
    #data = packer.readcmp(path)
    data = np.fromfile(path, dtype='<f', count=-1, sep='').reshape(-1, 25)
    data += source_mean
    data *= source_std
    data = data[:, 1:25]
    num_data_seg = data.shape[0]//segment_length + 1
    for i in range(1, num_data_seg+1):
      if data.shape[0] >= i * segment_length:
        data_seg = data[(i - 1) * segment_length : i * segment_length, :]
      else:
        data_seg = np.concatenate((data[(i - 1) * segment_length : , :], np.zeros((i * segment_length - data.shape[0], feat_dim))), axis=0)
      assert(data_seg.shape==(segment_length, feat_dim))  
      data_seg = nd.swapaxes(nd.array(data_seg), 1, 0)
      data_seg = nd.reshape(data_seg, (1, feat_dim, 1, -1))
      pred_seg = G_A(data_seg)
      pred_seg = nd.reshape(pred_seg, (feat_dim, -1))
      pred_seg = nd.swapaxes(pred_seg, 1, 0)
      pred_seg = pred_seg.asnumpy()
      pred_seg /= target_std[1:25]
      pred_seg -= target_mean[1:25]
      pred = pred_seg if i==1 else np.concatenate((pred, pred_seg), axis=0)
    pred = pred[:data.shape[0], :] 
    assert(data.shape==pred.shape)
    mgc_feat = []
    for i in range(pred.shape[0]):
      mgc_feat.append(pred[i].tolist())
    writer.writecmp(savepath, mgc_feat)

  '''
  files = codecs.open(target_scp).readlines()
  for _, f in enumerate(files):
    path = f.strip().split('\n')[0]
    print('processing target:' + path)
    basename = path.split('/')[-1].split('.')[0]
    savepath = t2s_outpath + '/' + basename + '.htk'
    #data = packer.readcmp(path)
    data = np.fromfile(path, dtype='<f', count=-1, sep='').reshape(-1, 25)
    data += target_mean
    data *= target_std
    data = data[:, 1:25]
    data = nd.swapaxes(nd.array(data), 1, 0)
    data = nd.reshape(data, (1, feat_dim, 1, -1))
    pred = G_B(data)
    pred = nd.reshape(pred, (feat_dim, -1))
    pred = nd.swapaxes(pred, 1, 0)
    pred = pred.asnumpy()
    pred /= target_std[1:25]
    pred -= target_mean[1:25]
    
    mgc_feat = []
    for i in range(pred.shape[0]):
      mgc_feat.append(pred[i].tolist())
    writer.writecmp(savepath, mgc_feat)
  '''
