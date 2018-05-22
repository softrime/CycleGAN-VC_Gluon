#!/usr/bin/python

import os, sys
import logging
import time

import mxnet as mx
from mxnet import gluon, autograd
from mxnet import nd

from config_util import parse_args, get_checkpoint_path, parse_contexts
from model import Generator, Discriminator
from data_iter import SentenceIter


def do_train(args, dataA_iter, dataB_iter, 
             G_A, G_B, D_A, D_B, 
             G_A_trainer, G_B_trainer, D_A_trainer, D_B_trainer, loss1, loss2):
  
  num_iteration    = args.config.getint('train', 'num_iteration')
  lambda_cyc       = args.config.getfloat('train', 'lambda_cyc')
  lambda_id        = args.config.getfloat('train', 'lambda_id')
  feat_dim         = args.config.getint('data', 'feat_dim')
  segment_length   = args.config.getint('train', 'segment_length')
  show_loss_every  = args.config.getint('train', 'show_loss_every')
  G_learning_rate  = args.config.getfloat('train', 'G_learning_rate')
  D_learning_rate  = args.config.getfloat('train', 'D_learning_rate')
  source_speaker   = args.config.get('data', 'source_speaker')
  target_speaker   = args.config.get('data', 'target_speaker')
  contexts         = parse_contexts(args)
  G_lr_decay = G_learning_rate / 200000
  D_lr_decay = D_learning_rate / 200000

  dataA_iter.reset()
  dataB_iter.reset()

  label = nd.zeros((1, 1), ctx=contexts[0])

  loss_cyc_A = 0
  loss_cyc_B = 0
  loss_D_A_fake   = 0
  loss_D_B_fake   = 0
  loss_D_A   = 0
  loss_D_B   = 0

  for p in G_A.collect_params():
    G_A.collect_params()[p].grad_req = 'add'
  for p in G_B.collect_params():
    G_B.collect_params()[p].grad_req = 'add'
  for p in D_A.collect_params():
    D_A.collect_params()[p].grad_req = 'add'
  for p in D_B.collect_params():
    D_B.collect_params()[p].grad_req = 'add'

  for iter in range(num_iteration):

    if iter==10000:
      lambda_id = 0

    if iter >= 200000:
      G_A_trainer.set_learning_rate(G_A_trainer.learning_rate - G_lr_decay)
      G_B_trainer.set_learning_rate(G_B_trainer.learning_rate - G_lr_decay)
      D_A_trainer.set_learning_rate(D_A_trainer.learning_rate - D_lr_decay)
      D_B_trainer.set_learning_rate(D_B_trainer.learning_rate - D_lr_decay)


    G_A.collect_params().zero_grad()
    G_B.collect_params().zero_grad()
    D_A.collect_params().zero_grad()
    D_B.collect_params().zero_grad()

    inputA = dataA_iter.next()
    inputB = dataB_iter.next()
    inputA = inputA.as_in_context(contexts[0])
    inputB = inputB.as_in_context(contexts[0])
      
    ##############################################################################
    # Train Generator                                                                    
    ##############################################################################

    # calculate loss for inputA
    inputA.attach_grad()
    with autograd.record():
      fakeB_tmp = G_A(inputA)
    fakeB = fakeB_tmp.copy()
    fakeB.attach_grad()
    with autograd.record():
      cycleA_tmp = G_B(fakeB)
    cycleA = cycleA_tmp.copy()
    cycleA.attach_grad()
    with autograd.record():
      print(cycleA, inputA)
      L_cycleA = loss1(cycleA, inputA)
    L_cycleA.backward()
    cycleA_grad = cycleA.grad * lambda_cyc
    cycleA_tmp.backward(cycleA_grad)

    label[:] = 1

    fakeB.attach_grad()
    with autograd.record():
      fakeB_D = nd.reshape(fakeB, (1, 1, feat_dim, segment_length))
      pred = D_B(fakeB_D)
      DlossB = loss2(pred, label)
    DlossB.backward()
    fakeB_grad_D = fakeB.grad
    fakeB_tmp.backward(fakeB_grad_D)

    # calculate loss for inputB
    inputB.attach_grad()
    with autograd.record():
      fakeA_tmp = G_B(inputB)
    fakeA = fakeA_tmp.copy()
    fakeA.attach_grad()
    with autograd.record():
      cycleB_tmp = G_A(fakeA)
    cycleB = cycleB_tmp.copy()
    cycleB.attach_grad()
    with autograd.record():
      L_cycleB = loss1(cycleB, inputB)
    L_cycleB.backward()
    cycleB_grad = cycleB.grad * lambda_cyc
    cycleB_tmp.backward(cycleB_grad)
    
    label[:] = 1
    fakeA.attach_grad()
    with autograd.record():
      fakeA_D = nd.reshape(fakeA, (1, 1, feat_dim, segment_length))
      pred = D_A(fakeA_D)
      DlossA = loss2(pred, label)
    DlossA.backward()
    fakeA_grad_D = fakeA.grad
    fakeA_tmp.backward(fakeA_grad_D)


    # identity loss
    inputB.attach_grad()
    with autograd.record():
      indenB_tmp = G_A(inputB)
    indenB = indenB_tmp.copy()
    indenB.attach_grad()
    with autograd.record():
      L = loss1(indenB, inputB)
    L.backward()
    indenB_grad = indenB.grad * lambda_id
    indenB_tmp.backward(indenB_grad)

    inputA.attach_grad()
    with autograd.record():
      indenA_tmp = G_B(inputA)
    indenA = indenA_tmp.copy()
    indenA.attach_grad()
    with autograd.record():
      L = loss1(indenA, inputA)
    L.backward()
    indenA_grad = indenA.grad * lambda_id
    indenA_tmp.backward(indenA_grad)


    ##############################################################################
    # Train Discriminator and Update                                                                 
    ##############################################################################

    fakeB = G_A(inputA)
    fakeA = G_B(inputB)
    def train_discriminator(modD, modD_trainer, real, fake):
      label[:] = 1
      real.attach_grad()
      with autograd.record():
        real = nd.reshape(real, (1, 1, feat_dim, segment_length))
        pred = modD(real)
        L_true = loss2(pred, label)
      L_true.backward()

      label[:] = 0
      fake.attach_grad()
      with autograd.record():
        fake = nd.reshape(fake, (1, 1, feat_dim, segment_length))
        pred = modD(fake)
        L_fake = loss2(pred, label)
      L_fake.backward()
      
      L = L_fake + L_true

      modD_trainer.step(1)
      return L/2
    
    lossD_A = train_discriminator(D_A, D_A_trainer, inputA, fakeA)
    lossD_B = train_discriminator(D_B, D_B_trainer, inputB, fakeB)



    ##############################################################################
    # Update Generator                                                                    
    ##############################################################################
    
    G_A_trainer.step(1)
    G_B_trainer.step(1)

    loss_cyc_A    += L_cycleA.asnumpy()[0]
    loss_cyc_B    += L_cycleB.asnumpy()[0]
    loss_D_B_fake += DlossB.asnumpy()[0]
    loss_D_A_fake += DlossA.asnumpy()[0]
    loss_D_A      += lossD_A.asnumpy()[0]
    loss_D_B      += lossD_B.asnumpy()[0]

    if iter % show_loss_every == 0 and iter != 0:
      loss_cyc_A    /= show_loss_every
      loss_cyc_B    /= show_loss_every
      loss_D_B_fake /= show_loss_every
      loss_D_A_fake /= show_loss_every
      loss_D_A      /= show_loss_every
      loss_D_B      /= show_loss_every
      logging.info('[%s] | iter[%d] | loss_cyc_A:%f | loss_cyc_B:%f | loss_D_B_fake:%f | loss_D_A_fake:%f | loss_D_A:%f | loss_D_B:%f', 
                    time.ctime(), iter, loss_cyc_A, loss_cyc_B, loss_D_B_fake, loss_D_A_fake, loss_D_A, loss_D_B)  
      loss_cyc_A    = 0
      loss_cyc_B    = 0
      loss_D_B_fake = 0
      loss_D_A_fake = 0
      loss_D_A      = 0
      loss_D_B      = 0
      
      G_A.collect_params().save('checkpoints/G_A/'+'G_A_'+source_speaker+'-'+target_speaker+'_mgc-'+str(feat_dim)+'_iteration-'+str(iter)+'_seglen-'+str(segment_length)+'_lambda-'+str(lambda_cyc)+'-'+str(lambda_id)+'_lr-'+str(G_learning_rate)+'-'+str(D_learning_rate)+'.params')
      G_B.collect_params().save('checkpoints/G_B/'+'G_B_'+source_speaker+'-'+target_speaker+'_mgc-'+str(feat_dim)+'_iteration-'+str(iter)+'_seglen-'+str(segment_length)+'_lambda-'+str(lambda_cyc)+'-'+str(lambda_id)+'_lr-'+str(G_learning_rate)+'-'+str(D_learning_rate)+'.params')
      D_A.collect_params().save('checkpoints/D_A/'+'D_A_'+source_speaker+'-'+target_speaker+'_mgc-'+str(feat_dim)+'_iteration-'+str(iter)+'_seglen-'+str(segment_length)+'_lambda-'+str(lambda_cyc)+'-'+str(lambda_id)+'_lr-'+str(G_learning_rate)+'-'+str(D_learning_rate)+'.params')
      D_B.collect_params().save('checkpoints/D_B/'+'D_B_'+source_speaker+'-'+target_speaker+'_mgc-'+str(feat_dim)+'_iteration-'+str(iter)+'_seglen-'+str(segment_length)+'_lambda-'+str(lambda_cyc)+'-'+str(lambda_id)+'_lr-'+str(G_learning_rate)+'-'+str(D_learning_rate)+'.params')


        
if __name__=='__main__':

  ##############################################################################
  # Get Parameters                                                                     
  ##############################################################################

  args = parse_args('default.cfg')
  train_source_scp = args.config.get('data', 'train_source_scp')
  train_target_scp = args.config.get('data', 'train_target_scp')
  feat_dim         = args.config.getint('data', 'feat_dim')
  segment_length   = args.config.getint('train', 'segment_length')
  num_iteration    = args.config.getint('train', 'num_iteration')
  G_learning_rate  = args.config.getfloat('train', 'G_learning_rate')
  D_learning_rate  = args.config.getfloat('train', 'D_learning_rate')
  momentum         = args.config.getfloat('train', 'momentum')
  source_speaker   = args.config.get('data', 'source_speaker')
  target_speaker   = args.config.get('data', 'target_speaker')
  lambda_cyc       = args.config.getfloat('train', 'lambda_cyc')
  lambda_id        = args.config.getfloat('train', 'lambda_id')
  source_gv        = args.config.get('data', 'source_gv')
  target_gv        = args.config.get('data', 'target_gv')
  contexts         = parse_contexts(args)

  ##############################################################################
  # Log Configure                                                                     
  ##############################################################################

  logfilename = 'LOG/%s-%s_mgc-%d_iteration-%d_seglen-%d_lambda-%f-%f_lr-%f-%f.log_%s'%(
                source_speaker, target_speaker, feat_dim, num_iteration, segment_length, 
                lambda_cyc, lambda_id, G_learning_rate, D_learning_rate, time.time())
  logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)-15s %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            filename=logfilename,
                            filemode='w')
        
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
  console.setFormatter(formatter)
  logging.getLogger('').addHandler(console)

  ##############################################################################
  # Load Data                                                                     
  ##############################################################################

  dataA_iter = SentenceIter(contexts, train_source_scp, feat_dim, source_gv,  
                            is_train=True, segment_length=segment_length)
  dataB_iter = SentenceIter(contexts, train_target_scp, feat_dim, target_gv, 
                            is_train=True, segment_length=segment_length)


  ##############################################################################
  # Create Models                                                                     
  ##############################################################################

  G_A = Generator()
  G_B = Generator()
  D_A = Discriminator()
  D_B = Discriminator()

  G_A.collect_params().initialize(ctx=contexts[0])
  G_B.collect_params().initialize(ctx=contexts[0])
  D_A.collect_params().initialize(ctx=contexts[0])
  D_B.collect_params().initialize(ctx=contexts[0])

  G_A_trainer = gluon.Trainer(G_A.collect_params(), 
                              optimizer='adam', 
                              optimizer_params={'learning_rate':G_learning_rate}
                              )

  G_B_trainer = gluon.Trainer(G_B.collect_params(), 
                              optimizer='adam', 
                              optimizer_params={'learning_rate':G_learning_rate}
                              )

  D_A_trainer = gluon.Trainer(D_A.collect_params(), 
                              optimizer='adam', 
                              optimizer_params={'learning_rate':D_learning_rate}
                              )

  D_B_trainer = gluon.Trainer(D_B.collect_params(), 
                              optimizer='adam', 
                              optimizer_params={'learning_rate':D_learning_rate}
                              )

  loss1 = gluon.loss.L1Loss()
  loss2 = gluon.loss.L2Loss()
  
                  
  ##############################################################################
  # Training                                                                     
  ##############################################################################

  do_train(args, dataA_iter, dataB_iter, 
           G_A, G_B, D_A, D_B, 
           G_A_trainer, G_B_trainer, D_A_trainer, D_B_trainer, loss1, loss2)
