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
             G_A_trainer, G_B_trainer, D_A_trainer, D_B_trainer, loss):
  
  num_iteration    = args.config.getint('train', 'num_iteration')
  lambda_cyc       = args.config.getfloat('train', 'lambda_cyc')
  lambda_id        = args.config.getfloat('train', 'lambda_id')
  feat_dim         = args.config.getint('data', 'feat_dim')
  segment_length   = args.config.getint('train', 'segment_length')
  show_loss_every  = args.config.getint('train', 'show_loss_every')


  dataA_iter.reset()
  dataB_iter.reset()

  label = nd.zeros((1, 1))

  loss_cyc_A = 0
  loss_cyc_B = 0
  loss_D_A_fake   = 0
  loss_D_B_fake   = 0
  loss_D_A   = 0
  loss_D_B   = 0

  for iter in range(num_iteration):
    inputA = dataA_iter.next()
    inputB = dataB_iter.next()

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

    if iter==43:
      #print(G_A.upsample2, G_A.outputs);input()
      for i, p in enumerate(G_A.collect_params()):
        print(p, G_A.collect_params()[p].data())




    cycleA.attach_grad()
    with autograd.record():
      L_cycleA = loss(cycleA, inputA)
    L_cycleA.backward()
    cycleA_grad = cycleA.grad * lambda_cyc
    cycleA_tmp.backward(cycleA_grad)
    fakeB_grad_G = fakeB.grad

    label[:] = 1

    fakeB.attach_grad()
    with autograd.record():
      fakeB_D = nd.reshape(fakeB, (1, 1, feat_dim, segment_length))
      pred = D_B(fakeB_D)
      DlossB = loss(pred, label)
    DlossB.backward()
    fakeB_grad_D = fakeB.grad
    fakeB_tmp.backward(fakeB_grad_G + fakeB_grad_D)

    gradG_A = [G_A.collect_params()[p].grad().copy() for i, p in enumerate(G_A.collect_params())]
    gradG_B = [G_B.collect_params()[p].grad().copy() for i, p in enumerate(G_B.collect_params())]


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
      L_cycleB = loss(cycleB, inputB)
    L_cycleB.backward()
    cycleB_grad = cycleB.grad * lambda_cyc
    cycleB_tmp.backward(cycleB_grad)
    fakeA_grad_G = fakeA.grad
    
    label[:] = 1
    fakeA.attach_grad()
    with autograd.record():
      fakeA_D = nd.reshape(fakeA, (1, 1, feat_dim, segment_length))
      pred = D_A(fakeA_D)
      DlossA = loss(pred, label)
    DlossA.backward()
    fakeA_grad_D = fakeA.grad
    fakeA_tmp.backward(fakeA_grad_G + fakeA_grad_D)


    # update G
    for i, p in enumerate(G_A.collect_params()):
      gradsr = G_A.collect_params()[p].grad()
      gradsf = gradG_A[i]
      assert(gradsr.shape==gradsf.shape)
      assert(len(G_A.collect_params()[p]._grad)==1)
      G_A.collect_params()[p]._grad[0] = gradsr + gradsf



    for i, p in enumerate(G_B.collect_params()):
      gradsr = G_B.collect_params()[p].grad()
      gradsf = gradG_B[i]
      assert(gradsr.shape==gradsf.shape)
      assert(len(G_B.collect_params()[p]._grad)==1)
      G_B.collect_params()[p]._grad[0] = gradsr + gradsf
      
    gradG_A = [G_A.collect_params()[p].grad().copy() for i, p in enumerate(G_A.collect_params())]
    gradG_B = [G_B.collect_params()[p].grad().copy() for i, p in enumerate(G_B.collect_params())]


    # identity loss
    inputB.attach_grad()
    with autograd.record():
      indenB_tmp = G_A(inputB)
    indenB = indenB_tmp.copy()
    indenB.attach_grad()
    with autograd.record():
      L = loss(indenB, inputB)
    L.backward()
    indenB_grad = indenB.grad * lambda_id
    indenB_tmp.backward(indenB_grad)

    inputA.attach_grad()
    with autograd.record():
      indenA_tmp = G_B(inputA)
    indenA = indenA_tmp.copy()
    indenA.attach_grad()
    with autograd.record():
      L = loss(indenA, inputA)
    L.backward()
    indenA_grad = indenA.grad * lambda_id
    indenA_tmp.backward(indenA_grad)


    # update G
    for i, p in enumerate(G_A.collect_params()):
      gradsr = G_A.collect_params()[p].grad()
      gradsf = gradG_A[i]
      assert(gradsr.shape==gradsf.shape)
      assert(len(G_A.collect_params()[p]._grad)==1)
      G_A.collect_params()[p]._grad[0] = gradsr + gradsf

    for i, p in enumerate(G_B.collect_params()):
      gradsr = G_B.collect_params()[p].grad()
      gradsf = gradG_B[i]
      assert(gradsr.shape==gradsf.shape)
      assert(len(G_B.collect_params()[p]._grad)==1)
      G_B.collect_params()[p]._grad[0] = gradsr + gradsf

    gradG_A = [G_A.collect_params()[p].grad().copy() for i, p in enumerate(G_A.collect_params())]
    gradG_B = [G_B.collect_params()[p].grad().copy() for i, p in enumerate(G_B.collect_params())]


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
        L_true = loss(pred, label)
      L_true.backward()
      gradmodD = [modD.collect_params()[p].grad().copy() for i, p in enumerate(modD.collect_params())]

      label[:] = 0
      fake.attach_grad()
      with autograd.record():
        fake = nd.reshape(fake, (1, 1, feat_dim, segment_length))
        pred = modD(fake)
        L_fake = loss(pred, label)
      L_fake.backward()
      
      L = L_fake + L_true
      for i, p in enumerate(modD.collect_params()):
        gradsr = modD.collect_params()[p].grad()
        gradsf = gradmodD[i]
        assert(gradsr.shape==gradsf.shape)
        assert(len(modD.collect_params()[p]._grad)==1)
        modD.collect_params()[p]._grad[0] = gradsr + gradsf
      modD_trainer.step(1)
      return L/2
    
    lossD_A = train_discriminator(D_A, D_A_trainer, inputA, fakeA)
    lossD_B = train_discriminator(D_B, D_B_trainer, inputB, fakeB)



    ##############################################################################
    # Update Generator                                                                    
    ##############################################################################

    for i, p in enumerate(G_A.collect_params()):
      gradsr = G_A.collect_params()[p].grad()
      gradsf = gradG_A[i]
      assert(gradsr.shape==gradsf.shape)
      assert(len(G_A.collect_params()[p]._grad)==1)
      G_A.collect_params()[p]._grad[0] = gradsr + gradsf
    G_A_trainer.step(1)

    for i, p in enumerate(G_B.collect_params()):
      gradsr = G_B.collect_params()[p].grad()
      gradsf = gradG_B[i]
      assert(gradsr.shape==gradsf.shape)
      assert(len(G_B.collect_params()[p]._grad)==1)
      G_B.collect_params()[p]._grad[0] = gradsr + gradsf
    G_B_trainer.step(1)

    loss_cyc_A    += L_cycleA.asnumpy()[0]
    loss_cyc_B    += L_cycleB.asnumpy()[0]
    loss_D_B_fake += DlossB.asnumpy()[0]
    loss_D_A_fake += DlossA.asnumpy()[0]
    loss_D_A      += lossD_A.asnumpy()[0]
    loss_D_B      += lossD_B.asnumpy()[0]
    #print(loss_cyc_A, loss_cyc_B, loss_D_B_fake, loss_D_A_fake, loss_D_A, loss_D_B)

    if iter % show_loss_every == 0 and iter != 0:
      loss_cyc_A    /= show_loss_every
      loss_cyc_B    /= show_loss_every
      loss_D_B_fake /= show_loss_every
      loss_D_A_fake /= show_loss_every
      loss_D_A      /= show_loss_every
      loss_D_B      /= show_loss_every
      logging.info('iter[%d] | loss_cyc_A:%f | loss_cyc_B:%f | loss_D_B_fake:%f | loss_D_A_fake:%f | loss_D_A:%f | loss_D_B:%f', 
                    iter, loss_cyc_A, loss_cyc_B, loss_D_B_fake, loss_D_A_fake, loss_D_A, loss_D_B)  
      loss_cyc_A    = 0
      loss_cyc_B    = 0
      loss_D_B_fake = 0
      loss_D_A_fake = 0
      loss_D_A      = 0
      loss_D_B      = 0


        
if __name__=='__main__':

  ##############################################################################
  # Get Parameters                                                                     
  ##############################################################################

  args = parse_args('default.cfg')
  contexts = mx.cpu()
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

  dataA_iter = SentenceIter(train_source_scp, feat_dim, 
                            is_train=True, segment_length=segment_length)
  dataB_iter = SentenceIter(train_target_scp, feat_dim, 
                            is_train=True, segment_length=segment_length)


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

  loss = gluon.loss.L2Loss()
                    

  ##############################################################################
  # Training                                                                     
  ##############################################################################

  do_train(args, dataA_iter, dataB_iter, 
           G_A, G_B, D_A, D_B, 
           G_A_trainer, G_B_trainer, D_A_trainer, D_B_trainer, loss)