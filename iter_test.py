from data_iter import SentenceIter

scp = '/home/szw73/Study/TTS/experiments/cycle_gan/CycleGAN-gluon/testing_scp/SF1.scp'

test_iter = SentenceIter(scp=scp, feat_dim=24)
test_iter.reset()

for i, data in enumerate(test_iter):
  print(data)
  input()
