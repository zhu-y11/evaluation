# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Multilingual Sentiment Analysis 
@Author Yi Zhu
Upated 23/01/2018
"""

#************************************************************
# Imported Libraries
#************************************************************
import random
from random import shuffle
from tqdm import tqdm
import string

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1234)
random.seed(1234)

import pdb


class absaAnalyzer(nn.Module):
  def __init__(self, word2idx, idx2word, ent2idx, idx2ent, asp2idx, idx2asp, pol2idx, idx2pol, ea2idx, idx2ea, emb_vocab, emb, max_seq_len):
    super(absaAnalyzer, self).__init__()

    self.word2idx = word2idx
    self.idx2word = idx2word
    self.ent2idx = ent2idx
    self.idx2ent = idx2ent
    self.asp2idx = asp2idx
    self.idx2asp = idx2asp
    self.pol2idx = pol2idx
    self.idx2pol = idx2pol
    self.ea2idx = ea2idx
    self.idx2ea = idx2ea
    self.vocab_size = len(word2idx)
    self.vocab_pad = self.vocab_size
    self.ent_size = len(ent2idx)
    self.asp_size = len(asp2idx)
    self.pol_size = len(pol2idx)
    self.ea_size = len(ea2idx)
 
    self.max_seq_len = max_seq_len

    # word embedding dim = pre-trained emb dim
    self.embed_dim = emb.size()[1]
    self.word_embeds = self.initEmbeds(emb_vocab, emb)    
    # conv layer
    self.filter_n = 300
    self.win = 5

    self.conv1 = nn.Conv2d(1, self.filter_n, (self.win, self.embed_dim))
    self.tanh = nn.Tanh()
    # polling
    self.maxpool = nn.MaxPool1d(self.max_seq_len - self.win + 1)
    #MLP
    self.h_dim = 100
    self.l1 = nn.Linear(self.filter_n, self.h_dim)
    self.l2 = nn.Linear(self.h_dim, self.ea_size)
    #sigmoid
    self.t = 0.2
    self.sigmoid = nn.Sigmoid()

    '''
    # entity dim and aspect dim are the same
    self.ea_dim = ea_dim
    # initialize both randomly
    self.ent_embeds = nn.Embedding(self.ent_size, self.ea_dim)
    self.asp_embeds = nn.Embedding(self.asp_size, self.ea_dim)

    self.dropout = nn.Dropout(p = 0.3)

    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.bidirectional = bidirectional 
    self.lstm = nn.LSTM(self.embed_dim, self.hidden_dim // 2, 
                        num_layers = self.num_layers, dropout = 0.3, bidirectional = self.bidirectional)
    self.hidden = self.init_hidden()
    
    # scoring network for attentions
    self.mlp_intoh = nn.Linear(self.hidden_dim + 2 * self.ea_dim, mlph_dim)
    self.mlp_htoo = nn.Linear(mlph_dim, 1)

    # projection from hidden_dim to target size
    self.atttoo = nn.Linear(self.hidden_dim, self.target_size)
    '''


  def initEmbeds(self, emb_vocab, emb):
    print('initializing embedding layers ...')
    # initialize randomly with padding
    embeds = nn.Embedding(self.vocab_size + 1, self.embed_dim, padding_idx = self.vocab_size)
    for w, idx in tqdm(self.word2idx.items(), total = self.vocab_size):
      # skip words that are not in pre-trained embeddings and paddings
      if w not in emb_vocab:
        continue
      # initialize words that are in the pretrained embeddings
      embeds.weight.data[idx] = emb[emb_vocab.index(w)]
    return embeds


  def init_hidden(self):
    n_lstm = 2 if self.bidirectional else 1
    return (torch.randn(self.num_layers * n_lstm, 1, self.hidden_dim // 2, requires_grad = True),
            torch.randn(self.num_layers * n_lstm, 1, self.hidden_dim // 2, requires_grad = True))


  def forward(self, in_seqs): 
    sent_embeds = self.word_embeds(in_seqs).unsqueeze(1)

    conv_layer = self.tanh(self.conv1(sent_embeds).squeeze())

    pooling_layer = self.maxpool(conv_layer).squeeze()
    h1 = self.l1(pooling_layer)
    pdb.set_trace()
    h2 = self.l2(h1)
    probs = self.sigmoid(h2)


    '''
    self.hidden = self.init_hidden()

    embeds = self.word_embeds(sent).view(len(sent), 1, -1)
    ent_embeds = self.ent_embeds(ents).view(len(ents), -1)
    asp_embeds = self.asp_embeds(asps).view(len(asps), -1)

    if self.training:
      embeds = self.dropout(embeds)
   
    # n = len(sent), m = len(ents)
    lstm_out, self.hidden = self.lstm(embeds) 
    # m, n, hidden_dim
    lstm_out = lstm_out.squeeze().unsqueeze(0).repeat(len(ents), 1, 1)
    # m, n, 2 * ea_dim
    ent_asp_embeds = torch.cat((ent_embeds, asp_embeds), 1).unsqueeze(1).repeat(1, len(sent), 1)
    # m, n, hidden_dim + 2 * ea_dim
    h_eas_cat = torch.cat((lstm_out, ent_asp_embeds), 2)

    # m, n, mlph_dim
    scr_h = F.relu(self.mlp_intoh(h_eas_cat))
    # m, n, 1
    scrs = self.mlp_htoo(scr_h)
    atts = F.softmax(scrs, 1)
    
    # m, n, hidden_dim
    lstm_out_att = lstm_out * atts
    # m, hidden_dim
    lstm_out_att = lstm_out_att.sum(1)
 
    os = self.atttoo(lstm_out_att)
    prob_os = F.log_softmax(os, 1)
    return prob_os
    '''


def evalEmbed(emb_vocab, emb, train_data, dev_data, test_data, lang, lower_case, cuda, bs = 100, max_seq_len = 50, epoch = 20, report_every = 5):
  if cuda:
    torch.cuda.manual_seed(1234)
  word2idx, idx2word, \
  ent2idx, idx2ent,\
  asp2idx, idx2asp,\
  pol2idx, idx2pol,\
  ea2idx, idx2ea = get_vocab_entity_aspect_polarity(train_data, dev_data, test_data, lower_case)
  analyzer = absaAnalyzer(word2idx, idx2word, 
                          ent2idx, idx2ent, 
                          asp2idx, idx2asp,
                          pol2idx, idx2pol, 
                          ea2idx, idx2ea,
                          emb_vocab, emb, max_seq_len)
  criterion = nn.NLLLoss()
  optimizer = optim.Adam(analyzer.parameters()) 
  #optimizer = optim.SGD(analyzer.parameters(), lr = 0.01, momentum = 0.9) 
  if cuda:
    analyzer.cuda()

  ''' 
    [
      [
        Tensor([w1_idx, w2_idx, ... wn_idx, padding_idx, padding_idx, ...]),
        Tensor(slot1 target)
      ],
      ...
    ]
  '''
  train_seqs = prepSeqs(dev_data, word2idx, ent2idx, asp2idx, pol2idx, ea2idx, lower_case, max_seq_len, cuda)
  dev_seqs = prepSeqs(dev_data, word2idx, ent2idx, asp2idx, pol2idx, ea2idx, lower_case, max_seq_len, cuda)
  test_seqs = prepSeqs(dev_data, word2idx, ent2idx, asp2idx, pol2idx, ea2idx, lower_case, max_seq_len, cuda)

  best_acc_dev = .0
  corbest_acc_test = .0
  # total batch number
  n_batch = len(train_seqs) // bs if len(train_seqs) % bs == 0 else len(train_seqs) // bs + 1
  for i in range(epoch):
    # shuflle training data
    shuffle(train_seqs)
    for j in range(n_batch):
      batch_train = train_seqs[j * bs: (j + 1) * bs]
      total_loss_train, r_train = train(batch_train, analyzer, criterion, optimizer, cuda) 
      '''
      if j % report_every == 0:
        _, acc_dev = test(train_seqs, analyzer, criterion, cuda)
        _, acc_test = test(test_seqs, analyzer, criterion, cuda)
        print('current dev_acc = {:.3f}%, current test_acc = {:.3f}%'.format(acc_dev, acc_test))
        if acc_dev > best_acc_dev:
          best_acc_dev = acc_dev
          corbest_acc_test = acc_test 
          #torch.save(analyzer, 'best_analyzer_{}.model'.format(lang))
      print("epoch {}, batch {}/{}\nloss = {:.5f}  train_acc = {:.3f}%\nbest dev_acc = {:.3f}%  corresponding test_acc = {:.3f}%".format(i + 1, j + 1, n_batch, total_loss_train / len(batch_train), acc_train, best_acc_dev, corbest_acc_test))
      '''


def get_vocab_entity_aspect_polarity(train_data, dev_data, test_data, lower_case):
  vocab = [] # vocabulary
  ents = [] # entities
  asps = [] # aspects
  pols = [] # polarity labels
  eas = [] # entity#aspect tuples
  get_all_terms(train_data, vocab, ents, asps, pols, eas, lower_case, is_train_data = True)
  get_all_terms(dev_data, vocab, ents, asps, pols, eas, lower_case)
  get_all_terms(test_data, vocab, ents, asps, pols, eas, lower_case)

  # append UNK to the last index
  vocab.append('UNK')
  vocab = set(vocab)
  word2idx = dict(zip(vocab, range(len(vocab))))
  idx2word = dict(zip(range(len(vocab)), vocab))

  ents = set(ents)
  ent2idx = dict(zip(ents, range(len(ents))))
  idx2ent = dict(zip(range(len(ents)), ents))

  asps = set(asps)
  asp2idx = dict(zip(asps, range(len(asps))))
  idx2asp = dict(zip(range(len(asps)), asps))

  pols = set(pols)
  pol2idx = dict(zip(pols, range(len(pols))))
  idx2pol = dict(zip(range(len(pols)), pols))

  eas = set(eas)
  ea2idx = dict(zip(eas, range(len(eas))))
  idx2ea = dict(zip(range(len(eas)), eas))

  return word2idx, idx2word, ent2idx, idx2ent, asp2idx, idx2asp, pol2idx, idx2pol, ea2idx, idx2ea


def get_all_terms(input_data, vocab, ents, asps, pols, eas, lower_case, is_train_data = False):
  for sent, ops in input_data.items():
    # only add words in trainin data to vocab
    if is_train_data:
      # remove punctuations
      vocab.extend([word.lower() if lower_case else word for word in sent.strip().split(' ') if word not in string.punctuation])
    for op in ops:
      ea = op['category']
      cats = ea.split('#')
      eas.append(ea)
      ents.append(cats[0])
      asps.append(cats[1])
      pols.append(op['polarity']) 


def prepSeq(seq, to_idx, cuda, max_seq_len = -1):
  idxs = [to_idx[w] if w in to_idx else to_idx['UNK'] for w in seq] + [len(to_idx)] * (max_seq_len - len(seq))
  tensor = torch.LongTensor(idxs)
  return tensor.cuda() if cuda else tensor


def prepTarget(ops, ent2idx, asp2idx, pol2idx, ea2idx, cuda):
  target = []
  for op in ops:
    #ent, asp = op['category'].split('#')
    ea = op['category']
    target.append(ea)
  target = list(set(target))
  return prepSeq(target, ea2idx, cuda)
  '''
    ents.append(cats[0])
    asps.append(cats[1])
    lbs.append(op['polarity'])
  ent_seq = prepSeq(ents, ent2idx, cuda)
  asp_seq = prepSeq(asps, asp2idx, cuda)
  lb_seq = prepSeq(lbs, lb2idx, cuda)
  '''


def prepSeqs(data, word2idx, ent2idx, asp2idx, pol2idx, ea2idx, lower_case, max_seq_len, cuda):
  new_data = []
  for sent, ops in data.items():
    tokens = [w.lower() if lower_case else w for w in sent.strip().split(' ') if w not in string.punctuation]
    input_seq = prepSeq(tokens, word2idx, cuda, max_seq_len)
    target_seq = prepTarget(ops, ent2idx, asp2idx, pol2idx, ea2idx, cuda)
    new_data.append([input_seq, target_seq])
  return new_data


def train(train_seqs, analyzer, criterion, optimizer, cuda):
  # clear grad
  optimizer.zero_grad()
  # training mode
  analyzer.train()
  # eval metrics
  pre = 0
  rec = 0
  tot = 0
  in_seqs, target_seqs = zip(*train_seqs)
  in_seqs = torch.stack(in_seqs)
  probs = analyzer(in_seqs)
  '''
  # number of instances
  n_inst = 0
  for train_seq, ent_seq, asp_seq, lb_seq in train_seqs:
    probs = analyzer(train_seq, ent_seq, asp_seq)
    loss = criterion(probs, lb_seq)
    acc += (probs.max(1)[1] == lb_seq).data[0]
    total_loss += loss.data
    n_inst += len(lb_seq) 
    loss.backward()
  acc = acc * 100.0 / n_inst 
  total_loss = total_loss[0]
  # update minibatch
  optimizer.step()
  return total_loss, acc
  '''


def test(seqs, analyzer, criterion, cuda):
  analyzer.eval()
  total_loss = torch.cuda.FloatTensor([0]) if cuda else torch.FloatTensor([0])
  acc = 0
  n_inst = 0
  for seq, ent_seq, asp_seq, lb_seq in seqs:
    probs = analyzer(seq, ent_seq, asp_seq)
    loss = criterion(probs, lb_seq)
    acc += (probs.max(1)[1] == lb_seq).data[0]
    total_loss += loss.data
    n_inst += len(lb_seq) 
  acc = acc * 100.0 / n_inst 
  total_loss = total_loss[0]
  return total_loss, acc
  
