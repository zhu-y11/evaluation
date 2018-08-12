# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
@Author Yi Zhu
Upated  06/04/2018 
delete embeddings that are not in the training/dev/test set
"""

#************************************************************
# Imported Libraries
#************************************************************
import argparse
import os
from tqdm import tqdm

def create_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--in_embeds", type = str, required = True,
      help = "word embedding files (txt)")
  parser.add_argument("--input_files", type = str, required = True,
      help = "input dir or files")
  parser.add_argument("--file_type", type = str, default = 'conllu',
      choices = ['conllu'],
      help = "file types to be precessed")
  parser.add_argument("--out_embeds", type = str, default = 'vectors.txt',
      help = "word embedding files")
  parser.add_argument("--case", action = 'store_true', default = False, 
      help = "case sensitive")

  args = parser.parse_args()
  return args


def get_words(args):
  words = []
  if os.path.isdir(args.input_files):
    get_words_from_dir(args, args.input_files, words)
  else:
    get_words_from_file(args, args.input_files, words)
  words = list(set(words))
  words = dict(zip(words, [1] * len(words)))
  return words


def get_words_from_dir(args, indir, words):
  for root, dirs, files in os.walk(indir):
    for infile in files:
      infile = os.path.join(root, infile)
      if infile.endswith('.conllu'):
        get_words_from_file(args, infile, words)


def get_words_from_file(args, infile, words):
  if args.file_type == 'conllu':
    read_conllu_file(args, infile, words)


def read_conllu_file(args, infile, words):
  # conllu format
  with open(infile, 'r') as fin:
    for line in fin:
      line = line.strip()
      if not line:
        continue
      word = line.split()[1]
      if not args.case:
        word = word.lower()
      words.append(word)


def crop(args, words):
  ct = 0
  dim = 0
  cropped_embeds = []
  with open(args.in_embeds, 'r') as fin:
    emb_ct, dim = fin.readline().strip().split()
    for line in tqdm(fin, total = int(emb_ct)):
      linevec = line.strip().split()
      # skip head
      if len(linevec) == 2 and int(linevec[1]) > 0:
        dim = int(linevec[1])
        continue
      word = linevec[0]
      if not args.case:
        word = word.lower()
      if word not in words:
        continue
      ct += 1
      cropped_embeds.append(line)
  with open(args.out_embeds, 'w') as fout:
    fout.write('{} {}\n'.format(ct, dim))
    fout.write(''.join(cropped_embeds).strip())



if __name__ == '__main__':
  args = create_args()
  words = get_words(args)
  crop(args, words)
