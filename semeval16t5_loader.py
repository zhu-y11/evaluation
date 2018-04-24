# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Loading data for SemEval 16 t5
@Author Yi Zhu
Upated  23/01/2018
"""

#************************************************************
# Imported Libraries
#************************************************************
import os
from lang_map import lang_map
from nltk.tokenize.moses import MosesTokenizer
import xml.etree.ElementTree as ET


class AutoVivification(dict):
  """
    Implementation of perl's autovivification feature.
  """
  def __getitem__(self, item):
    try:
      return dict.__getitem__(self, item)
    except KeyError:
      value = self[item] = type(self)()
      return value


def loadData(dir_path):
  data = AutoVivification()
  for root, dirs, files in os.walk(dir_path):
    for file_name in files:
      if not file_name.endswith('.xml'):
        continue
      suffix = file_name.find('.xml')
      lang = lang_map[file_name[:suffix].split('_')[2].lower()]
      data_type = os.path.basename(root)
      domain = os.path.basename(os.path.dirname(root))
      task = os.path.basename(os.path.dirname(os.path.dirname(root)))
      
      tokenizer = MosesTokenizer(lang = lang) 

      tree = ET.parse(os.path.join(root, file_name))
      revs = tree.getroot()
      for rev in revs:
        for sents in rev:
          for sent in sents:
            text = None
            ops = []
            for c in sent:
              if c.tag == 'text':
                text = tokenizer.tokenize(c.text, escape = False, return_str = True)
              elif c.tag == 'Opinions':
                for op in c:
                  ops.append(op.attrib)
            if not ops:
              continue
            data[data_type][lang][task][domain][text] = ops
  return data['train'], data['dev'], data['test']
