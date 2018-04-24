# -*- coding: UTF-8 -*-
#!/usr/bin/python3
"""
Configurations for the main scripts
@Author Yi Zhu
Upated 04/24/2018
"""

#************************************************************
# Imported Libraries
#************************************************************
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description = 'Evaluations for Word Embeddings')
    
    # Embeddings file paths
    parser.add_argument('--emb_path', nargs = '+', 
        default = ['/home/yz568/Documents/code/w2v_subword/code/de.mo.posadd.035.vec.txt',
                   '/home/yz568/Documents/code/w2v_subword/code/de.bpe.addpos.035.vec.txt'],
        help = 'Embedding file paths')

    # Languages
    parser.add_argument('--lang', '-l', nargs = '+',
        default = ['de', 'de'],
        help = 'Languages to evaluate (same length as embedding paths, can be repeated)')

    # Evaluation Task
    parser.add_argument('--task', nargs = '+',
        # current evaluations we have
        choices = ['word_similarity', 'pos_tagging', 'parsing'],
        default = ['word_similarity'],
        help = 'Task Type (same length as the evaluation data dir)')

    # Evaluation Data
    parser.add_argument('--evaldata_path', nargs = '+', 
        default = ['/mnt/hdd/yz568/data/word_similarity'],
        help = 'Data Path for Evaluation Data')
    
    # Others
    parser.add_argument('--lower_case', action = 'store_true', default = True)
    parser.add_argument('--cuda', action = 'store_true', default = True)

    args = parser.parse_args()
    assert(len(args.lang) == len(args.emb_path) and len(args.task) == len(args.evaldata_path))
    return args
