#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, getopt, collections
import string
import io, os
import re
import numpy as np
import argparse
import nltk
import yaml
from nltk.translate.bleu_score import SmoothingFunction


with open('configs/pythia_config.yml', 'r') as stream:
    args = yaml.safe_load(stream)['val']

def get_acc():
  # init some stuff
  count = 0
  total_count = 0
  candidates = []

  # read model output
  with open(args['output'], 'r') as outf:
    candidates = outf.read().splitlines()
    # for line in outf:
      # candidates.append(line.strip())

  # get output groupings if n > 1 and check it is correct
  total_num = len(candidates)
  candidates = [x.split('[seperator]')[1] for x in candidates]
  candidates = [candidates[i: i+args['n']] for i in range(0, len(candidates), args['n'])]
  assert len(candidates) == total_num / args['n']

  # read ground truth references and compare
  with io.open(args['ref'], 'r', encoding='utf8') as reff:
    for out, ref in zip(candidates, reff):
      if ref.strip() in out:
        count += 1
      total_count += 1
  acc = count * 1.0 / total_count
  print("Acc = %.3f" % acc)


def get_bleu():
    # init some stuff
    candidates = []
    refs = []

    # read model output
    with io.open(args['output'], 'r', encoding='utf8') as outf:
        for line in outf:
            ans = line.split('[seperator]')[1]
            candidates.append(ans.split())
    cand_length = len(candidates)

    # read ground truth reference
    with io.open(args['ref'], 'r', encoding='utf8') as reff:
        for line in reff:
            refs.append([line.split()])

    # only check first N numbers, this will change based on what was specified in eval.py
    refs = refs[0:5000]
    assert cand_length == len(refs)

    # calculate BLEU
    score = 0
    cc = SmoothingFunction() # add this to sentence_bleu call if n-gram count is too low
    for c, r in zip(candidates, refs):
        temp = nltk.translate.bleu_score.sentence_bleu(r, c)
        score += temp
    avg = score/cand_length 
    print("BLEU = %.3f" % avg)


if __name__ == "__main__":
  get_acc()
  if args['n'] == 1:
      get_bleu()
  else:
      print("N > 1 so did not calculate BLEU")
