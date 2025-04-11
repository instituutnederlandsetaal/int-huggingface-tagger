import os
import sys
import re

from dataclasses import dataclass
training_lexicon = "/media/projecten/Corpora/TrainingDataForTools/CobaltExport/2024_2/Lemmatizer/lem.all.tsv"

# items.append({'lemma': lemma, 'lemma_gigpos': lemma_gigpos, 'lemma_id': lemma_id, 'wordform_gigpos' : row['wordform_gigpos']}) 

@dataclass
class LexiconFromTrainingData:
    fn: str = training_lexicon

    index: dict = None

    def createIndex(self, fn: str=training_lexicon):
      print(f"creating training data lexicon from {training_lexicon}",file=sys.stderr)
      f = open(fn,"r")
      index = {}
      for l in f.readlines():
        cols = re.split("\t", l.strip())
        [word,pos,lemma,freq] = cols[0:4]
        item = {'lemma': lemma, 'wordform_gigpos' : pos, 'lemma_gigpos' : pos, 'lemma_id' : 'blabla', 'frequency': int(freq)}
        if word in index:
          index[word].append(item)
        else:
          index[word] = [item]
      self.index = index
      #print(str(index))
      return index
