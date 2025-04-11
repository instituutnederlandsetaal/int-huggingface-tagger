import sys
import re
import json
import gzip

from datasets import Dataset, DatasetDict
from typing import List

def default_field_dict() :
  return  {'tokens' : 0, 'tags' : 1, 'lemmata': 2}

def to_dict(s,k, fields: dict):
  l = len(s)

  d = {'id': str(k)}

  for f in fields:
    i = fields[f]
    values = [w[i] if len(w) > i else '_' for w in s]
    d[f] = values

  # d  = {'id': str(k), 'tokens' : words, 'tags' : pos, 'lemmata' : lemmata}
  return d

def open_maybe_unzip(filename):
  if filename.endswith(".gz"):
      return gzip.open(filename,"rt")
  else:
      return open(filename,'r')

def read_sentences_to_list_of_dicts(filename: str, fields: dict, max_chunk_size=None) -> List[dict]:
  f = open_maybe_unzip(filename)
  sentence = []
  n_sentences = 0
  n_tokens = 0
  sentences = []
  keys = list(fields.keys())

  def flush_sentence():
    nonlocal sentence
    nonlocal n_sentences
    nonlocal n_tokens

    if (len(sentence) > 0):
        n_sentences += 1
        d = to_dict(sentence,n_sentences,fields)
        n_tokens = 0
        # print(f"output sentence of length {len(sentence)}, dict={d}")
        sentences.append(d)
    sentence = [];

  for l in f.readlines():
       l = l.strip()
       if fields is None: # fields should be in first line if nog given
          fields = {}
          cols = re.split("\t", l)
          for (i,f) in enumerate(cols):
            fields[f] = i
          continue
       if (l == '' or l.startswith('#')):
         flush_sentence()
         continue

       if (max_chunk_size is not None and n_tokens >= max_chunk_size):
         flush_sentence()

       columns = re.split("\t",l)
       #if (len(columns) < len(keys)):
       #    print("Fields missing in " + 
       # str(len(columns)) + ':' + str(columns), file=sys.stderr)
       sentence.append(columns)
       n_tokens += 1

  if (len(sentence) > 0):
     flush_sentence()

  return sentences

def create_dataset(filename, fields: dict, max_chunk_size=None):
  files = filename
  if isinstance(filename, str):
    files =  [filename]
  all_items = []
  for f in files:
    sentences = read_sentences_to_list_of_dicts(f, fields,max_chunk_size=max_chunk_size)
    all_items += sentences
  d1 = {}
  for f in fields:
    items = list(map(lambda x: x[f], all_items))
    d1[f] = items
  dataset = Dataset.from_dict(d1)
  return dataset

def load_datasets_tsv(train, test, fields: dict,max_chunk_size=None) -> Dataset:
    train = create_dataset(train, fields,max_chunk_size)
    test = create_dataset(test, fields,max_chunk_size)
    d = DatasetDict({
        'train' : train,
        'test' :  test})

    #print(d)
    return d

if __name__=='__main__':
    files = ['/media/projecten/Corpora/TrainingDataForTools/galahad-corpus-data/public-corpora/clariah-evaluation-corpora/gtbcit_18/test_train//gtbcit_18.train.tsv.gz', '/media/projecten/Corpora/TrainingDataForTools/galahad-corpus-data/public-corpora/clariah-evaluation-corpora/gtbcit_18/test_train//gtbcit_18.dev.tsv.gz']
    fields = {'tokens' : 0, 'tags' : 1, 'lemmata': 2}
    d = create_dataset(files, fields)
    print(d)


'''
form    lemma   POS     morph
CORNEILLEP_SUIVANTE_TRAIN       …       PONfrt  MORPH=empty
Ami     ami     NOMcom  NOMB.=s|GENRE=m
,       ,       PONfbl  MORPH=empty
j'      je      PROper  PERS.=1|NOMB.=s|GENRE=x|CAS=n
ai      avoir   VERcjg  MODE=ind|TEMPS=pst|PERS.=1|NOMB.=s
beau    beau    ADVgen  MORPH=empty
rêver   rêver   VERinf  MORPH=empty
,       ,       PONfbl  MORPH=empty
toute   tout    DETind  NOMB.=s|GENRE=f
ma      mon     DETpos  PERS.=1|NOMB.=s|GENRE=f
rêverie rêverie NOMcom  NOMB.=s|GENRE=f
Ne      ne      ADVneg  MORPH=empty
me      je      PROper  PERS.=1|NOMB.=s|GENRE=x|CAS=r
fait    faire   VERcjg  MODE=ind|TEMPS=pst|PERS.=3|NOMB.=s
rien    rien    PROind  NOMB.=s|GENRE=x
comprendre      comprendre      VERinf  MORPH=empty
en      en      PRE     MORPH=empty
ta      ton     DETpos  PERS.=2|NOMB.=s|GENRE=f
galanterie      galanterie      NOMcom  NOMB.=s|GENRE=f
.       .       PONfrt  MORPH=empty
'''


