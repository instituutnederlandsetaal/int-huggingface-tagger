from transformers import AutoTokenizer, T5ForConditionalGeneration
import re
import sys
import lexicon_databases
from typing import Callable, Mapping, List
import lemmatizer

lemmatizer = lemmatizer.Lemmatizer(sys.argv[2],sys.argv[3])
lemmatizer.init()

def get_lines(f):
   with open(f,'r') as file:
     return file.readlines()

lines= map(lambda x: re.sub('\t\t','\t', x), get_lines(sys.argv[1]))

tokens = []

for l in lines:
  if re.match('.*\S\t.*\t.*', l):
    [word,pos,tok] = re.split('\t',l)
    pos_1  = re.sub(';.*', '', pos)
    c = { 'sentence_bound' : False , 'word': word, 'pos' : pos_1, 'pos_head' : re.sub('[(;].*', '', pos), 'tok' : tok }
    c['word_pos'] = word + ' ' + c['pos_head']
  else:
    c = { 'sentence_bound' : True }
  tokens.append(c)

verbose=True

for token in tokens:

   if token['sentence_bound']:
       print("\n")
       continue

   lemmatizer.lemmatize_token(token)

   if verbose:
      print(f"{token['word']}\t{token['pos']}\t{token['lemma']}\t{token['lexiconMatch']}\t{token['info']}")
   else:
      print(f"{token['word']}\t{token['pos']}\t{token['lemma']}")
   sys.stdout.flush()

