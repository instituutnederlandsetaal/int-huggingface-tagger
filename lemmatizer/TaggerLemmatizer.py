from transformers import AutoTokenizer, T5ForConditionalGeneration
import re
import sys
from typing import Callable, Mapping, List
import os
import io
from dataclasses import dataclass
import json

from .lemmatizer import Lemmatizer

sys.path.append('../tagging')

import tagging.tei as tei
import tagging.tagger as tagger
import tagging.readFromTSV as tsv

@dataclass

class Processor:
  config_path: str = None

  def init(self):
    pass

  def process(self,input_filename,output_filename):
    raise NotImplementedError

@dataclass

class TaggerLemmatizer(Processor):
   config_path: str = None

   model_name : str = None
   tokenizer_name : str = None
   lemmatizer = None
   tagger = None
   verbose : bool = False

   choose_compatible: bool = False
   just_tag: bool = False

   def lemmatize_sentence_action(self):
     return lambda s : self.lemmatizer.lemmatize_tokenized_sentence(s)

   def tag_and_lemmatize_sentence_action(self):
     def process(s):
         if self.just_tag:
           self.tagger.tag_tokenized_sentence(s,choose_compatible=self.choose_compatible)
         else:
           self.lemmatizer.lemmatize_tokenized_sentence(self.tagger.tag_tokenized_sentence(s,choose_compatible=self.choose_compatible))
     return process

   #return lambda s: self.lemmatizer.lemmatize_tokenized_sentence(self.tagger.tag_tokenized_sentence(s))

   def init(self):
     if self.config_path is None:
       self.config_path = os.environ.get('TAGGER_CONFIG_PATH') 

     try:
       json_file =  open(self.config_path)
       config = json.load(json_file)
       if 'lem_tokenizer' not in config:
         config['lem_tokenizer'] = config['lem_model']
       tag_mapping_path = config['tag_mapping_path'] if 'tag_mapping_path' in config else None
       chop_pos_to_main = config['chop_pos_to_main'] if 'chop_pos_to_main' in config else True
       self.lemmatizer = Lemmatizer(config['lem_model'], config['lem_tokenizer'], config['lexicon_path'], tag_mapping_path, chop_pos_to_main=chop_pos_to_main)
       self.lemmatizer.init()

       self.choose_compatible = config['choose_compatible'] if 'choose_compatible' in config else False
       self.just_tag = config['just_tag'] if 'just_tag' in config else False
       tagging_model_name = config['tagging_model']
       self.tagger = tagger.find_model(tagging_model_name,return_alternatives=self.choose_compatible)
     except Exception as exception:
        print("Oeeeeps!")
        print(str(exception))
        raise exception 

   def tagPlainText(self, text, output):
     pos_results = self.tagger.tag_text_to_dicts(text)
     #print(f"pos_results={pos_results}")
     a = self.lemmatize_sentence_action()
     lemmatized = list(map(a,pos_results))
     #print(f"lemmatized={lemmatized}")
     self.printSentences(lemmatized, output)

   def process(self, input_filename, output_filename):
       with io.open(input_filename, mode="r", encoding="utf-8") as f: # encoding="utf-8"
          contents = f.read() # //.decode('utf-8')
          self.tagPlainText(contents, output_filename)

   def printSentences(self,sentences, output):
     f = open(output, 'w+')
     f.write("token\tpos\tlemma\n")
     for s in sentences:
       for token in s:
         if self.verbose:
           f.write(f"{token['word']}\t{token['pos']}\t{token['lemma']}\t{token['lexiconMatch']}\t{token['info']}\n")
         else:
           f.write(f"{token['word']}\t{token['pos']}\t{token['lemma']}\n")
       f.write("\n")
     f.flush()
     f.close()

   def tagTSV(self, input_path, output_path, fields: dict = None): # assumes that 'word' is the first property
     sentences = tsv.read_sentences_to_list_of_dicts(input_path)
     for sentence in sentences:
       self.tag_and_lemmatize_sentence_action()(sentence)
       print(sentence)
     pass

   def tagTEI(self, input_path, output_path):
     if os.path.isfile(input_path):
       tei.doWithTEISentences(input_path,output_path,self.tag_and_lemmatize_sentence_action(),also_keep_old=self.choose_compatible)
     elif os.path.isdir(input_path):
      files = os.listdir(input_path)
      for f in files:
        print(f)
        sys.stdout.flush()
        try:
          tei.doWithTEISentences(input_path + "/" + f,output_path + "/" + f , self.tag_and_lemmatize_sentence_action(),also_keep_old=self.choose_compatible)
        except Exception as e:
          print(f"Exception {e} for file {f}", file=sys.stderr)

if __name__ == '__main__':
  # t = TaggerLemmatizer("combi.config")
  processor_class = globals()["TaggerLemmatizer"]
  t : Processor = processor_class("combi.config")
  t.init()

  if len(sys.argv) > 2:
    t.process(sys.argv[1], sys.argv[2])
  else:
    t.tagPlainText('Dit is een "" zinnetje. En dit ook hoor!', '/tmp/out.txt')
