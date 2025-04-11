from transformers import AutoTokenizer, T5ForConditionalGeneration
import re
import sys
import torch
from typing import Callable, Mapping, List
from dataclasses import dataclass

try:
  from lexicon_databases import Lexicon
  from token_properties import postprocess_tagged_token, connect_linked_tokens
except:
  from .lexicon_databases import Lexicon
  from .token_properties import postprocess_tagged_token, connect_linked_tokens

@dataclass

class Lemmatizer:
   model_name : str = None
   tokenizer_name : str = None
   lexicon_path: str = None
   tag_mapping_path: str = None
   chop_pos_to_main: bool = True
   ##
   lexicon: Lexicon = None
   tokenizer =  None
   model =  None
   device = None
   use_cuda = None

   def init(self):
     if self.tokenizer_name is None:
       self.tokenizer_name = self.model_name
     self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
     self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
     self.lexicon = Lexicon(self.lexicon_path, self.tag_mapping_path)
     self.lexicon.init()
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     use_cuda = True if torch.cuda.is_available() else False

   cache = { }

   def lemmatize_byt5(self, token: dict):
    
    wp = token['word_pos']
    lemma = None
    if wp in self.cache:
        lemma = self.cache[wp]
    else:
        model_inputs = self.tokenizer(wp, max_length=128, truncation=True, return_tensors="pt")
        if self.use_cuda:
          model_inputs = {key: value.to(self.device) for key, value in model_inputs.items()}
        else:
          pass  
        outputs = self.model.generate(**model_inputs, max_length=1000)
        lemma = re.sub("<pad>|</s>", "", self.tokenizer.decode(outputs[0]))
        self.cache[wp] = lemma
    #print("byt5 lemma:" + lemma + " token: " + str(token),file=sys.stderr)
    return lemma

   def connect_joined_tokens(sentence: List[dict]):
      pass
   
   def lemmatize_tokenized_sentence(self, sentence: List[dict]):
      connect_linked_tokens(sentence) # werkt totaal niet voor telwoorden etc...
      for t in sentence:
         # print(f"Token: {t}")
         self.lemmatize_token(t)
      return sentence

   def prepare_token(self, token: dict):
      wordform = token['grouped_wordform'] if 'grouped_wordform' in token else token['word']
      if 'token' in token:
         token['word'] = token['token']
      if 'pos' in token:
        token['pos_head'] = re.sub("\(.*", "", self.lexicon.map_pos(token['pos']))
        if self.chop_pos_to_main:
          token['word_pos'] = wordform + ' ' + token['pos_head']
        else:
          token['word_pos'] = wordform + ' ' + token['pos']
      else: 
        token['word_pos'] = wordform
        token['pos_head'] = 'hadjememaar'
        token['pos'] = 'hadjememaar'

   def special_cases(self, token):
       if (token['pos_head'] == "NUM") and re.match('.*[0-9].*', token["word"]): # TODO regels voor telwoorden implementeren!
          token['lemma'] = token['word']
          #print(str(token),file=sys.stderr)
          return True
       if (token['pos_head'] == "PD") and re.match('[Mm][Ee][Nn]', token["word"]):
          token['lemma'] = "men"
          return True
       return False

   def lemmatize_token(self, token: dict):

       self.prepare_token(token)
       wordform = token['grouped_wordform'] if 'grouped_wordform' in token else token['word']

       p = token['pos']
       if re.match(".*VRB.*_.A.*", p): # vreselijk tijdelijk hackje ....
          p = 'ADV'
       (compatible_lexical_items, incompatible_lexical_items, info) = self.lexicon.lookup_compatible(wordform, p)
 
       token['info'] = info

       if ('sentence_bound' in token) and token['sentence_bound']:
         return
  
       lemma = None
       token['lemma'] = None

       token['lexiconMatch'] = 'None'
  
       if (len(compatible_lexical_items) > 0):
         lemma = compatible_lexical_items[0]['lemma']
         token['lexicon'] = compatible_lexical_items[0]['lexicon']
         token['info'] = info
       elif self.special_cases(token):
         token['lexicon'] = 'special_cases'
         token['info'] = 'special_cases'
         lemma = token['lemma']
         #print(str(token),file=sys.stderr)
       else:
         if (token['pos'] != 'LET' and token['pos'] != 'PC'):
           lemma = self.lemmatize_byt5(token)
           token['info'] = 'byt5'
           token['lexicon'] = 'byt5' 
       if (token['pos'] != 'LET'):
         token['lemma'] = lemma
  
def lemmatizer_from_config(config: dict):
    if 'lem_tokenizer' not in config:
       config['lem_tokenizer'] = config['lem_model']

    tag_mapping_path = config['tag_mapping_path'] if 'tag_mapping_path' in config else None
    chop_pos_to_main = config['chop_pos_to_main'] if 'chop_pos_to_main' in config else True

    lemmatizer = Lemmatizer(config['lem_model'], config['lem_tokenizer'], config['lexicon_path'], tag_mapping_path, chop_pos_to_main=chop_pos_to_main)
    lemmatizer.init()
    return lemmatizer

