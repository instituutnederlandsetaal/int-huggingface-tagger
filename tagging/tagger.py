from transformers import AutoTokenizer, AutoModelForTokenClassification
from tokenizers import pre_tokenizers
from transformers import pipeline
from transformers.pipelines import PIPELINE_REGISTRY
from transformers.pipelines.token_classification import TokenClassificationPipeline, AggregationStrategy
import regex as re
import sys
import os
from typing import Callable, Mapping, List
from functools import lru_cache
import nltk
#nltk.download('punkt')

from nltk.tokenize import sent_tokenize, word_tokenize
from dataclasses import dataclass
try:
  from token_classification_with_alternatives import register, TokenClassificationWithAlternativesPipeline
  from pie_tokenizer import pie_tokenizer, pie_tokenize_sentence
  import tdn_tagset
  from tdn_tagset import *
except ImportError:
  from .token_classification_with_alternatives import register, TokenClassificationWithAlternativesPipeline
  from .pie_tokenizer import pie_tokenizer, pie_tokenize_sentence
  from .tdn_tagset import *

register()

# cf https://huggingface.co/docs/transformers/add_new_pipeline

def simpleTokenize(text):
    s0 = re.sub('-', '下線', text)
    s1 = re.sub(r"(\p{P}+)", r" \1 ", s0)
    s2 = re.split(r'\s+', s1)
    s3 = " ".join(s2)
    s4 = re.sub('下線', '-', s3)
    return s4

def simpleTokenizeToListOfDicts(text) -> List[dict]:
    s0 = list(pie_tokenize_sentence(text)) # re.split(r'\s+', simpleTokenize(text))
    s1 = map(lambda x: {'token' : x }, s0)
    return list(s1)

def token_splitter():
    #return "切"
    return " "

def ja_sentsplit(text: str) -> List[str]:
    return re.sub("。", "。_x_x_x_y", text).split('_x_x_x_y')

# arbitrary splitting of long sentence
def chunk_sentence(sentence, max_sentence_length, check_integrity=True):
    tokens = re.split("\\s+", sentence)  # word_tokenize(sentence) # hier gaat het dus mis ....
    chunks=[]
    chunk=[]
    n_tokens = 0

    def flush_chunk():
      nonlocal n_tokens
      nonlocal chunk
      if (len(chunk) > 0):
        chunks.append(chunk)
      n_tokens = 0;
      chunk=[]

    for t in tokens:
       if (n_tokens >= max_sentence_length):
         flush_chunk()
       chunk.append(t)
       n_tokens += 1

    flush_chunk()
    res = list(map(lambda x: " ".join(x), chunks))
    if check_integrity:
        reconstructed = " ".join(res)
        s1 = re.sub("\s+", "", sentence)
        s2 = re.sub("\s+", "", reconstructed)
        if s1 != s2:
          print(f"Ramp S: {s1} Chunks: {s2}")
    return res

# use nltk for sentence splitting
def sentsplit(text, language='dutch', max_sentence_length=50, check_integrity=False): # arbitrary split of long sentences is ugly
    if language == "japanese":
      return ja_sentsplit(text)

    sentences =  sent_tokenize(text, language)
    
    if check_integrity:
      print(f"Splitting: #####\n{text}\n#####")
      
      reconstructed = " ".join(sentences)
      s1 = re.sub("\s+", "", text)
      s2 = re.sub("\s+", "", reconstructed)
      if s1 != s2:
         print(f"Botched!!!!!\n{text}\n{sentences}\n{s1}\n{s2}")
      for s in sentences:
        print(f"Sentence: {s}")
        if re.match('.*["`”].*', s):
            print("Pas op, na split: " + s)

    if (max_sentence_length is not None):
      subsentences = []
      for s in sentences:
         parts = chunk_sentence(s,  max_sentence_length)
         if len(parts) > 1:
           subsentences += parts
         else:
           subsentences.append(s)
      if check_integrity:
        reconstructed = " ".join(subsentences)
        s1 = re.sub("\s+", "", text)
        s2 = re.sub("\s+", "", reconstructed)
        if s1 != s2:
           print(f"Botched!!!!!\n{text}\n{sentences}\n{s1}\n{s2}")
           exit
        for su in subsentences:
          print(f"subsentence: {su}")
      return subsentences
    else:
      return sentences;

def read_mapping(filename):
   m = {}
   f = open(filename,"r")
   for line in f.readlines():
     [tag,mapped] = re.split("\t",line)
     m[tag] = re.sub("\\s+", "", mapped)
   return m

@dataclass
class TaggingModel:
    name : str = None
    model : AutoModelForTokenClassification = None ## type?
    tokenizer : AutoTokenizer = None ## type?
    other_tokenizer :  AutoTokenizer = None

    is_first : Callable[[str,str], bool] = lambda token, tag: not token.startswith('##') ## type ? # Callable[[str,str], bool]
    language : str = 'dutch'
    return_alternatives: bool = False
    ####
    the_processor = None
    the_pretokenized_processor = None
    tag_mapping = None
    pipeline_tag=  "token-classification" # "token-classification-with-alternatives"
    n_sents = 0
    @property
    def processor(self):
        if self.return_alternatives:
          self.pipeline_tag = "token-classification-with-alternatives"
          #print(f"Use customized pipeline {self.pipeline_tag}")
        if self.the_processor is None:
          self.the_processor = pipeline(self.pipeline_tag, model=self.model, tokenizer=self.tokenizer)
        return self.the_processor

    def pretokenized_processor(self):
        if self.return_alternatives:
          self.pipeline_tag = "token-classification-with-alternatives"
          #print(f"Use customized pipeline {self.pipeline_tag}")
        if self.the_pretokenized_processor is None:
            self.the_pretokenized_processor = pipeline(self.pipeline_tag, model=self.model, tokenizer=self.other_tokenizer)
        return self.the_pretokenized_processor

    def tag_sentence(self, text : str) -> List[List[str]]:
        tokens =  simpleTokenizeToListOfDicts(text)
        tokens_tagged = self.tag_tokenized_sentence(tokens)
        def p(t):
          x = t['pos'] if 'pos' in t else ''
          return x
        r = list(map(lambda t : [t['token'], p(t), 'subtokens_not_available'], tokens_tagged))
        return r
    
    # deze is afgeschaft omdat we transformers niet zover konden krijgen niet woorden te splitsen op '-'

    def tag_sentence_old(self, text : str) -> List[List[str]]:

        toksx = simpleTokenize(text)

        print(f"Pretokenized: {toksx}")
        if self.n_sents % 100 == 0:
          print(f"[{self.n_sents}] {text}",file=sys.stderr)

        self.n_sents += 1
        tokens = (self.other_tokenizer)(toksx).tokens() # self.tokenizer(text).tokens()
        print("Untokenized: " + text + " ,Tokens: " + str(tokens))
        try:
          results = (self.pretokenized_processor())(toksx) # self.processor(text)
        except:
          print(f"!!!could not process text: {text}")

        z = list(map(lambda x: 'O', tokens))
        literals = list(map(lambda x: None, tokens))

        for r in results:
            literal = toksx[r['start']:r['end']]
            literals[r['index']] = literal
            z[r['index']] = self.mapped_tag(r['entity'])

        for i in range(0,len(literals)):
            if literals[i] is None:
                #print(f"None literal at {i}: {tokens[i]}")
                literals[i] = re.sub('##|▁|<s>|</s>', '', tokens[i]) # hier gaat het dus fout

        check_integrity = False

        if check_integrity:
          s1 = re.sub("\[(CLS|SEP)\]", "", re.sub("\s+", "", " ".join(literals)))
          s2 = re.sub("\s+", "", text)

          if s1 != s2:
             print(f"Sentence plain text changed:\n{text}\n{s2}\n{s1}")
           
        res = self.regroup(list(zip(tokens, z, literals)))
        if check_integrity and '"' in text:
           print(f"In: {text}, Results: {res} ")

        return res

    def tag_text_to_dicts(self, text: str)-> List[List[dict]]:
       tagged = self.tag_text(text, False)
       return self.sentencesToDict(tagged)

    # zie https://huggingface.co/course/chapter6/8?fw=pt 
    # dit is geen goede oplossing, de enige manier is er voor te zorgen dat de tokenizering van het basismodel ook goed is

    
    def choose_compatible_alternative(self,token, r):
        # return tdn_choose_compatible_alternative(token,r)
        alternatives = r['alternatives']
        best_pos = self.mapped_tag(r['entity'])
        if 'ref_pos' in token:
          pos_old = token['ref_pos']
          compatible_options = list(filter(lambda x: main_pos_compatible(x[0],pos_old), alternatives))
          if len(compatible_options) > 0:
             best_compatible = compatible_options[0][0] 
             
             #if (best_compatible != best_pos):
             #   print(f"{token['token']}/{pos_old}: {best_pos} -- {best_compatible}")
             # adapted_pos = transfer_features(pos_old,best_compatible)
             return best_compatible
             #return f"{pos_old}\t{best_compatible}"
        return best_pos

    def tag_tokenized_sentence(self, sentence: List[dict], choose_compatible: bool=False) -> List[dict]:
         #self.model.resize_token_embeddings(len(self.other_tokenizer))

         try: 
           toks = list(map(lambda x: x['token'], sentence))
  
           toksx = token_splitter().join(toks)
           starts_at = {}
           p = 0
           for t in sentence:
              starts_at[p] = t
              p += len(t['token']) + 1
  
           tokens = self.other_tokenizer(toksx).tokens()
    
           results = (self.pretokenized_processor())(toksx)
         
           # dit stukje werkt niet voor FremyCompany/roberta-large-nl-oscar23

           for r in results:
              start = r['start']
              r['starts_word'] = start in starts_at
              if start in starts_at:
                 entity = self.choose_compatible_alternative(starts_at[start], r) if choose_compatible else self.mapped_tag(r['entity'])
                 starts_at[start]['pos'] = entity # self.mapped_tag(r['entity'])
                 starts_at[start]['unmapped_pos'] = r['entity']
          
           z = " ".join(map(lambda t: t['token'] + '/' + t['pos'] if ('pos' in t) else 'Nope', sentence))
         except Exception as e:
            print(f"Exception {e} processing sentence of length {len(sentence)}, too bad! {sentence}")
            raise e
         return sentence

    def mapped_tag(self,tag):
        if self.tag_mapping is not None and not tag in self.tag_mapping:
            raise Exception(f"Fatal: No mapping for {tag}!!!!")
        if self.tag_mapping is not None and tag in self.tag_mapping:
           return self.tag_mapping[tag]
        else:
           return tag
 
    def flatten(self, t: List[List[List[str]]]) -> List[List[str]]:
        return [item for sublist in t for item in sublist]

    def tag_sentences(self, text : List[str], flatten: bool=True) -> List[List[str]]:
        print("Start tagging sentences",file=sys.stderr)
        tagged_sentences: List[List[List[str]]] = list(map(self.tag_sentence, text))
        if flatten:
          return self.flatten(tagged_sentences)
        else:
          return tagged_sentences

    def tag_text(self, text: str, flatten: bool=True) -> List[List[str]]:
        return self.tag_sentences(sentsplit(text,self.language), flatten)

    def join_subwords(self, sublst,show_subword_tags=False):
        token = ''.join(list(map(lambda x: x[0], sublst)))
        tokens = '|'.join(list(map(lambda x: x[0], sublst)))
        tags = ';'.join(list(map(lambda x: x[1], sublst))) 
        tag = tags if show_subword_tags else re.sub(";.*", "", tags)
        return (token, tag, tokens)

    # raap de door Bert uit elkaar gesloopte woorden weer bij elkaar
    def regroup(self, lst):
        sublists = []
        current_sublist = []
        #sublists.append(current_sublist)
        for (token, tag, literal) in lst:
            # print(f"token={token}, tag={tag}, literal={literal}")
            if (self.is_first(token, tag)):
                new_sub = [(literal, tag)]
                sublists.append(new_sub)
                current_sublist = new_sub
            else:
                current_sublist.append((literal, tag))
        #print(str(sublists))
        return list(map(lambda x : self.join_subwords(x), sublists))

    def sentenceToDict(self,sentence) -> List[dict]:
       tokenDicts = []
       for tt in sentence:
           #print(f"tt={tt} {sentence}")
           token, tag, tokens = tt
           if not token in set(["[SEP]","[CLS]"]):
              tokenDicts.append({'token' : token, 'pos': tag })
       return tokenDicts

    def sentencesToDict(self, sentences):
       #print(f"sentencesToDict: sentences={list(sentences)}")
       asDicts = list(map(lambda s : self.sentenceToDict(s), sentences))
       return asDicts

def model(name, is_first = lambda token, tag: not token.startswith('##'), language='dutch', return_alternatives=False):
    m = AutoModelForTokenClassification.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name)
    other_tokenizer = AutoTokenizer.from_pretrained(name)
    other_tokenizer.pretokenizer = pre_tokenizers.CharDelimiterSplit(token_splitter())
    #print(other_tokenizer.pretokenizer)
    tm = TaggingModel(name, m, tokenizer, other_tokenizer, is_first=is_first, language=language,return_alternatives=return_alternatives)
    print(f"Model {name} created, alternatives={return_alternatives}", file=sys.stderr)
    return tm

def printSentences(fileName, sentences):
    with open(fileName, 'w') as f:
        for ns in sentences:
            # f.write("\n ##### \n")
            for tt in ns:

                token, tag, tokens = tt
                if not token in set(["[SEP]","[CLS]"]):
                   f.write(token + "\t\t" + tag + "\t\t" + tokens + "\n")
        f.close()

is_first_ja = lambda token, tag: not (tag.startswith("I-") or tag == 'AUX' or (tag == 'SCONJ' and token=='て' ) or (tag == 'CCONJ' and token=='ば' ))
# models

LazyModel = Callable[[], TaggingModel]

fukai_kaeru: LazyModel = lambda : model("proycon/bert-pos-cased-deepfrog-nld")
sophie_ner: LazyModel = lambda : model("CLTL/gm-ner-xlmrbase", lambda token, tag : token.startswith('▁'))
bertje_pos: LazyModel =  lambda : model("wietsedv/bert-base-dutch-cased-finetuned-udlassy-pos")
deepfrog_lem: LazyModel  = lambda: model("proycon/bert-lemma-cased-cgn_elex-nld")
silly_mini: LazyModel  = lambda: model("../data/tagging/tagging_models/pos_tagging_model")
multatuli: LazyModel  = lambda: model("../data/tagging/tagging_models/pos_tagging_model_multatuli")
krantjes: LazyModel  = lambda: model("../data/tagging/tagging_models/pos_tagging_model_krantjes")
japanese_upos: LazyModel  = lambda : model("KoichiYasuoka/bert-base-japanese-upos", language='japanese', is_first = is_first_ja) # https://github.com/KoichiYasuoka/esupar
tdn_19_from_lassy =  lambda: model("../data/tagging/tagging_models/pos_tagging_model_19_from_bertje_udlassy_pos")
pos_tagging_model_19 = lambda: model("../data/tagging/tagging_models/pos_tagging_model_19")
bab_pos_tagging_model = lambda: model("../data/tagging/tagging_models/bab_pos_tagging_model")
bab_pos_tagging_model_bertje17 = lambda: model("../data/tagging/tagging_models/bab_pos_tagging_model_bertje17")

models = {
    'deepfrog' : fukai_kaeru,
    'sophie_ner' : sophie_ner,
    'bertje_pos' : bertje_pos,
    'deepfrog_lem' : deepfrog_lem,
    'japanese_upos' : japanese_upos,
    'silly_mini' : silly_mini,
    'multatuli' : multatuli,
    'krantjes' : krantjes,
    'tdn_19_from_lassy' : tdn_19_from_lassy,
    'pos_tagging_model_19' : pos_tagging_model_19,
    'bab_pos_tagging_model' : bab_pos_tagging_model,
    'bab_pos_tagging_model_bertje17' : bab_pos_tagging_model_bertje17
}

def find_model(model_name, tag_mapping: dict = None, return_alternatives=False):
    the_model = None
    if (model_name in models):
        the_model =  models[model_name]()
    else:
        if os.path.exists(model_name):
            print(f"Found {model_name} alt={return_alternatives} on filesystem" , file=sys.stderr)
            models[model_name]  = lambda: model(model_name, return_alternatives=return_alternatives)
            the_model =  models[model_name]()
    if tag_mapping is not None and the_model is not None:
        the_model.tag_mapping = tag_mapping
    return the_model

def slurp_file(fileName: str) -> str:
    with open(fileName, 'r') as f:
        return f.read()

if __name__ == '__main__':
    pass
