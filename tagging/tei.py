from lxml import etree as ET
import os
from glob import glob
import os.path
from os import path
import sys
import re
from dataclasses import dataclass
from collections import  namedtuple
from typing import Callable, Mapping, List
#TeiSentence = namedtuple('ParlaSentence', ['words', 'word_ids', 's', 'word_elements','lang', 'sentence_id'])

@dataclass
class TEISentence:
  words: []
  word_ids: [] 
  s: str
  word_elements: []
  lang: str
  sentence_id: str   

  def getSentence(self): 
    return ' '.join(self.words)
 
p5=True
ns_p5 = "http://www.tei-c.org/ns/1.0"
ns_tei=""
ns_xml=""
if p5:
# ns_tei="{http://www.tei-c.org/ns/1.0}"
 ns_xml="{http://www.w3.org/XML/1998/namespace}"

def getNamespace(x):
  return ET.QName(x.getroot()).namespace

def getElementNamespace(x):
  return ET.QName(x).namespace

maximum_sentence_length = 80 # 190 # 200

def findAllXmlIn(path):
  result = [y for x in os.walk(path) for y in glob(os.path.join(x[0], '*.*ml'))]
  return result

def getTextContent(elem: ET.Element) -> str:
  z = elem.xpath(".//text()")
  if (z is None):
      return "_none_"
  txt = ("".join(z)).strip()
  if len(txt) == 0:
      txt = "_none_"
  return re.sub('<[^<>]*>', '', txt)

def getContent(word: ET.Element, ns_tei) -> str:
  segje = word.find(ns_tei + 'seg')

  if segje is None:
      # print(f'Segje is er niet, tragedie! ns_tei=<{ns_tei}> actual ns={getElementNamespace(word)}: content: {getTextContent(word)}')
      if 'norm' in word.attrib:
        return word.attrib['norm']
      else:
        return getTextContent(word)
  else:
      return getTextContent(segje)

def tagje(n):
    return ET.QName(n).localname

def get_lang_attrib(elem):
  try:
     lang = elem.attrib[ns_xml + 'lang']
     return lang
  except:
     return None

def find_language(elem):
  global ns_xml
  lang = get_lang_attrib(elem)
  if lang is not None:
      return lang
  else: 
    parent = elem.getparent()
    if parent is not None:
        return find_language(parent)
    return None

def has_word_as_child(w):
    l = list(filter(lambda x: tagje(x)=='w', w.findall("./*")))
    #print(str(l))
    return len(l) > 0

def getSentencesFromFile(file:str, use_xml_ns=True, use_tei_ns=False):
    root = ET.parse(file)
    namespace = getNamespace(root)
    use_tei_ns =  (namespace == ns_p5)
    ns_xml = "{http://www.w3.org/XML/1998/namespace}" if use_xml_ns else ""
    ns_tei = "{http://www.tei-c.org/ns/1.0}" if use_tei_ns else ""
    return getSentences(root, ns_tei, ns_xml)

def getAtt(elem, name):
  if name in elem.attrib:
    return elem.attrib[name]
  else:
    return ''

'''
  Argument root: een TEI corpusdocument (als lxml element tree) met <w>- en <s>-tagging
  Keert een lijst tupels [words,word_ids,s,word_elements,lang] terug
  ToDo: wat doe je als er geen zinnen in het document zitten......
'''

def getSentences(root, ns_tei="{http://www.tei-c.org/ns/1.0}", ns_xml="{http://www.w3.org/XML/1998/namespace}", get_token_chunks=False):
  sentences = []
  sents =  root.findall('.//' + ns_tei + 's')
  if len(sents) == 0:
     sents =  root.findall('.//' + ns_tei + 'q')
  docelem = next(root.iter())

  sents = list(sents)
  if (len(sents)  == 0 or get_token_chunks):
     print("Not using sentences", file=sys.stderr)
     return getTokenChunks(root, ns_tei, ns_xml)
  docid = 'no_doc_id' # docelem.attrib['{http://www.w3.org/XML/1998/namespace}id']
  print("docid: " + docid + f" ns_tei={ns_tei}, sentences: {len(sents)}")
  n_sent = 0
  for s in sents:
      sentence=[]
      lang = find_language(s)
      sentence_id = docid + '.s.' + str(n_sent)

      s.attrib['{http://www.w3.org/XML/1998/namespace}id'] = sentence_id
      if lang is not None:
        s.attrib['{http://www.w3.org/XML/1998/namespace}lang'] = lang

      word_elements = list(filter(lambda e: ((tagje(e)=='pc') | (tagje(e)=='w')) and not has_word_as_child(e), s.findall(".//*"))) # [self::{http://www.tei-c.org/ns/1.0}w or self::{http://www.tei-c.org/ns/1.0}pc]") # |.//{http://www.tei-c.org/ns/1.0}pc")

      if len(word_elements) >= maximum_sentence_length:
        words = list(map(lambda w: getContent(w,ns_tei), word_elements))
        print("Warning: Long sentence in document with id " + docid + " " + str(words))
        chunks =  getTokenChunks(s, ns_tei, ns_xml)
        for c in chunks: # ToDo iets met de ids van de chunks
           sentences.append(c)
           n_sent = n_sent+1

      if (len(word_elements) > 0 and len(word_elements) < maximum_sentence_length): #  lange zinnen zijn erg traag en meestal opsommingen van aanwezigen. Die later structureel uitfilteren
        first_word_id = getAtt(word_elements[0],ns_xml + 'id')
        words = list(map(lambda w: getContent(w,ns_tei), word_elements))
        word_ids = list(map(lambda w: getAtt(w,ns_xml + 'id'), word_elements))
        #words = list(map(lambda w: w.text, word_elements))
        if (len(words) > 0 and len(words) < maximum_sentence_length and (None not in words)):
           sentences.append(TEISentence(words, word_ids, s, word_elements,lang, sentence_id)) # sentence id ook toevoegen!
      n_sent = n_sent+1
  return sentences

def getTokenChunks(root,ns_tei="{http://www.tei-c.org/ns/1.0}", ns_xml="{http://www.w3.org/XML/1998/namespace}", max_chunk_size=50):
   token_elements = list(filter(lambda e: ((tagje(e)=='pc') | (tagje(e)=='w')) and not has_word_as_child(e), root.findall(".//*")))
   chunks=[]
   chunk=[]
   n_tokens = 0

   def flush_chunk():
      nonlocal n_tokens
      nonlocal chunk
      if (len(chunk) > 0):
        words = list(map(lambda w: getContent(w,ns_tei), chunk))
        word_ids = list(map(lambda w: getAtt(w,ns_xml + 'id'), chunk))
        sentence_id = 'abritrary'
        sentence = TEISentence(words, word_ids, chunk[0], chunk,'nl', sentence_id)
        chunks.append(sentence)
      n_tokens = 0
      chunk=[]

   for t in token_elements:
       if (n_tokens >= max_chunk_size):
         flush_chunk()
       chunk.append(t)
       n_tokens += 1

   flush_chunk()
   return chunks
######## stuff for huggingface tagger/lemmatizer 

def getSentencesDict(filename, ns_tei="{http://www.tei-c.org/ns/1.0}", ns_xml="{http://www.w3.org/XML/1998/namespace}", get_token_chunks=False):
   root = ET.parse(filename) 
   namespace= getNamespace(root)
   if (namespace== ns_p5):
       ns_tei = "{" + namespace  + "}"
   else: 
       ns_tei = ""

   print(f"{filename}--> namespace={ns_tei}")

   sentences = getSentences(root, ns_tei, ns_xml, get_token_chunks)
   sents = []
   for s in sentences:
      tokens = []
      for i in range(0, len(s.words)):
          elem = s.word_elements[i]
          ref_lemma = elem.attrib['lemma'] if 'lemma' in elem.attrib else 'no_lemma'
          ref_pos = elem.attrib['pos'] if 'pos' in elem.attrib else 'no_pos'
          t = {'token' : s.words[i], 'id' :s.word_ids[i],  'element': s.word_elements[i], 'ref_pos' : ref_pos, 'ref_lemma' : ref_lemma }
          tokens.append(t)
      sents.append(tokens)
   return root, sents

def putAnnotationBack(filename: str, root, sentences:List[List[dict]], also_keep_old=False):
  new_prefix = 'new_' if also_keep_old else ''
  old_prefix = 'old_' if also_keep_old else ''
  n_sentences = 0
  for s in sentences:
    for t in s:
           w = t['element']
           for a in ['pos', 'lemma', 'lexicon']:
               if also_keep_old and a in w.attrib and a in t and t[a] is not None:
                  w.attrib[old_prefix + a] = w.attrib[a]
               if a in t and t[a] is not None: 
                   if a == 'lexicon':
                     w.attrib['resp'] = '#lexicon.' + t[a]
                   else:
                     w.attrib[a] = t[a]
           if 'group_id' in t and t['group_id'] is not None:
              group_id = t['group_id']
              join_id = f'mw_{n_sentences}_{group_id}'
              join = ET.Element('join')
              join.attrib['n']  = join_id
              if 'part'  in t and t['part'] is not None:
                 w.attrib['part'] = t['part']
              if 'nonverb'  in t and t['nonverb'] is not None:
                 w.attrib['ana'] = '#not_the_verbal_part'
              w.append(join)
              
           
  n_sentences += 1              

  with open(filename, 'wb') as f:
     root.write(f, encoding='utf-8')

def doWithTEISentences(in_file: str, out_file: str, action, get_token_chunks=False, also_keep_old=False):
   root, sents = getSentencesDict(in_file, ns_tei='', get_token_chunks=False)
   for s in sents:
     action(s)
   putAnnotationBack(out_file,root,sents,also_keep_old)

