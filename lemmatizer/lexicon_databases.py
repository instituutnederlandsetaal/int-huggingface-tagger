#from tabulate import tabulate
import sys
import re
import pickle

try:
  from LexiconDataConnector import LexiconDataConnector
except:
  from .LexiconDataConnector import LexiconDataConnector

from dataclasses import dataclass

def uniqbystr(l):
    d = {}
    for x in l:
      s = str(x)
      d[s] = x
    return list(d.values())

@dataclass
class Lexicon:
  lexicon_path: str = None
  tag_mapping_path: str = None
  database_config_path: str = None
  database_connector: LexiconDataConnector = None

  indexes = {
    'hilex' : None,
    'molex' : None,
    'frequencies': None,
    'training_lexicon' : None,
    'combined' : {}
  }

  tag_mapping = None

  def init(self):
    if self.lexicon_path is not None:
      self.loadFromPickle(self.lexicon_path)
    if self.tag_mapping_path is not None:
      self.read_mapping(self.tag_mapping_path)
    print(self.database_connector)

  def read_mapping(self, filename):
    m = {}
    f = open(filename,"r")
    for line in f.readlines():
      [tag,mapped] = re.split("\t",line)
      m[tag] = re.sub("\\s+", "", mapped)
    self.tag_mapping = m


  def add(self,dict,key,value):
      dict[key] = value
      return dict
  
  def lexicon_priority(self,index_name):
      if (index_name == 'training_lexicon'):
         return -1000000000
      if (index_name=='molex'):
         return -100000
      return 0
 
  def createIndex(self, index_name):
      if index_name not in self.indexes or self.indexes[index_name] is None:
         print(f"Creating index {index_name}", file=sys.stderr)
         self.indexes[index_name] = self.database_connector.loadIndex(index_name)

         if self.indexes['frequencies'] is None:
            f : dict = self.database_connector.loadIndex('frequencies')
            self.indexes['frequencies'] = f
         freq =  self.indexes['frequencies']
         for key, words in self.indexes[index_name].items():
            for w in words:
               #print(index_name + ":" + str(w), file=sys.stderr)
               if 'frequency' in w:
                 pass
               elif 'n_w' in w: # hilex: neem 'frequentie' van de woordvorm/lemma combinatie (=aantal attestaties)
                  w['frequency'] = w['n_w']
               elif w['lemma'] in freq:
                  w['frequency'] = freq[w['lemma']]
               else:
                  w['frequency'] = 0
            if key in self.indexes['combined']:
               known = self.indexes['combined'][key] + words
            else:
               known = words
            self.indexes['combined'][key] = sorted(words,key = lambda x: -1 * x['frequency'] + self.lexicon_priority(index_name))
            # nog steeds er dubieus, ic wordt dan altijd IC, etc
  
            # dit werkt niet goed
            # sorteer hilex lemmata op woordvorm_attestaties
            # sorteer molex lemmata of frequentie lemma
            # en dan?? wie wint??

  def lookup(self, word, index_name='hilex'):
      #print("Lookup:" + index_name)
      self.createIndex(index_name)
      index = self.indexes[index_name]
      if word in index:
          items = uniqbystr(index[word])
          return list(map(lambda x : self.add(x,'lexicon', index_name), items))
      return []
  
  def lookup_both(self,w):
      return self.lookup(w,'training_lexicon') + self.lookup(w,'molex') + self.lookup(w,'hilex')
      #return self.lookup(w,'molex') + self.lookup(w,'hilex')
  
  def info(self,lemmata):
      return "; ".join(list(map(lambda l: f"{l['lexicon']}:{l['lemma']}:{l['lemma_gigpos']}:{l['lemma_id']}", lemmata)))
 

  def map_pos(self, pos):
    p = self.tag_mapping[pos] if self.tag_mapping is not None and pos in self.tag_mapping else pos
    return p

  def lookup_compatible(self,w, pos):
      lookups = self.lookup_both(w.lower())
      pos_mapped = self.tag_mapping[pos] if self.tag_mapping is not None and pos in self.tag_mapping else pos
      pos_head = re.sub("\(.*","",pos_mapped) 
  
      def compatible(item):
          #print(f"{item} ? {pos_head} {pos}", file=sys.stderr)
          if item['lexicon'] == 'training_lexicon':
             return pos==item['wordform_gigpos']
          parts = set(map(lambda x: re.sub('\(.*', '', x),
                  re.split("\s+", item['lemma_gigpos'])))
          return pos_head in parts or (item['lexicon'] == 'molex' and re.match('VRB.*part.*', item['wordform_gigpos']) and pos_head=='AA') 
  
      matching = list(filter(compatible, lookups))
  
      msorted = sorted(matching,key=lambda w: -1 * w['frequency'])
      not_matching = list(filter(lambda x : not compatible(x), lookups))
      information = self.info(msorted) + ' || ' + self.info(not_matching) 
      return (msorted, not_matching, information)
  
  
  def saveToPickle(self,filename: str):
    filehandler = open(filename, 'wb')
    pickle.dump(self.indexes, filehandler)
  
  def loadFromPickle(self,filename: str):
    filehandler = open(filename, 'rb')
    self.indexes = pickle.load(filehandler)
    print(f"loaded self.indexes from file {filename}")
    print(self.indexes.keys())


