import json
import tagger
import re
import sys
import gzip
import readFromTSV

def turn_around(s):
  sent = []
  tokens = s['tokens']
  tags = s['tags']
  lemmata = s['lemmata'] if 'lemmata' in s else []
  for i in range(0,len(tokens)):

     token = {'token' : tokens[i], 'ref_pos_complete' : tags[i] }
     token['ref_pos'] = re.sub("_.*", "", tags[i]) 

     if 'lemmata' in s:
        token['ref_lemma'] = lemmata[i]

     if 'relevances' in s:
        token['relevant'] = s['relevances'][i]
     sent.append(token)

  #print(f"Sentence: {sent}", file=sys.stderr)
  return sent

def open_maybe_unzip(filename):
  if filename.endswith(".gz"):
      return gzip.open(filename,"rt")
  else:
      return open(filename,'r')

def read_dataset(filename, data_format: str='json', fields: dict=readFromTSV.default_field_dict(), max_chunk_size: int=None):
  sentences = []
  with open_maybe_unzip(filename) as f:
    print("Test data: " + filename)
    if data_format=='tsv' or re.match(".*\.(tsv)(\.gz)?", filename):
      print(f"Loading test data from TSV {filename}")
      sentences_as_dicts = readFromTSV.read_sentences_to_list_of_dicts(filename, fields,max_chunk_size)
      # this is a bit silly:
      sentences = list(map(turn_around, sentences_as_dicts))
    else: 
      lines = f.readlines()
      for line in lines:
        sentence = json.loads(line)
        sentences.append(turn_around(sentence))
  return sentences
   
def tag_dataset(sentences,tagger,lemmatizer=None,choose_compatible:bool=False):
   for s in sentences:
      tagger.tag_tokenized_sentence(s, choose_compatible)
      if lemmatizer is not None:
         lemmatizer.lemmatize_tokenized_sentence(s)
         #print(f'Yep, lemmatized {s}!!!', file=sys.stderr)

def print_sentence_tabsep(s,r):
    print("\n",file=r)
    for t in s:
        pos = t['pos'] if 'pos' in t else 'no_pos'
        lemma = f"\t{t['lemma']}" if 'lemma' in t else '\t'
        print(f"{t['token']}\t{pos}\t{lemma}",file=r)

def print_tabsep(s,r):
    print("\n##########",file=r)
    for t in s:
        pos = t['pos'] if 'pos' in t else 'no_pos'
        group_id = t['group_id'] if 'group_id' in t else None
        lex_part = f" lexicon={t['lexicon']}" if 'lexicon' in t else ""
        lemma_error = f"\t <=L= {t['ref_lemma']}({lex_part})" if 'lemma_error' in t and t['lemma_error'] else ""
        lemma = f"\t{t['lemma']}" if 'lemma' in t else '\t'
        if 'error' in t and t['error']:
            print(f"{t['token']}\t{pos}\t{lemma}\t{group_id}\t<=== {t['ref_pos']}{lemma_error}",file=r)
        else:
            print(f"{t['token']}\t{pos}\t{lemma}\t{group_id}\t{lemma_error}",file=r)

def incr(d,t1,t2):
    if (t1,t2) in d:
        d[(t1,t2)] = d[(t1,t2)] + 1
    else:
       d[(t1,t2)] =1

def evaluate_tagged(tagged_dataset, model_name, dataset_filename, report=None):
   n_tokens = 0
   n_errors = 0
   n_coarse_errors = 0
   n_missed_tokens = 0

   n_lemma_errors = 0
   has_lemmata = False
   lemma_errors_per_lexicon = {}
   tokens_per_lexicon = {}

   r = None
   error_types={}
   coarse_error_types={}
   if report is not None:
       r = open(report,'w')
   for s in tagged_dataset:
       error_in_sentence = False

       for token in s:
           if not 'pos' in token: 
             n_missed_tokens += 1
             continue
           n_tokens += 1
           pos_head = re.sub("\(.*?\)|_[bif]|(\|.*)", "", token['pos'])
           ref_pos_head = re.sub("\(.*?\)|_[bif]|(\|.*)", "", token['ref_pos'])
           token['error'] = False
           if re.sub('\(\)', '', token['pos']) != re.sub('\(\)', '', token['ref_pos']):
               n_errors += 1
               token['error'] = True
               incr(error_types,token['ref_pos'], token['pos'])
               error_in_sentence = True
           if pos_head != ref_pos_head:
             unmapped = token['unmapped_pos'] if 'unmapped_pos' in token else ''
             incr(coarse_error_types, ref_pos_head,  pos_head)
             #print(f"hyp {pos_head} ({unmapped}) != ref {ref_pos_head} for '{token['token']}'")
             n_coarse_errors += 1

           if 'lexicon' in token:
             l = token['lexicon']
             if l in lemma_errors_per_lexicon:
               tokens_per_lexicon[l] += 1
             else:
               tokens_per_lexicon[l] = 1

           if 'lemma' in token and 'ref_lemma' in token and token['lemma'] != token['ref_lemma'] and not (token['pos'] == 'LET' or token['pos'] == 'PC'):
               if 'lexicon' in token:
                  l = token['lexicon']
                  if l in lemma_errors_per_lexicon:
                     lemma_errors_per_lexicon[l] += 1
                  else:
                     lemma_errors_per_lexicon[l] = 1
                  l = token['lexicon']
                  if l in lemma_errors_per_lexicon:
                     tokens_per_lexicon[l] += 1
                  else:
                     tokens_per_lexicon[l] = 1
               token['lemma_error'] = True
               has_lemmata=True
               n_lemma_errors += 1

       if (error_in_sentence or True) and report is not None:
           print_tabsep(s,r)
           r.flush()

   score = 1 - (n_errors / n_tokens)
   coarse_score = 1 - (n_coarse_errors / n_tokens)
   lemma_score = 1 - (n_lemma_errors / n_tokens)  if has_lemmata else None

   results = {
             'score' : score,
             'coarse_score' : coarse_score,
             'model_name' : model_name,
             'test_data' : dataset_filename,
             'test_tokens' : n_tokens,
             'lemmatization' : lemma_score
            }
   status =  f"Model: {model_name}; Data: {dataset_filename}\n- Tokens {n_tokens}, missed {n_missed_tokens}; score {score}, coarse {coarse_score}, lemmata {lemma_score}\n"
   
   if has_lemmata:
      lemma_score_per_lexicon = {}
      for k in lemma_errors_per_lexicon:
         lemma_score_per_lexicon[k] = 1 - lemma_errors_per_lexicon[k] / tokens_per_lexicon[k]
      results['lemma_errors_per_lexicon'] = lemma_errors_per_lexicon
      results['tokens_per_lexicon'] = tokens_per_lexicon
      results['lemma_score_per_lexicon'] = lemma_score_per_lexicon

   print(status)


   error_log = "\nError analysis:\n- POS head errors\n"

   error_type_info = list(sorted(coarse_error_types.items(), key=lambda x: -1 * x[1]))
   for i in error_type_info:
       if i[1] > 2:
         error_log += str(i) + "\n"

   error_log += "\n- With features\n"
   error_type_info = list(sorted(error_types.items(), key=lambda x: -1 * x[1]))
   for i in error_type_info:
       if i[1] > 2:
         error_log += str(i) + "\n"


   if r is not None:
      r.close()
      with open(report, 'r') as original: data = original.read()
      with open(report, 'w') as modified: modified.write(status + "" + error_log + "\n" + data)
   return results

def evaluate(model_name, dataset_filename, mapping_filename: str = None, report: str=None, lemmatizer=None, data_format: str='json', fields: dict=None, max_chunk_size: int=None):
  the_tagger = tagger.find_model(model_name)
  if mapping_filename is not None:
     mapping = tagger.read_mapping(mapping_filename)
     the_tagger.tag_mapping = mapping
  testdata = read_dataset(dataset_filename, data_format=data_format,fields=fields,max_chunk_size=max_chunk_size)
  tag_dataset(testdata, the_tagger, lemmatizer)
  results = evaluate_tagged(testdata, model_name, dataset_filename,report=report)
  return results

def tag_pretagged(model_name, dataset_filename, output_filename, mapping_filename: str = None, report: str=None, lemmatizer=None, data_format: str='json', fields: dict=None, max_chunk_size: int=None, choose_compatible: bool = False):

  print(f"Load model: {model_name}, {choose_compatible}")
  the_tagger = tagger.find_model(model_name, return_alternatives=choose_compatible)
  if mapping_filename is not None:
     mapping = tagger.read_mapping(mapping_filename)
     the_tagger.tag_mapping = mapping
  testdata = read_dataset(dataset_filename, data_format=data_format,fields=fields,max_chunk_size=max_chunk_size)
  tag_dataset(testdata, the_tagger, lemmatizer, choose_compatible)
  output_file = open(output_filename,"w")
  for s in testdata:
    print_sentence_tabsep(s,output_file)
  output_file.close()

if __name__=='__main__':
  dataset_filename='../data/nederval/json/19_thomas.test.json'
  model_name='../data/tagging/pos_tagging_model_19/'
  model_name='../data/tagging/pos_tagging_model_19_from_bertje_udlassy_pos'
  model_name='../data/tagging/pos_tagging_model_19_finetuned'
  model_name='deepfrog'
  mapping_filename='mappings/cgn_core.txt'
  evaluate(model_name, dataset_filename,mapping_filename)
