import re
from typing import Callable, Mapping, List
import sys
import json
sys.path.append('../tagging')
sys.path.append('./tagging')
try:
  import readFromTSV as tsv
except:
  import tagging.readFromTSV as tsv
  
def postprocess_tagged_token(token: dict):
    if 'word' not in token and 'token' in token:
        token['word']  = token['token'] # Bah!
    if not 'pos' in token:
        return
    else:
        pos = token['pos']
        if '_' in pos:
            [pos, annex] = re.split('_', pos)
            token['pos'] = pos
            if annex.startswith('b'):
                token['part'] = 'initial'
            elif annex.startswith('i'):
                token['part']  = 'internal'
            elif annex.startswith('f'):
                token['part'] = 'final'
            if 'A' in annex:
                token['nonverb'] = True
            else:
                token['nonverb'] = None
        else:
            token['part']  = None


def exists(f, l):
    return len(list(filter(f,l))) > 0

def connect_linked_tokens(tokens: List[dict]):
    for t in tokens:
        postprocess_tagged_token(t)
    initial = None
    current_group = None
    n_groups = 0
    for t in tokens: 
        if 'part' not in t:
            continue
        elif t['part'] == 'initial':
            current_group = [t]
            t['group'] = current_group
            t['group_id'] = n_groups
            n_groups += 1 # beter: kies random guid, id is nu alleen uniek binnen de zin
        elif t['part']  == 'internal':
            if current_group is not None:
                current_group.append(t)
                t['group']  = current_group
        elif t['part'] == 'final':
            if current_group is not None:
                current_group.append(t)
                t['group']  = current_group
            current_group = None
   
    groups = {}
    for t in tokens:
        if 'group' in  t:
            group = t['group']
            if len(group) <= 1:
              del t['group']
              del t['group_id']
              continue
            
            for x in group:
              if 'group_id' in x:
                t['group_id']  = x['group_id']
            def x(t):
                return 0 if t['nonverb'] else 1
            
            word_parts = [t['word'] for t in group]
            if exists(lambda x: x['nonverb'] == True, group) and exists(lambda x: x['nonverb'] == None, group) and exists(lambda x: x['pos'].startswith('VRB'), group):
                sorted_group = sorted(group, key=x)
                word_parts = [t['word'] for t in sorted_group]
            
            t['grouped_wordform'] = ''.join(word_parts)
            del t['group'] # dit liever niet eigenlijk als je in de TEI nog joins wil toevoegen....
            
        
    
def test(fn):
    fields = {'word': 0, 'pos' : 1}
    sentences = tsv.read_sentences_to_list_of_dicts(fn, fields)
    for s in sentences:
        tokens = []
        for [i,w] in enumerate(s['word']):
           token = {}
           token['word'] = w
           pos = s['pos'][i]
           token['pos']  = pos
           tokens.append(token)
        connect_linked_tokens(tokens)
    pass

if __name__ == '__main__':
    test('TestData/connect_tokens_test.txt')

