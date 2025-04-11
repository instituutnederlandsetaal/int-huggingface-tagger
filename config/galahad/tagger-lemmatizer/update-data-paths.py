import json
import sys
import os
import re


example = {
 "tagging_model" : "data/tagging/tagging_models/galahad_models/pos_model_tdn_ALL/",
 "lem_tokenizer" : "data/lemmatizer/galahad/tdn_byt5_model/",
 "lem_model" : "data/lemmatizer/galahad/tdn_byt5_model/",
 "lexicon_path" : "data/lexicon/lexicon.pickle",
 "chop_pos_to_main" : False
}

lem_path = "models/galahad/lemmatizer/"
tagger_path = "models/galahad/tagger/"
lexicon_path = "models/galahad/lexicon/"

for f in os.listdir("."):
  if f.endswith("config"):
    obj = json.load(open(f))
    obj['tagging_model']=re.sub(".*/(.)",tagger_path + r"\1",obj['tagging_model'])
    obj['lem_model']=re.sub(".*/(.)", lem_path + r"\1" , obj['lem_model'])
    obj['lem_tokenizer']=re.sub(".*/(.)",lem_path + r"\1" ,obj['lem_tokenizer'])
    obj['lexicon_path']=re.sub(".*/(.)", lexicon_path + r"\1",obj['lexicon_path'])
    string = json.dumps(obj,indent=4)
    out=open(f"fixed/{f}", "w")
    print(string,file=out)
    out.close()
    print(string)


