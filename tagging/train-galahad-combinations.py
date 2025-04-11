from train import train_from_config, usage
import sys
import json
import os
import re

template=json.load(open(sys.argv[1],'r'))
training_data_dir = template['base_dir']
combi_dir = re.sub('training-data','combinations', training_data_dir)
files = os.listdir(combi_dir)

def clone(o):
  return json.loads(json.dumps(o))

for f in files:
  combi = json.load(open(combi_dir + '/' + f,'r'))
  combi_name = combi['name']

  instance  = clone(template)
  instance['combination'] = combi_name
  instance['model_output_path'] =  f"../data/tagging/tagging_models/galahad_models/pos_model_tdn_{combi_name}/"
  instance['name'] = combi_name

  print(json.dumps(instance,indent=4))
  train_from_config(config_path=instance)

