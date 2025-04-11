from datasets import load_dataset, Dataset, ClassLabel
from transformers import AutoModelForSequenceClassification, DataCollatorForLanguageModeling, AutoModelForMaskedLM, \
    AutoModelForTokenClassification, DataCollatorForTokenClassification
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer
import numpy as np
from datasets import load_metric, Dataset
import json
import eval_tagger
import sys
import transfer
from  transfer import TaskTuning
import re
import socket
import os
from os.path import exists
from datetime import date
import git
sys.path.append('../lemmatizer')
from lemmatizer import Lemmatizer, lemmatizer_from_config

# export CUDA_VISIBLE_DEVICES="" to really disable GPU
# resume_from_checkpoint=True als je verder wilt trainen
# "python3 {TAGGERS}/{tagger_name}/train.py {train_set_path} {dev_set_path} {config_path} {docker_path}

def resolve_combination(config):
   if 'combination' in config:
     base_dir = config['base_dir']
     combi_name = config['combination']
     combi_path = f"{base_dir}/../combinations/{combi_name}.combination.json"
     if exists(combi_path):
       combi_file = open(combi_path ,"r")
       combi_object = json.load(combi_file)
       datasets = combi_object['datasets']
       training = [f"{x}/{x}.train.tsv" for x in datasets]
       testing = [f"{x}/{x}.test.tsv" for x in datasets]
       dev =  [f"{x}/{x}.dev.tsv" for x in datasets][0]
       print(f"{training} {testing} {dev}")    
       config['training_data'] = training
       config['dev_data'] = dev
       config['test_data'] = testing
     else:
        print(f"{combi_path} does not exist!", file=sys.stderr)

def train_from_config(train_path=None, dev_path=None, config_path=None, model_path=None, skip_training: bool = None):

   local_params = locals()
   params = {}
   for k in local_params.keys():
      v = str(local_params[k])
      params[k] = v


   use_data_paths_from_config = train_path is None and dev_path is None

   #print(config_path)
   #print(type(config_path))
   if type(config_path)==dict: # messy, please change
     config=config_path
     config_name=config['name']
   else:
     json_file =  open(config_path)
     config = json.load(json_file)
     config_name = re.sub("^(.*/)?(.*)\.[^/]*$", r"\2", config_path)
     config['name'] = config_name

   resolve_combination(config)

   log_path = '/tmp/'
   if 'log_path' in config:
     log_path = config['log_path']

   if skip_training is None:
     skip_training = config['skip_training'] if 'skip_training' in config else False
   base_dir = config['base_dir']
   base_model = config['base_model']

   training_data =  list(map(lambda x: base_dir + '/' + x, config['training_data'] )) if (train_path is None or train_path=='-') else [train_path]

   # Training data partitions in config for instance:
   # "training_data" :  ["bab.train.N.tsv.gz"],
   # "training_partition" : { "variable" : "N", "number" : 10},

   use_cpu = config["use_cpu"] if "use_cpu" in config else False

   training_partitions = training_data
   n = 1 
   partitioned = False
   if ("training_partition" in config):
     partitioning = config["training_partition"]
     variable = partitioning["variable"]
     n = partitioning["number"]
     partitioned = True
     training_partitions = [re.sub(variable, str(k), training_data[0]) for k in range(0,n)]

   experiment = {}

   experiment['hostname'] = socket.gethostname()
   experiment['date'] = str(date.today())
   experiment['cwd'] = os.getcwd()
   if 'VIRTUAL_ENV' in os.environ:
     experiment['virtualenv'] = os.environ['VIRTUAL_ENV']

   try:
     git_info = {}
     repo = git.Repo(search_parent_directories=True)
     sha = repo.head.object.hexsha
     git_info['commit'] = sha
     repo_url = re.sub('://.*@', '://', repo.remotes['github'].url)
     git_info['repository'] = repo_url

     branch = repo.active_branch
     branch_name = branch.name
     git_info['branch'] = str(branch_name)
     experiment['git'] = git_info
     print(json.dumps(git_info,indent=2))
     
   except:
     print("Error getting git info, maybe not in git repository")

   if type(config_path)==str:
     experiment['config_path'] = config_path

   experiment['config'] = config
   experiment['function_params'] = params
   experiment['sys_argv'] = sys.argv

   for k in range(0,n):
     
     training_data_k = training_partitions[:k+1] if partitioned else training_partitions

     print(f"At {k}: {training_data_k}")
     
     model_output_dir = config['model_output_path'] if (model_path is None or model_path=='-') else model_path
     dev_data = base_dir + '/' + config['dev_data'] if (dev_path is None or dev_path=='-') else dev_path

     epochs = config['epochs']

     fields = config['fields'] if 'fields' in config else {'tokens' : 0, 'tags' : 1, 'lemmata': 2}
     max_chunk_size = config['max_chunk_size'] if 'max_chunk_size' in config else None

     per_epoch_eval = {}
     data_format = config['data_format'] if 'data_format' in config else None
     if not skip_training:
       tuning: TaskTuning = TaskTuning(base_model, label_column_name='label', num_train_epochs=epochs)
       dataset = tuning.create_pos_dataset(training_data_k, dev_data,data_format=data_format, fields=fields, max_chunk_size=max_chunk_size)
       per_epoch_eval = tuning.tasktune_token_classification_model(base_model, dataset, label_column_name='label', output_dir=model_output_dir, num_train_epochs=epochs, use_cpu=use_cpu)
     else:
       print('skip training', file=sys.stderr)
     results = {}

     if 'test_data' in config and use_data_paths_from_config:
       
       results['test_set_results']  = {}
       lemmatizer = None
       test_data_setting = config['test_data']
       test_data_files = []

       if type(test_data_setting) == list:
         test_data_files = test_data_setting
       else:
         test_data_files = [test_data_setting]

       for filename in test_data_files:  
        test_data = base_dir + '/' + filename
        filename_short = re.sub(".*/", "", filename)
        filename_short = re.sub("[.].*", "", filename_short)

        if 'lemmatizer' in config and skip_training: # lemmatizer is used only when training is skipped (we do not want training to fail if the lemmatizer fails)
          lemconfig = config['lemmatizer']
          lemmatizer = lemmatizer_from_config(lemconfig)

        log_file=config['log_path'] + "/" + filename_short + "." +  config_name + ".log"

        test_results = eval_tagger.evaluate(model_output_dir, test_data, report=log_file,
                               data_format=data_format, fields=fields, max_chunk_size=max_chunk_size, lemmatizer=lemmatizer)
        results['test_set_results'][filename_short] = test_results

     results['stage'] = k
     results['per_epoch_dev_set_eval'] = per_epoch_eval
     experiment['dev_and_test_set_results'] = {} 
     experiment['dev_and_test_set_results'][f"stage_{k}"] = results
   
   if not skip_training:
     save_results_to = open(f"{model_output_dir}/provenance.{config_name}.json",'w') # waar bewaren we dit iha...???
     json.dump(experiment, save_results_to,indent=4)
     save_results_to.close()
   else:
     save_results_to = open(f"{log_path}/{config_name}.test.results.json","w")
     json.dump(results, save_results_to,indent=4)
     save_results_to.close()


def usage():
    print("Usage: python train.py <TRAIN_PATH> <DEV_PATH> <CONFIG_PATH> <MODEL_PATH>\n\t\t(or just <CONFIG_PATH> to the use data and model paths from the config)")

if __name__ == '__main__':
   if (len(sys.argv) == 2):
     train_from_config(config_path=sys.argv[1])
   elif len(sys.argv) == 5:
     train_from_config(
           train_path=sys.argv[1],
           dev_path=sys.argv[2],
           config_path=sys.argv[3],
           model_path=sys.argv[4])
   else:
      usage()
