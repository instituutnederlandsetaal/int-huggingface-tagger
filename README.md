
# int-hugginface-tagger

Use models from huggingface transformers for historical Dutch PoS tagging and lemmatisation.

Caution: This is a prerelease.

### GaLAHaD-related Repositories
- [galahad](https://github.com/INL/galahad)
- [galahad-train-battery](https://github.com/INL/galahad-train-battery)
- [galahad-taggers-dockerized](https://github.com/INL/galahad-taggers-dockerized)
- [galahad-corpus-data](https://github.com/INL/galahad-corpus-data/)
- [int-pie](https://github.com/INL/int-pie)
- [int-huggingface-tagger](https://github.com/INL/huggingface-tagger) [you are here]
- [galahad-huggingface-models](https://github.com/INL/galahad-huggingface-models)

## Synopsis
This repository contains code to:
* run and train a simple huggingface token classifier based PoS tagger
* train and run a lemmatizer combining:
  * the INT historical lexicon
  * for out of vocabulary tokens, a ByT5 model (https://huggingface.co/docs/transformers/model_doc/byt5)

To use:
* clone this repository
* Create a virtual environment and activate it (python 3.10 or later)
* run `bash requirements.sh`
* [install Git Large File Storage]  (https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage) 
* clone the model repository

Assuming you have installed Git LFS and you work on linux:
```
git clone https://github.com/INL/int-huggingface-tagger
cd int-huggingface-tagger
python3.10 -m venv venv
source ./venv/bin/activate
bash requirements.sh
cd ..
git lfs clone https://github.com/INL/galahad-huggingface-models
cd int-huggingface-tagger
ln -s ../galahad-huggingface-models/models/ .
python example-usage.py config/galahad/tagger-lemmatizer/ALL.tdn.config example-data/eline.txt /tmp/output.txt
```

You need to enable git lfs to include the trained models in a clone.

Tagger-lemmatizer
=================

Run:
----

* On plain text: `python scripts/example-usage.py <configuration file> <input text> <output tsv> 
  ```python example-usage.py config/galahad/tagger-lemmatizer/ALL.tdn.config example-data/eline.txt /tmp/output.txt```

* On (tokenized!) TEI: `python scripts/example-usage-TEI.py config/galahad/tagger-lemmatizer/ALL.tdn.config example-data/example.tei /tmp/example.tagged.tei```
Your tei needs to be tokenized, with word id's in xml:id.


```xml
<TEI xmlns="http://www.tei-c.org/ns/1.0">
<teiHeader/>
<text>
  <body>
  <w xml:id="w1">Dit</w>
  <w xml:id="w2">is</w>
  <w xml:id="w3">een</w>
  <w xml:id="w4">corte</w>
  <w xml:id="w5">Zin</w>
  <pc>.</pc>
  </body>
</text>
</TEI>
```

```xml
<TEI xmlns="http://www.tei-c.org/ns/1.0">
<teiHeader/>
<text>
  <body>
  <w xml:id="w1" pos="PD(type=d-p,position=free)" lemma="dit" resp="#lexicon.molex">Dit</w>
  <w xml:id="w2" pos="VRB(finiteness=fin,tense=pres)" lemma="zijn" resp="#lexicon.molex">is</w>
  <w xml:id="w3" pos="PD(type=indef,subtype=art,position=prenom)" lemma="een" resp="#lexicon.molex">een</w>
  <w xml:id="w4" pos="AA(degree=pos,position=prenom)" lemma="kort" resp="#lexicon.hilex">corte</w>
  <w xml:id="w5" pos="NOU-C(number=sg)" lemma="zin" resp="#lexicon.molex">Zin</w>
  <pc pos="PC">.</pc>
  </body>
</text>
```			

The config file specifies the tagging model and lemmatization models and the (pickled) lexicon filename, e.g.
```json
{
 "tagging_model" : "../data/tagging/tagging_models/pos_tagging_model_combined_gysbert/",
 "lem_tokenizer" : "../data/byt5-lem-hilex-19",
 "lem_model" : "../data/byt5-lem-hilex-19/checkpoint-53500/",
 "lexicon_path" : "../data/lexicon/lexicon.pickle"
}
```

Training the token classifier for PoS tagging
---------------------------------------------

### From configuration file
```
python train.py <config file>
```

A typical configuration file example
```json
{
  "epochs": 10,
  "data_format" : "tsv",
  "fields" : {"tokens" : 0, "tags" : 1, "lemmata": 2},
  "max_chunk_size" : 50,
  "base_dir" : "/vol1/data/tagger/galahad-corpus-data/training-data/couranten/",
  "training_data" :  ["couranten.train.tsv.gz"],
  "test_data" : "couranten.test.tsv.gz",
  "dev_data" : "couranten.dev.tsv.gz",
  "model_output_path" : "../data/tagging/tagging_models/couranten/",
  "base_model" : "emanjavacas/GysBERT"
}
```

### From python: for instance: at `tagging/nineteen.py`

Finetuning a pretrained language model (adapt a general language model to a specific domain, using unlabeled text):
```python
def finetune_dbnl_19_lm(base_bert='GroNLP/bert-base-dutch-cased', 
                output_dir='../data/tagging/language_models/dbnl_19_lm'):
    training_data = '../data/tagging/unannotated/dbnl_stukje_19.json'
    test_data = '../data/tagging/unannotated/multatuli_ideen.json'
    dataset = load_dataset('json',
                           data_files={'train': [training_data], 'test': test_data},
                           #sep="\t",
                           download_mode='force_redownload')
    dataset = dataset.remove_columns('tags').rename_column('tokens', 'text')
    transfer.finetune_language_model(base_bert, dataset,output_dir)

```


Tasktuning a model to a training dataset (add a classification layer to a language model, using PoS-labeled data):
```python
def train_pos_dataset_gysbert(): # Use the Gysbert model from MacBerth, best results for now
    training_data = '../data/nederval/json/19_thomas.train.json'
    test_data =     '../data/nederval/json/19_thomas.test.json'
    task_output_dir = '../data/tagging/tagging_models/pos_tagging_model_19_gysbert'

    base_model = 'emanjavacas/GysBERT'
    dataset = transfer.create_pos_dataset(training_data, test_data)
    transfer.tasktune_token_classification_model(base_model, dataset, label_column_name='label', 
                   output_dir=task_output_dir, num_train_epochs=epochs())
``` 

### Data format for training files:
  TSV (Cf. [galahad-corpus-data](https://github.com/INL/galahad-corpus-data/)) or huggingface dataset JSON, with one object per line, which represents a tagged sentence:

```json
{"id":"168","tokens":["Ik","lees","voor","me","pleizier",",","meneer",",","als","ik","lees","."],"tags":["PD(type=pers,position=free)","VRB(finiteness=fin,tense=pres)","ADP(type=pre)","PD(type=poss,position=prenom)","NOU-C(number=sg)","LET","NOU-C(number=sg)","LET","CONJ(type=sub)","PD(type=pers,position=free)","VRB(finiteness=fin,tense=pres)","LET"]}
{"id":"102","tokens":["O","verbasterde","nazaet","!"],"tags":["INT","AA(degree=pos,position=prenom)","NOU-C(number=sg)","LET"]}
```




Training the byt5 model for unknown word lemmatisation
------------------------------------------------------

Cf. `lemmatizer/train_byt5_lemmatizer.py`.

Input is a tab separated file, exported from the historical lexicon or the training corpus data, containing at least word, pos, and lemma columns 
Training takes a long time (as does running). Byt5 is slow.

Besides the ByT5 model, the lemmatizer uses data from the INT historical lexicon (`data/lexicon/lexicon.pickle`).
The ByT5 model is used as a fallback for unknown words.
