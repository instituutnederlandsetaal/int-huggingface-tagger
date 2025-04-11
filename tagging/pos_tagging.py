from transformers import AutoTokenizer, AutoModelForTokenClassification
from tokenizers import pre_tokenizers
from transformers import pipeline
import tei
import re
import sys
import os
from typing import Callable, Mapping, List
from functools import lru_cache
import nltk
from tagger import find_model
import tagger

#nltk.download('punkt')

from nltk.tokenize import sent_tokenize, word_tokenize

example: str = "Batavia heeft om advies gevraagd. Maar er kwam geen antwoord uit Den Haag. Jammer!"
example_ja: str = "ドアが内側に向けて小さく開いた。ボーイはトレイを両手に持ち、軽く一礼して部屋の中に入っていった。僕は廊下の花瓶の陰で、彼が出てくるのを待ちながら、これからどうすればいいのか考えを巡らせた。"

def slurp_file(fileName: str) -> str:
    with open(fileName, 'r') as f:
        return f.read()

def test_japans():
    the_model = japanese_upos()
    print("model loaded")
    pos_results = the_model.tag_text(example_ja)
    printSentences('/tmp/pos.out.txt', [pos_results])

def test_tokenized():
    model_name = 'pos_tagging_model_19'
    text = "Dit is een behoorlijk oninteressant Jan's voorbeeldje ."
    sentence = list(map(lambda t: {'token': t} , re.split("\s+", text)))
    the_model = find_model(model_name)
    r = the_model.tag_tokenized_sentence(sentence)
    print(str(r))

# usage: input output <model name>
if __name__=='__main__':
    if (len(sys.argv) <= 1) or (sys.argv[1] == 'testtok'):
       test_tokenized();
       exit();
    model_name = 'pos_tagging_model_19'
    text_file = None
    text = 'Dit is een behoorlijk oninteressant voorbeeldje'
    output = '/tmp/pos.out.txt'

    if len(sys.argv) > 1:
        text_file = sys.argv[1]
        text = slurp_file(text_file)
    if len(sys.argv) > 2:
        output = sys.argv[2]
    if len(sys.argv) > 3:
        model_name = sys.argv[3]
    the_model = find_model(model_name)
    if len(sys.argv) > 4:
        mapping = tagger.read_mapping(sys.argv[4])
        the_model.tag_mapping = mapping

    pos_results = the_model.tag_text(text)
    tagger.printSentences(output, [pos_results])
