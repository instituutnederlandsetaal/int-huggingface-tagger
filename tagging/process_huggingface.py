import os
import re
import sys
from typing import TextIO, List
import requests
import urllib.parse
import jsons
import io

server = 'http://127.0.0.1:8000/tag_with_model/bertje_pos'

def tagTextFile(inputFileName: str) -> List[List[str]]:
   with io.open(inputFileName, mode="rb") as f: # encoding="utf-8"
       contents = f.read() # //.decode('utf-8')
       #print("sending : " + contents)
       response = requests.post(server, contents, headers={'Content-Type': 'text/plain; charset=UTF-8'})
       #print(response.content)
       object =  jsons.loads(response.content)
       return  list(filter(lambda x: x[1] != 'O' and x[0] != '', object))


def tagTextFileToTSV(in_file: str, f_out:  TextIO ):
    res : List[List[str]] = tagTextFile(in_file)
    for w in res:
        f_out.write(f"{w[0]}\t{w[2]}\t{w[1]}\n")

def process(in_file : str, out_file: str, tagger='bab-tagger'):
    f_out: TextIO = open(out_file, "w")
    f_out.write("Literal\tlemma\tpos\n") # write tsv header
    tagTextFileToTSV(in_file, f_out)
    f_out.close()

if __name__=='__main__':
    #tagTextFile(sys.argv[1])
    process(sys.argv[1], "/tmp/tagged.txt")
