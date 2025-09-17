# Usage: 
# python example-usage.py [config path] [input path] [output path]
# Example
# python example-usage.py config/galahad/tagger-lemmatizer/ALL.tdn.config example-data/eline.txt example-data/output.txt

import sys
from lemmatizer.TaggerLemmatizer import TaggerLemmatizer

def main():
  t = TaggerLemmatizer(sys.argv[1])
  t.init()
  t.process(sys.argv[2], sys.argv[3])

if __name__ == "__main__":
  main()
