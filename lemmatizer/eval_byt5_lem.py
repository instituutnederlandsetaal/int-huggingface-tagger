from dataclasses import dataclass
import re
import sys

# !! this is a 'type-level' evaluation. 
# A lemmatization is considered correct if it occurs in the test set gold standard
# errors on tokens ambiguous within the dev set are not penalized

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

@dataclass
class LemEval:
  training_file:str = None
  test_file:str = None
  output_file:str = None
  patch:bool = True
  log_all = False

  #

  known_words = {}
  known_lemmata = {}

  test_words = {}
  test_pos = {}
  has_lemma = {}
  has_training_lemma = {}

  def read_training(self):
     f = open(self.training_file,"r")
     lines = f.readlines()
     for line in lines:
       [word,pos,lemma,count] = re.split("\t", line)
       self.known_words[word] = 1
       self.known_lemmata[lemma] = 1
       self.store(line, self.has_training_lemma)

  def choose_training_lemma(self, wp):
    if wp in self.has_training_lemma:
       z= list(self.has_training_lemma[wp])
       if z is not None:
         possible =  sorted(z, key = lambda x: -1 * x[1]) # [['VAN', 35502]]
         return possible[0][0]
    return None

  def store(self, line, dictionary):
       line = line.strip()
       [word,pos,lemma,count] = re.split("\t", line)
       if count == 'count':
         return
       count = int(count)
       self.test_words[word] = 1
       self.test_pos[pos] = 1
       wp = word + ' ' + pos
       #print(f"<{wp}>")
       l = []
       if wp in dictionary:
         l = dictionary[wp]
         l.append([lemma,count])
       else:
         l = [[lemma,count]]
         dictionary[wp] = l

  def read_test_goldstandard(self):
     f = open(self.test_file,"r")
     lines = f.readlines()
     for line in lines:
        self.store(line, self.has_lemma)

  def read_output(self):
     f = open(self.output_file,"r")
     lines = f.readlines()
     correct = 0
     count = 0
     cumul_freq = 0
     correct_freq = 0
     miss = 0
     known_words = 0
     known_lemmata = 0
     known_word_correct = 0
     known_lemma_correct = 0
     lemma_known_word_unknown = 0
     lemma_known_word_unknown_correct = 0
     for line in lines:
       line = line.strip()
       lemma = re.sub("\s.*", "", line)
       wp = re.sub("^.*?\s+", "", line)
       wp = re.sub("^[(]|[)]$","",wp)
       word = re.sub(" .*","",wp)
       count += 1
       ok = False
       situation =  {}
       if not wp in self.has_lemma:
         miss += 1
         #print(f"Missing: lemma={lemma}, wp=<{wp}>", file=sys.stderr)
       else:
         if self.patch:
           guess_from_training = self.choose_training_lemma(wp)
           if guess_from_training is not None:
             lemma = guess_from_training

         possible_lemmata = list(map(lambda x: x[0], self.has_lemma[wp]))
         freqs = list(map(lambda x: x[1], self.has_lemma[wp]))
         freq = sum(freqs)
         cumul_freq += freq
         truth = ';'.join(possible_lemmata)

         plausible = list(filter(lambda x : x in self.known_lemmata , possible_lemmata))

         if lemma in possible_lemmata: # dit is nog steeds veel te optimistisch
            ok = True
            correct += 1
            chosen = filter(lambda x: x[0] == lemma, self.has_lemma[wp])
            # print(list(chosen))
            chosen_freq =  sum(map(lambda x: x[1], chosen))
            correct_freq +=  chosen_freq # dit anders...
         if word in self.known_words:
            if ok: known_word_correct += 1
            known_words += 1
            situation['known_word'] = True
         if len(plausible) > 0:
            situation['known_lemma'] = True
            if ok: known_lemma_correct += 1
            known_lemmata += 1
            if not word in self.known_words:
              lemma_known_word_unknown += 1
              if ok:
                lemma_known_word_unknown_correct +=1 
         if self.log_all:
           s = ', '.join(list(situation.keys()))
           if ok:
             print(f"+{line}  ({s})")
           else:
             print(f"-{line} !{truth} ({s})")
     unknown_words = count - known_words
     unknown_lemmata = count - known_lemmata
     unknown_word_correct = correct - known_word_correct
     unknown_lemma_correct = correct - known_lemma_correct
     print(
       f"\n###### type level evaluation, patch with most frequent lemma from training data: {self.patch} #####\n"
       f"All: {count} types, type level score: {correct/count}, missed: {miss}\n"
       f"All: {cumul_freq} tokens, token level score: {correct_freq/cumul_freq}, missed: {miss}\n"
       f"Known word:\t\t\t {known_word_correct/known_words} of {known_words}\n" 
       f"Unknown word:\t\t\t {unknown_word_correct/unknown_words} of {unknown_words}\n"
       f"Unknown word, known lemma:\t {lemma_known_word_unknown_correct/lemma_known_word_unknown} of {lemma_known_word_unknown}\n"
       f"Known lemma:\t\t\t {known_lemma_correct/known_lemmata} of {known_lemmata}\n"
       f"Unknown lemma:\t\t\t {unknown_lemma_correct/unknown_lemmata} of {unknown_lemmata}\n"
     )

  def eval(self):
     self.read_training()
     self.read_test_goldstandard()
     self.patch = False
     self.read_output()
     self.patch = True
     self.read_output()

# ../data/tagging/gys_training_lex.txt ../data/tagging/gys_dev_lex.txt /tmp/gystest.out
if __name__ == '__main__':
  [training, test, output] = [sys.argv[1], sys.argv[2], sys.argv[3]]  
  t = LemEval(training, test, output) 
  t.eval()
