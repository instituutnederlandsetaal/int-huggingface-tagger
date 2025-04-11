import re
import string

def regexsplitter(regex):
    def func(line):
        for line in re.split(regex, line):
            line = line.strip()
            if line:
                yield line

    return func


SECTION = r"([A-Z\. ]+\.)"
FULLSTOP = r"(.+?[\.!?])(?=[^0-9])"
# Never split on [ ] ' - as these can occur in words: 't, [ge]zien, ge-zien
PUNCTS = re.sub(r"[\[\]'-]", "", string.punctuation)
# Addionally, keep any type of punctuation within a word in tact (e.g. €1.000,50)
WORD = r"((?<!\S)[{}]|[{}](?!\S))".format(PUNCTS, PUNCTS)


# Even the tokenizer needs to adhere to the max_sent_len, 
# as it can cause out of memory errors.
def pie_tokenizer(text, max_sent_len=50):
    section = regexsplitter(SECTION)
    fullstop = regexsplitter(FULLSTOP)
    word = regexsplitter(WORD)

    for line in section(text):
        for sent in fullstop(line):
            sent = [w for raw in sent.split() for w in word(raw)]
            while len(sent) > max_sent_len:
                yield sent[:max_sent_len]
                sent = sent[max_sent_len:]
            if sent:
                yield sent

def pie_tokenize_sentence(line):
    word = regexsplitter(WORD)
    sent = [w for raw in line.split() for w in word(raw)]
    return list(sent)

if __name__ == "__main__":
  s = "Hallo, daar he(b) je (maar) weer..... 'een zin-ne-tje'!"
  print(list(pie_tokenizer(s,100)))
