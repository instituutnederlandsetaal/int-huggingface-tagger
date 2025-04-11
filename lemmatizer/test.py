from transformers import AutoTokenizer, T5ForConditionalGeneration
import re
import sys

def  chunk_text(text, tokenizer, size):

    # print(text)
    print(str(tokenizer.model_max_length))
    sentences = text

    chunks = []

    chunk = ''

    length = 0

    for sentence in sentences:
        tokenized_sentence = tokenizer.encode(sentence, truncation=False, max_length=None, return_tensors='pt')[0]

        if len(tokenized_sentence) > size:
            print("Too long, truncating" + str(tokenized_sentence))
            sentence = sentence[:size]
            #continue

        length += len(tokenized_sentence)

        if length <= size:
            chunk = chunk + sentence
        else:
            if len(chunk.strip()) > 0:
                chunks.append(chunk.strip())
            chunk = sentence
            length = len(tokenized_sentence)

    if len(chunk.strip()) > 0:
        chunks.append(chunk.strip())

    return chunks


example_sentence = "Ben algoritme dat op ba8i8 van kunstmatige inte11i9entie vkijwel geautomatiseerd een tekst herstelt met OCR fuuten."
havelaartje="Toenikeendagdaarnavandebeurskwam,zeiFritsdateriemandkeweestwasommytespreken.NaardebeschryvingwashetdeSjaalman.Hoehymegevondenhad...nuja,'tadreskaartje!Ikdachterover,mynkinderenvanschooltenemen,wanthetislastig,nogtwintig,dertigjarenlatertewordennagezetendooreenschoolkameraaddieeensjaaldraagtinplaatsvaneenjas,endienietweethoelaathetis.OokhebikFritsverbodennaardeWestermarkttegaan,alserkramenstaan."
normaal_zinnetje="Gelukkigisditeenerggewoonzinnetje.Fijntoch?"
engels_zinnetje="Iwouldnotknowwhetherthiscouldwork.Wouldit?"
volkskrantje="In een wetenschapstak waar sombere berichten de maat zijn, is het zowaar een lichtpuntje. Als alle landen zich houden aan hun klimaatbeloftes, kan de opwarming van de aarde écht beperkt blijven tot net onder de 2 graden Celsius ten opzichte van de pre-industriële tijd."
wnt="Volgens Oudgermaansch gebruik beschouwd als het begin van een nieuwen dag; thans alleen nog in samenstellingen (zie hieronder)."
dorine="Het stortregende en Dorine van Lowe was doodmoê, toen zij, dien middag, vóor het diner nog even bij Karel en Cateau aanwipte, maar Dorine was tevreden over zichzelve. Zij was na het lunch dadelijk uitgegaan en had geheel Den Haag doortrippeld en doortramd; zij had veel bereikt zoo niet alles en haar vermoeide gezicht stond heel blij en hare levendige zwarte oogen flonkerden."

model_name = sys.argv[2]
tokenizer = AutoTokenizer.from_pretrained(sys.argv[3])

def get_text(f):
   with open(f,'r') as file:
     z = file.read()
     return z


gt="geiteklederen NOU-C; werckten VRB; werckten NOU; saghen VRB" # get_text('../tagging/zieltjes.txt')
gt=get_text(sys.argv[1])
chunks=re.split("\n",gt)
#print(chunks)

model = T5ForConditionalGeneration.from_pretrained(model_name)

#print("start generating")

for chunk in chunks:
   fields = re.split("\t", chunk)
   word = fields[0]
   word_pos = fields[0] + " " + fields[1]
   lemma = fields[2]
   model_inputs = tokenizer(word_pos, max_length=128, truncation=True, return_tensors="pt")
   outputs = model.generate(**model_inputs, max_length=1000)
   #l = len(outputs)
   #for i in range(0,l):
       #x = tokenizer.decode(outputs[i])
       #print(f"{i} {x}", file=sys.stderr)
   res = re.sub("<pad>|</s>", "", tokenizer.decode(outputs[0]))
   #print(f"Hypothesis: {res}\nGT:{gt}")
   print(f"{word} -> {res}\t\t\t\t({chunk})")

