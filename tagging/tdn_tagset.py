import re
import sys
'''
Ad-hoc code used to tag Corpus Gysseling in 'compatibility mode' (Assign missing TDN core features (degree,position,WF,representation) to be compatible with Gysseling tags)
'''

tdn_core_features = {
    "NOU-C" : ["number", "WF"],
    "NOU-P" : ["WF"],
    "AA" : ["degree", "position", "WF"],
    "ADV" : ["type", "WF"],
    "VRB" : ["finiteness", "tense", "WF"],
    "NUM" : ["type", "position", "representation", "WF"],
    "PD" : ["type", "subtype", "position", "WF"],
    "ADP" : ["type", "WF"],
    "CONJ" : ["type", "WF"],
    "INT" : ["WF"],
    "RES" : ["type", "WF"]
  } 

def not_less_specific(v1, v2):
  return (v1==v2 or v1=='uncl' or re.match(".*[|].*",v1)) and not (v2=='uncl') # deugt niet helemaal
    
def parse_features(tag):
      features = {}
      pos_no_feats = re.sub("\(.*?\)", "", tag)
      for m in re.finditer(r"([a-zA-Z]+)=([^=,()]+)", tag):
         feature = m.group(1)
         value = m.group(2)
         #print(f"{feature} {value}")
         features[feature]=value
      return [pos_no_feats,features]

def main_pos_compatible(pos, main_pos):
      pos_no_feats = re.sub("\(.*?\)", "", pos)
      main_pos_no_feats = re.sub("\(.*?\)", "", main_pos)
      # print(f"{pos_no_feats} {main_pos_no_feats}")
      return (pos_no_feats == main_pos_no_feats)


def compatible_with_features_part(old_pos,new_pos):
      [old_main,old_features] = parse_features(old_pos)
      [new_main,new_features] = parse_features(new_pos)

      if old_main != new_main:
         return False
      else:
     
         for f in old_features.keys():
            if f in tdn_core_features[old_main]:
     
              old_value = old_features[f]
     
              if f in new_features:
                 if f in ["type", "position", "degree"]:
                   return True # weakly compatible: new must have all core features present in old
                 new_value = new_features[f]
                 return old_value == new_value or not_less_specific(old_value,new_value)   
              else:
                 return False
            #print("{f} is not a core feature for {old_main}")
      return True      

def compatible_with_features(old_pos,new_pos):
   
    old_parts = list(re.split("[+]", old_pos))
    new_parts =  list(re.split("[+]", new_pos))
    return len(old_parts) == len(new_parts) and all(compatible_with_features_part(o,n) for [o,n]  in zip(old_parts,new_parts))

def transfer_features_part(old_pos, new_pos):
      old={}
      new={}
      change={}

      adapted_pos=old_pos
      [old_main,old] = parse_features(old_pos)
      [new_main,new] = parse_features(new_pos)

      for k in new.keys():
         if k in old and (old[k]=='uncl' or re.match(".*[|].*", old[k])):
            change[k] = new[k]
            replace=f"{k}={new[k]}"
            search=f"{k}=[^=,()]+"
            adapted_pos=re.sub(search, replace, adapted_pos)

      return adapted_pos

def transfer_features(old_pos, new_pos):
      old_posses = list(re.split("[+]", old_pos))
      new_posses = list(re.split("[+]", new_pos))
      adapted_posses = []
      if len(old_posses) == len(new_posses):
        for i in range(0,len(old_posses)):
          adapted_pos = transfer_features_part(old_posses[i],new_posses[i])
          adapted_posses.append(adapted_pos)
        return "+".join(adapted_posses)
      else:
        return old_pos

def set_feature(pos, f, v):
   find = f"({f})=([^=,()]+)"
   replace = f"{f}={v}"
   return re.sub(find,replace,pos)

def remove_bad_features(pos):
   bad = ["WF", "degree"]
   for b in bad:
      pos = set_feature(pos, b, "uncl")
   return pos

def strip_features(pos):
   #print("Stripping:" + pos)
   parts = list(re.split("[+]", pos))
   new_parts = []
   for p in parts:
      [pos_part,features] = parse_features(p)
      toomuch = list(filter(lambda k: (k == 'WF') or (not k in tdn_core_features[pos_part]), features.keys())) # Delete WF, the assignment is very bad
      
      if (len(toomuch) == 0):
         new_parts.append(p)
         continue
      find = f"({'|'.join(toomuch)})=([^=,()]+)" 
      replace = ""
      pstripped = re.sub(find,replace,p)
      p1 = re.sub(",,",",",pstripped)
      p2 = re.sub("[(],","(",p1)
      p3 = re.sub(",[)]",")",p2)
      new_parts.append(p3)
   stripped = "+".join(new_parts)
   return stripped

def guess_degree(token):
   type = token['token']
   lemma = token['lemma'] if 'lemma' in token else 'no_lemma'
   pos = token['ref_pos'] if 'ref_pos' in token else 'no_pos'

   if ('ref_pos' in token and 'lemma' in token and pos.startswith('AA') and not lemma.endswith('ER') and re.match(".*r[e]?[n]?$", type)):
      print(f"Heel misschien comp: {type} {pos} {lemma}")


def tdn_choose_compatible_alternative(token, r):
        guess_degree(token)
        alternatives = r['alternatives']
        best_pos = r['entity']
        if 'ref_pos' in token:
          pos_old = token['ref_pos']
          compatible = list(filter(lambda x: compatible_with_features(pos_old, x[0]), alternatives))
          if len(compatible) > 0:
             best_compatible = compatible[0][0] 
             
             #if (best_compatible != best_pos):
             #   print(f"{token['token']}/{pos_old}: {best_pos} -- {best_compatible}")
             # adapted_pos = transfer_features(pos_old,best_compatible)
             return remove_bad_features(strip_features(best_compatible))
             #return f"{pos_old}\t{best_compatible}"
          else:
           stripped = strip_features(pos_old)
           if pos_old != "no_pos":
             #print(f"No compatible alternative found for {pos_old}, stripped to {stripped}!")
             pass
           return stripped
        return best_pos

def test_compatibility():
  dir = "/media/proj/Corpora/TrainingDataForTools/CobaltExport/2024/TagStuff/"
  gystags = dir + "gystagjes.txt"
  tdntags = dir + "used_tags.txt"
  gystaglist = map(lambda x: x.strip(), open(gystags,"r").readlines())
  tdntaglist = list(map(lambda x: x.strip(), open(tdntags,"r").readlines()))

  for gystag in gystaglist:
    if gystag.startswith('NOU-C') and re.match(".*[+].*",gystag) is None:
      compatible = list( filter(lambda t: compatible_with_features(gystag,t), tdntaglist) )
      print(f"{gystag} {compatible} {list(map(lambda x: strip_features(x), compatible))}")

if __name__ == "__main__":
    #parse_features("NOU(type=common)")
    test_compatibility() 
  
