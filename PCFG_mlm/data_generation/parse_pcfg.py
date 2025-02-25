import nltk
import pickle
import os

# NOTE: change your path here
REPO_DIR = '.'
#######

PCFG_DEF_DIR = os.path.join(REPO_DIR, 'PCFG_def')


cfg3f = """
22|->20 21
22|->20 19 21
22|->21 19 19
22|->20 20
 19|->18 16 18
 19|->17 18
 19|->18 18
 20|->16 16
 20|->16 17
 20|->17 16 18
 21|->18 17
 21|->17 16
 21|->16 17 18
 21|->16 18
 16|->15 15
 16|->13 15 13
 16|->14 13
 16|->14 14
 17|->15 14 13
 17|->14 15
 17|->15 14
 18|->14 15 13
 18|->15 13 13
 18|->13 15
 13|->11 12
 13|->12 11 12
 13|->10 12 11
 14|->10 12
 14|->12 10 12
 14|->12 11
 14|->10 12 12
 15|->10 11 11
 15|->11 11 10
 15|->10 10
 15|->12 12 11
 10|->8 9 9
 10|->9 7 9
 10|->7 9 9
 11|->8 8
 11|->9 7
 11|->9 7 7
 12|->7 9 7
 12|->9 8
 12|->8 8 9
 7|->2 2 1
 7|->3 2 2
 7|->3 1 2
 7|->3 2
 8|->3 1 1
 8|->1 2
 8|->3 3 1
 9|->1 2 1
 9|->3 3
 9|->1 1
""" 



cfg3b = """
22|->21 20
22|->20 19
19|->16 17 18
19|->17 18 16
20|->17 16 18
20|->16 17
21|->18 16
21|->16 18 17
16|->15 13
16|->13 15 14
17|->14 13 15
17|->15 13 14
18|->15 14 13
18|->14 13
13|->11 12
13|->12 11
14|->11 10 12
14|->10 11 12
15|->12 11 10
15|->11 12 10
10|->7 9 8
10|->9 8 7
11|->8 7 9
11|->7 8 9
12|->8 9 7
12|->9 7 8
7|->3 1
7|->1 2 3
8|->3 2
8|->3 1 2
9|->3 2 1
"""



cfg3i = """
22|->19 19 20
22|->21 20 19
19|->18 16 18
19|->16 16
20|->17 16 17
20|->18 18
21|->16 16 18
21|->18 17
16|->13 13
16|->14 14
17|->15 15
17|->15 14
18|->14 15 13
18|->14 15
13|->12 11
13|->10 12 11
14|->10 10 10
14|->10 10
15|->11 11 10
15|->11 10 12
10|->8 7 7
10|->9 9
11|->7 7 7
11|->7 7 8
12|->7 9 9
12|->8 7
7|->3 1 2
7|->2 3 1
8|->1 1
8|->2 2
9|->1 1 3
9|->1 2
"""

cfg3h="""
22|->20 20 21
22|->19 21
19|->16 17
19|->18 17
20|->18 16
20|->17 16
21|->17 17 18
21|->17 18 17
16|->14 13
16|->15 13
17|->13 14
17|->15 13 15
18|->15 13 13
18|->15 14 14
18|->14 15 15
13|->12 11
13|->11 10
14|->10 12 12
14|->10 10
14|->12 12 10
15|->10 12
15|->11 11 10
10|->8 7 9
10|->9 7
10|->8 8
11|->8 7 7
11|->7 7
11|->7 9 9
12|->7 9
12|->8 7
12|->9 8
7|->2 3 2
7|->1 2 3
7|->1 3 1
8|->1 2
8|->3 3 1
8|->1 3
9|->2 1 3
9|->1 3 3
"""

cfg3g="""
22|->19 20
22|->20 20 19
22|->20 19 21
19|->17 17 16
19|->18 17 16
19|->18 16 17
20|->16 17
20|->18 18
20|->16 17 17
21|->16 16
21|->16 16 18
21|->18 16
16|->14 13 13
16|->13 14
16|->13 13
17|->14 13 14
17|->14 15 13
17|->15 14
18|->15 13
18|->15 15
18|->14 13 15
13|->10 12
13|->11 11 11
13|->11 11
14|->11 12
14|->10 11 10
14|->10 10
15|->10 11
15|->12 10 10
15|->12 11
10|->8 8 8
10|->7 7 7
10|->7 7
11|->8 8 9
11|->9 7
11|->8 9 7
12|->7 9
12|->7 8
12|->9 9 9
7|->2 3 1
7|->1 1
7|->2 2
8|->1 3 2
8|->1 3
8|->3 3 1
9|->2 3 3
9|->2 3
9|->2 1
"""

orig_str = cfg3i
replaced_str = orig_str.replace("|->", " -> ")
# replace each number with a letter, by ord 
replace_dict = {1:'a', 2:'b', 3:'c', 7:'d', 8:'e', 9:'f', 10:'g', 11:'h', 12:'i', 13:'j', 14:'k', 15:'l', 16:'m', 17:'n', 18:'o', 19:'p', 20:'q', 21:'r', 22:'s', 23:'t', 24:'u', 25:'v', 26:'w', 27:'x', 28:'y', 29:'z'}

for i in range(26, -1, -1):
  if i in replace_dict:
    replaced_str = replaced_str.replace(str(i), replace_dict[i])
print(replaced_str)

# count the number of rules per non-terminal, and assign equal probabilities to each rule
prob_dict = {}
for i in range(26, -1, -1):
  if i in replace_dict:
    match = f"{replace_dict[i]} ->"
    cnt = replaced_str.count(match)
    if cnt == 0:
      continue
    prob = round(1/cnt, 3)
    probs = [prob] * (cnt-1)
    probs += [1-sum(probs)]
    prob_dict[replace_dict[i]] = probs


# gather the rules
clause_dict = {}

start_symbol = None 
for line in replaced_str.split("\n"):
  if line == "":
    continue
  lhs, rhs = line.split(" -> ")
  lhs = lhs.strip()
  if start_symbol is None:
    start_symbol = lhs
  if lhs not in clause_dict:
    clause_dict[lhs] = []
  clause_dict[lhs].append(rhs)


# assign probabilities to the rules

rules = []
for lhs in clause_dict:
  clauses = clause_dict[lhs]
  probs = prob_dict[lhs]
  clauses_with_probs = []
  for clause, prob in zip(clauses, probs):
    curr_clause_prob = f"{clause} [{prob:.3f}]"
    clauses_with_probs.append(curr_clause_prob)
  rule = f"{lhs} -> " + " | ".join(clauses_with_probs)
  rules.append(rule)

rule_str = "\n".join(rules)
# add terminals to the rules
# NOTE: there are only 3 terminals
rule_str += "\n"
rule_str += "a -> 'a' [1.000]\n"
rule_str += "b -> 'b' [1.000]\n"
rule_str += "c -> 'c' [1.000]\n"
print(rule_str)



grammar = nltk.PCFG.fromstring(rule_str)
char2int = {'a':0, 'b':1, 'c':2}
char2int['BOS'] = len(char2int)
char2int['EOS'] = len(char2int)
char2int['PAD'] = len(char2int)
char2int['MASK'] = len(char2int)

int2char = {v:k for k,v in char2int.items()}


pcfg_def = {
  'grammar': grammar,
  'map': char2int,
  'BOS': char2int['BOS'],
  'start_symbol': start_symbol,
  'vocab_size': len(char2int),
}
pcfg_def['reverse_map'] = int2char 
pcfg_def['shift'] = len(char2int) / 2
pcfg_def['scale'] = 2 / len(char2int)



fname = os.path.join(PCFG_DEF_DIR, 'yuanzhi_cfg3i.pkl')
with open(fname, 'wb') as f:
  pickle.dump(pcfg_def, f)