from tqdm import tqdm
import entropy
from experiments import estimators, funcs
import random
import csv
from collections import Counter, defaultdict
import glob
import unidecode
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import ward, fcluster
import matplotlib.pyplot as plt
import numpy as np

def cond_H(p, q, funcs, counts_q, N_q):
    res = [0 for func in funcs]
    for val in counts_q:
        sample = []
        for i in range(len(q)):
            if q[i] == val:
                sample.append(p[i])
        for i, func in enumerate(funcs):
            S, N, counts = entropy.prob(sample)
            est = func(S, N, counts)
            res[i] += (counts_q[val] / N_q) * est
    return res

def mi(p, q, funcs):
    S_p, N_p, counts_p = entropy.prob(p)
    S_q, N_q, counts_q = entropy.prob(q)
    res = [func(S_p, N_p, counts_p) for func in funcs]
    sub = cond_H(p, q, funcs, counts_q, N_q)
    for i in range(len(funcs)):
        res[i] -= sub[i]
    if res[0] < 0:
        print(sub)
        input()
    return res

def vi(p, q, funcs):
    S_p, N_p, counts_p = entropy.prob(p)
    S_q, N_q, counts_q = entropy.prob(q)
    res = cond_H(p, q, funcs, counts_q, N_q)
    add = cond_H(q, p, funcs, counts_p, N_p)
    for i in range(len(funcs)):
        res[i] += add[i]
    return res

def permutation_test(p, q, true_mi, true_vi, funcs, perms=1000):
    p = [x for x in p]
    q = [x for x in q]
    gq_mi, lq_vi = [1 for func in funcs], [1 for func in funcs]
    for i in tqdm(range(perms - 1)):
        p = np.random.permutation(p)
        q = np.random.permutation(q)
        mi_t = mi(p, q, funcs)
        vi_t = vi(p, q, funcs)
        for num, func in enumerate(funcs):
            if mi_t[num] >= true_mi[num]: gq_mi[num] += 1
            if vi_t[num] <= true_vi[num]: lq_vi[num] += 1
    return [x / perms for x in gq_mi], [x / perms for x in lq_vi]


# print(mi([0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], funcs))
# print(mi([0, 0, 0, 1, 1, 1], [0, 0, 1, 0, 0, 1], funcs))
# print(mi([0, 0, 0, 1, 1, 1], [0, 0, 1, 0, 1, 1], funcs))
# input()
LS = len('data/mccarthy/gender-dictionaries/')
fout = open('logs/mccarthy.txt', 'w')

genders = defaultdict(dict)
for file in glob.glob('data/mccarthy/gender-dictionaries/*'):
    print(file)
    with open(file, 'r') as fin:
        for row in fin:
            word, gender = row.strip().split('\t')
            genders[file[LS:LS+2]][word] = gender

swadesh = ['hr', 'sk', 'uk', 'ru', 'bg', 'fr', 'ca', 'it', 'es', 'pt']
northeuralex = ['he', 'el', 'hi', 'lt', 'lv', 'hr', 'sk', 'uk', 'ru', 'bg', 'sv', 'da', 'fr', 'ca', 'it', 'es', 'pt']
res = defaultdict(list)
missing = []

dropped = 0
with open('data/mccarthy/inanimate_swadesh.csv', 'r') as fin:
    reader = csv.DictReader(fin)
    for i, row in enumerate(reader):
        # print(row)
        for lang in row:
            if lang not in swadesh:
                continue
            word = row[lang].split(',')[0].strip()
            if word not in genders[lang]:
                t = unidecode.unidecode(word)
                if t not in genders[lang]:
                    missing.append([lang, word])
                    print([lang, word])
                else:
                    res[lang].append(genders[lang][t])
            else:
                res[lang].append(genders[lang][word])

# with open('data/mccarthy/inanimate_northEuraLex.tsv', 'r') as fin:
#     reader = csv.DictReader(fin, delimiter='\t')
#     for i, row in enumerate(reader):
#         # print(row)
#         for lang in row:
#             if lang not in swadesh:
#                 continue
#             word = row[lang].split(',')[0].strip()
#             if word not in genders[lang]:
#                 t = unidecode.unidecode(word)
#                 if t not in genders[lang]:
#                     missing.append([lang, word])
#                     print([lang, word])
#                 else:
#                     res[lang].append(genders[lang][t])
#             else:
#                 res[lang].append(genders[lang][word])

print(len(missing))
print(Counter([x[0] for x in missing]))

langs = list(res.keys())
final = []
for num1, lang1 in enumerate(langs):
    # l = []
    for num2, lang2 in enumerate(langs[num1 + 1:]):
        mi_c = mi(res[lang1], res[lang2], funcs)
        vi_c = vi(res[lang1], res[lang2], funcs)
        mi_stats, vi_stats = permutation_test(res[lang1], res[lang2], mi_c, vi_c, funcs)
        fout.write(f'{lang1} vs. {lang2}:\n')
        for i in range(len(funcs)):
            fout.write(f'{estimators[i]}:\t{mi_c[i]}\t({mi_stats[i]})\t{vi_c[i]}\t({vi_stats[i]})\n')
        print(lang1, lang2, mi_c, vi_c)
        # l.append(mi_c[0])
        final.append([x for x in mi_c])
    # final.append(l)

# clustering = AgglomerativeClustering(linkage='average', affinity='precomputed').fit(final)
# Z = ward([x[0] for x in final])

fout.close()

print(langs)
plt.figure()
# dn = hierarchy.dendrogram(Z, labels=langs)
fig, axes = plt.subplots(2, 3, figsize=(8, 3))
for i in range(6):
    Z = ward([x[i] for x in final])
    axes[i // 3][i % 3].title.set_text(estimators[i])
    dn = hierarchy.dendrogram(Z, ax=axes[i // 3][i % 3], labels=langs)
plt.show()
