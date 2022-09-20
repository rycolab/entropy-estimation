from conllu import parse_incr
from nltk.corpus import wordnet as wn
from collections import deque, defaultdict
from tqdm import tqdm
import entropy
from experiments import estimators, funcs
import random

# ['eng', 'als', 'arb', 'bul', 'cat', 'cmn', 'dan', 'ell', 'eus', 'fas', 'fin', 'fra', 'glg', 'heb', 'hrv', 'ind', 'ita', 'jpn', 'nld', 'nno', 'nob', 'pol', 'por', 'qcn', 'slv', 'spa', 'swe', 'tha', 'zsm']

data = {
    'hrv': [
        'hr_set-ud-train.conllu',
        'hr_set-ud-dev.conllu',
        'hr_set-ud-test.conllu'
    ],
    'heb': [
        'he_htb-ud-train.conllu',
        'he_htb-ud-dev.conllu',
        'he_htb-ud-test.conllu'
    ],
    'ell': [
        'el_gdt-ud-train.conllu',
        'el_gdt-ud-dev.conllu',
        'el_gdt-ud-test.conllu'
    ],
    'arb': [
        'ar_padt-ud-train.conllu',
        'ar_padt-ud-dev.conllu',
        'ar_padt-ud-test.conllu'
    ],
    'pol': [
        'pl_pdb-ud-train.conllu',
        'pl_pdb-ud-dev.conllu',
        'pl_pdb-ud-test.conllu'
    ],
    'spa': [
        'es_ancora-ud-train.conllu',
        'es_ancora-ud-dev.conllu',
        'es_ancora-ud-test.conllu',
        'es_gsd-ud-train.conllu',
        'es_gsd-ud-dev.conllu',
        'es_gsd-ud-test.conllu'
    ],
    'ita': [
        'it_isdt-ud-train.conllu',
        'it_isdt-ud-dev.conllu',
        'it_isdt-ud-test.conllu',
        'it_vit-ud-train.conllu',
        'it_vit-ud-dev.conllu',
        'it_vit-ud-test.conllu'
    ],
    'por': [
        'pt_gsd-ud-train.conllu',
        'pt_gsd-ud-dev.conllu',
        'pt_gsd-ud-test.conllu',
        'pt_bosque-ud-train.conllu',
        'pt_bosque-ud-dev.conllu',
        'pt_bosque-ud-test.conllu',
    ],
    'fra': [
        'fr_gsd-ud-train.conllu',
        'fr_gsd-ud-dev.conllu',
        'fr_gsd-ud-test.conllu'
    ]
}

def permutation_test(al, funcs, true, S_adj, adjs, N, perms=1000):
    data = defaultdict(lambda: defaultdict(list))
    # al: noun_lemma, gender, adj_lemma
    for x in al:
        data[x[1]][x[0]].append(x[2])
    shuffled_genders = []
    for g in data:
        for n in data[g]:
            shuffled_genders.append(g)

    orig_ents = [func(S_adj, N, adjs) for func in funcs]
    geq = [1 for func in funcs]
    for i in tqdm(range(perms - 1)):
        random.shuffle(shuffled_genders)

        i = 0
        counts = defaultdict(lambda: defaultdict(int))
        genders = defaultdict(int)
        for g in data:
            for n in data[g]:
                new_g = shuffled_genders[i]
                # print(n, g, new_g)
                genders[new_g] += len(data[g][n])
                for adj in data[g][n]:
                    counts[new_g][adj] += 1
                i += 1
                pass

        mi = [x for x in orig_ents]
        for gender in genders:
            try:
                S_cur = entropy.prob_counts(counts[gender], genders[gender])
                for num, func in enumerate(funcs):
                        mi[num] -= (genders[gender] / N) * func(S_cur, genders[gender], counts[gender])
            except Exception as e:
                print(e)
                print(S_cur)
        for num, m in enumerate(mi):
            if m >= true[num]:
                geq[num] += 1

    return [x / perms for x in geq]

animates = dict()
bfs = deque([wn.synset('living_thing.n.01')])
print('Getting list of animate wordnet entries')
while bfs:
    cur = bfs.popleft()
    if cur not in animates:
        animates[cur] = True
        for synset in cur.hyponyms():
            bfs.append(synset)
print('Done')


for lang in data:
    print(f'LANGUAGE: {lang}')
    al = []
    counts = defaultdict(lambda: defaultdict(int))
    genders = defaultdict(int)
    adjs = defaultdict(int)
    N = 0
    for f in data[lang]:
        file = f'data/williams/{lang}/{f}'
        with open(file, 'r') as fin:
            for sent in tqdm(parse_incr(fin)):
                sent = sent.filter(id=lambda x: type(x) is int)
                for word in sent:
                    if word['deprel'] == 'amod':
                        try:
                            lemma = word['lemma']
                            head = sent[word['head'] - 1]
                            if head['upos'] != 'NOUN':
                                continue
                            is_animate = False
                            head_lemma = head['lemma']
                            a = wn.synsets(head_lemma, lang=lang, pos=wn.NOUN)
                            # print(head_lemma, a)
                            for synset in a:
                                if synset in animates:
                                    is_animate = True
                                    break
                            if not is_animate and a:
                                gender = head['feats'].get('Gender', word['feats']['Gender'])
                                genders[gender] += 1
                                counts[gender][lemma] += 1
                                adjs[lemma] += 1
                                # print([head_lemma, gender, lemma])
                                al.append([head_lemma, gender, lemma])
                                N += 1
                        except Exception as e:
                            pass

    # MI = H(adjectives) - Σₓ p(gender=x) H(adjectives|gender=x)
    S_gender = entropy.prob_counts(genders, N)
    S_adj = entropy.prob_counts(adjs, N)
    true_mis = []
    for num, func in enumerate(funcs):
        H_adj = func(S_adj, N, adjs)
        H_gender = func(S_gender, N, genders)
        print(H_adj, H_gender, N)
        mi = H_adj
        divide = min(H_adj, H_gender)
        for gender in genders:
            S_cur = entropy.prob_counts(counts[gender], genders[gender])
            mi -= (genders[gender] / N) * func(S_cur, genders[gender], counts[gender])
        print(estimators[num], mi, mi / divide)
        true_mis.append(mi)
    print(permutation_test(al, funcs, true_mis, S_adj, adjs, N))