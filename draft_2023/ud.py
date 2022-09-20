"""
Experiment on structured entropy prediction using Markov model of POS tags from UD.
"""

import conllu
from tqdm import tqdm
from collections import defaultdict

TERMINAL = '$'

def get_pos_transitions(file):
    """Get observed Markov transitions between POS tags from a conllu treebank."""
    
    transitions = defaultdict(lambda: defaultdict(int))
    with open(file, 'r') as f:
        for sent in tqdm(conllu.parse_incr(f)):
            # filter out multi-word tokens, then collect pos tags
            pos = [token['upos'] for token in sent if type(token['id']) is int] + [TERMINAL]
            # collect transitions
            for i in range(1, len(pos)):
                transitions[pos[i - 1]][pos[i]] += 1
    return transitions

def main():
    pos = get_pos_transitions('data/es_ancora-ud-train.conllu')
    print(pos)

if __name__ == '__main__':
    main()