import os
import argparse
import random
import json
from operator import itemgetter

def load_corpus(corpus):
    with open(corpus, 'r') as f:
        sents = f.read().split('\n')
    sents = [sent.lower().split() for sent in sents]
    return sents

def load_alignment(alignment):
    with open(alignment, 'r') as f:
        alignment = json.load(f)
    return alignment

def is_aligned(A, f_ind, sent, e_start, e_end):
    f_align = []
    for (e, f) in A:
        if f == f_ind:
            f_align.append(e)
    if f_ind < 1 or f_ind > len(sent):
        return False
    if len(f_align) > 0 and (min(f_align) < e_start or max(f_align) > e_end):
        return False
    return True

def extract(f_start, f_end, e_start, e_end, f_sent, e_sent, A):
    if f_end == 0:
        return []
    # Check if alignment points violate consistency
    for (e, f) in A:
        if (e < e_start or e > e_end) and f_start <= f <= f_end:
            # Notice the textbook pseudo-code line 4 lacks the last condition,
            # the algorithm will return wrong result (full sentence pairs) without it.
            return []
    # Add phrase pairs (incl. additional unaligned f)
    E = []
    f_s = f_start
    while True:
        f_e = f_end
        while True:
            e_phrase = ' '.join(e_sent[e_start-1: e_end])
            f_phrase = ' '.join(f_sent[f_s-1: f_e])
            E.append((e_phrase, f_phrase))
            f_e += 1
            if not is_aligned(A, f_e, f_sent, e_start, e_end):
                break
        f_s -= 1
        if not is_aligned(A, f_s, f_sent, e_start, e_end):
            break
    return E

def phrase_extraction(f_sent, e_sent, A):
    BP = []
    for e_start in range(1, len(e_sent) + 1):
        for e_end in range(e_start, len(e_sent) + 1):
            # Find the minimally matching foreign phrase
            f_start, f_end = len(f_sent), 0
            for (e, f) in A:
                if e_start <= e <= e_end:
                    f_start = min(f, f_start)
                    f_end = max(f, f_end)
            BP += extract(f_start, f_end, e_start, e_end, f_sent, e_sent, A)
    return BP

def demo():
    f_sent = 'michael geht davon aus , dass er im haus bleibt'.split()
    e_sent = 'michael assumes that he will stay in the house'.split()
    A = [(1,1), (2,2), (2,3), (2,4), (3,6), (4,7), (5,10), (6,10), (7,8), (8,9), (9,9)]

    phrases = phrase_extraction(f_sent, e_sent, A)
    print(phrases)
    print('Total {} phrases.'.format(len(phrases)))

def main(f_corpus, e_corpus, alignment, save_dir):
    f_sents = load_corpus(f_corpus)
    e_sents = load_corpus(e_corpus)
    alignment = load_alignment(alignment)
    extracted_phrases = []
    for f_sent, e_sent in zip(f_sents, e_sents):
        # Create a specific alignment for sentence pair.
        A = {}
        for e in e_sent:
            d = {f: alignment[e][f] for f in f_sent if f in alignment[e]}
            f = max(d, key=d.get)
            A[e_sent.index(e) + 1] = f_sent.index(f) + 1
        A = sorted(A.items(), key=itemgetter(0))
        # Extract phrases
        extracted_phrases += phrase_extraction(f_sent, e_sent, A)
    with open(save_dir + 'phrases.txt', 'w') as f:
        for e_phrase, f_phrase in extracted_phrases:
            f.write(e_phrase + '  --  ' + f_phrase + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--f-corpus', type=str)
    parser.add_argument('--e-corpus', type=str)
    parser.add_argument('--alignment', type=str)
    parser.add_argument('--save-dir', type=str, default='./output/')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    main(args.f_corpus, args.e_corpus, args.alignment, args.save_dir)
