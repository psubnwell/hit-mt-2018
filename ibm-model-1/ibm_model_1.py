import os
import argparse
import json
import math
import time
import multiprocessing
from operator import itemgetter
from copy import deepcopy
import pandas as pd

def load_corpus(corpus):
    with open(corpus, 'r') as f:
        sents = f.read().split('\n')
    sents = [sent.lower() for sent in sents]
    return sents

def build_vocab(corpus):
    sents = load_corpus(corpus)
    vocab = set()
    for sent in sents:
        for word in sent.split():
            vocab.add(word)
    return list(vocab)

# def init_t(f_vocab, e_vocab):
#     """
#     Initialize all translation probabilities.
#     The conditional probabilities are written as below:
#     t(e|f) <=> t[e][f]
#     """
#     t = {e: {f: 1/len(e_vocab)
#              for f in f_vocab}
#          for e in e_vocab}
#     return t

def init_t(f_sents, e_sents, e_vocab):
    t = {e: {} for e in e_vocab}
    for f_sent, e_sent in zip(f_sents, e_sents):
        for e in e_sent.split():
            for f in f_sent.split():
                t[e][f] = 1 / len(e_vocab)
    return t

def distance(t_1, t_2):
    delta = 0
    for e in list(t_1.keys()):
        for f in list(t_1[e].keys()):
            delta += (t_1[e][f] - t_2[e][f]) ** 2
    return math.sqrt(delta)

# def distance(t_1, t_2):
#     es = list(t_1.keys())
#     fs = list(t_1.values())[0].keys()

#     delta = 0
#     for e, f in zip(es, fs):
#         delta += (t_1[e][f] - t_2[e][f]) ** 2
#     return math.sqrt(delta)

def is_converged(t_prev, t_curr, epsilon):
    delta = distance(t_prev, t_curr)
    return delta < epsilon, delta

# def train_iter_parallel(f_sents, e_sents, f_vocab, e_vocab, t_prev, core_num):
#     t = deepcopy(t_prev)

#     # Initialize count(e|f) and total(f)
#     count = {e: {f: 0 for f in f_vocab}
#              for e in e_vocab}
#     total = {f: 0 for f in f_vocab}

#     pool = multiprocessing.Pool(core_num)
#     res = pool.map(_collect_count,
#                    list(zip(f_sents, e_sents, [t_prev] * len(f_sents))))
#     counts, totals = zip(*res)

#     for c in counts:
#         for e in c.keys():
#             for f in c[e].keys():
#                 count[e][f] += c[e][f]
#     for to in totals:
#         for f in to.keys():
#             total[f] = to[f]

#     # Estimate probabilities
#     # Eq 4.14
#     for e in t.keys():
#         for f in t[e].keys():
#             t[e][f] = count[e][f] / total[f]

#     return t

# def _collect_count(param):
#     return collect_count(*param)

# def collect_count(f_sent, e_sent, t):
#     fs = f_sent.split()
#     es = e_sent.split()

#     count = {e: {f: 0 for f in fs}
#              for e in es}
#     total = {f: 0 for f in fs}
#     s_total = {e: 0 for e in es}

#     # Compute normalization
#     # Eq 4.13 denominator part
#     for e in es:
#         s_total[e] = 0
#         for f in fs:
#             s_total[e] += t[e][f]

#     # Collect counts
#     for e in es:
#         for f in fs:
#             # Eq 4.14 numerator part
#             count[e][f] = t[e][f] / s_total[e]
#             # Eq 4.14 denominator part
#             total[f] = t[e][f] / s_total[e]
#     return count, total


def train_iter(f_sents, e_sents, f_vocab, e_vocab, t_prev):
    t = deepcopy(t_prev)

    # Initialize count(e|f) and total(f)
    count = {e: {f: 0 for f in f_vocab}
             for e in e_vocab}
    total = {f: 0 for f in f_vocab}

    for f_sent, e_sent in zip(f_sents, e_sents):
        fs = f_sent.split()
        es = e_sent.split()
        # In fact s_total is a float,
        # we make it a dict for better readability
        s_total = {e: 0 for e in e_vocab}

        # Compute normalization
        # Eq 4.13 denominator part
        for e in es:
            s_total[e] = 0
            for f in fs:
                s_total[e] += t[e][f]

        # Collect counts
        for e in es:
            for f in fs:
                # Eq 4.14 numerator part
                count[e][f] += t[e][f] / s_total[e]
                # Eq 4.14 denominator part
                total[f] += t[e][f] / s_total[e]

    # Estimate probabilities
    # Eq 4.14
    for e in t.keys():
        for f in t[e].keys():
            t[e][f] = count[e][f] / total[f]

    return t

def train(f_corpus, e_corpus, epsilon, iter_num, save_dir, save_iteration=False, save_alignment=True):
    f_sents = load_corpus(f_corpus)
    e_sents = load_corpus(e_corpus)
    f_vocab = build_vocab(f_corpus)
    e_vocab = build_vocab(e_corpus)

    t_prev = init_t(f_sents, e_sents, e_vocab)

    converged = False
    i = 0
    if save_iteration:
        output_iteration(t_prev, save_dir, i)

    while not converged and i < iter_num:
        t = train_iter(f_sents, e_sents, f_vocab, e_vocab, t_prev)
        # t = train_iter_parallel(f_sents, e_sents, f_vocab, e_vocab, t_prev, multiprocessing.cpu_count())
        converged, delta = is_converged(t_prev, t, epsilon)
        i += 1

        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) +
              '  Iteration {} finished! Delta = {}.'.format(i, delta))
        if save_iteration:
            output_iteration(t, save_dir, i)
            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) +
                  '    Iteration information saved!')
        if save_alignment:
            output_alignment(t, save_dir)
            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) +
                  '    Alignment result saved!')

        t_prev = t
    return t

def output_iteration(t, save_dir, iter_num):
    if iter_num == 0:
        e_vocab = list(t.keys())
        df = pd.DataFrame(columns=['e', 'f', str(iter_num)+' it.'], index=[0])
        for e in e_vocab:
            for f in t[e].keys():
                df.loc[df.index.max() + 1] = [e, f, t[e][f]]
        df.drop([0], inplace=True)
        df.to_csv(save_dir + 'iterations.csv', index=False)
    else:
        df = pd.read_csv(save_dir + 'iterations.csv')
        df[str(iter_num) + ' it.'] = list(map(lambda e, f: t[e][f], df['e'], df['f']))
        df.to_csv(save_dir + 'iterations.csv', index=False)

def output_alignment(t, save_dir):
    # Save all candidate alignment with probabilities.
    with open(save_dir + 'alignment_all.json', 'w') as f:
        json.dump(t, f, ensure_ascii=False)

    # Save one to one alignment without probabilities.
    res = {e: sorted(fs.items(),
                     key=itemgetter(1),
                     reverse=True)[0][0]
           for e, fs in t.items()}
    with open(save_dir + 'alignment.json', 'w') as f:
        json.dump(res, f, ensure_ascii=False)

    # Save one to one alignment with probabilities in descending order.
    res = {e: sorted(fs.items(),
                     key=itemgetter(1),
                     reverse=True)[0]
           for e, fs in t.items()}
    res = sorted(res.items(),
                 key=lambda x: x[1][1],
                 reverse=True)
    with open(save_dir + 'alignment.txt', 'w') as f:
        for e, f_prob in res:
            f.write('{}\t{}\t{}\n'.format(e, f_prob[0], f_prob[1]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--f-corpus', type=str)
    parser.add_argument('--e-corpus', type=str)
    parser.add_argument('--save-dir', type=str, default='./output/')
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--iter-num', type=int, default=10)
    parser.add_argument('--save-iteration', dest='save_iteration', action='store_true')
    parser.add_argument('--no-save-iteration', dest='save_iteration', action='store_false')
    parser.set_defaults(save_iteration=False)
    parser.add_argument('--save-alignment', dest='save_alignment', action='store_true')
    parser.add_argument('--no-save-alignment', dest='save_alignment', action='store_false')
    parser.set_defaults(save_alignment=True)
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + '  ' + 'Start!')
    train(args.f_corpus, args.e_corpus, args.epsilon, args.iter_num, args.save_dir, args.save_iteration, args.save_alignment)
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + '  ' + 'Finished!')
