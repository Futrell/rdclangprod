""" Mandarin classifiers """
import sys
import math
import random
from collections import Counter

import numpy as np
import pandas as pd
import scipy.special
import scipy.optimize
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer
from plotnine import *

from generalmodel import EMPTY, policy, SimpleFixedLengthListener, entropy

# ideas
# redo the below with a full vocabulary -- restricted experimental vocabulary is nonrepresentative.
# there is epsilon funniness -- more common classifiers have stronger differentials (ok?) and if epsilon is very small the generic classifier is never used -- resolve this by setting epsilon very small

FREQ_COL = 'most_produced_noun_frequency_sogouw'
SPEC_COL = 'preferred_spec_cl'
NOUN_COL = 'most_freq_used_noun'
CORPUS_NOUN_COL = 'noun'
CORPUS_FREQ_COL = 'log_noun_freq'
CORPUS_SPEC_COL = 'specl'
CORPUS_PATH = 'classifiers/heldout_data_complete_20171128.csv'

DISCOUNT = 1

EPSILON = 2 ** -10

def empirical_classifier_lang(d, spec_col=SPEC_COL, noun_col=NOUN_COL):
    classifier_set = set(d[spec_col])
    num_classifiers = len(classifier_set)
    num_nouns = len(d)
    lang = []
    vocab = {'个': 0}
    for c, classifier in enumerate(classifier_set):
        vocab[classifier] = c+1
    for n, noun in enumerate(d[noun_col]):
        vocab[noun] = n+num_classifiers+1
    N = len(vocab)
    for noun, classifier in zip(d[noun_col], d[spec_col]):
        noun_str = str(vocab[noun]).zfill(math.ceil(math.log10(N)))
        classifier_str = str(vocab[classifier]).zfill(math.ceil(math.log10(N)))
        lang.append([('0', noun_str), (classifier_str, noun_str)])
    return lang, vocab

def random_classifier_lang(num_nouns=100, num_classifiers=2, seed=0):
    """ Language where every noun can go with a generic classifier, or
    one of `num_classifiers` specific classifiers, evenly distributed. """
    assert num_nouns % num_classifiers == 0
    nouns_per_classifier = num_nouns // num_classifiers
    vocab = {'ge': 0}
    
    nouns = ['N' + str(n).zfill(math.ceil(math.log10(num_nouns))) for n in range(num_nouns)]
    for n, noun_str in enumerate(nouns):
        vocab[noun_str] = 1 + num_classifiers + n
        
    classifiers = []
    for c in range(num_classifiers):
        classifier_str = 'C'+str(c).zfill(math.ceil(math.log10(num_classifiers)))
        classifiers.extend([classifier_str] * nouns_per_classifier)
        vocab[classifier_str] = c + 1
        
    random.seed(seed)
    random.shuffle(classifiers)
    
    lang = [
        [
            (vocab['ge'], vocab[noun_str]),
            (vocab[classifier_str], vocab[noun_str])            
        ]
        for noun_str, classifier_str in zip(nouns, classifiers)
    ]

    return lang, vocab
        
def analyze_classifier_lang(lang, p_g, gamma=1, alpha=1, epsilon=EPSILON, **kwds):
    """
    Predicts observed frequency effect at low gamma (~1), inverse at high gamma (~2), because for
    low-frequency words, there is very high uncertainty about the right classifier.
    Alpha needs to be very high.
    Next step: Try to get predictions using the real nouns.
    """
    classifiers = np.array(lang)[:, -1, 0].astype(int)
    num_classifiers = len(set(classifiers))
    R = SimpleFixedLengthListener.from_strings(lang).R(epsilon=epsilon)
    lnp = policy(R, p_g=p_g, gamma=gamma, alpha=alpha, **kwds)
    specific_entropy = entropy(lnp[:, EMPTY, 1:num_classifiers+1], -1) # entropy over specific classifier
    lnp_specific = np.take_along_axis(lnp[:, EMPTY, :], np.expand_dims(classifiers, -1), axis=-1).T[0]
    lnp_generic = lnp[:, EMPTY, 0]
    return specific_entropy, lnp_specific, lnp_generic, lnp

def zipf_mandelbrot(N, s, q=0):
    k = np.arange(N) + 1
    p = 1/(k+q)**s
    Z = p.sum()
    return p / Z

def fit_zipf_mandelbrot(x, init_s=1, init_q=0):
    N = len(x)
    p = x.sort_values(ascending=False)
    def loss(sq):
        phat = zipf_mandelbrot(N, *sq)
        return -p @ np.log(phat)
    return scipy.optimize.minimize(loss, (init_s, init_q)).x

def read_vectors(filename):
    with open(filename) as infile:
        first_line = next(infile)
        V, D = map(int, first_line.strip().split())
        embeddings = np.zeros((V, D))
        words = {}
        for i, line in enumerate(infile):
            word, *rest = line.strip("\n").split(" ")
            words[word] = i
            embeddings[i] = list(map(float, rest))
    return words, embeddings

def make_prefix_tensor(one, two, ungram_penalty=-100):
    # one : W x C
    # two : W x C x N
    # result: W x (N+C+1) x (N+C)
    *W, C, N = two.shape
    R = np.ones(tuple(W)+(N+C+1,)+(N+C,)) * ungram_penalty
    R[..., EMPTY, :C] = one
    R[..., :C, C:] = two
    return R

def delta_listener(num_nouns, lnp_n_given_c, epsilon=EPSILON):
    lnp_w_given_n = np.log(np.eye(num_nouns) + epsilon)
    lnp_w_given_c = marginalize(lnp_n_given_c, lnp_w_given_n)

    # now need lnp(w|cn)/lnp(w|c)
    # we already have lnp(w|cn) = lnp(w|n)
    R_n_given_c = lnp_w_given_n[:, None, :] - lnp_w_given_c[None, :, :] # shape N x C x W

    # problem: this does not penalize wrong classifier in the long term

    return lnp_w_given_c.T, R_n_given_c.T # shapes W x C, W x C x N

def lm_logprob(model, tokens):
    return model(torch.tensor(tokens)).logits.log_softmax(-1)[:, tokens].diag(1).sum().item()

def marginalize(lnp_x_given_c, lnp_y_given_x):
    # input: C x X, X x Y
    # output: C x Y
    return np.log(np.exp(lnp_x_given_c) @ np.exp(lnp_y_given_x))

def get_lm_probs(model, tokenizer, prefix, classifiers, nouns, generic="个", demonstrative="一", strip=2):
    # return tensors of shape C and C x N
    # assert are to ensure that tokenization is monotonic
    lnp_c = np.zeros(len(classifiers))
    lnp_n_given_c = np.zeros((len(classifiers), len(nouns)))
    prefix_tokens = tokenizer.encode(prefix)[:-strip] # strip is how many control tokens to eliminate
    lnp_prefix = lm_logprob(model, prefix_tokens)
    n_prefix_tokens = len(prefix_tokens)
    for c, classifier in enumerate(classifiers):
        prefix_classifier = prefix + demonstrative + classifier
        prefix_classifier_tokens = tokenizer.encode(prefix_classifier)[:-strip]
        assert prefix_classifier_tokens[:n_prefix_tokens] == prefix_tokens
        print("Querying %s" % prefix_classifier, file=sys.stderr)        
        lnp_c[c] = lm_logprob(model, prefix_classifier_tokens)
        n_prefix_classifier_tokens = len(prefix_classifier_tokens)
        for n, noun in enumerate(nouns):
            full = prefix_classifier + noun
            full_tokens = tokenizer.encode(full)[:-strip]
            assert full_tokens[:n_prefix_classifier_tokens] == prefix_classifier_tokens
            print("Querying %s" % full, file=sys.stderr)
            lnp_n_given_c[c,n] = lm_logprob(model, full_tokens) - lnp_c[c]
    return lnp_c - lnp_prefix, lnp_n_given_c # first is lnp_c, then lnp_n_given_c

def lm_exp(items_filename,
           vectors_filename=None,
           prefix="图片中有",
           model_str="TsinghuaAI/CPM-Generate",
           demonstrative="一", # or 那、这、
           generic="个",
           ungram_penalty=-100,
           epsilon=EPSILON,
           strip=2,
           **kwds):
    classifiers, nouns = read_items(items_filename)
    classifier_set = [generic] + list(set(classifiers)) # generic classifier is element 0

    # TODO: Maybe R needs to consider a wider range of nouns? (Load in different files)
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    model = AutoModelWithLMHead.from_pretrained(model_str)
    lnp_c, lnp_n_given_c = get_lm_probs(
        model,
        tokenizer,
        prefix,
        classifier_set,
        nouns,
        generic=generic,
        demonstrative=demonstrative,
        strip=strip
    ) # C, C x N
    R_c, R_n_given_c = delta_listener(len(nouns), lnp_n_given_c, epsilon=epsilon) # W x C, W x C x N
    # for p_g, use p_n or Zipf distribution fit to frequency data
    lnp_n = marginalize(lnp_c, lnp_n_given_c) # to use as p_g
    R = make_prefix_tensor(R_c, R_n_given_c, ungram_penalty=ungram_penalty)
    lnp0 = make_prefix_tensor(lnp_c, lnp_n_given_c, ungram_penalty=ungram_penalty)
    lnp = policy(R, p_g=np.exp(lnp_n), lnp0=lnp0, **kwds)
    return lnp

def random_classifier_simulation(num_nouns=20, num_classifiers=5, alpha=DISCOUNT, gamma=1.1, s=1, seed=0, **kwds):
    p_g = zipf_mandelbrot(num_nouns, s=s)
    lang, vocab = random_classifier_lang(num_nouns, num_classifiers, seed=seed)
    specific_entropy, lnp_specific, lnp_generic, lnp = analyze_classifier_lang(lang, p_g, gamma=gamma, alpha=alpha, **kwds)
    d = pd.DataFrame({
        'H_spec': specific_entropy,
        'lnp_specific': lnp_specific,
        'lnp_generic': lnp_generic,
        'classifier': np.array(lang)[:, -1, 0].astype(str),
        'p_g': p_g,
        'r': np.arange(num_nouns) + 1,
    })
    d['p_spec'] = np.exp(d['lnp_specific']) / (np.exp(d['lnp_specific']) + np.exp(d['lnp_generic']))
    d['lnp_g'] = np.log(d['p_g'])
    d['surprisal'] = -d['lnp_g']
    return d

def main():
    d1 = random_classifier_simulation(num_nouns=200, num_classifiers=10, num_iter=10, monitor=True, gamma=1.1)
    d1['gamma'] = 1.1
    d2 = random_classifier_simulation(num_nouns=200, num_classifiers=10, num_iter=10, monitor=True, gamma=1.05)
    d2['gamma'] = 1.05
    d = d1.append(d2)
    d.to_csv("classifiers/classifier_model.csv")
    
def load_dataset(filename, cutoff=20):
    # filter spec rank <11 to get 616 nouns
    d = pd.read_csv(filename, encoding="ISO-8859-1")
    classifiers = Counter(d[CORPUS_SPEC_COL])
    classifier_stats = pd.DataFrame({CORPUS_SPEC_COL: list(classifiers.keys()),
                                     'frequency': list(classifiers.values())})
    top_classifiers = classifier_stats.sort_values('frequency', ascending=False).head(n=cutoff)[CORPUS_SPEC_COL]
    df = pd.merge(d, top_classifiers)[[CORPUS_NOUN_COL, CORPUS_FREQ_COL, CORPUS_SPEC_COL]].drop_duplicates()
    return df

def main_empirical(filename=CORPUS_PATH, cutoff=20, alpha=.5, gamma=1.1, **kwds):
    df = load_dataset(filename, cutoff=cutoff)
    lang, vocab = empirical_classifier_lang(df, noun_col=CORPUS_NOUN_COL, spec_col=CORPUS_SPEC_COL)
    p_g = np.exp(df[CORPUS_FREQ_COL])
    specific_entropy, lnp_specific, lnp_generic, lnp = analyze_classifier_lang(
        lang,
        p_g,
        gamma=gamma,
        alpha=alpha,
        **kwds,
    )
    df['p_spec'] = np.exp(lnp_specific) / (np.exp(lnp_specific) + np.exp(lnp_generic))
    df['H_spec'] = specific_entropy
    df['logfreq'] = np.log(df[FREQ_COL])
    df['surprisal'] = -df['logfreq']
    return df
    
    

    

if __name__ == '__main__':
    main(*sys.argv[1:])
    


            
            
            


