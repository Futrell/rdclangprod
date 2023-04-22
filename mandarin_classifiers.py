""" Mandarin classifiers """
import sys
import math
import random

import numpy as np
import pandas as pd

from generalmodel import EMPTY, SimpleFixedLengthListener, entropy, policy_iteration

DISCOUNT = .9
EPSILON = 10 ** -3
NUM_NOUNS = 200
NUM_CLASSIFIERS = 10
ZIPF_EXPONENT = 1.5

def random_classifier_lang(num_nouns=NUM_NOUNS, num_classifiers=NUM_CLASSIFIERS, seed=0):
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
        
def analyze_classifier_lang(lang, p_g, gain=1, discount=1, epsilon=EPSILON, method=policy_iteration, debug=False, **kwds):
    # Predicts observed frequency effect at low gain (~1), because for low-frequency words, there is very high uncertainty about the right classifier.
    classifiers = np.array(lang)[:, -1, 0].astype(int)
    num_classifiers = len(set(classifiers))
    R = SimpleFixedLengthListener.from_strings(lang).R(epsilon=epsilon)
    lnp = method(R, p_g=p_g, gain=gain, discount=discount, **kwds)
    specific_entropy = entropy(lnp[:, EMPTY, 1:num_classifiers+1], -1) # entropy over specific classifier
    lnp_specific = np.take_along_axis(lnp[:, EMPTY, :], np.expand_dims(classifiers, -1), axis=-1).T[0]
    lnp_generic = lnp[:, EMPTY, 0]
    if debug:
        breakpoint()
    return specific_entropy, lnp_specific, lnp_generic, lnp

def zipf_mandelbrot(N, s, q=0):
    k = np.arange(N) + 1
    p = 1/(k+q)**s
    Z = p.sum()
    return p / Z

def random_classifier_simulation(num_nouns=NUM_NOUNS,
                                 num_classifiers=NUM_CLASSIFIERS,
                                 discount=DISCOUNT,
                                 gain=1.1,
                                 s=ZIPF_EXPONENT,
                                 seed=0,
                                 **kwds):
    p_g = zipf_mandelbrot(num_nouns, s=s)
    lang, vocab = random_classifier_lang(num_nouns, num_classifiers, seed=seed)
    specific_entropy, lnp_specific, lnp_generic, lnp = analyze_classifier_lang(lang, p_g, gain=gain, discount=discount, **kwds)
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
    d['discount'] = discount
    d['gain'] = gain
    return d

def main(num_nouns=NUM_NOUNS,
         num_classifiers=NUM_CLASSIFIERS,
         gain1=1.1,
         gain2=1.01,
         s=ZIPF_EXPONENT,
         num_iter=100,
         **kwds):
    d1 = random_classifier_simulation(
        num_nouns=num_nouns,
        num_classifiers=num_classifiers,
        monitor=True,
        gain=gain1,
        s=s,
        num_iter=num_iter,
        **kwds
    )
    d2 = random_classifier_simulation(
        num_nouns=num_nouns,
        num_classifiers=num_classifiers,
        monitor=True,
        gain=gain2,
        s=s,
        num_iter=num_iter,
        **kwds
    )
    d = d1.append(d2)
    d.to_csv("classifiers/classifier_model.csv")
    return d
    
if __name__ == '__main__':
    main(*sys.argv[1:])
