import sys
import itertools

import numpy as np
import torch
import pandas as pd
import tqdm
from transformers import AutoTokenizer, AutoModelWithLMHead

from generalmodel import EMPTY, grid, encode_simple_lang

GAMMA_MAX = 3
EPSILON = 10 ** -3
NUM_ITER = 100
STEPS = 1

# Predictions:
# (1) Automatic policy p(that) -> more that (ns or negative...)
# (2) Higher surprisal RC -> more that (old pred, but done with modern LM's) -- got it with size 1 and 2!
# (3) Higher entropy over RC's -> more that (new pred) -- check it.

# TODO: Try to fit RDC model directly with the real RCsubj_string and NPhead?

# Entropy estimation ideas:
# 1. k-step: only feasible for k=1; causes forced kill for k=2
# 2. Importance sampling: need an analysis of complexity and accuracy

def lm_probs(tokenizer, model, prefix, suffix, complementizer="that", steps=STEPS):
    """ get p(suffix | prefix) + p(that suffix | prefix) and p(that | prefix) """
    # assumes: no control characters; whitespaces force tokenization; complementizer is one token; prefix space behavior
    prefix_tokens = tokenizer.encode(prefix)
    len_prefix = len(prefix_tokens)
    suffix1_tokens = tokenizer.encode(" " + suffix)
    suffix2_tokens = tokenizer.encode(" " + " ".join([complementizer, suffix]))
    suffix1_tokens = suffix1_tokens[:steps]
    suffix2_tokens = suffix2_tokens[:steps+1]
    one = model(torch.tensor(prefix_tokens + suffix1_tokens)).logits.log_softmax(-1)[len_prefix-1:, suffix1_tokens].diag().sum()
    two_parts = model(torch.tensor(prefix_tokens + suffix2_tokens)).logits.log_softmax(-1)[len_prefix-1:, suffix2_tokens].diag()
    lnp_comp = two_parts[0]
    two = two_parts.sum()
    return one.item(), two.item(), torch.logaddexp(one, two).item(), lnp_comp.item()

def lm_entropy(tokenizer, model, prefix, complementizer="that", steps=STEPS):
    # steps=2 cannot be computed
    assert 1 <= steps <= 2
    prefix_tokens = tokenizer.encode(prefix + ' ' + complementizer)
    N = len(prefix_tokens)
    input = torch.tensor([prefix_tokens])
    if steps == 2:
        V = len(tokenizer)
        continuations = torch.arange(V).unsqueeze(-1)
        input = torch.cat([input.expand(V, N), continuations], -1)
    probs = model(input).logits.log_softmax(-1)[..., -steps:, :].sum(-2) # shape V^steps
    H = -(probs.exp() * probs).sum()
    return H.item()

def compdrop_entropy_grid(**kwds):
    # good freq and entropy results with gamma=1.01, alpha=.95
    # results: when RC is lower probability in p(g), higher probability of complementizer.
    num_verbs = 2
    num_rcs = 4
    
    def f(policies):
        lnpcomp_hf = policies[..., 0, EMPTY, 2, 1]
        lnpcomp_lf = policies[..., num_rcs-1, EMPTY, 2, 1]
        lnpcomp_le = policies[..., num_rcs, EMPTY, 3, 1]
        return {
            'lnpcomp_hf': lnpcomp_hf,
            'lnpcomp_lf': lnpcomp_lf,
            'entropy_diff': lnpcomp_le - lnpcomp_hf, # positive when LOW entropy has HIGHER probability of complementizer (?) -- so expect NEGATIVE, and indeed it's NEGATIVE almost everywhere.
            'freq_diff': lnpcomp_lf - lnpcomp_hf, # positive when LOW probability has HIGHER probability of complementizer (expected)
        }
    # in compdrop_R, 0 is stop, 1 is that, 2-3 are verbs, 4 is the first noun
    R = compdrop_R(
        num_verbs=num_verbs,
        num_rcs=num_rcs,
        num_nouns=0
    )
    p_g = np.array([
        1/2, 3/12, 2/12, 1/12, # the 3/12 
        1/2, 1/2, 0, 0,
    ])
    # predicts complementizer before low-prob RC (in line with Levy & Jaeger 2007) 
    # predicts complementizer before high-entropy RC (new prediction)
    # both of these predictions are in terms of the need probability and thus implicitly p_0, not controlled p_g.
    
    # result requires alpha < 1
    # if there's high uncertainty about *which RC*, then that RC gets less signal strength, and goes later.
    # is what matters p(the specific RC | it is an RC) or p(the specific RC, it is an RC)? seems like the former, interestingly.
    return grid(f, R, p_g=p_g, **kwds)

def compdrop_rcprob_grid(k=3, **kwds):
    # generally negative correlation for gamma<1, gamma>1 no effect.
    # doesn't seem to work because the automatic policy effect overwhelms whatever frequency effect might exist?
    # might require lower gamma for RC after low-RC-prob verb (why?)
    num_verbs = 2
    def f(policies):
        d = {
            'lnpcomp_lf_raw': policies[..., 0, EMPTY, 2, 1],
            'lnpcomp_hf_raw': policies[..., 2, EMPTY, 3, 1],
            'lnpcomp_lf': policies[..., 0, EMPTY, 2, 1] - np.log( # p(that | V1, g0) + p(R1 | V1, g0)
                np.exp(policies[..., 0, EMPTY, 2, 1]) + np.exp(policies[..., 0, EMPTY, 2, 2+num_verbs])
            ),
            'lnpcomp_hf': policies[..., 2, EMPTY, 3, 1] - np.log(
                np.exp(policies[..., 2, EMPTY, 3, 1]) + np.exp(policies[..., 2, EMPTY, 3, 2+num_verbs])
            ),
        }
        return d
    R = compdrop_R(num_verbs=num_verbs,
                   num_rcs=2,
                   num_nouns=2,
                   epsilon=EPSILON)
    p_g = np.array([ 
        2,1,   # V1Ri # p(Ri | V1) = 1
        2*k,k, # V2Ri # p(Ri | V2) = k
        2*k,k, # V1Nj # p(Nj | V1) = k
        2,1,   # V2Nj # p(Nj | V2) = 1
    ])
    p_g = p_g / p_g.sum()
    return grid(f, R, p_g=p_g, **kwds)

def compdrop_lang(num_verbs=1, num_nouns=1, num_rcs=1, the=False):
    # need a way to control strength of r, not c
    # 0 = stop symbol
    # 1 = that
    # optionally 2=the
    # 2:2+num_verbs = verbs
    # 2+num_verbs:2+num_verbs+num_rcs = rcs
    # finally nouns
    verbs = range(2+the, 2+the+num_verbs)
    rcs = range(2+the+num_verbs, 2+the+num_verbs+num_rcs)
    nouns = range(2+the+num_verbs+num_rcs, 2+the+num_verbs+num_rcs+num_nouns)

    rc_sentences = list(itertools.product(verbs, rcs))
    rc_c_sentences = [(v,1,r) for v,r in rc_sentences]
    rc_null_sentences = [(v,r,0) for v,r in rc_sentences]
    if the:
        noun_sentences = [(v,2,n) for v,n in itertools.product(verbs, nouns)]
    else:
        noun_sentences = [(v,n,0) for v,n in itertools.product(verbs, nouns)]
    rc_utterances = list(zip(rc_c_sentences, rc_null_sentences))
    noun_utterances = [(ns,) for ns in noun_sentences]

    utterances = rc_utterances + noun_utterances
    return utterances

def compdrop_R(num_verbs=1, num_nouns=1, num_rcs=1, the=False, **kwds):
    lang = compdrop_lang(num_verbs=num_verbs, num_nouns=num_nouns, num_rcs=num_rcs, the=the)
    return encode_simple_lang(lang, **kwds)

def main():
    print("Generating grids...", file=sys.stderr)
    df = compdrop_entropy_grid(num_iter=NUM_ITER, gamma_max=GAMMA_MAX)
    df.to_csv("compdrop/compdrop_sims.csv")
    print("Done.", file=sys.stderr)    

    print("Loading data...", file=sys.stderr)
    data = pd.read_csv("compdrop/data.csv")

    print("Loading models...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelWithLMHead.from_pretrained("gpt2")
    
    print("Getting probabilities...", file=sys.stderr) 
    results = [
        lm_probs(tokenizer, model, prefix, suffix)
        for prefix, suffix in tqdm.tqdm(list(zip(data['prefix'], data['suffix'])))
    ]
    data['lnp_suffix'], data['lnp_thatsuffix'], data['lnp_both'], data['lnp_comp'] = zip(*results)
    
    print("Getting entropy...", file=sys.stderr)
    entropy = [lm_entropy(tokenizer, model, prefix) for prefix in tqdm.tqdm(data['prefix'])]
    data['entropy'] = entropy
    data.to_csv("compdrop/lm_output.csv")
    print("Done.", file=sys.stderr)
    return 0

if __name__ == '__main__':
    sys.exit(main())

