import sys
import itertools

import numpy as np
import torch
import pandas as pd
import tqdm
from transformers import AutoTokenizer, AutoModelWithLMHead

from generalmodel import EMPTY, grid, encode_simple_lang

GAIN_MAX = 3
EPSILON = 10 ** -3
NUM_ITER = 100
STEPS = 1
MODEL = "gpt2"

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

def compdrop_grid(**kwds):
    num_nouns = 2
    num_rcs = 4
    
    def f(policies):
        lnpcomp_hf = policies[..., 0, EMPTY, 2, 1]           # p(that | N0, g=N0 R0) -- high prob,    high entropy
        lnpcomp_lf = policies[..., 1, EMPTY, 2, 1]           # p(that | N0, g=N0 R1) -- low prob,     high entropy
        lnpcomp_hf2 = policies[..., num_rcs+1, EMPTY, 3, 1]  # p(that | N1, g=N1 R1) -- high prob,    low entropy

        return {
            'lnpcomp_hf': lnpcomp_hf,
            'lnpcomp_lf': lnpcomp_lf,
            'lnpcomp_hf2': lnpcomp_hf2,
            'freq_diff': lnpcomp_lf - lnpcomp_hf, 
            'freq_diff2': lnpcomp_lf - lnpcomp_hf2, 
        }
    # in compdrop_R, 0 is stop, 1 is that, 2-3 are verbs, 4 is the first noun
    R = compdrop_R(
        num_nouns=num_nouns,
        num_rcs=num_rcs,
        num_other=0
    )
    p_g = np.array([
        1/2, 3/12, 2/12, 1/12, # the 3/12 
        1/2, 1/2, 0, 0,
    ])
    return grid(f, R, p_g=p_g, **kwds)

def compdrop_lang(num_nouns=1, num_other=1, num_rcs=1):
    # 0 = stop symbol
    # 1 = that
    # optionally 2=the
    # 2:2+num_nouns = verbs
    # 2+num_nouns:2+num_nouns+num_rcs = rcs
    # finally nouns
    nouns = range(2, 2+num_nouns)
    rcs = range(2+num_nouns, 2+num_nouns+num_rcs)
    others = range(2+num_nouns+num_rcs, 2+num_nouns+num_rcs+num_other)

    rc_sentences = list(itertools.product(nouns, rcs))
    rc_c_sentences = [(v,1,r) for v,r in rc_sentences]
    rc_null_sentences = [(v,r,0) for v,r in rc_sentences]
    other_sentences = [(v,n,0) for v,n in itertools.product(nouns, others)]
    rc_utterances = list(zip(rc_c_sentences, rc_null_sentences))
    other_utterances = [(ns,) for ns in other_sentences]

    utterances = rc_utterances + other_utterances
    return utterances

def compdrop_R(num_nouns=1, num_other=1, num_rcs=1, **kwds):
    lang = compdrop_lang(num_nouns=num_nouns, num_other=num_other, num_rcs=num_rcs)
    return encode_simple_lang(lang, **kwds)

def main():
    print("Generating grids...", file=sys.stderr)
    df = compdrop_grid(num_iter=NUM_ITER, gain_max=GAIN_MAX)
    df.to_csv("compdrop/compdrop_sims.csv")
    print("Done.", file=sys.stderr)    

    print("Loading data...", file=sys.stderr)
    data = pd.read_csv("compdrop/data.csv")

    print("Loading models...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelWithLMHead.from_pretrained(MODEL)

    print("Getting probabilities...", file=sys.stderr) 
    results = [
        lm_probs(tokenizer, model, prefix, suffix)
        for prefix, suffix in tqdm.tqdm(list(zip(data['prefix'], data['suffix'])))
    ]
    data['lnp_suffix'], data['lnp_thatsuffix'], data['lnp_both'], data['lnp_comp'] = zip(*results)
    print("Done.", file=sys.stderr)
    
    data.to_csv("compdrop/lm_output.csv")
    return 0

if __name__ == '__main__':
    sys.exit(main())

