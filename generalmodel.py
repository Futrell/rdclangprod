import sys
import itertools

import tqdm
import numpy as np
import scipy.special
import pandas as pd
from plotnine import *
import einops
import rfutils

INF = float('inf')
EMPTY = -1
EMPTYSLICE = slice(-1, None, None)
NONEMPTY = slice(None, -1, None)
COLON = slice(None, None, None)

def the_unique(xs):
    xs_it = iter(xs)
    first_value = next(xs_it)
    for x in xs_it:
        assert x == first_value
    return first_value

def first(xs):
    for x in xs:
        return x
    else:
        raise ValueError("Empty iterable passed to first")

def all_same(xs):
    first_time = True
    for x in xs:
        if first_time:
            first_value = x
            first_time = False
        elif x != first_value:
            return False
    else:
        return True

def cartesian_indices(k, n):
    return itertools.product(*[range(k)]*n)

def last(xs):
    for x in xs:
        pass
    return x

def rindex(xs, y):
    return last(i for i, x in enumerate(xs) if x == y)

def shift_left(x, leading=1):
    """ convert array shape B1...X -> B...X1 """
    return np.expand_dims(x.squeeze(leading), -1)

def shift_right(x, leading=0):
    """ convert array shape B...X1 -> B1...X """
    return np.expand_dims(x.squeeze(-1), leading)

def integrate(x, B=0):
    """ transform a prefix tensor lnp(x_t | x_{<t}) into lnp(x_{\le t}) """
    T = x.ndim - B - 1
    y = x.copy()    
    for _, prev, curr in reversed(list(backward_iterator(T))):
        # y[0,0,a] = x[0,0,a]
        # y[0,a,b] = x[0,a,b] + x[0,0,a]       
        # y[a,b,c] = x[a,b,c] + (x[0,a,b] + x[0,0,a])
        #          = x[a,b,c] + y[0,a,b]
        y[curr] = y[curr] + shift_left(y[prev], B + 1)
    return y

def conditionalize(x, B=0):
    """ transform a prefix tensor
    lnp(x_{\le t})
    into
    lnp(x_t | x_{<t}) = lnp(x_{\le t}) - lnp(x_{<t})
    """
    T = x.ndim - B - 1
    y = x.copy()
    for _, prev, curr in reversed(list(backward_iterator(T))):
        # y[0,0,a] = x[0,0,a]
        # y[0,a,b] = x[0,a,b] - x[0,0,a]
        # y[a,b,c] = x[a,b,c] - x[0,a,b]
        y[curr] = x[curr] - shift_left(x[prev], B + 1)
    return y

def backward_iterator(T):
    for t in range(1, T):
        # current address is 00c,x  ... ex. initially with X=2 and T=3 and t=2, 00,x  -- 1x1x2
        # continuation is    0cx,y      ex. initially with X=2 and T=3 and t=2, 0x,Y  -- 1x2x2
        #                                                             then t=1, 0x,y  -- 1x2x2
        #                                                                  t=1, xy,Z  -- 2x2x2
        curr_address = (...,) + (EMPTYSLICE,)*t + (NONEMPTY,)*(T-t-1) + (COLON,)
        cont_address = (...,) + (EMPTYSLICE,)*(t-1) + (NONEMPTY,)*(T-t) + (COLON,)
        yield t, curr_address, cont_address

def passive_dynamics(p_g, lnp, B=0):
    """ transform p(g) and ln p(x_t | g, x_{<t}) into ln p(x_t | x_{<t}) """
    joint = integrate(lnp, B=B) # ln p(x_{\le t} | g)
    T = lnp.ndim - B - 1
    marginal = scipy.special.logsumexp(np.log(p_g[(...,) + (None,)*T]) + joint, B, keepdims=True) # ln p(x_{\le t})
    return conditionalize(marginal, B=B)

def value_to_go(lnp, local_value, alpha=1, B=0):
    T = lnp.ndim - B - 1
    v = np.zeros_like(alpha) + local_value # trick to broadcast to correct shape and make a copy
    p = np.exp(lnp)
    for _, curr, cont in backward_iterator(T):
        # v[a,b,c] = l[a,b,c]
        # v[0,a,b] = l[0,a,b] + alpha * v[a,b,:].sum(-1)
        continuation_value = (p[cont] * v[cont]).sum(-1, keepdims=True)
        v[curr] = v[curr] + alpha * shift_right(continuation_value, B + 1)
    return v

def control_signal(R, lnp, lnp0, gamma, alpha, B=0):
    control_cost = lnp - lnp0
    local_value = gamma*R - control_cost
    v = value_to_go(lnp, local_value, alpha, B=B)
    return v + control_cost

def policy(R,
           p_g=None,
           gamma=1,
           alpha=1,
           num_iter=100,
           init_temperature=100,
           B=0,
           debug=False,
           force_index=0,
           force=0,
           monitor=False,
           tie_init=True,
           return_default=False):
    # R_g : a tensor G x C ... C x X with conditional rewards
    assert B >= 0
    if p_g is None:
        G = R.shape[B]
        p_g = np.ones(G) / G

    # Initialize with potential forced initial condition
    if not tie_init:
        a, g, *_ = np.broadcast_arrays(alpha, gamma)[0].shape
        R = einops.repeat(R, "1 1 ... -> a g ...", a=a, g=g)
    init = 1/init_temperature * np.random.randn(*R.shape)
    init[..., force_index] += force
    lnp = scipy.special.log_softmax(init, -1)

    # Policy iteration
    iterations = tqdm.tqdm(range(num_iter)) if monitor else range(num_iter)
    for i in iterations:
        lnp0 = passive_dynamics(p_g, lnp, B=B)
        u = control_signal(R, lnp, lnp0, gamma, alpha, B=B)
        if debug:
            breakpoint()
        lnp = scipy.special.log_softmax(lnp0 + u, -1)

    if return_default:
        return lnp, lnp0
    else:
        return lnp

def pad_R(R, T=1):
    """ extend an R tensor by length T with no stop action """
    assert T >= 0
    G, *Cs, X = R.shape
    old_T = len(Cs) + 1
    C = X + 1

    # extend length
    new_R = np.zeros((G,) + tuple(C for _ in range(old_T + T - 1)) + (X,))
    # fill in old values where appropriate
    new_R[(COLON,) + (EMPTY,)*T + (COLON,)*old_T] = R
    return new_R

def fp_R(R, value=0):
    """ add a null symbol e at index 0 satisfying the following condition:
    R(xey) = R(xy) + value, so
    R(e | x) = value, R(y | ex) = R(y | x).
    """
    G, *rest = R.shape
    T = len(rest)
    new_R = np.zeros((G,) + tuple(x+1 for x in rest))
    new_R[(COLON,) + (slice(1,None,None),)*T] = R # old values maintained        
    new_R[..., 0] = value # R(e | x) = value
    # now need to fill in values for R(y | xe) = R(y | x).
    V = rest[-1]
    for context in cartesian_indices(V+2, T-1): # plus EMPTY and e
        # strip filled-pauses 0 out of the utterance, unless they are final
        old_context = tuple(x-1 for x in context if x != 0) # context stripped of e
        S = len(old_context)
        old_context = (T-S-1)*(EMPTY,) + old_context
        new_R[(COLON,) + context + (slice(1, None, None),)] = R[(COLON,) + old_context + (COLON,)]
    return new_R
    
def stop_R(R):
    """ add absorbing action at index 0 everywhere with value 0
    and everything has value 0 after the absorbing action.

    input is G x C ... C x X
    output is G x (C+1) ... (C+1) x (X+1)
    """
    G, *rest = R.shape
    T = len(rest)
    # add # to the vocabulary
    # previously like (-1,0)->1, (-1,1)->2, ..., (0,0)->3, etc.
    # now like        (-1,0)->0, (-1,1)->1, (-1,2)->2, ...,
    # basically anything with an index 0 will have value 0, otherwise things are the same
    new_R = np.zeros((G,) + tuple(x+1 for x in rest)) 
    # fill in old values where appropriate
    new_R[(COLON,) + (slice(1,None,None),)*T] = R
    return new_R

def encode_simple_lang(lang, epsilon=0.2, strength=None, init_pL=None):
    """ Reward tensor for listener model with
    p(w | x) \propto \epsilon + [x fits w]
    """
    lang = list(lang)
    G = len(lang)
    if init_pL is None:
        init_pL = np.ones(G) / G    
    T = len(lang[0][0])
    assert all(len(x) == T for y in lang for x in y)
    vocab = {x:i for i, x in enumerate(sorted(set(
        char for goal in lang for utterance in goal for char in utterance
    )))}
    V = len(vocab)
    assoc = np.zeros((G,) + (V+1,)*(T-1) + (V,))
    for g, strings in enumerate(lang):
        for string in strings:
            for prefix in rfutils.buildup(str(string)):
                S = len(prefix)
                loc = (EMPTY,)*(T-S) + tuple(vocab[x] for x in prefix)
                assoc[(g,) + loc] = 1
    if strength is not None:
        assoc *= strength[(COLON,) + (None,)*T]
    assoc += epsilon
    # p(w | x) = 1/Z epsilon + assoc[w, x], where
    #        Z = \sum_w epsilon + assoc[w, x]
    Z = assoc.sum(0, keepdims=True)
    p_L = assoc / Z
    R = conditionalize(np.log(p_L))
    R[(COLON,) + (EMPTY,)*(T-1) + (COLON,)] -= np.log(init_pL)[:, None]
    return R

# p(w | xyz) = p(w) p(w|x)/p(w) p(w|xy)/p(w|x) p(w|xyz)/p(w|xy)

def add_corr(R, corr_value=0):
    """ add correction action ! at index 0 which cancels all previous actions.
    R(x!y) = y
    R(x!) = 0
    so R(! | x) = R(x!) - R(x) = -R(x).

    input: unpadded R tensor of shape G x C... x X
    output: unpadded R tensor of shape G x (C+1)... x (X+1) including ! action
    """
    G, *Cs, X = R.shape
    T = len(Cs)
    C = the_unique(Cs)
    corr_index = 0
    # add ! to the vocabulary initially with value 0
    new_R = np.zeros((G,) + tuple(C+1 for _ in range(T)) + (X+1,))
    # fill in old values
    new_R[(COLON,) + (slice(1,None,None),)*(T+1)] = R
    # fill in values of R(y | x!) = R(y)
    # that is,
    # R( z | 0x!y ) = R( z | y )

    # first fill in values so R(x | y!z) = R(x | z).
    for address in cartesian_indices(C+1, T):
        if corr_index in address:
            corr_loc = rindex(address, corr_index)
            # Ex. if address is (1,0,2,3), so corr_loc = 1,
            # replacement should be at (EMPTY,EMPTY,2,3)
            replacement_address = (EMPTY,)*(corr_loc+1) + address[(corr_loc+1):]
            new_R[(COLON,) + address] = new_R[(COLON,) + replacement_address]
            
    # next fill in values of R(! | x) = -R(x)
    R_integrated = np.zeros_like(new_R)
    V = C # vocabulary size, including the editing term
    for t in range(1, T+2): # start with short utterances first. 
        for utterance in cartesian_indices(V, t):
            padded_utterance = (EMPTY,)*(T-t+1) + utterance # eg, 0ba
            padded_prefix = (EMPTY,)*(T-t+2) + utterance[:-1] # eg, 00b
            # set the special value for the editing term
            if padded_utterance[-1] == 0:
                # R(!|00b) = -R(00b)
                new_R[(COLON,) + padded_utterance] = -R_integrated[(COLON,) + padded_prefix] 
            # update the integrated values
            R_integrated[(COLON,)+ padded_utterance] = R_integrated[(COLON,)+padded_prefix] + new_R[(COLON,)+ padded_utterance] # R(0ba) = R(00b) + R(a|0b)
               # R(0b!) = R(00b) - R(00b) = 0, good.
    return new_R

def analyze_stutter_policy(lnp, stop=False):
    num_G = 4
    T = lnp.ndim - 2 - 1
    # T is the padding length, so length - 2.
    p = np.exp(lnp)

    # p(! | wrong) = \sum_g p(!, g | wrong) 
    #              = \sum_g p(g, !, wrong) / p(wrong)
    #              = \sum_g p(g) p(wrong | g) p(! | wrong, g) / \sum_g p(g) p(wrong | g)
    #              = \sum_g p(wrong | g) p(! | wrong, g) / \sum_g p(wrong | g) for uniform p(g)

    # correct context locations: (.*!)?a for 1,2 and (.*!)?b for 3,4

    # below, test only in the initial position... maybe expand to all relevant T.
    pathological_numerator = ( # for p(! | right)
        p[(..., 0,) + (EMPTY,)*(T-1) + (stop+1,)] * p[(..., 0,) + (EMPTY,)*(T-2) + (stop+1,stop+0)] +
        p[(..., 1,) + (EMPTY,)*(T-1) + (stop+1,)] * p[(..., 1,) + (EMPTY,)*(T-2) + (stop+1,stop+0)] +
        p[(..., 2,) + (EMPTY,)*(T-1) + (stop+2,)] * p[(..., 2,) + (EMPTY,)*(T-2) + (stop+2,stop+0)] +
        p[(..., 3,) + (EMPTY,)*(T-1) + (stop+2,)] * p[(..., 3,) + (EMPTY,)*(T-2) + (stop+2,stop+0)]
    )

    correct_denominator = (
        p[(..., 0,) + (EMPTY,)*(T-1) + (stop+1,)] +
        p[(..., 1,) + (EMPTY,)*(T-1) + (stop+1,)] + 
        p[(..., 2,) + (EMPTY,)*(T-1) + (stop+2,)] +
        p[(..., 3,) + (EMPTY,)*(T-1) + (stop+2,)]
    )
    
    healthy_numerator = (  # for p(! | wrong)
        p[(..., 0,) + (EMPTY,)*(T-1) + (stop+2,)] * p[(..., 0,) + (EMPTY,)*(T-2) + (stop+2,stop+0)] +
        p[(..., 1,) + (EMPTY,)*(T-1) + (stop+2,)] * p[(..., 1,) + (EMPTY,)*(T-2) + (stop+2,stop+0)] +
        p[(..., 2,) + (EMPTY,)*(T-1) + (stop+1,)] * p[(..., 2,) + (EMPTY,)*(T-2) + (stop+1,stop+0)] +
        p[(..., 3,) + (EMPTY,)*(T-1) + (stop+1,)] * p[(..., 3,) + (EMPTY,)*(T-2) + (stop+1,stop+0)]
    )

    incorrect_denominator = (
        p[(..., 0,) + (EMPTY,)*(T-1) + (stop+2,)] +
        p[(..., 1,) + (EMPTY,)*(T-1) + (stop+2,)] +
        p[(..., 2,) + (EMPTY,)*(T-1) + (stop+1,)] +
        p[(..., 3,) + (EMPTY,)*(T-1) + (stop+1,)]
    )

    correct_numerator = (  # for p(right | right)
        p[(..., 0,) + (EMPTY,)*(T-1) + (stop+1,)] * p[(..., 0,) + (EMPTY,)*(T-2) + (stop+1,stop+1)] +
        p[(..., 1,) + (EMPTY,)*(T-1) + (stop+1,)] * p[(..., 1,) + (EMPTY,)*(T-2) + (stop+1,stop+2)] +
        p[(..., 2,) + (EMPTY,)*(T-1) + (stop+2,)] * p[(..., 2,) + (EMPTY,)*(T-2) + (stop+2,stop+1)] +
        p[(..., 3,) + (EMPTY,)*(T-1) + (stop+2,)] * p[(..., 3,) + (EMPTY,)*(T-2) + (stop+2,stop+2)]        

    )

    # also interesting:
    # expected number of !s per ultimately correct utterance
    # expected number of !s per entirely correct utterance
    V = 2
    B = 2
    p_joint = np.exp(integrate(lnp, B=B))[(..., COLON,) + (NONEMPTY,)*(T-1) + (COLON,)]
    expected_delay = np.zeros_like(p_joint)
    expected_stutter = np.zeros_like(p_joint)
    delay = np.zeros_like(p_joint)
    stutter = np.zeros_like(p_joint)
    for utterance in cartesian_indices(V+1, T):
        # "expected delay" is ...
        # p(x | g) [ corr(x, g) ] * d(x)  / p(x | g) [ corr(x, g) ]
        # where d(x) is the first onset of the correct substring.
        if stop:
            utterance = tuple(i+1 for i in utterance)
        delay[(..., 0, *utterance)] = 0 if utterance[:2] == (stop+1,stop+1) else (substring_index(utterance, (stop+0,stop+1,stop+1)) + 1)
        delay[(..., 1, *utterance)] = 0 if utterance[:2] == (stop+1,stop+2) else (substring_index(utterance, (stop+0,stop+1,stop+2)) + 1)
        delay[(..., 2, *utterance)] = 0 if utterance[:2] == (stop+2,stop+1) else (substring_index(utterance, (stop+0,stop+2,stop+1)) + 1)
        delay[(..., 3, *utterance)] = 0 if utterance[:2] == (stop+2,stop+2) else (substring_index(utterance, (stop+0,stop+2,stop+2)) + 1)

        # "expected stutter" is...
        # p(x | g) [ all_corr(x,g) ] * d(x) / p(x | g) [ all_corr(x, g) ]
        stutter[(..., 0, *utterance)] = count_stutters(utterance, (stop+1,stop+1))
        stutter[(..., 1, *utterance)] = count_stutters(utterance, (stop+1,stop+2))
        stutter[(..., 2, *utterance)] = count_stutters(utterance, (stop+2,stop+1))
        stutter[(..., 3, *utterance)] = count_stutters(utterance, (stop+2,stop+2))

    delay_mask = delay != -1
    delay_numerator = (p_joint * delay_mask * delay).sum(tuple(range(3,3+T))).mean(-1)
    delay_denominator = (p_joint * delay_mask).sum(tuple(range(3,3+T))).mean(-1)

    stutter_mask = stutter != -1
    stutter_numerator = (p_joint * stutter_mask * stutter).sum(tuple(range(3,3+T))).mean(-1)
    stutter_denominator = (p_joint * stutter_mask).sum(tuple(range(3,3+T))).mean(-1)    

    return {
        'pathological_corr': pathological_numerator/correct_denominator,
        'healthy_corr': healthy_numerator/incorrect_denominator,
        'stutter_pref': np.log(pathological_numerator) - np.log(correct_numerator), # stutter test: positive if stutter preferred over correct continuation
        'expected_delay': delay_numerator/delay_denominator,
        'expected_stutter': stutter_numerator/stutter_denominator,
    }

def count_stutters(utterance, target):
    i = substring_index(utterance, target)
    target_start = target[0]
    if i == -1:
        return -1
    else:
        prefix = utterance[:i]
        if any(x not in {target_start, 0} for x in prefix): # includes some "by accident" correct utterances...
            return -1
        else:
            return i

def substring_index(xs, ys):
    xs = tuple(xs)
    ys = tuple(ys)
    N = len(ys)
    for i in range(len(xs)):
        if xs[i:i+N] == ys:
            return i
    else:
        return -1

def ab_listener_R(eps=1/5):
    assoc = ab_R(good=1, bad=0)
    p_xw = assoc + eps
    p_x = p_xw.sum(0, keepdims=True)
    return np.log(p_xw) - np.log(p_x) + np.log(4)

def ab_R(good=1, bad=-1):
    # aa, ab, ba, bb
    R = np.array([
        [[1, 0],  # aa
         [0, 0],
         [1, 0]], # 0a
        [[0, 1],
         [0, 0],
         [1, 0]],
        [[0, 0],
         [1, 0],
         [0, 1]],
        [[0, 0],
         [0, 1],
         [0, 1]]
    ])
    return np.ones_like(R)*bad + R*(good-bad)

def fp_figures(V=5, epsilon=1/5, weight=1, gamma=1.5, alpha=1):
    R = uneven_listener(V,
                        epsilon=epsilon,
                        weight=weight)
    R_padded = fp_R(pad_R(R, T=1), value=0)
    lnp, lnp0 = policy(R_padded,
                       gamma=gamma,
                       alpha=alpha,
                       return_default=True,
                       monitor=True,
                       init_temperature=100,
                       num_iter=1000)

    R_diag = np.diag(R)
    R_offdiag = np.zeros(V)
    R_offdiag[1:] = np.diag(R, -1)
    R_offdiag[0] = R[0,1]
    DR = R_diag - R_offdiag

    policy_df = pd.DataFrame({
        'g': np.repeat(range(V), V+1) + 1,
        'x': np.tile(range(V+1), V),
        'lnp_g(x)': einops.rearrange(lnp[:, EMPTY, :], "g x -> (g x)"),
        'R_g': einops.rearrange(R_padded[:, EMPTY, :], "g x -> (g x)"),
    })

    df = pd.DataFrame({
        'DR': DR,
        'p0(x)': np.exp(lnp0)[0, EMPTY, 1:],
        'p0(x|ε)': np.exp(lnp0)[0, 0, 1:],
        'p_{g_x}(x)': np.diag(np.exp(lnp)[:, EMPTY, :], 1),
        'p_{g_x}(ε)': np.exp(lnp)[:, EMPTY, 0]
    })

    dfm = pd.melt(df, id_vars=['DR'])

    return df, policy_df

def uneven_listener(V, epsilon=1/5, weight=1, offset=0):
    # p(w | x) \propto \epsilon + [L(x) = w]
    # good with V=5, epsilon=.2, gamma=2, alpha=1
    weights = np.zeros((V, V))
    fill = np.array(range(1, V+1)) * weight + offset
    np.fill_diagonal(weights, fill)
    weights += epsilon
    logprobs = np.log(weights) - np.log(weights.sum(-1, keepdims=True)) # p(w | x), shape W x X
    denominator = -np.log(V)
    return logprobs - denominator


def shortlong_grid(eps=1/5, offset=0, **kwds):
    def shortlong_pref(policies):
        p_short = np.exp(policies[:, :, 0, EMPTY, EMPTY, 0])
        p_long = np.exp(policies[:, :, 0, EMPTY, EMPTY, 1])
        p_weird = np.exp(policies[:, :, 0, EMPTY, EMPTY, 2])
        p_short2 = np.exp(policies[:, :, 1, EMPTY, EMPTY, 3])
        p_long2 = np.exp(policies[:, :, 1, EMPTY, EMPTY, 1])
        return {
            'p_short': p_short,
            'p_long': p_long,
            'p_weird': p_weird,
            'p_short2': p_short2,
            'p_long2': p_long2,
        }
    R = shortlong_R(eps=eps) + offset
    df1 = grid(shortlong_pref, R, **kwds)
    return df1

def codability_grid(**kwds):
    def codability_pref(policies):
        return {'statistic': np.exp(policies[:, :, 0, EMPTY, 0])}
    R = codability_R()
    # Weirdly, control costs here are zero, but the codable-first effect
    # does come through because it maximizes probability of a good move
    # in the second step during early BA iterations.
    return grid(codability_pref, R, **kwds)

def stutter_grid(eps=1/5, pad=2, **kwds):
    R = add_corr(pad_R(ab_listener_R(eps=eps), T=pad))
    return grid(analyze_stutter_policy, R, **kwds)

def grid(f, R, gamma_min=0, gamma_max=5, gamma_steps=100, alpha_min=0, alpha_max=1, alpha_steps=100, init_temperature=1000, num_iter=1000, **kwds):
    alphas = np.linspace(alpha_max, alpha_min, alpha_steps)
    gammas = np.linspace(gamma_min, gamma_max, gamma_steps)
    T = R.ndim
    R = R[(None, None) + (COLON,)*T]
    alphas = alphas[(COLON,) + (None,) + (None,)*T]
    gammas = gammas[(None,) + (COLON,) + (None,)*T]    
    policies = policy(
        R,
        gamma=gammas,
        alpha=alphas,
        B=2,
        init_temperature=init_temperature,
        num_iter=num_iter,
        monitor=True,
        **kwds
    )

    alpha_expanded = einops.repeat(alphas.squeeze(), "a -> a g", g=gamma_steps)
    gamma_expanded = einops.repeat(gammas.squeeze(), "g -> a g", a=alpha_steps)

    df = pd.DataFrame({
        'alpha': einops.rearrange(alpha_expanded, "a g -> (a g)"),
        'gamma': einops.rearrange(gamma_expanded, "a g -> (a g)"),
    })

    statistics = f(policies)    
    for statistic_name, statistic_value in statistics.items():
        df[statistic_name] = einops.rearrange(statistic_value, "a g -> (a g)")
        

    return df

def codability_R():
    # g1 -> {ab, ac, ba, ca}
    # g2 -> {aa, bb, cc}
    R = np.array([
        [
            [0, 1, 1],  # a_
            [1, 0, 0],  # b_
            [1, 0, 0],  # c_
            [1, 1, 1],  # 0_
        ],
        [
            [1, 0, 0], # a_
            [0, 1, 0], # b_
            [0, 0, 1], # c_
            [1, 1, 1]
        ]
    ])
    return R

def shortlong_R(eps=1/5):
    return encode_simple_lang(
        [
            ["abc", "bca"],
            ["bcd", "dbc"],
            ["ada", "adb", "adc", "add",
             "daa", "dab", "dac", "dad"],
        ],
        epsilon=eps
    )

#    p_xw_smoothed = p_xw + eps
#    p_x_smoothed = p_xw_smoothed.sum(0, keepdims=True)
#    return np.log(p_xw_smoothed) - np.log(p_x_smoothed) + np.log(3)

def test_control_signal():
    # g1 -> aa
    # g2 -> ab
    # g3 -> ba
    # g4 -> bb
    R = np.array([
        [[1, 0],  # a_
         [0, 0],  # b_
         [1, 0]], # 0_
        [[0, 1],  # a_
         [0, 0],  # b_
         [1, 0]], # 0_
        [[0, 0],  # a_
         [1, 0],  # b_
         [0, 1]], # 0_                
        [[0, 0],  # a_
         [0, 1],  # b_
         [0, 1]], # 0_                
    ])
    lnp = scipy.special.log_softmax(R * 100, -1) # nearly deterministic
    lnp0 = np.log(np.array(
        [[[0.6250, 0.3750],
          [0.5000, 0.5000],
          [0.8000, 0.2000]]]
    ))

    # first try no future planning, then we should get u = R.
    u = control_signal(R, lnp, lnp0, gamma=1, alpha=0)
    assert is_close(u, R).all()

    u = control_signal(R, lnp, lnp0, gamma=1, alpha=1)    
    # check u(x_2 | g, x_1)
    assert is_close(u[0, 0, 0], 1)
    assert is_close(u[1, 0, 1], 1)
    assert is_close(u[2, 1, 0], 1)
    assert is_close(u[3, 1, 1], 1)

    # check u(x_1 | g, empty)
    # for u(a | g=0, empty) we should have approximately
    # l(0, 0a) + l(0, aa) + lnp0(a|a)
    # 1 + 1 - 0.4700
    assert is_close(u[0, EMPTY, 0], 1 + 1 - 0.4700)

    # g1 -> aaa
    # g2 -> abb
    p = np.array([
        [  # g1 -> aaa
            [ 
                [1, 0],  # aa_
                [0, 0],  # ab_
                [0, 0],    # a0_
            ],
            [
                [0, 0],  # ba_
                [0, 0],  # ba_
                [0, 0],    # b0_
            ],
            [
                [1, 0],   # 0a_
                [0, 0],   # 0b_
                [1, 0],     # 00_
            ]
        ],
        [  # g2 -> bab but no reward for initial b
            [ 
                [0, 0],  # aa_
                [0, 0],  # ab_
                [0, 0],    # a0_
            ],
            [
                [0, 1],  # ba_
                [0, 0],  # ba_
                [0, 0],    # b0_
            ],
            [
                [0, 0],   # 0a_
                [1, 0],   # 0b_
                [0, 1],     # 00_
            ]
        ],        
    ])
    
    R = np.array([
        [  # g1 -> aaa
            [ 
                [1, 0],  # aa_
                [0, 0],  # ab_
                [0, 0],    # a0_
            ],
            [
                [0, 0],  # ba_
                [0, 0],  # ba_
                [0, 0],    # b0_
            ],
            [
                [0, 0],   # 0a_
                [0, 0],   # 0b_
                [0, 0],     # 00_
            ]
        ],
        [  # g2 -> bab
            [ 
                [0, 0],  # aa_
                [0, 0],  # ab_
                [0, 0],    # a0_
            ],
            [
                [0, 1],  # ba_
                [0, 0],  # ba_
                [0, 0],    # b0_
            ],
            [
                [0, 0],   # 0a_
                [0, 0],   # 0b_
                [0, 0],     # 00_
            ]
        ],        
    ])
    p_g = np.array([.25, .75])
    lnp = scipy.special.log_softmax(p*100, -1)
    lnp0 = passive_dynamics(p_g, lnp) 
    # first try no future planning, then we should get u = R
    u = control_signal(R, lnp, lnp0, gamma=1, alpha=0)
    #assert is_close(u, R).all() # NANs!

    u = control_signal(R, lnp, lnp0, gamma=1, alpha=1)
    # utility of 00b is 0 + 0 + 1 + ln p(b | ba). p(b | ba) = 1 so we should get 1
    assert u[1, 1, 0, 1] == 1
    assert u[1, EMPTY, 1, 0] == 1    
    assert u[1, EMPTY, EMPTY, 1] == 1    
        
def test_value_to_go():
    # example local value tensor for...
    # aa -> 1+10
    # ab -> 1+20
    # ba -> 2+100
    # bb -> 2+200
    # so with alpha=1,
    # v(a) = 1 + 10 + 20 = 31
    # v(b) = 2 + 100 + 200 = 302
    l = np.array([[ # 1 x 3 x 2
        [10, 20],
        [100, 200],
        [1, 2],
    ]]).astype(float)
    assert (value_to_go(np.zeros(l.shape), l, 1)
            == np.array([[[10, 20], [100, 200], [31, 302]]])).all()

    # aaa -> 1+10+100
    # aab -> 1+10+200
    # aba -> 1+20+100
    # abb -> 1+20+200 etc.

    l = np.array([[  # 1 x 3 x 3 x 2
        [
            [100, 200],  # aa_
            [100, 200],  # ab_
            [4000, 5000],    # a0_
        ],
        [
            [100, 200],  # ba_
            [100, 200],  # ba_
            [6000, 7000],    # b0_
        ],
        [
            [10, 20],   # 0a_
            [10, 20],   # 0b_
            [1, 2],     # 00_
        ]
    ]]).astype(float)
    a, b = range(2)
    result = value_to_go(np.zeros(l.shape), l, 1)
    assert result[0, EMPTY, EMPTY, a] == 631
    assert result[0, EMPTY, EMPTY, b] == 632
    assert result[0, EMPTY, a, a] == 310
    assert result[0, EMPTY, a, b] == 320

def test_conditionalize():
    p = np.array(
        [[[111, 112],
          [123, 124],
          [.1, .2]],
         [[235, 236],
          [247, 248],
          [.3, .4]],
         [[110, 120],
          [230, 240],
          [100, 200]]]
    )
    dp = conditionalize(np.expand_dims(p, 0)).squeeze(0)
    a,b = range(2)
    assert dp[EMPTY,EMPTY,a] == p[EMPTY,EMPTY,a]
    assert dp[EMPTY,EMPTY,b] == p[EMPTY,EMPTY,b]
    assert dp[a,a,a] == p[a,a,a] - p[EMPTY,a,a]
    assert dp[b,b,a] == p[b,b,a] - p[EMPTY,b,b]
    assert dp[a,b,a] == p[a,b,a] - p[EMPTY,a,b]

def test_integrate():
    dp = np.array(
        [[[1, 2], # aa_
         [3, 4],  # ab_
         [-1.6224, -0.2199]], # a0_

        [[5, 6],  # ba_
         [7, 8],  # bb_
         [-0.4187, -1.0726]], # b0_

        [[10, 20],  # 0a_
         [30, 40],  # 0b_
         [100, 200]]] # 00_
    )
    p = integrate(np.expand_dims(dp, 0)).squeeze(0)
    # lnp(aaa) = lnp(a | 00) + lnp(a | 0a) + lnp(a | aa)
    a, b = range(2)
    assert p[EMPTY,EMPTY,a] == dp[EMPTY,EMPTY,a]
    assert p[EMPTY,EMPTY,b] == dp[EMPTY,EMPTY,b]
    assert p[EMPTY,a,a] == dp[EMPTY, EMPTY, a] + dp[EMPTY, a, a]
    assert p[EMPTY,a,b] == dp[EMPTY, EMPTY, a] + dp[EMPTY, a, b]
    assert p[EMPTY,b,a] == dp[EMPTY, EMPTY, b] + dp[EMPTY, b, a]
    assert p[EMPTY,b,b] == dp[EMPTY, EMPTY, b] + dp[EMPTY, b, b]    
    assert p[a,a,a] == dp[EMPTY,EMPTY,a] + dp[EMPTY,a,a] + dp[a,a,a] # -2.0956
    assert p[b,b,b] == dp[EMPTY,EMPTY,b] + dp[EMPTY,b,b] + dp[b,b,b]
    assert p[a,a,b] == dp[EMPTY,EMPTY,a] + dp[EMPTY,a,a] + dp[a,a,b]
    assert p[b,a,b] == dp[EMPTY,EMPTY,b] + dp[EMPTY,b,a] + dp[b,a,b]

def is_close(x, y, eps=10**-5):
    return abs(x-y) < eps

def test_integrate_conditionalize():
    """ test that conditionalize(integrate(x)) == x and integrate(conditionalize(x)) == x """
    for i in range(4):
        dim = (1,) + (5,)*i + (4,)
        dp = scipy.special.log_softmax(np.random.randn(*dim), -1)
        p = integrate(dp)
        dp2 = conditionalize(p)
        p2 = integrate(dp2)
        assert is_close(dp, dp2).all()
        assert is_close(p, p2).all()

def test_passive_dynamics():
    p_g = np.array([.1, .9, 0])
    # active dynamics is 0 -> aa, 1 -> bb, 0 -> cc
    # passive dynamics should have p(a|00) = .1, p(b|00) = .9, p(c|00) = 0
    #                              p(a|0a) = 1, etc.           p(c|0c) = undefined
    lnp = np.log(np.array([
        [[1, 0, 0],  # a_
         [0, 0, 0],  # b_
         [0, 0, 0],  # c_
         [1, 0, 0]], # 0_
        [[0, 0, 0],  # a_
         [0, 1, 0],  # b_
         [0, 0, 0],  # c_
         [0, 1, 0]], # 0_        
        [[0, 0, 0],  # a_
         [0, 0, 0],  # b_
         [0, 0, 1],  # c_
         [0, 0, 1]], # 0_
    ]))
    lnp0 = passive_dynamics(p_g, lnp).squeeze(0)
    a, b, c = range(3)
    assert lnp0[a,a] == 0
    assert lnp0[EMPTY,a] == np.log(p_g[0])
    assert lnp0[b,b] == 0
    assert lnp0[EMPTY,b] == np.log(p_g[1])
    assert np.isinf(lnp0[a,b]) and lnp0[a,b] < 0
    assert np.isinf(lnp0[b,a]) and lnp0[b,a] < 0

    p = np.array([
        [  # g1 -> aaa
            [ 
                [1, 0],  # aa_
                [0, 0],  # ab_
                [0, 0],    # a0_
            ],
            [
                [0, 0],  # ba_
                [0, 0],  # ba_
                [0, 0],    # b0_
            ],
            [
                [1, 0],   # 0a_
                [0, 0],   # 0b_
                [1, 0],     # 00_
            ]
        ],
        [  # g2 -> bab but no reward for initial b
            [ 
                [0, 0],  # aa_
                [0, 0],  # ab_
                [0, 0],    # a0_
            ],
            [
                [0, 1],  # ba_
                [0, 0],  # ba_
                [0, 0],    # b0_
            ],
            [
                [0, 0],   # 0a_
                [1, 0],   # 0b_
                [0, 1],     # 00_
            ]
        ],        
    ])
    p0 = np.exp(passive_dynamics(np.array([.25, .75]), np.log(p)).squeeze())
    assert p0[EMPTY,EMPTY,a] == 1/4
    assert p0[EMPTY,EMPTY,b] == 3/4
    assert p0[a,a,a] == 1
    assert p0[b,a,b] == 1

def test_add_corr():
    R = np.array([ # R(aa) = 11, R(ab)=12, R(ba)=23, R(bb)=24
        [[ 1.,  2.], # a_
         [ 3.,  4.], # b_
         [10., 20.]] # 0_
    ])
    new_R = add_corr(R)
    assert (new_R == np.array([[
        [  0.,  10.,  20.], # R(c|c)=0, R(a|c)=10, etc.
        [-10.,   1.,   2.],
        [-20.,   3.,   4.],
        [  0.,  10.,  20.] # R(c|0)=0
    ]])).all()

def t_e_s_t_stop_R():
    # start with aa,ab,ba,bb
    R = np.array([[
        [1, 2],
        [3, 4],
        [10, 20],
    ]])
    new_R = stop_R(R, T=0)
    assert (new_R == np.array([[
        [ 0., -INF, -INF],  # #_    after # can only continue with #
        [ 0.,  1.,  2.],    # a_    after a, probably continue with b
        [ 0.,  3.,  4.],    # b_    after b, probably continue with b
        [ 0., 10., 20.]     # 0_    after EMPTY, probably continue with b
    ]])).all()

    new_R = stop_R(R, T=1)
    assert (new_R == np.array([[
        [[ 0., -INF, -INF],  # ##_
          [ 0.,  0.,  0.],   # junk etc.
          [ 0.,  0.,  0.],
          [ 0.,  0.,  0.]],

         [[ 0., -INF, -INF],  # a#_
          [ 0.,  0.,  0.],    # aa_
          [ 0.,  0.,  0.],
          [ 0.,  0.,  0.]],

         [[ 0., -INF, -INF],
          [ 0.,  0.,  0.],
          [ 0.,  0.,  0.],
          [ 0.,  0.,  0.]],

         [[ 0., -INF, -INF],
          [ 0.,  1.,  2.],
          [ 0.,  3.,  4.],
          [ 0., 10., 20.]]
    ]])).all()


def figures():
    # short before long
    print("Generating shortlong grid...", file=sys.stderr)
    dfsl = shortlong_grid(tie_init=False, eps=1/5)
    dfsl.to_csv("output/shortlong.csv")

    # filled pause
    print("Generating filled-pause simulations...", file=sys.stderr)
    df, policy = fp_figures()
    df.to_csv("output/fp_summary.csv")
    policy.to_csv("output/fp_policy.csv")

    print("Generating correction simulations...", file=sys.stderr)
    dfs = stutter_grid(pad=4, gamma_max=5, alpha_min=.95, gamma_steps=50, alpha_steps=5, tie_init=False, eps=1/5)
    dfs.to_csv("output/stutter.csv")
    
if __name__ == '__main__':
    np.seterr(all="ignore")    
    import nose    
    nose.runmodule()
