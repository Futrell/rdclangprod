""" generalmodel

Contains the general vectorized policy-iteration implementation of the RDC language production model.

Reward functions and policies are represented using a data structure called a prefix tensor.
Suppose we want to represent probabilities of symbols given sequences of previous symbols,
up to a fixed maximum length, for example the probability of "c" after "ab". Given a finite
vocabulary "abc", the symbol "a" has index 0, "b" has index 1, and "c" has index 2. Then p(c|ab)
is p[0,1,2]. The prefix tensor has a special index EMPTY (-1) so that p(b | a) is p[EMPTY,0,1].
The probability p("abc") = p[EMPTY,EMPTY,0] * p[EMPTY,0,1] * p[0,1,2]. Therefore for a vocabulary
of size V, and a maximum length of T, the shape of a prefix tensor is (V+1) x (V+1) x ... x (V+1) x V,
with the (V+1) repeated T times.

Prefix tensors can also have leading dimensions which allow indexing by goal or by batch. So a policy
is represented as a prefix tensor G x (V+1) x ... x (V+1) x V, where the entries are log probabilities.

The code includes some utility functions for dealing with log probability prefix tensors, such as
integrate (which turns a prefix tensor where p[0,1,2] representing p(2 | 0,1) into one where
p[0,1,2] represents p(0,1,2)), conditionalize (the inverse of integrate), and marginalize, which
takes a prefix tensor with a leading dimension and marginalizes over that dimension.

Reward functions are also represented as prefix tensors. There are some utility functions to
make it easier to generate reward functions for languages of interest. The class
SimpleFixedLengthListener creates a reward tensor for a language defined using a list of
lists of strings (for example, [['ab', 'ba'], ['cd', 'dc'], ['ef', 'fe']] corresponds to
a language with three world states, each of which is expressible with one of two utterances
ab or ba, cd or dc, ef or fe respectively).
"""

import sys
import math
import random
import itertools

import tqdm
import numpy as np
import scipy.special
import pandas as pd
import einops

INF = float('inf')
EMPTY = -1
EMPTYSLICE = slice(-1, None, None)
NONEMPTY = slice(None, -1, None)
COLON = slice(None, None, None)

TOL = 10 ** -5

def the_unique(xs):
    xs_it = iter(xs)
    first_value = next(xs_it)
    for x in xs_it:
        assert x == first_value
    return first_value

def buildup(iterable):
    """ Build up

    Example:
    >>> list(buildup("abcd"))
    [('a',), ('a', 'b'), ('a', 'b', 'c'), ('a', 'b', 'c', 'd')]

    """
    so_far = []
    for x in iterable:
        so_far.append(x)
        yield tuple(so_far)

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
    """ Give indices to iterate backwards through a prefix tensor.

    For example, for sequences xyz, yield addresses:
       curr=0xy, cont=xyz
       curr=00y, cont=0xy

    Note that indices for curr=000,cont=00x are never yielded, because curr=000 is not
    represented within a prefix tensor.
    
    """
    for t in range(1, T):
        # current address is 00c,x  ... ex. initially with X=2 and T=3 and t=2, 00,x  -- 1x1x2
        # continuation is    0cx,y      ex. initially with X=2 and T=3 and t=2, 0x,Y  -- 1x2x2
        #                                                             then t=1, 0x,y  -- 1x2x2
        #                                                                  t=1, xy,Z  -- 2x2x2
        curr_address = (...,) + (EMPTYSLICE,)*t + (NONEMPTY,)*(T-t-1) + (COLON,)
        cont_address = (...,) + (EMPTYSLICE,)*(t-1) + (NONEMPTY,)*(T-t) + (COLON,)
        yield t, curr_address, cont_address

def automatic_policy(p_g, lnp, B=0):
    """ transform p(g) and ln p(x_t | g, x_{<t}) into ln p(x_t | x_{<t}) """
    joint = integrate(lnp, B=B) # ln p(x_{\le t} | g)
    T = lnp.ndim - B - 1
    marginal = scipy.special.logsumexp(np.log(p_g[(...,) + (None,)*T]) + joint, B, keepdims=True) # ln p(x_{\le t} | t)
    return conditionalize(marginal, B=B)

def value(lnp, local_value, discount=1, B=0):
    """ One pass of value iteration, starting with the later actions and going backward. """
    T = lnp.ndim - B - 1
    v = np.zeros_like(discount) + local_value # trick to broadcast to correct shape and make a copy
    p = np.exp(lnp)
    for _, curr, cont in backward_iterator(T):
        # v[a,b,c] = l[a,b,c]
        # v[0,a,b] = l[0,a,b] + discount * v[a,b,:].sum(-1)
        continuation_value = (p[cont] * v[cont]).sum(-1, keepdims=True)
        v[curr] = v[curr] + discount * shift_right(continuation_value, B + 1)
    return v

def control_signal(R, lnp, lnp0, gain, discount, costgain=1, lmbda=1, B=0):
    control_cost = (1/lmbda)*(lnp - lnp0)
    local_value = gain * R - costgain * control_cost
    v = value(lnp, local_value, discount, B=B) # v = gain*R - control_cost + discount*v'
    return v + costgain * control_cost # u = gain*R + discount*v' = v + control_cost

def z_iteration(R,
                p_g=None,
                gain=1,
                discount=1,
                init_temperature=100,
                lnp0=None,
                tie_init=True,
                num_iter=1000,
                B=0,
                debug=False,
                monitor=False,
                return_default=False,
                ba_steps=1,
                init_lnp=None,
                tol=TOL,
                print_loss=False):
    """ Solve for RDC policy using InfoRL algorithm from Rubin et al. (2012),
    a variant of z-iteration from Todorov (2009), interlaced with Blahut-Arimoto. """
    assert B >= 0
    T = R.ndim - 1
    G = R.shape[B]
    if p_g is None:
        p_g = np.ones(G) / G
    fit_lnp0 = lnp0 is None        

    if not tie_init:
        a, g, *_ = np.broadcast_arrays(discount, gain)[0].shape
        R = einops.repeat(R, "1 1 ... -> a g ...", a=a, g=g)

    if init_lnp is None:
        init = 1/init_temperature * np.random.randn(*R.shape).mean(B, keepdims=True).repeat(G, axis=B)
        lnp = scipy.special.log_softmax(init, -1)
    else:
        lnp = init_lnp
        
    F = np.zeros(R.shape)
    iterations = tqdm.tqdm(range(num_iter)) if monitor else range(num_iter)
    for i in iterations:
        old_lnp = lnp.copy()
        if fit_lnp0:
            lnp0 = automatic_policy(p_g, lnp, B=B)
        old_F = F.copy()
        for t, prev, curr in backward_iterator(T):
            energy = lnp0[curr] + gain*R[curr] + discount*old_F[curr]
            lnZ = scipy.special.logsumexp(energy, -1, keepdims=True)
            F[prev] = shift_right(lnZ, B + 1)
        if print_loss:
            if T == 1:
                print((p_g @ scipy.special.logsumexp(lnp0 + gain*R + discount*F, -1)).item())
            else:
                print((p_g @ scipy.special.logsumexp(lnp0 + gain*R + discount*F, -1)[:, (EMPTY,)*(T-1)]).item())
        lnp = scipy.special.log_softmax(lnp0 + gain*R + discount*F, -1)
        
        # Check convergence
        diff = np.sum(np.abs(old_F - F))
        if diff < tol:
            break

    if debug: breakpoint()
    if return_default:
        return lnp, lnp0
    else:
        return lnp

def policy_iteration(R,                    
                     p_g=None,             
                     gain=1,              
                     discount=1,              
                     lmbda=1,              
                     costgain=1,           
                     num_iter=1000,
                     tol=TOL,
                     init_temperature=100, 
                     lnp0=None,
                     init_lnp=None,
                     B=0,
                     extra_ba_steps=0,
                     extra_pi_steps=0,
                     tie_init=True,        
                     debug=False,          
                     monitor=False,        
                     return_default=False,
                     print_loss=False,
                     ):
    """ Compute RDC policy by Blahut-Arimoto policy iteration.

    Inputs:
    
    R: A tensor of shape GC*X where R[g,c1,...,cn,x] is the reward for action x given context c1,...,cn with goal g.
    p_g: A need distribution over goals, shape B*G where B is a batch dimension and G is goals.
    gain: Control gain.
    discount: Discount rate.
    lmbda: Inverse penalty for control information.
    costgain: Relative importance of cost in future planning calculation. Set to 0 to constrain information rate.
    num_iter: Maximum number of policy iteration steps.
    tol: If not None, the sum-squared-error convergence criterion.    
    init_temperature: Temperature for random energy initialization.
    lnp0: If specified, a fixed automatic policy.
    B: Number of batch dimensions, default 0.
    tie_init: If true, then all batches start with the same initialization.
    debug: If true, go into the debugger at each iteration.
    monitor: If true, show a progress bar.
    return_default: If true, return the default policy lnp0 in addition to the controlled policy lnp.

    """
    assert B >= 0
    if p_g is None:
        G = R.shape[B]
        p_g = np.ones(G) / G

    if not tie_init:
        a, g, *_ = np.broadcast_arrays(discount, gain)[0].shape
        R = einops.repeat(R, "1 1 ... -> a g ...", a=a, g=g)

    T = R.ndim - B - 1
    fit_lnp0 = lnp0 is None    

    # initialization not dependent on goal, so control cost at init = 0
    if init_lnp is None:
        init = 1/init_temperature * np.random.randn(*R.shape).mean(B, keepdims=True)
        lnp = scipy.special.log_softmax(init, -1)
    else:
        lnp = init_lnp


    # Policy iteration
    iterations = tqdm.tqdm(range(num_iter)) if monitor else range(num_iter)
    for i in iterations:
        old_lnp = lnp.copy()
        if fit_lnp0:
            lnp0 = automatic_policy(p_g, lnp, B=B)

        u = control_signal(R, lnp, lnp0, gain, discount, lmbda=lmbda, costgain=costgain, B=B)
        
        if print_loss:
            if T == 1:
                print((p_g @ scipy.special.logsumexp(lnp0 + lmbda*u, -1)).item())
            else:
                print((p_g @ scipy.special.logsumexp(lnp0 + lmbda*u, -1)[:, (EMPTY,)*(T-1)]).item())
        
        lnp = scipy.special.log_softmax(lnp0 + lmbda*u, -1)
        for k in range(extra_pi_steps):
            u = control_signal(R, lnp, lnp0, gain, discount, lmbda=lmbda, costgain=costgain, B=B)
            lnp = scipy.special.log_softmax(lnp0 + lmbda*u, -1)
        if fit_lnp0:
            for k in range(extra_ba_steps):
                lnp0 = automatic_policy(p_g, lnp, B=B)
                lnp = scipy.special.log_softmax(lnp0 + lmbda*u, -1)

        # Check convergence
        diff = np.sum(np.abs(np.exp(lnp) - np.exp(old_lnp)))
        if diff < tol:
            break

    if debug: breakpoint()
    
    if return_default:
        return lnp, lnp0
    else:
        return lnp

def loss_u(R, lnp, gain, discount, p_g=None, lmbda=1):
    T = R.ndim - 1
    if p_g is None:
        G = R.shape[0]
        p_g = np.ones(G)/G
    lnp0 = automatic_policy(p_g, lnp)
    u = control_signal(R, lnp, lnp0, gain, discount, lmbda=lmbda)
    return p_g @ (scipy.special.logsumexp(lnp0 + lmbda*u, -1))[:, (EMPTY,)*(T-1)]

    
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
    
class SimpleFixedLengthListener:
    def __init__(self, assoc, init_p_L=None):
        self.assoc = assoc
        self.V = self.assoc.shape[-1]
        self.T = len(self.assoc.shape) - 1
        self.G = self.assoc.shape[0]
        if init_p_L is None:
            self.init_p_L = np.ones(self.G) / self.G
        else:
            self.init_p_L = init_p_L

    @classmethod
    def from_strings(cls, lang, init_p_L=None):
        lang = list(lang)
        G = len(lang)
        T = len(lang[0][0])
        assert all(len(x) == T for y in lang for x in y)
        vocab = {x:i for i, x in enumerate(sorted(set(
            char for goal in lang for utterance in goal for char in utterance
        )))}
        V = len(vocab)
        assoc = np.zeros((G,) + (V+1,)*(T-1) + (V,))
        for g, strings in enumerate(lang):
            for string in strings:
                for prefix in buildup(string):
                    S = len(prefix)
                    loc = (EMPTY,)*(T-S) + tuple(vocab[x] for x in prefix)
                    assoc[(g,) + loc] = 1
        return cls(assoc, init_p_L)

    def p_L(self, epsilon=.02, strength=None, epsilon_multiplier=None):
        assoc = self.assoc
        if strength is not None:
            assoc = assoc * strength[(COLON,) + (None,)*self.T]
        if epsilon_multiplier is not None:
            assoc = assoc + epsilon * epsilon_multiplier[(COLON,) + (None,)*self.T]
        else:
            assoc = assoc + epsilon
        # p(w | x) = 1/Z epsilon + assoc[w, x], where
        #        Z = \sum_w epsilon + assoc[w, x]
        Z = assoc.sum(0, keepdims=True)
        p_L = assoc / Z
        return p_L

    def R(self, **kwds):
        """ Reward tensor. """
        p_L = self.p_L(**kwds)
        R = conditionalize(np.log(p_L))
        R[(COLON,) + (EMPTY,)*(self.T-1) + (COLON,)] -= np.log(self.init_p_L)[:, None]
        return R
        
def encode_simple_lang(lang,
                       epsilon=0.2,
                       strength=None,
                       epsilon_multiplier=None,
                       init_p_L=None):
    """ Reward tensor for listener model with
    p(w | x) \propto \epsilon + [x fits w]
    """
    L = SimpleFixedLengthListener.from_strings(lang, init_p_L=init_p_L)
    return L.R(epsilon=epsilon, strength=strength, epsilon_multiplier=epsilon_multiplier)

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
    return encode_simple_lang([
        ['aa'],
        ['ab'],
        ['ba'],
        ['bb'],
        ],
        epsilon=eps
    )
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

def fp_figures(V=5, epsilon=1/5, weight=1, gain=1.5, discount=1, method=policy_iteration, offset=0, which='distractors', **kwds):
    R = uneven_listener(V, epsilon=epsilon, weight=weight, which=which)
    R_diag = np.diag(R)
    R_offdiag = (R - np.eye(V)*R_diag).sum(-1) / (V-1)
    DR = R_diag - R_offdiag

    R_padded = fp_R(pad_R(R, T=1), value=0) + offset
    lnp, lnp0 = method(R_padded,
                       gain=gain,
                       discount=discount,
                       return_default=True,
                       monitor=True,
                       **kwds)
    policy_df = pd.DataFrame({
        'g': np.repeat(range(V), V+1) + 1,
        'x': np.tile(range(V+1), V),
        'lnp_g(x)': einops.rearrange(lnp[:, EMPTY, :], "g x -> (g x)"),
        'R_g': einops.rearrange(R_padded[:, EMPTY, :], "g x -> (g x)"),
    })

    df = pd.DataFrame({
        'g': range(V),
        'DR': DR,
        'p0(x)': np.exp(lnp0)[0, EMPTY, 1:],
        'p0(x|ε)': np.exp(lnp0)[0, 0, 1:],
        'p_{g_x}(x)': np.diag(np.exp(lnp)[:, EMPTY, :], 1),
        'p_{g_x}(ε)': np.exp(lnp)[:, EMPTY, 0]
    })

    dfm = pd.melt(df, id_vars=['DR'])
    return df, policy_df, lnp, lnp0, R_padded

def fp_listener(V, epsilon=1/5, weight=1, which='distractors'):
    lang = [
        [(v+1,0),
         (0,v+1)]
        for v in range(V)
    ]
    if which == 'distractors':
        return encode_simple_lang(lang,
                                  epsilon=epsilon,
                                  epsilon_multiplier=1/((np.arange(V)+1)*weight))
    elif which == 'target':
        return encode_simple_lang(lang,
                                  epsilon=epsilon,
                                  strength=weight * (np.arange(V)+1))
    elif which == 'both':
        return encode_simple_lang(lang,
                                  epsilon=epsilon,
                                  epsilon_multiplier=1/((np.arange(V)+1*weight)),
                                  strength=weight*(np.arange(V)+1))
    elif which == 'neither':
        return encode_simple_lang(lang, epsilon=epsilon)

def uneven_listener(V, epsilon=1/5, weight=1, which='distractors'):
    assert V <= 10
    # p(w | x) \propto \epsilon + [L(x) = w]
    if which == 'distractors':
        return encode_simple_lang(list(map(list, map(str, range(V)))),
                                  epsilon=epsilon,
                                  epsilon_multiplier=1/((np.arange(V)+1)*weight))
    elif which == 'target':
        return encode_simple_lang(list(map(list, map(str, range(V)))),
                                  epsilon=epsilon,
                                  strength=weight * (np.arange(V)+1))
    elif which == 'both':
        return encode_simple_lang(list(map(list, map(str, range(V)))),
                                  epsilon=epsilon,
                                  strength=weight * (np.arange(V)+1),
                                  epsilon_multiplier=1/((np.arange(V)+1)*weight))
        
        
        
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

def grid(f, R, gain_min=0, gain_max=5, gain_steps=100, discount_min=0, discount_max=1, discount_steps=100, init_temperature=1000, method=policy_iteration, **kwds):
    """ f is a function taking a tensor of policies and returning a dictionary of tensors of statistics """
    discounts = np.linspace(discount_max, discount_min, discount_steps)
    gains = np.linspace(gain_min, gain_max, gain_steps)
    T = R.ndim
    R = R[(None, None) + (COLON,)*T]
    discounts = discounts[(COLON,) + (None,) + (None,)*T]
    gains = gains[(None,) + (COLON,) + (None,)*T]    
    policies = method(
        R,
        gain=gains,
        discount=discounts,
        B=2,
        init_temperature=init_temperature,
        monitor=True,
        **kwds
    )

    discount_expanded = einops.repeat(discounts.squeeze(), "a -> a g", g=gain_steps)
    gain_expanded = einops.repeat(gains.squeeze(), "g -> a g", a=discount_steps)

    df = pd.DataFrame({
        'discount': einops.rearrange(discount_expanded, "a g -> (a g)"),
        'gain': einops.rearrange(gain_expanded, "a g -> (a g)"),
    })

    statistics = f(policies)    
    for statistic_name, statistic_value in statistics.items():
        df[statistic_name] = einops.rearrange(statistic_value, "a g -> (a g)")
        
    return df

def codability_R(eps=1/5):
    return encode_simple_lang(
        [
            ['ab', 'ac', 'ba', 'ca'],
            ['aa'],
            ['bb'],
            ['cc'],
        ],
        epsilon=eps,
    )

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

def entropy(x, axis=-1):
    p = np.exp(x)
    return -scipy.special.xlogy(p, p).sum(axis=axis)

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
    u = control_signal(R, lnp, lnp0, gain=1, discount=0)
    assert is_close(u, R).all()

    u = control_signal(R, lnp, lnp0, gain=1, discount=1)    
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
    lnp0 = automatic_policy(p_g, lnp) 
    # first try no future planning, then we should get u = R
    u = control_signal(R, lnp, lnp0, gain=1, discount=0)
    #assert is_close(u, R).all() # NANs!

    u = control_signal(R, lnp, lnp0, gain=1, discount=1)
    # utility of 00b is 0 + 0 + 1 + ln p(b | ba). p(b | ba) = 1 so we should get 1
    assert u[1, 1, 0, 1] == 1
    assert u[1, EMPTY, 1, 0] == 1    
    assert u[1, EMPTY, EMPTY, 1] == 1    
        
def test_value():
    # example local value tensor for...
    # aa -> 1+10
    # ab -> 1+20
    # ba -> 2+100
    # bb -> 2+200
    # so with discount=1,
    # v(a) = 1 + 10 + 20 = 31
    # v(b) = 2 + 100 + 200 = 302
    l = np.array([[ # 1 x 3 x 2
        [10, 20],
        [100, 200],
        [1, 2],
    ]]).astype(float)
    assert (value(np.zeros(l.shape), l, 1)
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
    result = value(np.zeros(l.shape), l, 1)
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

def test_automatic_policy():
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
    lnp0 = automatic_policy(p_g, lnp).squeeze(0)
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
    p0 = np.exp(automatic_policy(np.array([.25, .75]), np.log(p)).squeeze())
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

def figures():
    # short before long
    print("Generating shortlong grid...", file=sys.stderr)
    dfsl = shortlong_grid(tie_init=False, eps=1/5)
    dfsl.to_csv("output/shortlong.csv")

    # filled pause
    print("Generating filled-pause simulations...", file=sys.stderr)
    df, policy_df, lnp, lnp0, R_fp = fp_figures()
    df.to_csv("output/fp_summary.csv")
    policy_df.to_csv("output/fp_policy.csv")

    print("Generating correction simulations...", file=sys.stderr)
    dfs = stutter_grid(pad=4,
                       gain_max=5,
                       discount_min=.95,
                       gain_steps=50,
                       discount_steps=5,
                       tie_init=False,
                       eps=1/5)
    dfs.to_csv("output/stutter.csv")
    
if __name__ == '__main__':
    np.seterr(all="ignore")    
    import nose    
    nose.runmodule()
