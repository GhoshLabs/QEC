import numpy as np
import random

def metropolis_hastings_on_stabilizers(code, H_stab, init_e, stabilizer_vectors, q_error, n_samples=2000, burn_in=500):
    n = len(init_e)
    m_stab = len(stabilizer_vectors)

    cur_e = init_e.copy()
    cur_weight = int(cur_e.sum())
    if q_error == 0 or q_error == 1:
        raise ValueError("q_error cannot be 0 or 1 for MH numeric stability")
    log_odds = np.log(q_error/(1.0-q_error))
    cur_logp = cur_weight * log_odds

    trace_logp = []
    samples = []

    for it in range(n_samples):
        j = random.randrange(m_stab)
        svec = stabilizer_vectors[j]
        flip_indices = svec.nonzero()[0]
        delta = 0
        for idx in flip_indices:
            if cur_e[idx] == 1:
                delta -= 1
            else:
                delta += 1
        new_weight = cur_weight + delta
        new_logp = new_weight * log_odds

        accept_prob = min(1.0, np.exp(new_logp - cur_logp))
        if random.random() < accept_prob:
            cur_e[flip_indices] ^= 1
            cur_weight = new_weight
            cur_logp = new_logp

        trace_logp.append(cur_logp)
        samples.append(cur_e.copy())

    post_samples = np.array(samples[burn_in:])
    marginal = post_samples.mean(axis=0)
    e_map = (marginal > 0.5).astype(int)
    return {
        'trace_logp': np.array(trace_logp),
        'marginal': marginal,
        'e_map': e_map,
        'samples': post_samples
    }

