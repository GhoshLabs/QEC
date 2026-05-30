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
        # Vectorized calculation for the change in weight
        delta = len(flip_indices) - 2 * np.sum(cur_e[flip_indices])
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

    # Find the sample with the highest probability (lowest weight)
    trace_logp = np.array(trace_logp)
    best_idx = np.argmax(trace_logp)
    best_sample = samples[best_idx]

    return {
        'trace_logp': trace_logp,
        'marginal': marginal,
        'e_map': e_map,
        'samples': post_samples,
        'best_sample': best_sample
    }

def metropolis_hastings_joint(eX_init, eZ_init, all_stabs, n_X_stabs, q_error, n_samples):
    if q_error == 0 or q_error == 1:
         raise ValueError("q_error cannot be 0 or 1")
    
    log_odds = np.log(q_error / (1.0 - q_error))
    
    cur_eX = eX_init.copy()
    cur_eZ = eZ_init.copy()
    cur_weight = np.sum(cur_eX | cur_eZ)
    cur_logp = cur_weight * log_odds
    
    best_logp = cur_logp
    best_eX = cur_eX.copy()
    best_eZ = cur_eZ.copy()
    
    m_stab = len(all_stabs)
    
    for _ in range(n_samples):
        j = random.randrange(m_stab)
        svec = all_stabs[j]
        is_X_stab = (j < n_X_stabs)
        flip_indices = svec.nonzero()[0]
        
        if is_X_stab:
            delta_w = np.sum((cur_eX[flip_indices] ^ 1) | cur_eZ[flip_indices]) - np.sum(cur_eX[flip_indices] | cur_eZ[flip_indices])
            if (cur_logp + delta_w * log_odds) > cur_logp or random.random() < np.exp(delta_w * log_odds):
                cur_eX[flip_indices] ^= 1
                cur_weight += delta_w
                cur_logp += delta_w * log_odds
        else:
            delta_w = np.sum(cur_eX[flip_indices] | (cur_eZ[flip_indices] ^ 1)) - np.sum(cur_eX[flip_indices] | cur_eZ[flip_indices])
            if (cur_logp + delta_w * log_odds) > cur_logp or random.random() < np.exp(delta_w * log_odds):
                cur_eZ[flip_indices] ^= 1
                cur_weight += delta_w
                cur_logp += delta_w * log_odds
        
        if cur_logp > best_logp:
            best_logp = cur_logp
            best_eX = cur_eX.copy()
            best_eZ = cur_eZ.copy()
            
    return best_eX, best_eZ, best_logp

def metropolis_hastings_track_z(eX_init, eZ_init, all_stabs, n_X_stabs, q_error, n_samples, burn_in, logicals_X, logicals_Z):
    if q_error == 0 or q_error == 1:
         raise ValueError("q_error cannot be 0 or 1")
    
    log_odds = np.log(q_error / (1.0 - q_error))
    
    cur_eX = eX_init.copy()
    cur_eZ = eZ_init.copy()
    cur_weight = np.sum(cur_eX | cur_eZ)
    cur_logp = cur_weight * log_odds
    
    best_logp = cur_logp
    best_eX = cur_eX.copy()
    best_eZ = cur_eZ.copy()
    
    m_stab = len(all_stabs)
    
    num_classes = len(logicals_X)
    Z_ratios = np.zeros(num_classes)
    n_post_burn_in = 0

    for i in range(n_samples):
        j = random.randrange(m_stab)
        svec = all_stabs[j]
        is_X_stab = (j < n_X_stabs)
        flip_indices = svec.nonzero()[0]
        
        if is_X_stab:
            delta_w = np.sum((cur_eX[flip_indices] ^ 1) | cur_eZ[flip_indices]) - np.sum(cur_eX[flip_indices] | cur_eZ[flip_indices])
            if (cur_logp + delta_w * log_odds) > cur_logp or random.random() < np.exp(delta_w * log_odds):
                cur_eX[flip_indices] ^= 1
                cur_weight += delta_w
                cur_logp += delta_w * log_odds
        else:
            delta_w = np.sum(cur_eX[flip_indices] | (cur_eZ[flip_indices] ^ 1)) - np.sum(cur_eX[flip_indices] | cur_eZ[flip_indices])
            if (cur_logp + delta_w * log_odds) > cur_logp or random.random() < np.exp(delta_w * log_odds):
                cur_eZ[flip_indices] ^= 1
                cur_weight += delta_w
                cur_logp += delta_w * log_odds
        
        if cur_logp > best_logp:
            best_logp = cur_logp
            best_eX = cur_eX.copy()
            best_eZ = cur_eZ.copy()

        if i >= burn_in:
            n_post_burn_in += 1
            for k in range(num_classes):
                lX, lZ = logicals_X[k], logicals_Z[k]
                transformed_weight = np.sum((cur_eX ^ lX) | (cur_eZ ^ lZ))
                delta_w_logical = transformed_weight - cur_weight
                log_prob_ratio = delta_w_logical * log_odds
                Z_ratios[k] += np.exp(log_prob_ratio)

    if n_post_burn_in > 0:
        Z_ratios /= n_post_burn_in
    
    return best_eX, best_eZ, Z_ratios

def metropolis_hastings_avg_weight(eX_init, eZ_init, all_stabs, n_X_stabs, q_error, n_samples, burn_in):
    if q_error == 0 or q_error == 1:
         raise ValueError("q_error cannot be 0 or 1")
    
    log_odds = np.log(q_error / (1.0 - q_error))
    
    cur_eX = eX_init.copy()
    cur_eZ = eZ_init.copy()
    cur_weight = np.sum(cur_eX | cur_eZ)
    cur_logp = cur_weight * log_odds
    
    best_logp = cur_logp
    best_eX = cur_eX.copy()
    best_eZ = cur_eZ.copy()
    
    m_stab = len(all_stabs)
    
    total_weight = 0
    n_post_burn_in = 0
    min_weight = np.inf
    min_weight_configs = set()

    for i in range(n_samples):
        j = random.randrange(m_stab)
        svec = all_stabs[j]
        is_X_stab = (j < n_X_stabs)
        flip_indices = svec.nonzero()[0]
        
        if is_X_stab:
            delta_w = np.sum((cur_eX[flip_indices] ^ 1) | cur_eZ[flip_indices]) - np.sum(cur_eX[flip_indices] | cur_eZ[flip_indices])
            if (cur_logp + delta_w * log_odds) > cur_logp or random.random() < np.exp(delta_w * log_odds):
                cur_eX[flip_indices] ^= 1
                cur_weight += delta_w
                cur_logp += delta_w * log_odds
        else:
            delta_w = np.sum(cur_eX[flip_indices] | (cur_eZ[flip_indices] ^ 1)) - np.sum(cur_eX[flip_indices] | cur_eZ[flip_indices])
            if (cur_logp + delta_w * log_odds) > cur_logp or random.random() < np.exp(delta_w * log_odds):
                cur_eZ[flip_indices] ^= 1
                cur_weight += delta_w
                cur_logp += delta_w * log_odds
        
        if cur_logp > best_logp:
            best_logp = cur_logp
            best_eX = cur_eX.copy()
            best_eZ = cur_eZ.copy()
        
        if i >= burn_in:
            total_weight += cur_weight
            n_post_burn_in += 1
            if cur_weight < min_weight:
                min_weight = cur_weight
                min_weight_configs = { (tuple(cur_eX), tuple(cur_eZ)) }
            elif cur_weight == min_weight:
                min_weight_configs.add( (tuple(cur_eX), tuple(cur_eZ)) )

    avg_weight = total_weight / n_post_burn_in if n_post_burn_in > 0 else cur_weight
            
    return avg_weight, min_weight, len(min_weight_configs), best_eX, best_eZ

def metropolis_hastings_coset_probs(eX_init, eZ_init, all_stabs, n_X_stabs, q_error, n_samples, burn_in, logicals_X, logicals_Z):
    """
    Estimates relative probabilities of logical cosets by running independent
    MCMC chains starting in each logical sector.
    """
    if q_error == 0 or q_error == 1:
         raise ValueError("q_error cannot be 0 or 1")
    
    log_odds = np.log(q_error / (1.0 - q_error))
    m_stab = len(all_stabs)
    num_classes = len(logicals_X)
    
    # Aggregated distribution across all chains
    aggregated_probs = np.zeros(num_classes)

    # 1. Run an independent chain for each logical class
    for s in range(num_classes):
        # Initialize chain specifically in sector s
        cur_eX = eX_init ^ logicals_X[s]
        cur_eZ = eZ_init ^ logicals_Z[s]
        cur_weight = np.sum(cur_eX | cur_eZ)
        cur_logp = cur_weight * log_odds
        
        # Accumulate ratios Z_k / Z_s for this chain
        chain_Z_ratios = np.zeros(num_classes)
        n_post_burn_in = 0

        for i in range(n_samples):
            # Standard MH step
            j = random.randrange(m_stab)
            svec = all_stabs[j]
            is_X_stab = (j < n_X_stabs)
            flip_indices = svec.nonzero()[0]
            
            if is_X_stab:
                delta_w = np.sum((cur_eX[flip_indices] ^ 1) | cur_eZ[flip_indices]) - np.sum(cur_eX[flip_indices] | cur_eZ[flip_indices])
                if (cur_logp + delta_w * log_odds) > cur_logp or random.random() < np.exp(delta_w * log_odds):
                    cur_eX[flip_indices] ^= 1; cur_weight += delta_w; cur_logp += delta_w * log_odds
            else:
                delta_w = np.sum(cur_eX[flip_indices] | (cur_eZ[flip_indices] ^ 1)) - np.sum(cur_eX[flip_indices] | cur_eZ[flip_indices])
                if (cur_logp + delta_w * log_odds) > cur_logp or random.random() < np.exp(delta_w * log_odds):
                    cur_eZ[flip_indices] ^= 1; cur_weight += delta_w; cur_logp += delta_w * log_odds
            
            # 2. Track probability of all other sectors relative to current state
            if i >= burn_in:
                n_post_burn_in += 1
                # Precompute relative logical shifts to determine weight in other sectors
                for k in range(num_classes):
                    lX_rel = logicals_X[s] ^ logicals_X[k]
                    lZ_rel = logicals_Z[s] ^ logicals_Z[k]
                    
                    # Weight of the equivalent error configuration in sector k
                    transformed_weight = np.sum((cur_eX ^ lX_rel) | (cur_eZ ^ lZ_rel))
                    
                    # Probability ratio P(e_k) / P(e_s)
                    chain_Z_ratios[k] += np.exp((transformed_weight - cur_weight) * log_odds)

        if n_post_burn_in > 0:
            chain_dist = chain_Z_ratios / n_post_burn_in
            total_chain_mass = np.sum(chain_dist)
            if total_chain_mass > 0:
                aggregated_probs += (chain_dist / total_chain_mass)
    
    return aggregated_probs
