from utils import coset_weight_distr, generate_all_sectors, ge_initialize_given_syndrome, simpson_integral
import csv
import os
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from MH_sampler import metropolis_hastings_coset_probs, metropolis_hastings_avg_weight
from decoder import WormDecoder
from noise import depolarizing_noise
import syndrome as synd

def coset_probs_exact(eX, eZ, code, p):
    """Calculates exact probabilities and returns (probs, labels)."""
    all_sectors = generate_all_sectors(eX, eZ, code)
    num_logical_qubits = int(np.round(np.log2(len(all_sectors)) / 2))
    
    labels = []
    for c_bits in product([0, 1], repeat=num_logical_qubits):
        for b_bits in product([0, 1], repeat=num_logical_qubits):
            pauli_str = ""
            for i in range(num_logical_qubits):
                b, c = b_bits[i], c_bits[i]
                mapping = {(0,0):'I', (1,0):'X', (0,1):'Z', (1,1):'Y'}
                pauli_str += mapping[(b, c)]
            labels.append(pauli_str)
            
    P = []
    for lX, lZ in all_sectors:
        P.append(coset_weight_distr(lX, lZ, code, p))
    return P, labels

def coset_avg_wt_mcmc(eX, eZ, code, p, n_samples=20000, burn_in=5000):
    """
    Computes the average weight of errors per logical coset using MCMC.
    Returns (avg_weights, min_weights, labels, min_weight_counts).
    """
    q = p / (3 - 2 * p)
    HZ, HX = code.stabilizer_matrices()
    Zstab_vecs = [HZ[i] for i in range(HZ.shape[0])]
    Xstab_vecs = [HX[i] for i in range(HX.shape[0])]
    all_stabs = Xstab_vecs + Zstab_vecs
    n_X_stabs = len(Xstab_vecs)

    # Generate logical operators for tracking
    n = code.n
    log_X_supports = [s for s in [code.logical_X_support(), code.logical_X_conjugate()] if s]
    log_Z_supports = [s for s in [code.logical_Z_support(), code.logical_Z_conjugate()] if s]
    num_logical_qubits = len(log_X_supports)

    lX_vecs = []
    for support in log_X_supports:
        v = np.zeros(n, dtype=int); v[support] = 1
        lX_vecs.append(v)
    lZ_vecs = []
    for support in log_Z_supports:
        v = np.zeros(n, dtype=int); v[support] = 1
        lZ_vecs.append(v)

    logicals_X, logicals_Z = [], []
    labels = []
    for c_bits in product([0, 1], repeat=num_logical_qubits):
        for b_bits in product([0, 1], repeat=num_logical_qubits):
            lX, lZ = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
            pauli_str = ""
            for i, b in enumerate(b_bits):
                if b: lX ^= lX_vecs[i]
            for i, c in enumerate(c_bits):
                if c: lZ ^= lZ_vecs[i]
                mapping = {(0,0):'I', (1,0):'X', (0,1):'Z', (1,1):'Y'}
                pauli_str += mapping[(b_bits[i], c_bits[i])]
            logicals_X.append(lX)
            logicals_Z.append(lZ)
            labels.append(pauli_str)

    num_classes = len(logicals_X)
    avg_weights = np.zeros(num_classes)
    min_weights = np.full(num_classes, np.inf)
    min_weight_counts = np.zeros(num_classes)

    # Run an independent MCMC chain for each logical coset
    for s in range(num_classes):
        # Initialize chain in coset s
        eX_init = np.array(eX, dtype=int) ^ logicals_X[s]
        eZ_init = np.array(eZ, dtype=int) ^ logicals_Z[s]
        
        avg_wt, min_wt, mw_count, _, _ = metropolis_hastings_avg_weight(
            eX_init, eZ_init, all_stabs, n_X_stabs, 
            q, int(n_samples), int(burn_in)
        )
        
        avg_weights[s] = avg_wt
        min_weights[s] = min_wt
        min_weight_counts[s] = mw_count

    return avg_weights, min_weights, labels, min_weight_counts

def coset_probs_mcmc(eX, eZ, code, p, n_samples=20000, burn_in=5000):
    """
    Estimates the probability of all logical cosets using MCMC tracking.
    Returns (coset_probs, min_weight_error_probs).
    """
    # 1. Setup sampler parameters
    q = p / (3 - 2 * p) # Conversion for depolarizing noise
    HZ, HX = code.stabilizer_matrices()
    Zstab_vecs = [HZ[i] for i in range(HZ.shape[0])]
    Xstab_vecs = [HX[i] for i in range(HX.shape[0])]
    all_stabs = Xstab_vecs + Zstab_vecs
    n_X_stabs = len(Xstab_vecs)

    # 2. Generate logical operators for tracking
    n = code.n
    log_X_supports = [s for s in [code.logical_X_support(), code.logical_X_conjugate()] if s]
    log_Z_supports = [s for s in [code.logical_Z_support(), code.logical_Z_conjugate()] if s]
    num_logical_qubits = len(log_X_supports)

    # 2.1 Generate logical operators in the same order as generate_all_sectors
    lX_vecs = []
    for support in log_X_supports:
        v = np.zeros(n, dtype=int); v[support] = 1
        lX_vecs.append(v)
    lZ_vecs = []
    for support in log_Z_supports:
        v = np.zeros(n, dtype=int); v[support] = 1
        lZ_vecs.append(v)

    logicals_X, logicals_Z = [], []
    labels = []
    for c_bits in product([0, 1], repeat=num_logical_qubits):
        for b_bits in product([0, 1], repeat=num_logical_qubits):
            lX, lZ = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
            pauli_str = ""
            for i, b in enumerate(b_bits):
                if b: lX ^= lX_vecs[i]
            for i, c in enumerate(c_bits):
                if c: lZ ^= lZ_vecs[i]
                mapping = {(0,0):'I', (1,0):'X', (0,1):'Z', (1,1):'Y'}
                pauli_str += mapping[(b_bits[i], c_bits[i])]
            logicals_X.append(lX)
            logicals_Z.append(lZ)
            labels.append(pauli_str)

    # 3. Run the sampler (using independent parallel chains for each class)
    Z_ratios = metropolis_hastings_coset_probs(
        np.array(eX, dtype=int), np.array(eZ, dtype=int), all_stabs, n_X_stabs, 
        q, int(n_samples), int(burn_in), logicals_X, logicals_Z
    )

    # Normalize Z ratios to get relative coset probabilities
    return Z_ratios / np.sum(Z_ratios), labels


def coset_proxies_mcmc(eX, eZ, code, p, n_samples=20000, burn_in=5000):
    """
    Computes Z_ratios, avg_wts, min_wts, min_wt_counts, and labels using MCMC.
    Runs a single combined set of independent sector chains so work is not duplicated.
    """
    q = p / (3 - 2 * p)
    HZ, HX = code.stabilizer_matrices()
    Zstab_vecs = [HZ[i] for i in range(HZ.shape[0])]
    Xstab_vecs = [HX[i] for i in range(HX.shape[0])]
    all_stabs = Xstab_vecs + Zstab_vecs
    n_X_stabs = len(Xstab_vecs)

    # Generate logical operators for tracking
    n = code.n
    log_X_supports = [s for s in [code.logical_X_support(), code.logical_X_conjugate()] if s]
    log_Z_supports = [s for s in [code.logical_Z_support(), code.logical_Z_conjugate()] if s]
    num_logical_qubits = len(log_X_supports)

    lX_vecs = []
    for support in log_X_supports:
        v = np.zeros(n, dtype=int); v[support] = 1
        lX_vecs.append(v)
    lZ_vecs = []
    for support in log_Z_supports:
        v = np.zeros(n, dtype=int); v[support] = 1
        lZ_vecs.append(v)

    logicals_X, logicals_Z = [], []
    labels = []
    for c_bits in product([0, 1], repeat=num_logical_qubits):
        for b_bits in product([0, 1], repeat=num_logical_qubits):
            lX, lZ = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
            pauli_str = ""
            for i, b in enumerate(b_bits):
                if b: lX ^= lX_vecs[i]
            for i, c in enumerate(c_bits):
                if c: lZ ^= lZ_vecs[i]
                mapping = {(0,0):'I', (1,0):'X', (0,1):'Z', (1,1):'Y'}
                pauli_str += mapping[(b_bits[i], c_bits[i])]
            logicals_X.append(lX)
            logicals_Z.append(lZ)
            labels.append(pauli_str)

    num_classes = len(logicals_X)
    avg_weights = np.zeros(num_classes)
    min_weights = np.full(num_classes, np.inf)
    min_weight_counts = np.zeros(num_classes)
    aggregated_probs = np.zeros(num_classes)

    log_odds = np.log(q / (1.0 - q))
    m_stab = len(all_stabs)

    # Run one combined chain per logical sector
    for s in range(num_classes):
        cur_eX = np.array(eX, dtype=int) ^ logicals_X[s]
        cur_eZ = np.array(eZ, dtype=int) ^ logicals_Z[s]
        cur_weight = np.sum(cur_eX | cur_eZ)
        cur_logp = cur_weight * log_odds

        chain_Z_ratios = np.zeros(num_classes)
        total_weight = 0.0
        n_post_burn_in = 0
        min_weight = np.inf
        min_weight_configs = set()

        for i in range(int(n_samples)):
            j = np.random.randint(m_stab)
            svec = all_stabs[j]
            is_X_stab = (j < n_X_stabs)
            flip_indices = svec.nonzero()[0]

            if is_X_stab:
                delta_w = np.sum((cur_eX[flip_indices] ^ 1) | cur_eZ[flip_indices]) - np.sum(cur_eX[flip_indices] | cur_eZ[flip_indices])
                if (cur_logp + delta_w * log_odds) > cur_logp or np.random.rand() < np.exp(delta_w * log_odds):
                    cur_eX[flip_indices] ^= 1
                    cur_weight += delta_w
                    cur_logp += delta_w * log_odds
            else:
                delta_w = np.sum(cur_eX[flip_indices] | (cur_eZ[flip_indices] ^ 1)) - np.sum(cur_eX[flip_indices] | cur_eZ[flip_indices])
                if (cur_logp + delta_w * log_odds) > cur_logp or np.random.rand() < np.exp(delta_w * log_odds):
                    cur_eZ[flip_indices] ^= 1
                    cur_weight += delta_w
                    cur_logp += delta_w * log_odds

            if i >= int(burn_in):
                total_weight += cur_weight
                n_post_burn_in += 1

                if cur_weight < min_weight:
                    min_weight = cur_weight
                    min_weight_configs = { (tuple(cur_eX), tuple(cur_eZ)) }
                elif cur_weight == min_weight:
                    min_weight_configs.add((tuple(cur_eX), tuple(cur_eZ)))

                for k in range(num_classes):
                    lX_rel = logicals_X[s] ^ logicals_X[k]
                    lZ_rel = logicals_Z[s] ^ logicals_Z[k]
                    transformed_weight = np.sum((cur_eX ^ lX_rel) | (cur_eZ ^ lZ_rel))
                    chain_Z_ratios[k] += np.exp((transformed_weight - cur_weight) * log_odds)

        if n_post_burn_in > 0:
            avg_weights[s] = total_weight / n_post_burn_in
            min_weights[s] = min_weight if min_weight < np.inf else cur_weight
            min_weight_counts[s] = len(min_weight_configs)
            chain_dist = chain_Z_ratios / n_post_burn_in
            total_chain_mass = np.sum(chain_dist)
            if total_chain_mass > 0:
                aggregated_probs += (chain_dist / total_chain_mass)

    Z_ratios = aggregated_probs
    if np.sum(Z_ratios) > 0:
        Z_ratios = Z_ratios / np.sum(Z_ratios)

    return Z_ratios, avg_weights, min_weights, min_weight_counts, labels


def coset_probs_worm(eX, eZ, code, p, n_samples=500, n_burnin=100, seed=None):
    """
    Estimates the probability of all logical cosets using the worm decoder.
    Returns (coset_probs_normalized, labels).
    """
    # Initialize WormDecoder
    decoder = WormDecoder(code, p, n_samples=n_samples, n_burnin=n_burnin, seed=seed)
    
    # Generate logical operators and labels in standard order
    n = code.n
    log_X_supports = [s for s in [code.logical_X_support(), code.logical_X_conjugate()] if s]
    log_Z_supports = [s for s in [code.logical_Z_support(), code.logical_Z_conjugate()] if s]
    num_logical_qubits = len(log_X_supports)
    
    lX_vecs = []
    for support in log_X_supports:
        v = np.zeros(n, dtype=int); v[support] = 1
        lX_vecs.append(v)
    lZ_vecs = []
    for support in log_Z_supports:
        v = np.zeros(n, dtype=int); v[support] = 1
        lZ_vecs.append(v)
    
    logicals_X, logicals_Z = [], []
    labels = []
    for c_bits in product([0, 1], repeat=num_logical_qubits):
        for b_bits in product([0, 1], repeat=num_logical_qubits):
            lX, lZ = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
            pauli_str = ""
            for i, b in enumerate(b_bits):
                if b: lX ^= lX_vecs[i]
            for i, c in enumerate(c_bits):
                if c: lZ ^= lZ_vecs[i]
                mapping = {(0,0):'I', (1,0):'X', (0,1):'Z', (1,1):'Y'}
                pauli_str += mapping[(b_bits[i], c_bits[i])]
            logicals_X.append(lX)
            logicals_Z.append(lZ)
            labels.append(pauli_str)
    
    num_classes = len(logicals_X)
    coset_counts = np.zeros(num_classes)
    
    # Compute syndrome from reference error
    HZ, HX = code.stabilizer_matrices()
    eX_arr = np.array(eX, dtype=int)
    eZ_arr = np.array(eZ, dtype=int)
    syndZ = (HZ @ eX_arr) % 2
    syndX = (HX @ eZ_arr) % 2
    
    # Run worm decoder chains for each logical sector
    for s in range(num_classes):
        # Create reference error in sector s with same syndrome
        eX_ref = eX_arr ^ logicals_X[s]
        eZ_ref = eZ_arr ^ logicals_Z[s]
        
        # Run Z chain to get logical Z outcomes and edge reweighting
        z_counts, alpha = decoder._run_z_chain(eZ_ref)
        
        # Run X chain with reweighted edges
        log_w_X = decoder._reweight_x_edges(alpha)
        x_counts = decoder._run_x_chain(eX_ref, log_w_X)
        
        # Get the most probable outcomes from each chain
        best_z_bits = max(z_counts, key=z_counts.get)
        best_x_bits = max(x_counts, key=x_counts.get)
        
        # Map these outcomes to a logical coset index
        # best_x_bits are the logical X basis measurements (related to Z logical ops)
        # best_z_bits are the logical Z basis measurements (related to X logical ops)
        for k in range(num_classes):
            # For coset k, compute what the measurements should be
            ref_x_bits = tuple(int(np.dot(lv, logicals_X[k]) % 2) for lv in lZ_vecs)
            ref_z_bits = tuple(int(np.dot(lv, logicals_Z[k]) % 2) for lv in lX_vecs)
            
            if ref_x_bits == best_x_bits and ref_z_bits == best_z_bits:
                coset_counts[k] += 1
                break
    
    # Normalize to get coset probabilities
    coset_probs = coset_counts / np.sum(coset_counts) if np.sum(coset_counts) > 0 else np.ones(num_classes) / num_classes
    
    return coset_probs, labels


def coset_labels(code):
    log_X_supports = [s for s in [code.logical_X_support(), code.logical_X_conjugate()] if s]
    log_Z_supports = [s for s in [code.logical_Z_support(), code.logical_Z_conjugate()] if s]
    num_logical_qubits = len(log_X_supports)

    labels = []
    for c_bits in product([0, 1], repeat=num_logical_qubits):
        for b_bits in product([0, 1], repeat=num_logical_qubits):
            pauli_str = ""
            for i in range(num_logical_qubits):
                mapping = {(0,0):'I', (1,0):'X', (0,1):'Z', (1,1):'Y'}
                pauli_str += mapping[(b_bits[i], c_bits[i])]
            labels.append(pauli_str)
    return labels


def coset_integral_weights(eX, eZ, code, p, n_beta=41, n_samples=4000, burn_in=None, rng=None):
    """Estimate coset weights using thermodynamic integration and return unnormalized weights."""
    if rng is None:
        rng = np.random.default_rng()
    if burn_in is None:
        burn_in = max(1, n_samples // 4)

    beta = -np.log(p / (1.0 - p))
    betas = np.linspace(0.0, beta, n_beta)
    p_vals = 3.0 * np.exp(-betas) / (1.0 + 3.0 * np.exp(-betas))

    HZ, HX = code.stabilizer_matrices()
    Zstab_vecs = [HZ[i] for i in range(HZ.shape[0])]
    Xstab_vecs = [HX[i] for i in range(HX.shape[0])]
    all_stabs = Xstab_vecs + Zstab_vecs
    n_X_stabs = len(Xstab_vecs)

    sectors = generate_all_sectors(eX, eZ, code)
    num_classes = len(sectors)
    avg_weights = np.zeros((num_classes, n_beta), dtype=float)

    for k, (eX_rep, eZ_rep) in enumerate(sectors):
        for i, p_val in enumerate(p_vals):
            q_error = p_val / (3.0 - 2.0 * p_val)
            avg_weight, _, _, _, _ = metropolis_hastings_avg_weight(
                eX_rep, eZ_rep, all_stabs, n_X_stabs,
                q_error, n_samples, burn_in
            )
            avg_weights[k, i] = avg_weight

    free_energies = np.array([simpson_integral(betas, avg_weights[k]) for k in range(num_classes)])
    weights = np.exp(-free_energies)
    return weights, coset_labels(code)


def coset_probs_integral(eX, eZ, code, p, n_beta=41, n_samples=4000, burn_in=None, rng=None):
    weights, labels = coset_integral_weights(eX, eZ, code, p, n_beta, n_samples, burn_in, rng)
    probs = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights) / len(weights)
    return probs, labels


def get_all_syndromes(code):
    def _gen_single_type(n_stabs):
        for bits in product([0, 1], repeat=n_stabs):
            yield np.array(bits, dtype=int)

    syndZs = list(_gen_single_type(len(code.Z_stabilizers)))
    syndXs = list(_gen_single_type(len(code.X_stabilizers)))
    
    return [(sz, sx) for sz in syndZs for sx in syndXs]

def syndrome_probs(code, p):  
    HZ, HX = code.stabilizer_matrices()
    all_syndromes = get_all_syndromes(code)
    
    probs_dict = {}
    total_sum = 0.0
    
    for sz, sx in all_syndromes:
        # Find representative error vectors for the given syndrome bitstrings
        eX = ge_initialize_given_syndrome(HZ, sz)
        eZ = ge_initialize_given_syndrome(HX, sx)
        
        # Sum probabilities across all logical cosets to get total syndrome probability
        probs, _ = coset_probs_exact(eX, eZ, code, p)
        synd_prob = sum(probs)
        
        # Store using hashable tuple keys
        key = (tuple(sz), tuple(sx))
        probs_dict[key] = synd_prob
        total_sum += synd_prob
        
    # Normalize probabilities so they sum to 1
    if total_sum > 0:
        for key in probs_dict:
            probs_dict[key] /= total_sum
            
    return probs_dict

def bar_graph_syndrome_avg_with_exact(code, p, n_synd_samples=1000):
    HZ, HX = code.stabilizer_matrices()
    #probs_dict = syndrome_probs(code, p)
    n = code.n
    syndrome_counts = {}
    for _ in range(n_synd_samples):
        ex, ez = depolarizing_noise(n, p)
        sZ = synd.syndrome_from_eX(ex, code.Z_stabilizers)
        sX = synd.syndrome_from_eZ(ez, code.X_stabilizers)
        key = (tuple(sZ), tuple(sX))
        if key not in syndrome_counts:
            syndrome_counts[key] = 0
        syndrome_counts[key] += 1
    
    avg_probs = None
    avg_mcmc_probs = None
    avg_min_weight_probs = None
    labels = None

    total = sum(syndrome_counts.values())
    
    for (sz_tuple, sx_tuple), count in syndrome_counts.items():
        '''if p_syndrome == 0:
            continue
            
        sz = np.array(sz_tuple)
        sx = np.array(sx_tuple)
        
        # Find representative error configuration for this syndrome
        eX = ge_initialize_given_syndrome(HZ, sz)
        eZ = ge_initialize_given_syndrome(HX, sx)'''

        w = count / total  # empirical syndrome weight
        sz = np.array(sz_tuple)
        sx = np.array(sx_tuple)

        # Use a valid representative error for this syndrome
        eX = ge_initialize_given_syndrome(HZ, sz)
        eZ = ge_initialize_given_syndrome(HX, sx)
        
        # Get exact coset probabilities: P(L_i and S)
        probs, current_labels = coset_probs_exact(eX, eZ, code, p)
        
        # Get worm decoder estimates for this syndrome
        mcmc_probs, _ = coset_probs_worm(eX, eZ, code, p)
        
        # Calculate P(L_i | S) = P(L_i and S) / P(S)
        s_sum = sum(probs)
        if s_sum > 0:
            cond_probs = np.array(probs) / s_sum
            cond_mcmc_probs = np.array(mcmc_probs)
            
            if avg_probs is None:
                avg_probs = np.zeros(len(probs))
                avg_mcmc_probs = np.zeros(len(probs))
                labels = current_labels
            
            # Accumulate weighted contribution using empirical syndrome frequency
            avg_probs += w * cond_probs
            avg_mcmc_probs += w * cond_mcmc_probs
    
    if avg_probs is None:
        return

    num_cosets = len(avg_probs)
    indices = np.arange(num_cosets)
    width = 0.25

    plt.figure(figsize=(12, 7))
    plt.bar(indices - width/2, avg_probs, width, label='Exact Coset Prob', color='skyblue', alpha=0.8)
    plt.bar(indices + width/2, avg_mcmc_probs, width, label='Worm Decoder Coset Prob', color='orange', alpha=0.8)

    plt.xlabel('Logical Coset')
    plt.ylabel('Expected Probability')
    plt.title(f'Syndrome-Averaged Logical Coset Probabilities (L={code.L}, p={p})')
    if labels:
        plt.xticks(indices, labels, rotation=45)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'syndrome_avg_coset_probs_L{code.L}_p{p}.pdf')
    plt.close()

def bar_graph_proxies(code, p, use_exact=False):
    HZ, HX = code.stabilizer_matrices()
    #probs_dict = syndrome_probs(code, p)
    n = code.n
    eX, eZ = depolarizing_noise(n, p)

    if use_exact:
        exact_probs, labels = coset_probs_exact(eX, eZ, code, p)
        exact_norm = np.array(exact_probs) / np.sum(exact_probs)
    
    avg_wts, min_wts, labels, _ = coset_avg_wt_mcmc(eX, eZ, code, p, n_samples=code.L**4, burn_in=code.L**4//4)
    mcmc_norm = avg_wts / np.sum(avg_wts)
    
    p_min_wts = np.array([(p/3)**int(w) * (1-p)**(n-int(w)) for w in min_wts])
    min_wt_norm = p_min_wts / np.sum(p_min_wts)

    integral_norm, labels = coset_probs_integral(eX, eZ, code, p)
    coset_probs_norm, labels = coset_probs_mcmc(eX, eZ, code, p, n_samples=code.L**4, burn_in=code.L**4//4) 

    # --- Plotting ---
    num_cosets = len(integral_norm)
    indices = np.arange(num_cosets)
    
    plt.figure(figsize=(12, 7))
    
    if use_exact:
        width = 0.15
        plt.bar(indices - 2*width, exact_norm, width, label='Exact Prob', color='skyblue', alpha=0.8)
        plt.bar(indices - width, integral_norm, width, label='Integral Prob', color='orange', alpha=0.8)
        #plt.bar(indices, mcmc_norm, width, label='MCMC Avg Wt', color='green', alpha=0.8)
        plt.bar(indices + width, coset_probs_norm, width, label='MCMC Prob', color='purple', alpha=0.8)
        plt.bar(indices + 2*width, min_wt_norm, width, label='Min Weight Prob', color='red', alpha=0.8)
    else:
        width = 0.2
        plt.bar(indices - 1.5*width, integral_norm, width, label='Integral Prob', color='orange', alpha=0.8)
        #plt.bar(indices - 0.5*width, mcmc_norm, width, label='MCMC Avg Wt', color='green', alpha=0.8)
        plt.bar(indices + 0.5*width, coset_probs_norm, width, label='MCMC Prob', color='purple', alpha=0.8)
        plt.bar(indices + 1.5*width, min_wt_norm, width, label='Min Weight Prob', color='red', alpha=0.8)

    plt.xlabel('Logical Coset')
    plt.ylabel('Probability')
    plt.title(f'Coset Probabilities Proxies Comparison (L={code.L}, p={p})')
    plt.xticks(indices, labels, rotation=45)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'coset_proxies_L{code.L}_p{p}.pdf')
    plt.close()

def syndrome_sampling(code, p, n_total=10**5, n_batch=1000):
    """
    Samples syndromes in batches and computes the Total Variation Distance (TVD)
    between the empirical distributions of consecutive batches to monitor convergence.
    """
    n = code.n
    cumulative_counts = {}
    prev_probs = None
    tvd_history = []
    batch_checkpoints = []
    total_samples = 0

    for b_idx in range(n_total // n_batch):
        for _ in range(n_batch):
            ex, ez = depolarizing_noise(n, p)
            sZ = synd.syndrome_from_eX(ex, code.Z_stabilizers)
            sX = synd.syndrome_from_eZ(ez, code.X_stabilizers)
            key = (tuple(sZ), tuple(sX))
            cumulative_counts[key] = cumulative_counts.get(key, 0) + 1
        
        total_samples += n_batch
        # Normalize the cumulative counts to get the current probability distribution
        curr_probs = {k: v / total_samples for k, v in cumulative_counts.items()}

        if prev_probs is not None:
            # Compute TVD between cumulative distributions at step i and step i-1
            all_keys = set(curr_probs.keys()) | set(prev_probs.keys())
            tvd = 0.5 * sum(abs(curr_probs.get(k, 0) - prev_probs.get(k, 0)) for k in all_keys)
            tvd_history.append(tvd)
            batch_checkpoints.append(total_samples)
            
        prev_probs = curr_probs

    if tvd_history:
        plt.figure(figsize=(10, 6))
        plt.plot(batch_checkpoints, tvd_history, marker='o', linestyle='-')
        plt.xlabel("Total Samples")
        plt.ylabel("TVD (Cumulative Convergence)")
        plt.title(f"Convergence of Syndrome Distribution (L={code.L}, p={p})")
        plt.xscale('log')
        plt.grid(True)
        plt.savefig(f"syndrome_tvd_stability_L{code.L}_p{p}.pdf")
        plt.close()

def failure_mode_syndromes(code):
    HX, HZ = code.stabilizer_matrices()
    all_syndromes = get_all_syndromes(code)
    failure_syndromes = []
    for sz, sx in all_syndromes:
        ex = ge_initialize_given_syndrome(HZ, sz)
        ez = ge_initialize_given_syndrome(HX, sx)
        mcmc_probs, _ = coset_probs_worm(ex, ez, code, p=0.15)
        base_prob = mcmc_probs[0]  # Assuming the first coset is the trivial one
        if any(prob > base_prob for prob in mcmc_probs[1:]):
            failure_syndromes.append((sz, sx))
    return failure_syndromes


def _case_from_delta_s_u(delta_U, delta_S, beta):
    if delta_U > 0 and delta_S < beta * delta_U:
        return 1
    if delta_U < 0 and delta_S > beta * delta_U:
        return 2
    if delta_U < 0 and delta_S < beta * delta_U:
        return 3
    return 4


def _case_from_delta_f_e0(delta_E0, delta_F):
    if delta_E0 > 0 and delta_F > 0:
        return 1
    if delta_E0 < 0 and delta_F < 0:
        return 2
    if delta_E0 < 0 and delta_F > 0:
        return 3
    return 4


def _load_pt_history_csv(path):
    history = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV history file not found: {path}")

    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            beta = float(row['beta'])
            # Parse g values for type-G classification
            g = {
                'II': float(row['g_II']),
                'IX': float(row['g_IX']),
                'XI': float(row['g_XI']),
                'XX': float(row['g_XX'])
            }
            deltas = {}
            for e in ['IX', 'XI', 'XX']:
                delta_U = float(row[f'delta_U_{e}'])
                delta_S = float(row[f'delta_S_{e}'])
                case = _case_from_delta_s_u(delta_U, delta_S, beta)
                deltas[e] = {
                    'delta_E0': float(row[f'delta_E0_{e}']),
                    'delta_U': delta_U,
                    'delta_F': float(row[f'delta_F_{e}']),
                    'delta_S': delta_S,
                    'case': case
                }
            history.append({'g': g, 'deltas': deltas})
    return history


def _load_type_g_csv(path, beta):
    history = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV history file not found: {path}")

    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Parse g values for type-G classification
            g = {
                'II': float(row['g_II']),
                'IX': float(row['g_IX']),
                'XI': float(row['g_XI']),
                'XX': float(row['g_XX'])
            }
            E0 = {e: float(row[f'E0_{e}']) for e in ['II', 'IX', 'XI', 'XX']}
            U = {e: float(row[f'U_{e}']) for e in ['II', 'IX', 'XI', 'XX']}
            F = {e: float(row[f'F_{e}']) for e in ['II', 'IX', 'XI', 'XX']}
            deltas = {}
            for e in ['IX', 'XI', 'XX']:
                delta_E0 = E0[e] - E0['II']
                delta_U = U[e] - U['II']
                delta_F = F[e] - F['II']
                delta_S = beta * (delta_U - delta_F)
                case = _case_from_delta_s_u(delta_U, delta_S, beta)
                deltas[e] = {
                    'delta_E0': delta_E0,
                    'delta_U': delta_U,
                    'delta_F': delta_F,
                    'delta_S': delta_S,
                    'case': case
                }
            history.append({'g': g, 'deltas': deltas})
    return history


def _history_record_is_G(record):
    g = record.get('g') if isinstance(record, dict) else None
    if not isinstance(g, dict) or 'II' not in g:
        return False
    try:
        base = np.log(g['II'])
    except Exception:
        return False
    return any(np.log(g_val) - base > 0 for key, g_val in g.items() if key != 'II')


def _ensure_history(history):
    if isinstance(history, str):
        return _load_pt_history_csv(history)
    if isinstance(history, dict):
        return history.get('history', [])
    return history


def delta_s_vs_delta_u_plot(history, beta, filename='deltaS_vs_deltaU.pdf'):
    """Scatter plot of ΔS vs ΔU with the decision boundary ΔS = β ΔU."""
    history = _ensure_history(history)

    delta_U_vals = []
    delta_S_vals = []
    cases = []
    type_G_flags = []

    for rec in history:
        is_G = _history_record_is_G(rec)
        for coset, details in rec['deltas'].items():
            delta_U = details['delta_U']
            delta_S = details['delta_S']
            delta_U_vals.append(delta_U)
            delta_S_vals.append(delta_S)
            cases.append(_case_from_delta_s_u(delta_U, delta_S, beta))
            type_G_flags.append(is_G)

    case_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for c in cases:
        case_counts[c] += 1
    type_g_count = sum(type_G_flags)

    case_colors = {
        1: 'tab:green',
        2: 'tab:orange',
        3: 'tab:blue',
        4: 'gold'
    }
    fill_colors = {
        1: (0.0, 0.5, 0.0, 0.15),
        2: (1.0, 0.65, 0.0, 0.15),
        3: (0.0, 0.45, 0.7, 0.15),
        4: (1.0, 1.0, 0.0, 0.15)
    }

    x_min, x_max = min(delta_U_vals), max(delta_U_vals)
    x_range = x_max - x_min if x_max != x_min else 1.0
    x_pad = 0.1 * x_range
    x_plot_min = x_min - x_pad
    x_plot_max = x_max + x_pad

    y_min = min(delta_S_vals)
    y_max = max(delta_S_vals)
    y_range = y_max - y_min if y_max != y_min else 1.0
    y_pad = 0.1 * y_range
    y_plot_min = y_min - y_pad
    y_plot_max = y_max + y_pad

    x_pos = np.linspace(max(0, x_plot_min), x_plot_max, 300)
    x_neg = np.linspace(x_plot_min, min(0, x_plot_max), 300)

    plt.figure(figsize=(8, 6))
    plt.fill_between(x_pos, y_plot_min, beta * x_pos, facecolor=fill_colors[1], interpolate=True)
    plt.fill_between(x_neg, beta * x_neg, y_plot_max, facecolor=fill_colors[2], interpolate=True)
    plt.fill_between(x_neg, y_plot_min, beta * x_neg, facecolor=fill_colors[3], interpolate=True)
    plt.fill_between(x_pos, beta * x_pos, y_plot_max, facecolor=fill_colors[4], interpolate=True)

    for case_id, color in case_colors.items():
        x_case = [x for x, c in zip(delta_U_vals, cases) if c == case_id]
        y_case = [y for y, c in zip(delta_S_vals, cases) if c == case_id]
        if x_case:
            plt.scatter(x_case, y_case, c=color, alpha=0.85, label=f'Case {case_id}', s=50)

    # Overlay purple rings on type G points
    type_g_plotted = False
    for case_id, color in case_colors.items():
        x_case = [x for x, c, g in zip(delta_U_vals, cases, type_G_flags) if c == case_id and g]
        y_case = [y for y, c, g in zip(delta_S_vals, cases, type_G_flags) if c == case_id and g]
        if x_case:
            label = 'Type G' if not type_g_plotted else None
            plt.scatter(x_case, y_case, facecolors='none', edgecolors='purple', linewidths=1.2, s=50, label=label)
            type_g_plotted = True

    plt.plot(np.linspace(x_plot_min, x_plot_max, 2), beta * np.linspace(x_plot_min, x_plot_max, 2),
             color='tab:red', linestyle='--', label=r'$\Delta S = \beta \, \Delta U$')

    plt.xlabel(r'$\Delta U$')
    plt.ylabel(r'$\Delta S$')
    plt.title('ΔS vs ΔU with thermodynamic decision boundary')
    plt.xlim(x_plot_min, x_plot_max)
    plt.ylim(y_plot_min, y_plot_max)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    text_str = f"Case 1: {case_counts[1]}, Case 2: {case_counts[2]}, Case 3: {case_counts[3]}, Case 4: {case_counts[4]}, Type G: {type_g_count}"
    plt.figtext(0.5, 0.01, text_str, ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def delta_f_vs_delta_e0_plot(history, filename='deltaF_vs_deltaE0.pdf'):
    """Scatter plot of ΔF vs ΔE0 for false cosets."""
    history = _ensure_history(history)

    delta_E0_vals = []
    delta_F_vals = []
    cases = []
    type_G_flags = []

    for rec in history:
        is_G = _history_record_is_G(rec)
        for coset, details in rec['deltas'].items():
            delta_E0 = details['delta_E0']
            delta_F = details['delta_F']
            delta_E0_vals.append(delta_E0)
            delta_F_vals.append(delta_F)
            cases.append(details['case'])
            type_G_flags.append(is_G)

    if not delta_E0_vals:
        raise ValueError('No history records found for ΔF vs ΔE0 plotting.')

    case_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for c in cases:
        case_counts[c] += 1
    type_g_count = sum(type_G_flags)

    case_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for c in cases:
        case_counts[c] += 1

    case_colors = {
        1: 'tab:green',
        2: 'tab:orange',
        3: 'tab:blue',
        4: 'gold'
    }
    fill_colors = {
        1: (0.0, 0.5, 0.0, 0.15),
        2: (1.0, 0.65, 0.0, 0.15),
        3: (0.0, 0.45, 0.7, 0.15),
        4: (1.0, 1.0, 0.0, 0.15)
    }

    x_min, x_max = min(delta_E0_vals), max(delta_E0_vals)
    x_range = x_max - x_min if x_max != x_min else 1.0
    x_pad = 0.1 * x_range
    x_plot_min = x_min - x_pad
    x_plot_max = x_max + x_pad

    y_min = min(delta_F_vals)
    y_max = max(delta_F_vals)
    y_range = y_max - y_min if y_max != y_min else 1.0
    y_pad = 0.1 * y_range
    y_plot_min = y_min - y_pad
    y_plot_max = y_max + y_pad

    plt.figure(figsize=(8, 6))
    plt.fill([0, x_plot_max, x_plot_max, 0], [0, 0, y_plot_max, y_plot_max],
             color=fill_colors[1], alpha=0.15)
    plt.fill([x_plot_min, 0, 0, x_plot_min], [y_plot_min, y_plot_min, 0, 0],
             color=fill_colors[2], alpha=0.15)
    plt.fill([x_plot_min, 0, 0, x_plot_min], [0, 0, y_plot_max, y_plot_max],
             color=fill_colors[3], alpha=0.15)
    plt.fill([0, x_plot_max, x_plot_max, 0], [y_plot_min, y_plot_min, 0, 0],
             color=fill_colors[4], alpha=0.15)

    for case_id, color in case_colors.items():
        x_case = [x for x, c in zip(delta_E0_vals, cases) if c == case_id]
        y_case = [y for y, c in zip(delta_F_vals, cases) if c == case_id]
        if x_case:
            plt.scatter(x_case, y_case, c=color, alpha=0.85, label=f'Case {case_id}', s=50)

    # Overlay purple rings on type G points
    type_g_plotted = False
    for case_id, color in case_colors.items():
        x_case = [x for x, c, g in zip(delta_E0_vals, cases, type_G_flags) if c == case_id and g]
        y_case = [y for y, c, g in zip(delta_F_vals, cases, type_G_flags) if c == case_id and g]
        if x_case:
            label = 'Type G' if not type_g_plotted else None
            plt.scatter(x_case, y_case, facecolors='none', edgecolors='purple', linewidths=1.2, s=50, label=label)
            type_g_plotted = True

    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.axvline(0, color='black', linestyle='--', linewidth=1)

    plt.xlabel(r'$\,\Delta E_0$')
    plt.ylabel(r'$\,\Delta F$')
    plt.title('ΔF vs ΔE_0 for pathology candidate syndromes')
    plt.xlim(x_plot_min, x_plot_max)
    plt.ylim(y_plot_min, y_plot_max)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    text_str = f"Case 1: {case_counts[1]}, Case 2: {case_counts[2]}, Case 3: {case_counts[3]}, Case 4: {case_counts[4]}, Type G: {type_g_count}"
    plt.figtext(0.5, 0.01, text_str, ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def delta_f_vs_delta_u_plot(history, filename='deltaF_vs_deltaU.pdf'):
    """Scatter plot of ΔF vs ΔU for false cosets, showing quadrants for cases 1-4."""
    history = _ensure_history(history)

    delta_U_vals = []
    delta_F_vals = []
    cases = []
    type_G_flags = []

    for rec in history:
        is_G = _history_record_is_G(rec)
        for coset, details in rec['deltas'].items():
            delta_U_vals.append(details['delta_U'])
            delta_F_vals.append(details['delta_F'])
            cases.append(details['case'])
            type_G_flags.append(is_G)

    if not delta_U_vals:
        return

    case_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for c in cases:
        case_counts[c] += 1
    type_g_count = sum(type_G_flags)

    case_colors = {1: 'tab:green', 2: 'tab:orange', 3: 'tab:blue', 4: 'gold'}
    fill_colors = {
        1: (0.0, 0.5, 0.0, 0.15),
        2: (1.0, 0.65, 0.0, 0.15),
        3: (0.0, 0.45, 0.7, 0.15),
        4: (1.0, 1.0, 0.0, 0.15)
    }

    x_min, x_max = min(delta_U_vals), max(delta_U_vals)
    x_pad = 0.1 * (x_max - x_min if x_max != x_min else 1.0)
    x_plot_min, x_plot_max = x_min - x_pad, x_max + x_pad

    y_min, y_max = min(delta_F_vals), max(delta_F_vals)
    y_pad = 0.1 * (y_max - y_min if y_max != y_min else 1.0)
    y_plot_min, y_plot_max = y_min - y_pad, y_max + y_pad

    plt.figure(figsize=(8, 6))
    # Quadrant Fills
    plt.fill([0, x_plot_max, x_plot_max, 0], [0, 0, y_plot_max, y_plot_max], color=fill_colors[1], alpha=0.15) # Q1: Case 1
    plt.fill([x_plot_min, 0, 0, x_plot_min], [y_plot_min, y_plot_min, 0, 0], color=fill_colors[2], alpha=0.15) # Q3: Case 2
    plt.fill([x_plot_min, 0, 0, x_plot_min], [0, 0, y_plot_max, y_plot_max], color=fill_colors[3], alpha=0.15) # Q2: Case 3
    plt.fill([0, x_plot_max, x_plot_max, 0], [y_plot_min, y_plot_min, 0, 0], color=fill_colors[4], alpha=0.15) # Q4: Case 4

    for case_id, color in case_colors.items():
        x_case = [x for x, c in zip(delta_U_vals, cases) if c == case_id]
        y_case = [y for y, c in zip(delta_F_vals, cases) if c == case_id]
        if x_case:
            plt.scatter(x_case, y_case, c=color, alpha=0.85, label=f'Case {case_id}', s=50)

    type_g_plotted = False
    for case_id, color in case_colors.items():
        x_case = [x for x, c, g in zip(delta_U_vals, cases, type_G_flags) if c == case_id and g]
        y_case = [y for y, c, g in zip(delta_F_vals, cases, type_G_flags) if c == case_id and g]
        if x_case:
            label = 'Type G' if not type_g_plotted else None
            plt.scatter(x_case, y_case, facecolors='none', edgecolors='purple', linewidths=1.2, s=50, label=label)
            type_g_plotted = True

    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.axvline(0, color='black', linestyle='--', linewidth=1)

    plt.xlabel(r'$\Delta U$')
    plt.ylabel(r'$\Delta F$')
    plt.title('ΔF vs ΔU with case quadrants')
    plt.xlim(x_plot_min, x_plot_max)
    plt.ylim(y_plot_min, y_plot_max)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    text_str = f"Case 1: {case_counts[1]}, Case 2: {case_counts[2]}, Case 3: {case_counts[3]}, Case 4: {case_counts[4]}, Type G: {type_g_count}"
    plt.figtext(0.5, 0.01, text_str, ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def delta_f_minus_delta_u_vs_delta_e0_plot(history, filename='deltaF_minus_deltaU_vs_deltaE0.pdf'):
    """Scatter plot of (ΔF - ΔU) vs ΔE0. Since ΔF - ΔU = -ΔS/β, this shows entropy change vs ground state energy."""
    history = _ensure_history(history)

    delta_E0_vals = []
    delta_diff_vals = []
    cases = []
    type_G_flags = []

    for rec in history:
        is_G = _history_record_is_G(rec)
        for coset, details in rec['deltas'].items():
            delta_E0_vals.append(details['delta_E0'])
            delta_diff_vals.append(details['delta_F'] - details['delta_U'])
            cases.append(details['case'])
            type_G_flags.append(is_G)

    if not delta_E0_vals:
        return

    case_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for c in cases:
        case_counts[c] += 1
    type_g_count = sum(type_G_flags)

    case_colors = {1: 'tab:green', 2: 'tab:orange', 3: 'tab:blue', 4: 'gold'}
    fill_colors = {
        1: (0.0, 0.5, 0.0, 0.15),
        2: (1.0, 0.65, 0.0, 0.15),
        3: (0.0, 0.45, 0.7, 0.15),
        4: (1.0, 1.0, 0.0, 0.15)
    }

    x_min, x_max = min(delta_E0_vals), max(delta_E0_vals)
    x_pad = 0.1 * (x_max - x_min if x_max != x_min else 1.0)
    x_plot_min, x_plot_max = x_min - x_pad, x_max + x_pad

    y_min, y_max = min(delta_diff_vals), max(delta_diff_vals)
    y_pad = 0.1 * (y_max - y_min if y_max != y_min else 1.0)
    y_plot_min, y_plot_max = y_min - y_pad, y_max + y_pad

    plt.figure(figsize=(8, 6))
    # For this specific plane, we fill by ΔE0 sign but Case coloring handles the rest
    plt.fill([x_plot_min, 0, 0, x_plot_min], [y_plot_min, y_plot_min, y_plot_max, y_plot_max], 
             color=(0.5, 0.5, 0.5, 0.05), label='Type 1 Candidate' if x_plot_min < 0 else None)

    for case_id, color in case_colors.items():
        x_case = [x for x, c in zip(delta_E0_vals, cases) if c == case_id]
        y_case = [y for y, c in zip(delta_diff_vals, cases) if c == case_id]
        if x_case:
            plt.scatter(x_case, y_case, c=color, alpha=0.85, label=f'Case {case_id}', s=50)

    type_g_plotted = False
    for case_id, color in case_colors.items():
        x_case = [x for x, c, g in zip(delta_E0_vals, cases, type_G_flags) if c == case_id and g]
        y_case = [y for y, c, g in zip(delta_diff_vals, cases, type_G_flags) if c == case_id and g]
        if x_case:
            label = 'Type G' if not type_g_plotted else None
            plt.scatter(x_case, y_case, facecolors='none', edgecolors='purple', linewidths=1.2, s=50, label=label)
            type_g_plotted = True

    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.axvline(0, color='black', linestyle='--', linewidth=1)

    plt.xlabel(r'$\Delta E_0$')
    plt.ylabel(r'$\Delta F - \Delta U$ ($-\Delta S / \beta$)')
    plt.title('ΔF - ΔU vs ΔE_0')
    plt.xlim(x_plot_min, x_plot_max)
    plt.ylim(y_plot_min, y_plot_max)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    text_str = f"Case 1: {case_counts[1]}, Case 2: {case_counts[2]}, Case 3: {case_counts[3]}, Case 4: {case_counts[4]}, Type G: {type_g_count}"
    plt.figtext(0.5, 0.01, text_str, ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()