import matplotlib.pyplot as plt
import numpy as np
import itertools
import random
from noise import depolarizing_noise
from syndrome import syndrome_from_eX, syndrome_from_eZ
import utils
from MH_sampler import metropolis_hastings_on_stabilizers
from simulation import run_trial

def plot_mh_traces(code, p, decoder_type='MH', n_samples=3000, burn_in=500):
    """
    Run MH decoding once and plot trace diagnostics for X and Z chains.
    """
    if decoder_type == 'MH':
        # Marginal probability for independent chains: q = 2p/(3-p)
        # Odds: (2p/3) / (1 - p)
        q = 2 * p / (3 - p)
    else:
        # For joint decoding, exact log-odds requires q = p / (3 - 2p)
        q = p / (3 - 2 * p)

    # Sample true Pauli-frame noise
    eX, eZ = depolarizing_noise(code.n, p)

    # Compute syndromes
    syndZ = syndrome_from_eX(eX, code.Z_stabilizers)
    syndX = syndrome_from_eZ(eZ, code.X_stabilizers)

    # Stabilizer matrices
    HZ, HX = code.stabilizer_matrices()

    # Initial solutions (MWPM / Gaussian elim)
    eX_init = utils.mwpm_initialize_e_given_syndrome(HZ, syndZ)
    eZ_init = utils.mwpm_initialize_e_given_syndrome(HX, syndX)

    # Stabilizer vectors (proposal moves)
    Zstab_vecs = [HZ[i] for i in range(HZ.shape[0])]
    Xstab_vecs = [HX[i] for i in range(HX.shape[0])]

    if decoder_type == 'MH':
        # Run MH chains
        outX = metropolis_hastings_on_stabilizers(
            code, HZ, eX_init.copy(), Xstab_vecs,
            q_error=q, n_samples=n_samples, burn_in=burn_in
        )

        outZ = metropolis_hastings_on_stabilizers(
            code, HX, eZ_init.copy(), Zstab_vecs,
            q_error=q, n_samples=n_samples, burn_in=burn_in
        )

        # ---- Plot traces ----
        fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

        axs[0].plot(outX['trace_logp'], label='X-error chain')
        axs[0].axvline(burn_in, color='red', linestyle='--', label='burn-in')
        axs[0].set_ylabel('log posterior')
        axs[0].legend()

        axs[1].plot(outZ['trace_logp'], label='Z-error chain')
        axs[1].axvline(burn_in, color='red', linestyle='--', label='burn-in')
        axs[1].set_ylabel('log posterior')
        axs[1].set_xlabel('MH iteration')
        axs[1].legend()

        fig.tight_layout()
        plt.show()

    elif decoder_type in ['SingleChain', 'TrackZ']:
        # Single chain logic
        all_stabs = Xstab_vecs + Zstab_vecs
        n_X_stabs = len(Xstab_vecs)
        m_stab = len(all_stabs)
        
        cur_eX = eX_init.copy()
        cur_eZ = eZ_init.copy()
        cur_weight = np.sum(cur_eX | cur_eZ)
        log_odds = np.log(q / (1.0 - q))
        cur_logp = cur_weight * log_odds
        
        trace_logp = []
        
        for _ in range(n_samples):
            j = random.randrange(m_stab)
            svec = all_stabs[j]
            is_X_stab = (j < n_X_stabs)
            flip_indices = svec.nonzero()[0]
            
            if is_X_stab:
                delta_w = np.sum((cur_eX[flip_indices] ^ 1) | cur_eZ[flip_indices]) - np.sum(cur_eX[flip_indices] | cur_eZ[flip_indices])
            else:
                delta_w = np.sum(cur_eX[flip_indices] | (cur_eZ[flip_indices] ^ 1)) - np.sum(cur_eX[flip_indices] | cur_eZ[flip_indices])
            
            if (cur_logp + delta_w * log_odds) > cur_logp or random.random() < np.exp(delta_w * log_odds):
                if is_X_stab: cur_eX[flip_indices] ^= 1
                else: cur_eZ[flip_indices] ^= 1
                cur_weight += delta_w
                cur_logp += delta_w * log_odds
            
            trace_logp.append(cur_logp)
            
        plt.figure(figsize=(9, 5))
        plt.plot(trace_logp, label='Joint (X+Z) chain')
        plt.axvline(burn_in, color='red', linestyle='--', label='burn-in')
        plt.ylabel('log posterior')
        plt.xlabel('MH iteration')
        plt.title(f'MH {decoder_type} Trace (p={p})')
        plt.legend()
        plt.tight_layout()
        plt.show()

    elif decoder_type == 'Parallel':
        # Parallel logic: dynamically determine number of chains
        n = code.n

        # Dynamically generate logical operators
        log_X_supports = [s for s in [code.logical_X_support(), code.logical_X_conjugate()] if s]
        log_Z_supports = [s for s in [code.logical_Z_support(), code.logical_Z_conjugate()] if s]

        num_logical_qubits = len(log_X_supports)
        if num_logical_qubits != len(log_Z_supports):
            raise ValueError("Inconsistent number of logical X and Z operators.")

        log_X_op_vecs = []
        for support in log_X_supports:
            vec = np.zeros(n, dtype=int)
            vec[support] = 1
            log_X_op_vecs.append(vec)

        log_Z_op_vecs = []
        for support in log_Z_supports:
            vec = np.zeros(n, dtype=int)
            vec[support] = 1
            log_Z_op_vecs.append(vec)

        logicals_X = []
        logicals_Z = []

        lX_combinations = []
        for b_bits in itertools.product([0, 1], repeat=num_logical_qubits):
            lX = np.zeros(n, dtype=int)
            for i, b in enumerate(b_bits):
                if b: lX ^= log_X_op_vecs[i]
            lX_combinations.append(lX)

        lZ_combinations = []
        for c_bits in itertools.product([0, 1], repeat=num_logical_qubits):
            lZ = np.zeros(n, dtype=int)
            for i, c in enumerate(c_bits):
                if c: lZ ^= log_Z_op_vecs[i]
            lZ_combinations.append(lZ)

        for lZ in lZ_combinations:
            for lX in lX_combinations:
                logicals_X.append(lX)
                logicals_Z.append(lZ)

        all_stabs = Xstab_vecs + Zstab_vecs
        n_X_stabs = len(Xstab_vecs)
        m_stab = len(all_stabs)
        log_odds = np.log(q / (1.0 - q))
        
        traces = []
        
        num_classes = len(logicals_X)
        for k in range(num_classes):
            cur_eX = eX_init ^ logicals_X[k]
            cur_eZ = eZ_init ^ logicals_Z[k]
            cur_weight = np.sum(cur_eX | cur_eZ)
            cur_logp = cur_weight * log_odds
            
            chain_trace = []
            for _ in range(n_samples):
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
                
                chain_trace.append(cur_logp)
            traces.append(chain_trace)
            
        plt.figure(figsize=(10, 6))
        for k, trace in enumerate(traces):
            plt.plot(trace, alpha=0.5, label=f'Class {k}' if k%4==0 else None)
            
        plt.axvline(burn_in, color='red', linestyle='--', label='burn-in')
        plt.ylabel('log posterior')
        plt.xlabel('MH iteration')
        plt.title(f'MH Parallel Chains ({num_classes} classes) (p={p})')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
    
    else:
        raise ValueError(f"Unknown decoder_type: {decoder_type}")

def error_rate_vs_n_sample(code, p, decoder, n_samples=1000):
    failures = 0
    rates = []
    for i in range(n_samples):
        failures += run_trial(code, p, decoder)
        rates.append(failures / (i + 1))
    return rates