import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from code import ToricCode, PlanarSurfaceCode
from simulation import run_trial
import os
import glob
import re
import pandas as pd
import csv
from syndrome import syndrome_from_eX, syndrome_from_eZ
from decoder import MHDecoderSingleChain, MWPMDecoder, MHDecoderParallel, MHDecoderTrackZ, BPDecoder
from evaluation import coset_avg_wt_mcmc, coset_probs_worm, coset_integral_weights
from utils import d_kl, mbar, ge_initialize_given_syndrome
from noise import depolarizing_noise

def logical_error_rate(code, p, decoder, n_trials=1000):
    failures = 0
    for _ in range(n_trials):
        failures += run_trial(code, p, decoder)
    return failures / n_trials

def P_vs_L_plot(L_list, p_list, decoder_factory, trials=2000, code_type='Toric'):
    results = experiment(L_list, p_list, decoder_factory, trials, code_type)

    for p in p_list:
        rates = results[p]
        #plt.plot(L_list, rates, marker='s', label=f"p={p}")

    # Save data to CSV
    with open(f'p_vs_l_data_{code_type}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['p'] + [f'L={L}' for L in L_list])
        for p in p_list:
            rates = results[p]
            writer.writerow([p] + rates)

    '''plt.xlabel("Lattice Size L")
    plt.ylabel("Logical error rate")
    plt.legend()
    plt.yscale("log")
    plt.grid(True, which="both", ls="--")
    plt.show()'''

def threshold_plot(L_list, p_list, decoder_factory, trials=2000, code_type='Toric'):
    """Plots the logical error rate P vs physical error rate p for all L."""
    results = experiment(L_list, p_list, decoder_factory, trials, code_type)

    for i, L in enumerate(L_list):
        rates = [results[p][i] for p in p_list]
        #plt.plot(p_list, rates, marker='o', label=f"L={L}")

    # Use the decoder type name for output filenames when available
    sample_code = ToricCode(L_list[0]) if code_type == 'Toric' else PlanarSurfaceCode(L_list[0])
    sample_decoder = decoder_factory(sample_code, p_list[0])
    decoder_name = type(sample_decoder).__name__
    decoder_name = decoder_name.lower().replace('decoder', '').strip() or 'decoder'

    # Save data to CSV
    with open(f'threshold_data_{code_type}_{decoder_name}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['p'] + [f'L={L}' for L in L_list])  # header row
        for i, p in enumerate(p_list):
            rates = [results[p][j] for j in range(len(L_list))]
            writer.writerow([p] + rates)

    '''plt.xlabel("Physical error rate p")
    plt.ylabel("Logical error rate")
    plt.legend()
    plt.yscale("log")
    plt.grid(True, which="both", ls="--")
    plt.savefig(f'threshold_plot_{code_type}_{decoder_name}.pdf')''' 

def experiment(L_list, p_list, decoder_factory, trials=2000, code_type='Toric'):
    results = {} # rates for every L and p

    for p in p_list:
        rates = []
        for L in L_list:
            if code_type == 'Toric':
                code = ToricCode(L)
            elif code_type == 'Planar':
                code = PlanarSurfaceCode(L)
            else:
                raise ValueError(f"Unknown code_type: {code_type}")
            decoder = decoder_factory(code, p)
            rate = logical_error_rate(code, p, decoder, trials)
            rates.append(rate)

        results[p] = rates

    return results

def comparison_plot(p_list, trials=2000, L=8, code_type='Toric'):
    if code_type == 'Toric':
        code = ToricCode(L)
    elif code_type == 'Planar':
        code = PlanarSurfaceCode(L)
    mwpm_rates = []
    mcmc_rates = []
    ratio_rates = []

    for p in p_list:
        mwpm_decoder = MWPMDecoder(code)
        mcmc_decoder = MHDecoderParallel(code, q_error=p/(3-2*p), n_samples=L**4)
        ratio_decoder = MHDecoderTrackZ(code, q_error=p/(3-2*p), n_samples=L**4, burn_in=(L**4)//4)

        mwpm_rate = logical_error_rate(code, p, mwpm_decoder, trials)
        mcmc_rate = logical_error_rate(code, p, mcmc_decoder, trials)
        ratio_rate = logical_error_rate(code, p, ratio_decoder, trials)

        mwpm_rates.append(mwpm_rate)
        mcmc_rates.append(mcmc_rate)
        ratio_rates.append(ratio_rate)

    # Save data to CSV
    with open(f'comparison_data_L{L}_{code_type}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['p', 'MWPM_rate', 'MCMC_rate', 'Ratio_estimator_rate'])
        for i, p in enumerate(p_list):
            writer.writerow([p, mwpm_rates[i], mcmc_rates[i], ratio_rates[i]])

    plt.plot(p_list, mwpm_rates, marker='s', label="MWPM Decoder")
    plt.plot(p_list, mcmc_rates, marker='o', label="MCMC Decoder")
    plt.plot(p_list, ratio_rates, marker='^', label="Ratio-Estimator Decoder")
    plt.xlabel("Physical error rate p")
    plt.ylabel("Logical error rate")
    plt.legend()
    plt.yscale("log")
    plt.grid(True, which="both", ls="--")
    plt.savefig(f'comparison_plot_L{L}_{code_type}.pdf')                                                                                                                
    #plt.show()

def d_kl_vs_p(code, p_list, code_type='Toric'):
    n = code.n
    d_kl_values_1 = []
    d_kl_values_2 = []
    d_kl_values_3 = []
    for p in p_list:
        eX, eZ = depolarizing_noise(n, p)

        avg_wts, min_wts, labels, _ = coset_avg_wt_mcmc(eX, eZ, code, p, n_samples=n**4, burn_in=n**4//4)
        mcmc_norm = avg_wts / np.sum(avg_wts)
        p_min_wts = np.array([(p/3)**int(w) * (1-p)**(n-int(w)) for w in min_wts])
        min_wt_norm = p_min_wts / np.sum(p_min_wts)

        worm_norm, labels = coset_probs_worm(eX, eZ, code, p, n_samples=500, n_burnin=100)

        d_kl_1 = d_kl(worm_norm, mcmc_norm)
        d_kl_2 = d_kl(worm_norm, min_wt_norm)
        d_kl_3 = d_kl(mcmc_norm, min_wt_norm)

        d_kl_values_1.append(d_kl_1)
        d_kl_values_2.append(d_kl_2)
        d_kl_values_3.append(d_kl_3)

    # Save data to CSV
    with open(f'd_kl_data_{code_type}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['p', 'D_KL_1', 'D_KL_2', 'D_KL_3'])
        for i, p in enumerate(p_list):
            writer.writerow([p, d_kl_values_1[i], d_kl_values_2[i], d_kl_values_3[i]])

    plt.plot(p_list, d_kl_values_1, marker='o', label='D_KL(Worm || MCMC)')
    plt.plot(p_list, d_kl_values_2, marker='s', label='D_KL(Worm || Min Weight)')
    plt.plot(p_list, d_kl_values_3, marker='^', label='D_KL(MCMC || Min Weight)')
    plt.xlabel("Physical Error Rate $p$")
    plt.ylabel("KL Divergence")
    plt.yscale("log")
    plt.legend()
    plt.title(f'KL Divergence vs Physical Error Rate ({code_type})')
    plt.grid(True, which="both", ls="--")
    plt.savefig(f'd_kl_vs_p_plot_{code_type}.pdf')
    #plt.show()

def d_kl_vs_p_fixed_syndrome(code, p_phys, p_list, code_type='Toric'):
    n = code.n
    eX_fixed, eZ_fixed = depolarizing_noise(n, p_phys) # Generate a fixed syndrome

    d_kl_values_1 = []
    d_kl_values_2 = []
    d_kl_values_3 = []

    for p in p_list:
        avg_wts, min_wts, labels, _ = coset_avg_wt_mcmc(eX_fixed, eZ_fixed, code, p, n_samples=n**4, burn_in=n**4//4)
        mcmc_norm = avg_wts / np.sum(avg_wts)
        p_min_wts = np.array([(p/3)**int(w) * (1-p)**(n-int(w)) for w in min_wts])
        min_wt_norm = p_min_wts / np.sum(p_min_wts)

        worm_norm, labels = coset_probs_worm(eX_fixed, eZ_fixed, code, p, n_samples=500, n_burnin=100)

        d_kl_1 = d_kl(worm_norm, mcmc_norm)
        d_kl_2 = d_kl(worm_norm, min_wt_norm)
        d_kl_3 = d_kl(mcmc_norm, min_wt_norm)

        d_kl_values_1.append(d_kl_1)
        d_kl_values_2.append(d_kl_2)
        d_kl_values_3.append(d_kl_3)
    # Save data to CSV
    with open(f'd_kl_data_{code_type}_L{code.L}_{p_phys}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['p', 'D_KL_1', 'D_KL_2', 'D_KL_3'])
        for i, p_val in enumerate(p_list):
            writer.writerow([p_val, d_kl_values_1[i], d_kl_values_2[i], d_kl_values_3[i]])

    plt.plot(p_list, d_kl_values_1, marker='o', label='D_KL(Worm || MCMC)')
    plt.plot(p_list, d_kl_values_2, marker='s', label='D_KL(Worm || Min Weight)')
    plt.plot(p_list, d_kl_values_3, marker='^', label='D_KL(MCMC || Min Weight)')
    plt.xlabel("Physical Error Rate $p$")
    plt.ylabel("KL Divergence")
    plt.yscale("log")
    plt.title(f'KL Divergence vs Physical Error Rate (Fixed Syndrome at $p_{{phys}}$={p_phys}, {code_type}, L={code.L})')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(f'd_kl_vs_p_plot_{code_type}_L{code.L}_{p_phys}.pdf')
    #plt.show()

def d_kl_vs_p_syndrome_avg(code, p_list, code_type='Toric', n_syndromes=50):
    n=code.n
    d_kl_values_1 = []
    d_kl_values_2 = []
    d_kl_values_3 = []
    for p in p_list:
        # Dictionary to store syndrome results: (sZ, sX) -> [ex, ez, count]
        syndrome_samples = {}
        for _ in range(n_syndromes):
            ex, ez = depolarizing_noise(n, p)
            sZ = tuple(syndrome_from_eX(ex, code.Z_stabilizers))
            sX = tuple(syndrome_from_eZ(ez, code.X_stabilizers))
            key = (sZ, sX)
            if key not in syndrome_samples:
                syndrome_samples[key] = [ex, ez, 0]
            syndrome_samples[key][2] += 1

        d_kl_1_sum = 0
        d_kl_2_sum = 0
        d_kl_3_sum = 0

        for ex, ez, count in syndrome_samples.values():
            avg_wts, min_wts, _, _ = coset_avg_wt_mcmc(ex, ez, code, p, n_samples=10000, burn_in=1000)
            m_norm = avg_wts / np.sum(avg_wts)
            p_min_wts = np.array([(p/3)**int(w) * (1-p)**(n-int(w)) for w in min_wts])
            mw_norm = p_min_wts / np.sum(p_min_wts)

            w_norm, _ = coset_probs_worm(ex, ez, code, p, n_samples=500, n_burnin=100)

            w = count / n_syndromes
            d_kl_1_sum += w * d_kl(w_norm, m_norm)
            d_kl_2_sum += w * d_kl(w_norm, mw_norm)
            d_kl_3_sum += w * d_kl(m_norm, mw_norm)

        d_kl_values_1.append(d_kl_1_sum)
        d_kl_values_2.append(d_kl_2_sum)
        d_kl_values_3.append(d_kl_3_sum)

    # Save data to CSV
    with open(f'd_kl_synd_avg_data_{code_type}_L{code.L}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['p', 'D_KL_1', 'D_KL_2', 'D_KL_3'])
        for i, p in enumerate(p_list):
            writer.writerow([p, d_kl_values_1[i], d_kl_values_2[i], d_kl_values_3[i]])

    plt.plot(p_list, d_kl_values_1, marker='o', label='D_KL(Worm || MCMC)')
    plt.plot(p_list, d_kl_values_2, marker='s', label='D_KL(Worm || Min Weight)')
    plt.plot(p_list, d_kl_values_3, marker='^', label='D_KL(MCMC || Min Weight)')
    plt.xlabel("Physical Error Rate $p$")
    plt.ylabel("Average KL Divergence")
    plt.yscale("log")
    plt.title(f'Syndrome-Averaged KL Divergence ({code_type}, L={code.L}, samples={n_syndromes})')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(f'd_kl_synd_avg_plot_{code_type}_L{code.L}.pdf')
    plt.close()


def d_kl_vs_p_mbar(code, p_list, code_type='Toric', n_syndromes=20, n_samples_mcmc=None, burn_in=None):
    """Compute syndrome-averaged D_KL vs p using MBAR reweighting across syndrome samples."""
    if n_samples_mcmc is None:
        n_samples_mcmc = max(200, code.n ** 2)
    if burn_in is None:
        burn_in = max(1, n_samples_mcmc // 4)

    p_states = np.array(p_list, dtype=float)
    K = len(p_states)
    all_samples = []
    N_k = np.full(K, n_syndromes, dtype=float)

    HZ, HX = code.stabilizer_matrices()

    for state_idx, p_state in enumerate(p_states):
        for _ in range(n_syndromes):
            ex, ez = depolarizing_noise(code.n, p_state)
            sZ = syndrome_from_eX(ex, code.Z_stabilizers)
            sX = syndrome_from_eZ(ez, code.X_stabilizers)
            all_samples.append({
                'state_idx': state_idx,
                'sZ': sZ,
                'sX': sX,
                'p_state': p_state
            })

    N = len(all_samples)
    log_prob_matrix = np.zeros((K, N), dtype=float)
    dkl_worm_mcmc = np.zeros(N, dtype=float)
    dkl_worm_mwpm = np.zeros(N, dtype=float)
    dkl_mcmc_mwpm = np.zeros(N, dtype=float)

    for n, sample in enumerate(all_samples):
        eX = ge_initialize_given_syndrome(HZ, sample['sZ'])
        eZ = ge_initialize_given_syndrome(HX, sample['sX'])

        for k, p_state in enumerate(p_states):
            weights, _ = coset_integral_weights(eX, eZ, code, p_state,
                                               n_beta=41, n_samples=n_samples_mcmc, burn_in=burn_in)
            p_s = np.sum(weights)
            log_prob_matrix[k, n] = np.log(max(p_s, 1e-300))

        worm_norm, _ = coset_probs_worm(eX, eZ, code, sample['p_state'])
        avg_wts, min_wts, _, _ = coset_avg_wt_mcmc(
            eX, eZ, code, sample['p_state'], n_samples=n_samples_mcmc, burn_in=burn_in
        )
        mcmc_norm = avg_wts / np.sum(avg_wts)
        p_min_wts = np.array([(sample['p_state']/3)**int(w) * (1-sample['p_state'])**(code.n-int(w)) for w in min_wts])
        min_wt_norm = p_min_wts / np.sum(p_min_wts)

        dkl_worm_mcmc[n] = d_kl(worm_norm, mcmc_norm)
        dkl_worm_mwpm[n] = d_kl(worm_norm, min_wt_norm)
        dkl_mcmc_mwpm[n] = d_kl(mcmc_norm, min_wt_norm)

    u_kln = -log_prob_matrix
    mbar_results = mbar(u_kln, N_k=N_k)
    weights = mbar_results['weights']

    avg_dkl_worm_mcmc = np.sum(weights * dkl_worm_mcmc[None, :], axis=1) / np.sum(weights, axis=1)
    avg_dkl_worm_mwpm = np.sum(weights * dkl_worm_mwpm[None, :], axis=1) / np.sum(weights, axis=1)
    avg_dkl_mcmc_mwpm = np.sum(weights * dkl_mcmc_mwpm[None, :], axis=1) / np.sum(weights, axis=1)

    with open(f'd_kl_mbar_data_{code_type}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['p', 'D_KL(Worm||MCMC)', 'D_KL(Worm||MWPM)', 'D_KL(MCMC||MWPM)'])
        for i, p_state in enumerate(p_states):
            writer.writerow([
                p_state,
                avg_dkl_worm_mcmc[i],
                avg_dkl_worm_mwpm[i],
                avg_dkl_mcmc_mwpm[i]
            ])

    plt.figure(figsize=(10, 7))
    plt.plot(p_states, avg_dkl_worm_mcmc, marker='o', label='D_KL(Worm || MCMC)')
    plt.plot(p_states, avg_dkl_worm_mwpm, marker='s', label='D_KL(Worm || MWPM)')
    plt.plot(p_states, avg_dkl_mcmc_mwpm, marker='^', label='D_KL(MCMC || MWPM)')
    plt.xlabel('Physical Error Rate $p$')
    plt.ylabel('Syndrome-Averaged D_KL')
    plt.yscale('log')
    plt.title(f'Syndrome-Averaged D_KL from MBAR reweighting ({code_type})')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'd_kl_vs_p_mbar_{code_type}.pdf')
    plt.close()
    return {
        'p_states': p_states,
        'dkl_worm_mcmc': avg_dkl_worm_mcmc.tolist(),
        'dkl_worm_mwpm': avg_dkl_worm_mwpm.tolist(),
        'dkl_mcmc_mwpm': avg_dkl_mcmc_mwpm.tolist(),
        'mbar': mbar_results
    }

def d_kl_from_csv_samples(code, directory, target_p_list, code_type='Toric', n_samples_mcmc=None, burn_in=None, subsample=20):
    """
    Compute syndrome-averaged D_KL vs p using MBAR reweighting with syndromes
    drawn from multiple CSV files in a directory. Each CSV is treated as a set of biased 
    ensembles (replicas) to be combined.
    """
    if n_samples_mcmc is None:
        n_samples_mcmc = max(200, code.n ** 2)
    if burn_in is None:
        burn_in = max(1, n_samples_mcmc // 4)

    target_ps = np.array(target_p_list, dtype=float)
    all_syndromes = []
    all_phis = []
    all_F_II = []
    all_F_IX = []
    all_F_XI = []
    all_F_XX = []
    all_U_II = []
    all_U_IX = []
    all_U_XI = []
    all_U_XX = []
    all_E0_II = []
    all_E0_IX = []
    all_E0_XI = []
    all_E0_XX = []
    
    # Sampling states for MBAR: (p_phys, lambda)
    sampling_states = [] # list of (p, lmbda)
    sample_source_indices = [] # which state each sample came from

    pattern = os.path.join(directory, "parallel_tempering_history_p*.csv")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"No CSV files matching pattern {pattern} found.")
        return

    print(f"Loading data from {len(files)} CSV files...")
    for fpath in files:
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            continue

        # Extract p from file (filename or beta column)
        # Header includes 'beta', and p = 1 / (1 + exp(beta)) for X/Z independent? 
        # No, PT uses beta = -log(p/(1-p)). So p = 1 / (1 + exp(beta))
        beta_val = df['beta'].iloc[0]
        p_file = 1.0 / (1.0 + np.exp(beta_val))

        # Replicas correspond to different lambda values
        replicas = df['replica'].unique()
        for rep in replicas:
            rep_df = df[df['replica'] == rep].iloc[::subsample]
            lmbda = rep_df['lambda'].iloc[0]
            
            state_idx = len(sampling_states)
            sampling_states.append((p_file, lmbda))
            
            for _, row in rep_df.iterrows():
                # Parse syndrome string: "(0, 1, ...)"
                s_str = row['syndrome'].strip('()[]')
                synd = np.array([int(x.strip()) for x in s_str.split(',')], dtype=int)
                
                all_syndromes.append(synd)
                all_phis.append(row['phi'])
                all_F_II.append(row['F_II'])
                all_F_IX.append(row['F_IX'])
                all_F_XI.append(row['F_XI'])
                all_F_XX.append(row['F_XX'])
                all_U_II.append(row['U_II'])
                all_U_IX.append(row['U_IX'])
                all_U_XI.append(row['U_XI'])
                all_U_XX.append(row['U_XX'])
                all_E0_II.append(row['E0_II'])
                all_E0_IX.append(row['E0_IX'])
                all_E0_XI.append(row['E0_XI'])
                all_E0_XX.append(row['E0_XX'])
                sample_source_indices.append(state_idx)

    N = len(all_syndromes)
    K_samp = len(sampling_states)
    K_target = len(target_ps)
    
    if N == 0:
        print("No samples found.")
        return

    print(f"Total samples: {N} across {K_samp} sampling states.")
    
    # Unique p values we need to evaluate Z_S(p) for
    unique_ps = np.unique(np.concatenate([target_ps, [s[0] for s in sampling_states]]))
    p_to_idx = {p: i for i, p in enumerate(unique_ps)}
    
    # Evaluate log(Z_S(p)) and D_KL metrics for each sample
    # log_zs[sample_idx, p_idx]
    log_zs = np.zeros((N, len(unique_ps)))
    dkl_metrics = np.zeros((K_target, N, 3)) # (target_p, sample, metric_type)

    HZ, HX = code.stabilizer_matrices()
    for i in range(N):
        if i % 50 == 0: print(f"Processing sample {i}/{N}...")
        syndX = all_syndromes[i]
        eZ = ge_initialize_given_syndrome(HX, syndX)
        eX = np.zeros(code.n, dtype=int)

        for p_val in unique_ps:
            w, _ = coset_integral_weights(eX, eZ, code, p_val, n_beta=21, 
                                         n_samples=n_samples_mcmc, burn_in=burn_in)
            log_zs[i, p_to_idx[p_val]] = np.log(max(np.sum(w), 1e-300))

    # Compute D_KL metrics using CSV values
    for t, p_t in enumerate(target_ps):
        beta_t = -np.log(p_t / (1 - p_t))
        for i in range(N):
            F_vals = np.array([all_F_II[i], all_F_IX[i], all_F_XI[i], all_F_XX[i]])
            worm_norm = np.exp(-beta_t * F_vals)
            worm_norm /= np.sum(worm_norm)
            
            U_vals = np.array([all_U_II[i], all_U_IX[i], all_U_XI[i], all_U_XX[i]])
            mcmc_norm = np.exp(-beta_t * U_vals)
            mcmc_norm /= np.sum(mcmc_norm)
            
            E0_vals = np.array([all_E0_II[i], all_E0_IX[i], all_E0_XI[i], all_E0_XX[i]])
            mw_norm = np.exp(-beta_t * E0_vals)
            mw_norm /= np.sum(mw_norm)
            
            dkl_metrics[t, i, 0] = d_kl(worm_norm, mcmc_norm)
            dkl_metrics[t, i, 1] = d_kl(worm_norm, mw_norm)
            dkl_metrics[t, i, 2] = d_kl(mcmc_norm, mw_norm)

    # Build MBAR matrix u_kn: rows are sampling states, columns are samples
    # u_kn = -log_ZS(p_k) - lambda_k * phi_n
    u_kln = np.zeros((K_samp, N))
    N_k = np.zeros(K_samp)
    for k in range(K_samp):
        p_k, lmbda_k = sampling_states[k]
        p_idx = p_to_idx[p_k]
        u_kln[k, :] = -log_zs[:, p_idx] - lmbda_k * np.array(all_phis)
        N_k[k] = sample_source_indices.count(k)

    # MBAR to find free energies of sampling states
    mbar_res = mbar(u_kln, N_k=N_k)
    
    # Now for each target state (p_t, lambda=0), calculate weights for all samples
    results_dkl = np.zeros((K_target, 3))
    for t in range(K_target):
        p_t = target_ps[t]
        # Reduced potential of all samples in target state t
        u_t = -log_zs[:, p_to_idx[p_t]] 
        # Weight of sample n in target state t
        log_denom = logsumexp(np.log(N_k)[:, None] - mbar_res['f_k'][:, None] - u_kln, axis=0)
        weights_t = np.exp(-u_t - log_denom)
        weights_t /= np.sum(weights_t)
        
        for m in range(3):
            results_dkl[t, m] = np.sum(weights_t * dkl_metrics[t, :, m])

    # Plotting
    plt.figure(figsize=(10, 7))
    markers = ['o', 's', '^']
    labels = ['D_KL(Optimal||Avg_Wt)', 'D_KL(Optimal||MWPM)', 'D_KL(Avg_Wt||MWPM)']
    for m in range(3):
        plt.plot(target_ps, results_dkl[:, m], marker=markers[m], label=labels[m])
    
    plt.xlabel('Physical Error Rate $p$'); plt.ylabel('Average D_KL'); plt.yscale('log')
    plt.title(f'Syndrome-Averaged D_KL via MBAR ({code_type}, L={code.L})')
    plt.legend(); plt.grid(True, which='both', ls='--'); plt.tight_layout()
    plt.savefig(f'd_kl_from_csv_mbar_{code_type}_L{code.L}.pdf'); plt.close()
    print(f"Plot saved: d_kl_from_csv_mbar_{code_type}_L{code.L}.pdf")


def d_kl_Z_ratios_from_csv_samples(code, directory, target_p_list, code_type='Toric', n_samples_mcmc=None, burn_in=None, subsample=20):
    """
    Compute syndrome-averaged D_KL vs p using MBAR reweighting with syndromes
    drawn from multiple CSV files in a directory. Each CSV is treated as a set of biased 
    ensembles (replicas) to be combined.
    Uses Z_ratios from CSV for the "MCMC" probabilities.
    """
    if n_samples_mcmc is None:
        n_samples_mcmc = max(200, code.n ** 2)
    if burn_in is None:
        burn_in = max(1, n_samples_mcmc // 4)

    target_ps = np.array(target_p_list, dtype=float)
    all_syndromes = []
    all_phis = []
    all_F_II = []
    all_F_IX = []
    all_F_XI = []
    all_F_XX = []
    all_Z_II = []
    all_Z_IX = []
    all_Z_XI = []
    all_Z_XX = []
    all_E0_II = []
    all_E0_IX = []
    all_E0_XI = []
    all_E0_XX = []
    
    # Sampling states for MBAR: (p_phys, lambda)
    sampling_states = [] # list of (p, lmbda)
    sample_source_indices = [] # which state each sample came from

    pattern = os.path.join(directory, "parallel_tempering_history_p*.csv")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"No CSV files matching pattern {pattern} found.")
        return

    print(f"Loading data from {len(files)} CSV files...")
    # Check whether CSV files contain Z_ratios columns
    required_Z_cols = ['Z_II', 'Z_IX', 'Z_XI', 'Z_XX']
    has_Z_cols = None

    for fpath in files:
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            continue

        if has_Z_cols is None:
            has_Z_cols = all(col in df.columns for col in required_Z_cols)
            if not has_Z_cols:
                raise ValueError(
                    f"CSV files in '{directory}' are missing Z_ratios columns. "
                    f"Required columns: {required_Z_cols}. "
                    "Regenerate PT history CSV files with the updated Z_ratios output."
                )

        # Extract p from file (filename or beta column)
        # Header includes 'beta', and p = 1 / (1 + exp(beta)) for X/Z independent? 
        # No, PT uses beta = -log(p/(1-p)). So p = 1 / (1 + exp(beta))
        beta_val = df['beta'].iloc[0]
        p_file = 1.0 / (1.0 + np.exp(beta_val))

        # Replicas correspond to different lambda values
        replicas = df['replica'].unique()
        for rep in replicas:
            rep_df = df[df['replica'] == rep].iloc[::subsample]
            lmbda = rep_df['lambda'].iloc[0]
            
            state_idx = len(sampling_states)
            sampling_states.append((p_file, lmbda))
            
            for _, row in rep_df.iterrows():
                # Parse syndrome string: "(0, 1, ...)"
                s_str = row['syndrome'].strip('()[]')
                synd = np.array([int(x.strip()) for x in s_str.split(',')], dtype=int)
                
                all_syndromes.append(synd)
                all_phis.append(row['phi'])
                all_F_II.append(row['F_II'])
                all_F_IX.append(row['F_IX'])
                all_F_XI.append(row['F_XI'])
                all_F_XX.append(row['F_XX'])
                all_Z_II.append(row['Z_II'])
                all_Z_IX.append(row['Z_IX'])
                all_Z_XI.append(row['Z_XI'])
                all_Z_XX.append(row['Z_XX'])
                all_E0_II.append(row['E0_II'])
                all_E0_IX.append(row['E0_IX'])
                all_E0_XI.append(row['E0_XI'])
                all_E0_XX.append(row['E0_XX'])
                sample_source_indices.append(state_idx)

    N = len(all_syndromes)
    K_samp = len(sampling_states)
    K_target = len(target_ps)
    
    if N == 0:
        print("No samples found.")
        return

    print(f"Total samples: {N} across {K_samp} sampling states.")
    
    # Unique p values we need to evaluate Z_S(p) for
    unique_ps = np.unique(np.concatenate([target_ps, [s[0] for s in sampling_states]]))
    p_to_idx = {p: i for i, p in enumerate(unique_ps)}
    
    # Evaluate log(Z_S(p)) and D_KL metrics for each sample
    # log_zs[sample_idx, p_idx]
    log_zs = np.zeros((N, len(unique_ps)))
    dkl_metrics = np.zeros((K_target, N, 3)) # (target_p, sample, metric_type)

    HZ, HX = code.stabilizer_matrices()
    for i in range(N):
        if i % 50 == 0: print(f"Processing sample {i}/{N}...")
        syndX = all_syndromes[i]
        eZ = ge_initialize_given_syndrome(HX, syndX)
        eX = np.zeros(code.n, dtype=int)

        for p_val in unique_ps:
            w, _ = coset_integral_weights(eX, eZ, code, p_val, n_beta=21, 
                                         n_samples=n_samples_mcmc, burn_in=burn_in)
            log_zs[i, p_to_idx[p_val]] = np.log(max(np.sum(w), 1e-300))

    # Compute D_KL metrics using CSV values
    for t, p_t in enumerate(target_ps):
        beta_t = -np.log(p_t / (1 - p_t))
        for i in range(N):
            F_vals = np.array([all_F_II[i], all_F_IX[i], all_F_XI[i], all_F_XX[i]])
            worm_norm = np.exp(-beta_t * F_vals)
            worm_norm /= np.sum(worm_norm)
            
            Z_vals = np.array([all_Z_II[i], all_Z_IX[i], all_Z_XI[i], all_Z_XX[i]])
            mcmc_norm = Z_vals / np.sum(Z_vals)  # Already normalized in CSV
            
            E0_vals = np.array([all_E0_II[i], all_E0_IX[i], all_E0_XI[i], all_E0_XX[i]])
            mw_norm = np.exp(-beta_t * E0_vals)
            mw_norm /= np.sum(mw_norm)
            
            dkl_metrics[t, i, 0] = d_kl(worm_norm, mcmc_norm)
            dkl_metrics[t, i, 1] = d_kl(worm_norm, mw_norm)
            dkl_metrics[t, i, 2] = d_kl(mcmc_norm, mw_norm)

    # Build MBAR matrix u_kn: rows are sampling states, columns are samples
    # u_kn = -log_ZS(p_k) - lambda_k * phi_n
    u_kln = np.zeros((K_samp, N))
    N_k = np.zeros(K_samp)
    for k in range(K_samp):
        p_k, lmbda_k = sampling_states[k]
        p_idx = p_to_idx[p_k]
        u_kln[k, :] = -log_zs[:, p_idx] - lmbda_k * np.array(all_phis)
        N_k[k] = sample_source_indices.count(k)

    # MBAR to find free energies of sampling states
    mbar_res = mbar(u_kln, N_k=N_k)
    
    # Now for each target state (p_t, lambda=0), calculate weights for all samples
    results_dkl = np.zeros((K_target, 3))
    for t in range(K_target):
        p_t = target_ps[t]
        # Reduced potential of all samples in target state t
        u_t = -log_zs[:, p_to_idx[p_t]] 
        # Weight of sample n in target state t
        log_denom = logsumexp(np.log(N_k)[:, None] - mbar_res['f_k'][:, None] - u_kln, axis=0)
        weights_t = np.exp(-u_t - log_denom)
        weights_t /= np.sum(weights_t)
        
        for m in range(3):
            results_dkl[t, m] = np.sum(weights_t * dkl_metrics[t, :, m])

    # Plotting
    plt.figure(figsize=(10, 7))
    markers = ['o', 's', '^']
    labels = ['D_KL(Optimal||Z_ratios)', 'D_KL(Optimal||MWPM)', 'D_KL(Z_ratios||MWPM)']
    for m in range(3):
        plt.plot(target_ps, results_dkl[:, m], marker=markers[m], label=labels[m])
    
    plt.xlabel('Physical Error Rate $p$'); plt.ylabel('Average D_KL'); plt.yscale('log')
    plt.title(f'Syndrome-Averaged D_KL (Z_ratios) via MBAR ({code_type}, L={code.L})')
    plt.legend(); plt.grid(True, which='both', ls='--'); plt.tight_layout()
    plt.savefig(f'd_kl_Z_ratios_from_csv_mbar_{code_type}_L{code.L}.pdf'); plt.close()
    print(f"Plot saved: d_kl_Z_ratios_from_csv_mbar_{code_type}_L{code.L}.pdf")


def d_kl_vs_L(code_factory, p, L_list, code_type='Toric'):
    d_kl_values = []
    for L in L_list:
        code = code_factory(L)
        d_kl = D_KL_MCMC_MW(code, p)
        d_kl_values.append(d_kl)

    # Save data to CSV
    with open(f'd_kl_vs_L_data_p{p}_{code_type}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['L', 'D_KL'])
        for i, L in enumerate(L_list):
            writer.writerow([L, d_kl_values[i]])

    plt.plot(L_list, d_kl_values, marker='o')
    plt.xlabel("Lattice size L")
    plt.ylabel("D_KL(MH || MWPM)")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--")
    plt.savefig(f'd_kl_vs_L_plot_p{p}_{code_type}.pdf')
    #plt.show()

def threshold_comparison_plot():
    """
    Plots a comparison of different decoders using data from the 'data' directory.
    Styling follows the requested markers and color consistency for lattice sizes L.
    """
    data_dir = 'data'
    # Configuration for decoders: (filename_base, color)
    configs = {
        'BP': ('threshold_data_Toric_bp.csv', 'tab:blue'),
        'MH': ('threshold_data_Toric_mh.csv', 'tab:green'),
        'MWPM': ('threshold_data_Toric_mwpm_2.csv', 'tab:red'),
        'Worm': ('threshold_data_Toric_worm.csv', 'tab:orange')
    }

    plt.figure(figsize=(10, 7))
    # Mapping specific L values to requested marker shapes
    l_markers = {
        'L=5': '^',   # Triangle
        'L=10': 'v',  # Inverted Triangle
        'L=15': 's',  # Square
        'L=20': 'o'   # Circle
    }
    marker_list = ['D', 'p', '*', 'X', 'h', '8']
    marker_idx = 0

    for label, (fname, color) in configs.items():
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            continue

        df = pd.read_csv(fpath)
        p_vals = df['p'].values
        l_cols = [c for c in df.columns if c.startswith('L=')]

        for col in l_cols:
            if col not in l_markers:
                l_markers[col] = marker_list[marker_idx % len(marker_list)]
                marker_idx += 1

            plt.plot(p_vals, df[col], marker=l_markers[col], color=color,
                     label=f"{label} ({col})", linestyle='-', alpha=0.7)

    plt.xlabel("Physical error rate $p$")
    plt.ylabel("Logical error rate")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.title("Threshold Comparison: BP vs MH vs MWPM vs Worm")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig('threshold_comparison_plot.pdf')
    plt.close()