import matplotlib
matplotlib.use('Agg')  # Must be called before importing pyplot

from code import ToricCode, PlanarSurfaceCode
from noise import depolarizing_noise
from syndrome import syndrome_from_eX, syndrome_from_eZ
from decoder import MWPMDecoder, MHDecoder, GEDecoder, MHDecoderSingleChain, MHDecoderTrackZ, MHDecoderParallel, BPDecoder, WormDecoder
from logical import logical_parity
import numpy as np
import matplotlib.pyplot as plt
from simulation import run_trial
from mh_diagnostics import plot_mh_traces, error_rate_vs_n_sample
from threshold import threshold_comparison_plot, threshold_plot, P_vs_L_plot, comparison_plot
from plot_lattice import LatticePlotter, plot_failure_syndromes_on_lattice
from threshold import d_kl_vs_p, d_kl_vs_L, d_kl_vs_p_fixed_syndrome, d_kl_vs_p_syndrome_avg, d_kl_from_csv_samples, d_kl_Z_ratios_from_csv_samples
from evaluation import coset_probs_exact, coset_probs_worm, coset_avg_wt_mcmc, coset_proxies_mcmc, bar_graph_proxies, bar_graph_syndrome_avg_with_exact, syndrome_sampling, delta_s_vs_delta_u_plot, delta_f_vs_delta_e0_plot, delta_f_vs_delta_u_plot, delta_f_minus_delta_u_vs_delta_e0_plot, _load_pt_history_csv, _load_type_g_csv
from pathological_syndromes import parallel_tempering
import os
import csv
import pandas as pd
import utils

def run_single_experiment(L=5, p=0.05, decoder_type="MWPM", init_method='MWPM', code_type='Toric'):
    if code_type == 'Toric':
        code = ToricCode(L)
    elif code_type == 'Planar':
        code = PlanarSurfaceCode(L)
    else:
        raise ValueError(f"Unknown code_type: {code_type}")
    
    # --- Pauli-frame noise ---
    eX, eZ = depolarizing_noise(code.n, p)
    
    # --- Syndrome extraction ---
    syndZ = syndrome_from_eX(eX, code.Z_stabilizers)
    syndX = syndrome_from_eZ(eZ, code.X_stabilizers)

    n_samples = L**4
    burn_in = n_samples // 4

    # Choose decoder
    if decoder_type == "MWPM":
        decoder = MWPMDecoder(code)
    elif decoder_type == "MH":
        decoder = MHDecoder(code, q_error=2*p/(3-p))
    elif decoder_type == "SingleChain":
        # Use exact q for joint depolarizing noise: q/(1-q) = (p/3)/(1-p) => q = p/(3-2p)
        decoder = MHDecoderSingleChain(code, q_error=p/(3-2*p), n_samples=n_samples, burn_in=burn_in)
    elif decoder_type == "TrackZ":
        decoder = MHDecoderTrackZ(code, q_error=p/(3-2*p), n_samples=n_samples, burn_in=burn_in)
    elif decoder_type == "Parallel":
        decoder = MHDecoderParallel(code, q_error=p/(3-2*p), n_samples=n_samples, burn_in=burn_in)
    elif decoder_type == "GE":
        decoder = GEDecoder(code)
    elif decoder_type == "Worm":
        decoder = WormDecoder(code, p_phys=p, n_samples=n_samples, n_burnin=burn_in)
    else:
        raise ValueError(f"Unknown decoder_type: {decoder_type}")

    # --- Decode ---
    if decoder_type in ["MH", "SingleChain", "TrackZ", "Parallel"]:
        eX_hat, eZ_hat = decoder.decode(syndZ, syndX, init_method=init_method)
    else:
        eX_hat, eZ_hat = decoder.decode(syndZ, syndX)

    plotter = LatticePlotter(code, [eX,eZ], syndromes=(syndX, syndZ))
    plotter.plot(corrections=(eX_hat, eZ_hat))

    # --- Residual error ---
    rX = [a ^ b for a, b in zip(eX, eX_hat)]
    rZ = [a ^ b for a, b in zip(eZ, eZ_hat)]

    # --- Logical failure ---
    fail_X1 = logical_parity(rX, code.logical_Z_support())
    fail_X2 = logical_parity(rX, code.logical_Z_conjugate())
    fail_Z1 = logical_parity(rZ, code.logical_X_support())
    fail_Z2 = logical_parity(rZ, code.logical_X_conjugate())

    logical_failure = fail_X1 or fail_Z1 or fail_X2 or fail_Z2
    return logical_failure

def recompute_degeneracy_from_history(input_path, L=5, p=0.16):
    """
    Reads syndromes from a PT history CSV, recomputes degeneracy factors g 
    using mw_counts from MCMC, and saves a new CSV with updated values.
    """
    print(f"Loading history from {input_path}...")
    df = pd.read_csv(input_path)
    code = ToricCode(L)
    _, HX = code.stabilizer_matrices()
    
    unique_syndromes = df['syndrome'].unique()
    print(f"Found {len(unique_syndromes)} unique syndromes. Recomputing g values...")
    
    synd_cache = {}
    for s_str in unique_syndromes:
        # Parse syndrome string e.g. "(0, 1, 0, ...)"
        s_vals = np.fromstring(s_str.strip('()'), sep=',', dtype=int)
        
        # Initialize Z-error representative (detectable by X-stabilizers)
        eZ_ref = utils.ge_initialize_given_syndrome(HX, s_vals)
        eX_ref = np.zeros(code.n, dtype=int)
        
        # Run MCMC to get degeneracy factors (mw_counts)
        # Using 4000 samples and 1000 burn-in to match sampler configuration
        _, _, _, mw_counts, labels = coset_proxies_mcmc(
            eX_ref, eZ_ref, code, p, n_samples=4000, burn_in=1000
        )
        synd_cache[s_str] = dict(zip(labels, mw_counts))

    # Update the columns for each target coset
    for coset in ["II", "IX", "XI", "XX"]:
        df[f'g_{coset}'] = df['syndrome'].apply(lambda s: synd_cache[s].get(coset, 1))

    filename = os.path.basename(input_path)
    output_path = filename.replace(".csv", "_recomputed_g.csv")
    df.to_csv(output_path, index=False)
    print(f"Recomputed observables saved to: {output_path}")

if __name__ == "__main__":
    L=3
    p=0.16
    CODE_TYPE = 'Toric' # Options: 'Toric', 'Planar'
    
    # Example with selected Code and decoder: plots errors, syndromes and corrections on the lattice
    #failed = run_single_experiment(L=L, p=p, code_type=CODE_TYPE, decoder_type="MWPM")
    #print(f"Logical failure ({CODE_TYPE}):", failed)
    
    if CODE_TYPE == 'Planar':
        code = PlanarSurfaceCode(L)
    else:
        code = ToricCode(L)
    #plots traces of MH sampler for the given decoder type
    #plot_mh_traces(code, p=p, decoder_type="Parallel", n_samples=4000, burn_in=1000)

    #Initializes lists of lattice sizes, physical error rates and number of trials per point (for logical error rate estimation) for threshold and P vs L plots 
    L_list = np.arange(5, 20, 5)                 # lattice sizes
    p_list = np.arange(0.04, 0.24, 0.04)
    #p_list = [0.16,0.17,0.18,0.19]  # physical error rates
    trials = 1000                      # Monte Carlo trials per point

    #threshold plot (logical error rate vs physical error rate for different lattice sizes). The p at which the curves intersect gives the threshold
    #threshold_plot(L_list, p_list, lambda c, p_val: MWPMDecoder(c), trials, code_type=CODE_TYPE)
    #threshold_plot(L_list, p_list, lambda c, p_val: MHDecoderParallel(c, q_error=p_val/(3-2*p_val), n_samples=c.L**4, burn_in=(c.L**4)//4), trials, code_type=CODE_TYPE)
    #threshold_plot(L_list, p_list, lambda c, p_val: MHDecoderTrackZ(c, q_error=p_val/(3-2*p_val), n_samples=c.L**4, burn_in=(c.L**4)//4), trials, code_type=CODE_TYPE)
    #threshold_plot(L_list, p_list, lambda c, p_val: BPDecoder(c, p_val), trials, code_type=CODE_TYPE)
    #threshold_plot(L_list, p_list, lambda c, p_val: WormDecoder(c, p_phys=p_val, n_samples=375, n_burnin=125), trials, code_type=CODE_TYPE)

    #logical error rate vs lattice size for different physical error rates
    #P_vs_L_plot(L_list, p_list, lambda c, p_val: MHDecoderParallel(c, q_error=p_val/(3-2*p_val), n_samples=c.L**4, burn_in=(c.L**4)//4), trials, code_type=CODE_TYPE)

    #comparison of Ratio, MWPM and MH decoders. logical error rate vs physical error rate
    #comparison_plot(p_list, trials, code_type=CODE_TYPE)

    #plots the logical error rate vs the number of MH iterations
    #error_rate_vs_n_sample([10,25], code_type=CODE_TYPE, p=0.17, trials=trials, n_samples=4000)

    #plots bar graphs of exact coset probabilities, MCMC estimate and minimum weight error probability for each coset, averaged over syndromes
    #bar_graph_syndrome_avg_with_exact(code, p, n_synd_samples=4000)
    
    #plots bar graphs of exact coset probabilities, MCMC estimate and minimum weight error probability for each coset, given a single syndrome
    '''eX, eZ = depolarizing_noise(code.n, p)
    
    exact_probs, labels = coset_probs_exact(eX, eZ, code, p)
    exact_norm = np.array(exact_probs) / np.sum(exact_probs)

    mcmc_norm, min_weight_probs, labels, _ = coset_probs_mcmc(eX, eZ, code, p, n_samples=L**4, burn_in=L**4/4)

    bar_graph_with_exact(exact_norm, mcmc_norm, min_weight_probs, labels=labels,
              title=f"Coset Probabilities Comparison ({CODE_TYPE}, L={L}, p={p})")'''
    bar_graph_proxies(code, p, use_exact=True)
    
    #plots tvd of syndrome distribution over the number of batches
    #syndrome_sampling(code, p, n_total=10**6)

    # plots kl divergences over physical error rate
    #d_kl_vs_p(code, p_list, code_type=CODE_TYPE)

    #plots kl divergences for a fixed syndrome over physical error rate
    #d_kl_vs_p_fixed_syndrome(code, p, p_list, code_type=CODE_TYPE)

    #plots kl divergences averaged over syndromes over physical error rate
    #d_kl_vs_p_syndrome_avg(code, p_list, code_type=CODE_TYPE, n_syndromes=100)

    # plots kl divergence between MH and MW distributions over lattice size
    #code_factory = PlanarSurfaceCode if CODE_TYPE == 'Planar' else ToricCode
    #d_kl_vs_L(code_factory, p, L_list, code_type=CODE_TYPE)

    #plots failure mode syndromes on the lattice for L=2 Toric code
    #plot_failure_syndromes_on_lattice()

    #consolidated plot for threshold comparison of different decoders
    #threshold_comparison_plot()

    #fix parallel tempering history
    #csv_to_fix = 'pt_syndrome_samples/parallel_tempering_history_p16.csv'
    #recompute_degeneracy_from_history(csv_to_fix, L=5, p=0.16)

    '''#sample syndromes by parallel tempering of MH samplers with exponential tilting
    print("\n=== Running Parallel Tempering ===")
    
    # Initialize the global TC for pathological_syndromes module
    import pathological_syndromes as patho_synds
    patho_synds.TC = code
    
    p_pt = 0.16
    beta_pt = -np.log(p_pt / (1.0 - p_pt))
    pt_result = parallel_tempering(
        beta=beta_pt,
        phi_mode="phi_unified",
        alpha_hybrid=0.6,
        p=p_pt,
        n_replicas=24,
        n_steps=50,
        swap_interval=10,
        record_interval=10
    )
    
    # Extract and save history to CSV
    history = pt_result['history']
    csv_filename = "parallel_tempering_history_unified_corrected_1.csv"
    
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        header = [
            'step', 'replica', 'lambda', 'beta', 'phi', 'syndrome',
            'E0_II', 'E0_IX', 'E0_XI', 'E0_XX',
            'U_II', 'U_IX', 'U_XI', 'U_XX',
            'F_II', 'F_IX', 'F_XI', 'F_XX',
            'g_II', 'g_IX', 'g_XI', 'g_XX',
            'Z_II', 'Z_IX', 'Z_XI', 'Z_XX',
            'delta_E0_IX', 'delta_E0_XI', 'delta_E0_XX',
            'delta_U_IX', 'delta_U_XI', 'delta_U_XX',
            'delta_F_IX', 'delta_F_XI', 'delta_F_XX',
            'delta_S_IX', 'delta_S_XI', 'delta_S_XX'
        ]
        writer.writerow(header)
        
        # Write history entries
        for entry in history:
            row = [
                entry['step'],
                entry['replica'],
                entry['lambda'],
                entry['beta'],
                entry['phi'],
                str(entry['syndrome'])
            ]
            E0, U, F, g = entry['E0'], entry['U'], entry['F'], entry['g']
            row += [E0[e] for e in ['II', 'IX', 'XI', 'XX']]
            row += [U[e] for e in ['II', 'IX', 'XI', 'XX']]
            row += [F[e] for e in ['II', 'IX', 'XI', 'XX']]
            row += [g[e] for e in ['II', 'IX', 'XI', 'XX']]
            row += [entry['Z'][e] for e in ['II', 'IX', 'XI', 'XX']]
            
            deltas = entry['deltas']
            for e in ['IX', 'XI', 'XX']:
                if e in deltas:
                    row.append(deltas[e]['delta_E0'])
                else:
                    row.append(0.0)
            for e in ['IX', 'XI', 'XX']:
                if e in deltas:
                    row.append(deltas[e]['delta_U'])
                else:
                    row.append(0.0)
            for e in ['IX', 'XI', 'XX']:
                if e in deltas:
                    row.append(deltas[e]['delta_F'])
                else:
                    row.append(0.0)
            for e in ['IX', 'XI', 'XX']:
                if e in deltas:
                    row.append(deltas[e]['delta_S'])
                else:
                    row.append(0.0)
            
            writer.writerow(row)
    
    print(f"Parallel tempering history saved to '{csv_filename}'")
    print(f"Total history entries: {len(history)}")
    print(f"Final phis: {pt_result['phis']}")
    print(f"Lambda ladder: {pt_result['lambdas']}")'''

    '''#plots for delta_S vs delta_U and delta_F vs delta_E0 for the recorded history of the parallel tempering run
    p_pt = 0.16
    beta_pt = -np.log(p_pt / (1.0 - p_pt))
    csv_path = 'parallel_tempering_history_unified_corrected.csv'
    pt_history = _load_pt_history_csv(csv_path)
    # type_g_history = _load_type_g_csv('type_g_pathology_samples.csv', beta_pt)
    # combined_history = pt_history + type_g_history
    delta_s_vs_delta_u_plot(pt_history, beta_pt, filename='deltaS_vs_deltaU.pdf')
    delta_f_vs_delta_e0_plot(pt_history, filename='deltaF_vs_deltaE0.pdf')
    delta_f_vs_delta_u_plot(pt_history, filename='deltaF_vs_deltaU.pdf')
    delta_f_minus_delta_u_vs_delta_e0_plot(pt_history, filename='deltaF_minus_deltaU_vs_deltaE0.pdf')
    print("\nPlots generated:")
    print("  - deltaS_vs_deltaU.pdf")
    print("  - deltaF_vs_deltaE0.pdf")
    print("  - deltaF_vs_deltaU.pdf")
    print("  - deltaF_minus_deltaU_vs_deltaE0.pdf")'''

    '''# Verify d_kl plots using samples from all PT history files
    print("\n=== Running d_kl_from_csv_samples with MBAR aggregation ===")
    pt_dir = 'pt_samples_Z_ratios'
    p_targets = [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
    if os.path.exists(pt_dir):
        d_kl_from_csv_samples(code, pt_dir, p_targets, code_type=CODE_TYPE, subsample=50)
        d_kl_Z_ratios_from_csv_samples(code, pt_dir, p_targets, code_type=CODE_TYPE, subsample=50)
    else:
        print(f"Warning: Directory {pt_dir} not found.")'''
