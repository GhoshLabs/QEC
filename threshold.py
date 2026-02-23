import numpy as np
import matplotlib.pyplot as plt
from code import ToricCode, PlanarSurfaceCode
from simulation import run_trial
from decoder import MHDecoder, MWPMDecoder

def logical_error_rate(code, p, decoder, n_trials=1000):
    failures = 0
    for _ in range(n_trials):
        failures += run_trial(code, p, decoder)
    return failures / n_trials

def P_vs_L_plot(L_list, p_list, decoder_factory, trials=2000, code_type='Toric'):
    results = experiment(L_list, p_list, decoder_factory, trials, code_type)

    for p in p_list:
        rates = results[p]
        plt.plot(L_list, rates, marker='s', label=f"p={p}")

    plt.xlabel("Lattice Size L")
    plt.ylabel("Logical error rate")
    plt.legend()
    plt.yscale("log")
    plt.grid(True, which="both", ls="--")
    plt.show()

def threshold_plot(L_list, p_list, decoder_factory, trials=2000, code_type='Toric'):
    """Plots the logical error rate P vs physical error rate p for all L."""
    results = experiment(L_list, p_list, decoder_factory, trials, code_type)

    for i, L in enumerate(L_list):
        rates = [results[p][i] for p in p_list]
        plt.plot(p_list, rates, marker='o', label=f"L={L}")

    plt.xlabel("Physical error rate p")
    plt.ylabel("Logical error rate")
    plt.legend()
    plt.yscale("log")
    plt.grid(True, which="both", ls="--")
    plt.show()

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
    mh_rates = []
    mwpm_rates = []

    for p in p_list:
        mh_decoder = MHDecoder(code, q_error=2*p/(3-p))
        mwpm_decoder = MWPMDecoder(code)

        mh_rate = logical_error_rate(code, p, mh_decoder, trials)
        mwpm_rate = logical_error_rate(code, p, mwpm_decoder, trials)

        mh_rates.append(mh_rate)
        mwpm_rates.append(mwpm_rate)

    plt.plot(p_list, mh_rates, marker='o', label="MH Decoder")
    plt.plot(p_list, mwpm_rates, marker='s', label="MWPM Decoder")
    plt.xlabel("Physical error rate p")
    plt.ylabel("Logical error rate")
    plt.legend()
    plt.yscale("log")
    plt.grid(True, which="both", ls="--")
    plt.show()