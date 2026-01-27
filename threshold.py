import numpy as np
import matplotlib.pyplot as plt
from code import ToricCode
from simulation import run_trial
from decoder import MWPMDecoder

def logical_error_rate(code, p, decoder, n_trials=1000):
    failures = 0
    for _ in range(n_trials):
        failures += run_trial(code, p, decoder)
    return failures / n_trials

def threshold_plot(L_list, p_list, decoder, trials=2000):
    for L in L_list:
        code = ToricCode(L)
        rates = []
        for p in p_list:
            rate = logical_error_rate(code, p, decoder, trials)
            rates.append(rate)
        plt.plot(p_list, rates, label=f"L={L}")

    plt.xlabel("Physical error rate p")
    plt.ylabel("Logical error rate")
    plt.legend()
    plt.yscale("log")
    plt.show()

def threshold_experiment(L_list, p_list, trials=2000):
    results = {}

    for L in L_list:
        code = ToricCode(L)
        decoder = MWPMDecoder(code)
        rates = []

        for p in p_list:
            failures = 0
            for _ in range(trials):
                failures += run_trial(code, p, decoder)


            rates.append(failures / trials)

        results[L] = rates
        plt.plot(p_list, rates, label=f"L={L}")

    plt.xlabel("Physical error rate p")
    plt.ylabel("Logical error rate")
    plt.yscale("log")
    plt.legend()
    plt.show()

    return results