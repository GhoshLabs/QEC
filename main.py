from code import ToricCode
from noise import depolarizing_noise
from syndrome import syndrome_from_eX, syndrome_from_eZ
from decoder import MWPMDecoder, MHDecoder
from logical import logical_parity
import numpy as np
import matplotlib.pyplot as plt
from simulation import run_trial
from mh_diagnostics import plot_mh_traces
from threshold import threshold_experiment

def run_single_experiment(L=5, p=0.05, decoder_type="MWPM"):
    code = ToricCode(L)

    # Choose decoder
    if decoder_type == "MWPM":
        decoder = MWPMDecoder(code)
    else:
        q = 2*p/3
        decoder = MHDecoder(code, q_error=q)

    # --- Pauli-frame noise ---
    eX, eZ = depolarizing_noise(code.n, p)

    # --- Syndrome extraction ---
    syndZ = syndrome_from_eX(eX, code.Z_stabilizers)
    syndX = syndrome_from_eZ(eZ, code.X_stabilizers)

    # --- Decode ---
    eX_hat, eZ_hat = decoder.decode(syndZ, syndX)

    # --- Residual error ---
    rX = [a ^ b for a, b in zip(eX, eX_hat)]
    rZ = [a ^ b for a, b in zip(eZ, eZ_hat)]

    # --- Logical failure ---
    fail_Z = logical_parity(rZ, code.logical_Z_support())
    fail_X = logical_parity(rX, code.logical_X_support())

    logical_failure = fail_Z or fail_X
    return logical_failure


if __name__ == "__main__":
    L=8
    code = ToricCode(L)
    failed = run_single_experiment(L=8, p=0.08, decoder_type="MH")
    print("Logical failure:", failed)
    
    plot_mh_traces(code, p=0.08, n_samples=3000, burn_in=500)

    L_list = [3, 5, 7]                 # lattice sizes
    p_list = np.linspace(0.01, 0.15, 8)  # physical error rates
    trials = 2000                      # Monte Carlo trials per point

    threshold_experiment(
        L_list=L_list,
        p_list=p_list,
        trials=trials
    )
