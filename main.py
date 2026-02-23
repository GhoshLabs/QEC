from code import ToricCode, PlanarSurfaceCode
from noise import depolarizing_noise
from syndrome import syndrome_from_eX, syndrome_from_eZ
from decoder import MWPMDecoder, MHDecoder, GEDecoder, MHDecoderSingleChain, MHDecoderTrackZ, MHDecoderParallel
from logical import logical_parity
import numpy as np
import matplotlib.pyplot as plt
from simulation import run_trial
from mh_diagnostics import plot_mh_traces, error_rate_vs_n_sample
from threshold import comparison_plot, threshold_plot
from plot_lattice import LatticePlotter

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

    # Choose decoder
    if decoder_type == "MWPM":
        decoder = MWPMDecoder(code)
    elif decoder_type == "MH":
        decoder = MHDecoder(code, q_error=2*p/(3-p))
    elif decoder_type == "SingleChain":
        # Use exact q for joint depolarizing noise: q/(1-q) = (p/3)/(1-p) => q = p/(3-2p)
        decoder = MHDecoderSingleChain(code, q_error=p/(3-2*p))
    elif decoder_type == "TrackZ":
        decoder = MHDecoderTrackZ(code, q_error=p/(3-2*p))
    elif decoder_type == "Parallel":
        decoder = MHDecoderParallel(code, q_error=p/(3-2*p))
    elif decoder_type == "GE":
        decoder = GEDecoder(code)
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


if __name__ == "__main__":
    L=8
    p=0.17
    CODE_TYPE = 'Planar' # Options: 'Toric', 'Planar'
    
    # Example with selected Code and MWPM decoder
    failed = run_single_experiment(L=L, p=p, code_type=CODE_TYPE, decoder_type="MWPM")
    print(f"Logical failure ({CODE_TYPE}):", failed)
    
    if CODE_TYPE == 'Planar':
        code = PlanarSurfaceCode(L)
    else:
        code = ToricCode(L)
    plot_mh_traces(code, p=p, decoder_type="Parallel", n_samples=4000, burn_in=1000)

    L_list = [4,6,8]                 # lattice sizes
    #p_list = np.linspace(0.10, 0.20, 10)
    p_list = [0.16,0.17,0.18,0.19]  # physical error rates
    trials = 2000                      # Monte Carlo trials per point

    threshold_plot(L_list, p_list, lambda c, p_val: MHDecoder(c, q_error=2*p_val/(3-p_val)), trials, code_type=CODE_TYPE)

    #comparison_plot(p_list, trials, code_type=CODE_TYPE)

    '''rates = error_rate_vs_n_sample(code, p, MHDecoder(code, 2*p/(3-p)), n_samples=70000)
    plt.plot(rates)
    plt.xlabel("Number of samples")
    plt.ylabel("Logical error rate")
    plt.show()'''