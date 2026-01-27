import matplotlib.pyplot as plt
from noise import depolarizing_noise
from syndrome import syndrome_from_eX, syndrome_from_eZ
import utils
from MH_sampler import metropolis_hastings_on_stabilizers

def plot_mh_traces(code, p, n_samples=3000, burn_in=500):
    """
    Run MH decoding once and plot trace diagnostics for X and Z chains.
    """
    q = 2*p/3

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

    # Run MH chains
    outX = metropolis_hastings_on_stabilizers(
        code, HZ, eX_init.copy(), Zstab_vecs,
        q_error=q, n_samples=n_samples, burn_in=burn_in
    )

    outZ = metropolis_hastings_on_stabilizers(
        code, HX, eZ_init.copy(), Xstab_vecs,
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
