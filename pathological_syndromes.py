import numpy as np
import random
import csv
from code import ToricCode
from syndrome import syndrome_from_eZ
import utils
from evaluation import coset_avg_wt_mcmc, coset_probs_integral, coset_proxies_mcmc
from plot_n_beta_curves import metropolis_within_class
from plot_lattice import LatticePlotter
from noise import independent_XZ_noise
import matplotlib.pyplot as plt
import matplotlib
import syndrome as synd_mod

# ============================================================
# CONFIG
# ============================================================

L = 5
# TC = ToricCode(L) # Will be initialized in __main__
COSETS = ["II", "IX", "XI", "XX"] # Corresponding to I, LX2, LX1, LX1LX2 logical sectors for Z-errors

# ============================================================
# REQUIRED USER-DEFINED FUNCTIONS (PLUG THESE IN)
# ============================================================

def get_z_reps(syndrome):
    """Returns minimum-weight Z-error reps for all 4 logical cosets."""
    _, HX = TC.stabilizer_matrices()
    eZ_base = utils.mwpm_initialize_e_given_syndrome(HX, syndrome).astype(np.int8) # Base Z-error chain
    
    # Standard logical X operators for Toric code
    # These parities define the logical sectors for Z-errors
    x1_supp = TC.logical_X_support()
    x2_supp = TC.logical_X_conjugate()

    lX1 = np.zeros(TC.n, dtype=np.int8)
    lX2 = np.zeros(TC.n, dtype=np.int8)
    if x1_supp: lX1[x1_supp] = 1
    if x2_supp: lX2[x2_supp] = 1

    return {
        "II": eZ_base,          # Identity logical sector
        "XI": eZ_base ^ lX1,    # Logical X1 sector
        "IX": eZ_base ^ lX2,    # Logical X2 sector
        "XX": eZ_base ^ lX1 ^ lX2 # Logical X1X2 sector
    }

def compute_syndrome(error_chain):
    """Return syndrome from error chain"""
    return synd_mod.syndrome_from_eZ(error_chain, TC.X_stabilizers)

def weight(error_chain):
    """Return weight n(C)"""
    return int(np.sum(error_chain))

def propose(error_chain):
    """Propose a new error chain (e.g. flip a qubit)"""
    new_chain = error_chain.copy()
    idx = random.randrange(TC.n)
    new_chain[idx] ^= 1
    return new_chain

def base_sampler(p=0.1):
    """Initial error chain sampler"""
    _, eZ = independent_XZ_noise(TC.n, p)
    return np.array(eZ, dtype=np.int8)

# ============================================================
# CORE COMPUTATIONS
# ============================================================

def compute_all_observables(syndrome, beta, p_phys, n_beta=21):
    """
    Compute E0, U, F, and g by thermodynamic integration of U over beta.
    Calls coset_avg_wt_mcmc once per beta grid point to extract all statistics.
    """
    betas = np.linspace(0.0, beta, n_beta)
    all_avg_wts = np.zeros((n_beta, len(COSETS)))
    E0, U, F, g = {}, {}, {}, {}

    _, HX = TC.stabilizer_matrices()
    eZ_ref = utils.mwpm_initialize_e_given_syndrome(HX, syndrome).astype(np.int8)
    eX_ref = np.zeros_like(eZ_ref)

    for i, b in enumerate(betas):
        # Convert beta to p for depolarizing noise conversion used in coset_avg_wt_mcmc
        # Infinite temperature (beta=0) corresponds to p = 0.75 for depolarizing noise
        p_val = np.exp(-b) / (1.0 + np.exp(-b)) if b > 0 else 0.75
        
        Z_ratios, avg_wts, min_wts, mw_counts, labels = coset_proxies_mcmc(
            eX_ref, eZ_ref, TC, p_val, n_samples=4000, burn_in=1000
        )
        
        label_to_idx = {l: idx for idx, l in enumerate(labels)}
        for j, c_label in enumerate(COSETS):
            all_avg_wts[i, j] = avg_wts[label_to_idx[c_label]]
            
        # Extract final observables at the target physical error rate
        if i == n_beta - 1:
            for c_label in COSETS:
                idx = label_to_idx[c_label]
                E0[c_label] = min_wts[idx]
                U[c_label] = avg_wts[idx]
                g[c_label] = mw_counts[idx]

    # Compute Z_ratios using the final Z_ratios
    Z_ratios_dict = {c_label: Z_ratios[j] for j, c_label in enumerate(COSETS)}

    # Integrate delta U to get delta F (relative to II)
    for j, c_label in enumerate(COSETS):
        if c_label == "II":
            F[c_label] = 0.0
        else:
            delta_U = all_avg_wts[:, j] - all_avg_wts[:, 0]
            F[c_label] = utils.simpson_integral(betas, delta_U)

    return E0, U, F, g, Z_ratios_dict


# ============================================================
# PATHOLOGY SCORE FUNCTIONS
# ============================================================

def phi_1_new(E0):
    return E0["II"] - min(E0[e] for e in COSETS if e != "II")

def phi_2_new(F):
    return max(-(F[e] - F["II"]) for e in COSETS if e != "II")

def phi_3_new(U,F):
    return max(
        min(F[e] - F["II"], -(U[e] - U["II"]))
        for e in COSETS if e != "II"
    )

def phi_hybrid_new(E0, U, F, g, alpha=0.5):
    return alpha * phi_g(g) + (1 - alpha) * phi_2(U, F)

def phi_1(U, F):
    """
    Case 2 pathology score: both ΔF and ΔU are negative.
    φ1^E = min(-ΔF, -ΔU) for each false coset E, then take the maximum.
    """
    values = []
    for e in COSETS:
        if e == "II":
            continue
        delta_F = F[e] - F["II"]
        delta_U = U[e] - U["II"]
        values.append(min(-delta_F, -delta_U))
    return max(values) if values else 0.0


def phi_2(U, F):
    """
    Case 4 pathology score: crossing after β̄.
    φ2^E = min(+ΔU, -ΔF) for each false coset E, then take the maximum.
    """
    values = []
    for e in COSETS:
        if e == "II":
            continue
        delta_F = F[e] - F["II"]
        delta_U = U[e] - U["II"]
        values.append(min(delta_U, -delta_F))
    return max(values) if values else 0.0


def phi_3(U, F):
    """
    Case 3 pathology score: crossing before β̄.
    φ3^E = min(-ΔU, +ΔF) for each false coset E, then take the maximum.
    """
    values = []
    for e in COSETS:
        if e == "II":
            continue
        delta_F = F[e] - F["II"]
        delta_U = U[e] - U["II"]
        values.append(min(-delta_U, delta_F))
    return max(values) if values else 0.0


def phi_g(g):
    """
    Inter-coset degeneracy score
    """
    return max(np.log(g[e]) - np.log(g["II"]) for e in COSETS if e != "II")


def phi_hybrid(E0, U, F, g, alpha=0.5):
    """
    Stable hybrid score
    """
    return alpha * phi_g(g) + (1 - alpha) * phi_2(U, F)

def phi_unified(U, F):
    """
    Detects any pathology type by taking the maximum pathology score across
    all false cosets and all case-specific definitions.
    """
    values = []
    for e in COSETS:
        if e == "II":
            continue
        delta_F = F[e] - F["II"]
        delta_U = U[e] - U["II"]
        values.append(max(
            min(-delta_F, -delta_U),
            min(delta_U, -delta_F),
            min(-delta_U, delta_F)
        ))
    return max(values) if values else 0.0

def phi_tilt(U, F):
    values=[]
    for e in COSETS:
        if e == "II":
            continue
        delta_F = F[e] - F["II"]
        delta_U = U[e] - U["II"]
        values.append(min(-delta_F, np.abs(delta_U)))
    return max(values) if values else 0.0


# ============================================================
# PATHOLOGY CLASSIFICATION
# ============================================================

def classify_pathology(E0, U, F, g):
    I = "II"
    is_type1 = False
    is_type2 = False
    is_type3 = False

    for e in COSETS:
        if e == I:
            continue

        dE0 = E0[e] - E0[I]
        dU  = U[e]  - U[I]
        dF  = F[e]  - F[I]

        if dE0 <= 0:
            is_type1 = True
        if dE0 > 0 and dF < 0:
            is_type2 = True
        if dU < 0 and dF > 0:
            is_type3 = True

    if is_type1:
        label = "Type1"
    elif is_type2:
        label = "Type2"
    elif is_type3:
        label = "Type3"
    else:
        label = "None"

    if phi_g(g) > 0.0:
        label += "+G"

    return label


# ============================================================
# IMPORTANCE (TILTED) SAMPLER
# ============================================================

def tilted_sampler(beta, lambda_, phi_mode="phi_g", alpha_hybrid=0.5, p=0.1):
    """
    phi_mode options:
        "phi_1", "phi_2", "phi_3", "phi_g", "hybrid"
    """

    C = base_sampler(p)
    syndrome = compute_syndrome(C) # This is syndX from eZ
    E0, U, F, g, Z = compute_all_observables(syndrome, beta, p)

    phi_old = compute_phi(phi_mode, E0, U, F, g, alpha_hybrid)

    while True:
        C_new = propose(C)
        syndrome_new = compute_syndrome(C_new)
        
        E0_new, U_new, F_new, g_new, Z_new = compute_all_observables(syndrome_new, beta, p)

        phi_new = compute_phi(phi_mode, E0_new, U_new, F_new, g_new, alpha_hybrid)

        delta_weight = weight(C_new) - weight(C)

        log_accept = -beta * delta_weight + lambda_ * (phi_new - phi_old)

        if np.log(np.random.rand()) < log_accept:
            C = C_new
            phi_old = phi_new

        yield C


def compute_phi(mode, E0, U, F, g, alpha_hybrid):
    if mode == "phi_1":
        return phi_1(U,F)
    elif mode == "phi_2":
        return phi_2(U,F)
    elif mode == "phi_3":
        return phi_3(U, F)
    elif mode == "phi_g":
        return phi_g(g)
    elif mode == "phi_hybrid":
        return phi_hybrid(E0, U, F, g, alpha_hybrid)
    elif mode == "phi_unified":
        return phi_unified(U, F)
    elif mode == "phi_tilt":
        return phi_tilt(U,F)
    else:
        raise ValueError("Unknown phi mode")


# ============================================================
# SAMPLING LOOP
# ============================================================

def sample_pathologies(n_samples, beta, sampler):

    counts = {"Type1":0, "Type2":0, "Type3":0, "TypeG":0, "None":0}

    phi1_list, phi2_list, phi3_list, phig_list = [], [], [], []

    for _ in range(n_samples):

        C = next(sampler)
        syndrome = compute_syndrome(C)

        E0, U, F, g, Z = compute_all_observables(syndrome, beta, p_phys)

        # Scores
        p1 = phi_1(U, F)
        p2 = phi_2(U, F)
        p3 = phi_3(U, F)
        pg = phi_g(g)

        phi1_list.append(p1)
        phi2_list.append(p2)
        phi3_list.append(p3)
        phig_list.append(pg)

        label = classify_pathology(E0, U, F, g)
        counts[label] += 1

    return {
        "counts": counts,
        "phi1": np.array(phi1_list),
        "phi2": np.array(phi2_list),
        "phi3": np.array(phi3_list),
        "phig": np.array(phig_list)
    }

def _pt_history_entry(step, replica_idx, lambda_val, syndrome, observables, phi, beta):
    """Construct a history record for a single replica and step."""
    E0, U, F, g, Z = observables
    delta = {}
    for e in COSETS:
        if e == "II":
            continue
        delta_U = U[e] - U["II"]
        delta_F = F[e] - F["II"]
        delta_S = beta * (delta_U - delta_F)
        delta[e] = {
            "delta_E0": E0[e] - E0["II"],
            "delta_U": delta_U,
            "delta_F": delta_F,
            "delta_S": delta_S,
            "coset": e
        }

    return {
        "step": step,
        "replica": replica_idx,
        "lambda": lambda_val,
        "syndrome": tuple(syndrome.tolist()),
        "phi": float(phi),
        "beta": float(beta),
        "E0": E0,
        "U": U,
        "F": F,
        "g": g,
        "Z": Z,
        "deltas": delta
    }


def parallel_tempering(beta, phi_mode="phi_g", alpha_hybrid=0.5, p=0.1,
                       n_replicas=16, n_steps=1000, swap_interval=10, record_interval=10):
    """
    Run parallel tempering with a ladder of lambda values from 0 to 5.

    Each replica performs local tilted updates using the current lambda,
    and neighboring replicas attempt replica exchange according to:
        A(i↔j) = min(1, exp[(λ_i - λ_j)(φ(s_j) - φ(s_i))]).
    """
    lambdas = np.linspace(4.0, 10.0, n_replicas)

    replicas = [base_sampler(p) for _ in range(n_replicas)]
    syndromes = [compute_syndrome(C) for C in replicas]
    observables = [compute_all_observables(syndromes[i], beta, p) for i in range(n_replicas)]
    phis = [compute_phi(phi_mode, *observables[i][:4], alpha_hybrid) for i in range(n_replicas)]

    history = []
    for step in range(n_steps):
        for i in range(n_replicas):
            C = replicas[i]
            syndrome = syndromes[i]
            E0, U, F, g, Z = observables[i]
            phi_old = phis[i]

            C_new = propose(C)
            syndrome_new = compute_syndrome(C_new)
            E0_new, U_new, F_new, g_new, Z_new = compute_all_observables(syndrome_new, beta, p)
            phi_new = compute_phi(phi_mode, E0_new, U_new, F_new, g_new, alpha_hybrid)

            delta_weight = weight(C_new) - weight(C)
            log_accept = -beta * delta_weight + lambdas[i] * (phi_new - phi_old)
            if np.log(np.random.rand()) < log_accept:
                replicas[i] = C_new
                syndromes[i] = syndrome_new
                observables[i] = (E0_new, U_new, F_new, g_new, Z_new)
                phis[i] = phi_new

        if step % swap_interval == 0:
            for i in range(n_replicas - 1):
                j = i + 1
                lambda_i = lambdas[i]
                lambda_j = lambdas[j]
                phi_i = phis[i]
                phi_j = phis[j]

                log_exchange = (lambda_i - lambda_j) * (phi_j - phi_i)
                if np.log(np.random.rand()) < log_exchange:
                    replicas[i], replicas[j] = replicas[j], replicas[i]
                    syndromes[i], syndromes[j] = syndromes[j], syndromes[i]
                    observables[i], observables[j] = observables[j], observables[i]
                    phis[i], phis[j] = phis[j], phis[i]

        if step % record_interval == 0 or step == n_steps - 1:
            for i in range(n_replicas):
                history.append(_pt_history_entry(
                    step=step,
                    replica_idx=i,
                    lambda_val=lambdas[i],
                    syndrome=syndromes[i],
                    observables=observables[i],
                    phi=phis[i],
                    beta=beta
                ))

    return {
        "lambdas": lambdas,
        "replicas": replicas,
        "phis": np.array(phis),
        "observables": observables,
        "syndromes": syndromes,
        "history": history
    }

# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    matplotlib.use('Agg') # Use 'Agg' backend for non-interactive plotting
    # 1. Configure the code distance (L) here. This will update the global TC object.
    L = 5 
    global TC 
    TC = ToricCode(L) 

    # Configure other parameters
    p_phys = 0.20
    beta = -np.log(p_phys / (1 - p_phys))
    n_samples_to_generate = 200
    lambda_val = 2.0
    phi_mode = "phi_g" # Options: "phi_1", "phi_2", "phi_3", "phi_g", "phi_hybrid", "phi_unified", "phi_tilt"
    alpha_hybrid = 0.6 # Alpha for hybrid phi_mode, can be configured

    sampler = tilted_sampler(
        beta=beta,
        lambda_=lambda_val,
        phi_mode=phi_mode,
        alpha_hybrid=alpha_hybrid, 
        p=p_phys
    )

    phi_values_for_std = []
    csv_filename = "type_g_pathology_samples.csv"

    def _coset_row(prefix, values):
        return [values[e] for e in COSETS]

    header = [
        'sample_index', 'pathology_type', 'code_type', 'distance', 'p_phys', 'beta', 'phi_mode', 'phi_value',
        'eX_vector', 'eZ_vector', 'syndX_vector', 'syndZ_vector'
    ]
    header += [f'E0_{e}' for e in COSETS]
    header += [f'U_{e}' for e in COSETS]
    header += [f'F_{e}' for e in COSETS]
    header += [f'g_{e}' for e in COSETS]
    header += [f'Z_{e}' for e in COSETS]
    header += [f'integral_prob_{e}' for e in COSETS]

    print(f"Sampling {n_samples_to_generate} syndromes on L={L} Toric Code "
          f"with p_phys={p_phys} and phi_mode='{phi_mode}'...")
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)

        for i in range(n_samples_to_generate):
            C = next(sampler) # C is the Z-error chain
            syndX = compute_syndrome(C) # This is the X-stabilizer syndrome from the Z-error chain (from eZ)
            syndZ_for_plotter = np.zeros(len(TC.Z_stabilizers), dtype=int) # Z-stabilizer syndromes are for X-errors (from eX), which are not simulated here.

            E0, U, F, g, Z = compute_all_observables(syndX, beta, p_phys) # syndX is the Z-error syndrome
            label = classify_pathology(E0, U, F, g)
            current_phi = compute_phi(phi_mode, E0, U, F, g, alpha_hybrid)
            phi_values_for_std.append(current_phi)

            integral_probs, integral_labels = coset_probs_integral(
                np.zeros(TC.n, dtype=int), C, TC, p_phys,
                n_beta=41, n_samples=4000, burn_in=1000
            )
            integral_prob_map = {integral_labels[j]: integral_probs[j] for j in range(len(integral_labels))}

            print(f"\nSample {i+1} Pathological Metrics (Classification: {label}):")
            for coset in COSETS:
                print(f"  [{coset}] E0: {E0[coset]}, U: {U[coset]:.3f}, F: {F[coset]:.3f}, g: {g[coset]}, integral_prob: {integral_prob_map.get(coset, 0.0):.4f}")

            row = [
                i,
                label,
                TC.__class__.__name__,
                TC.L,
                p_phys,
                beta,
                phi_mode,
                current_phi,
                str(np.zeros(TC.n, dtype=int).tolist()),
                str(C.tolist()),
                str(syndX.tolist()),
                str(syndZ_for_plotter.tolist())
            ]
            row += _coset_row('E0', E0)
            row += _coset_row('U', U)
            row += _coset_row('F', F)
            row += _coset_row('g', g)
            row += _coset_row('Z', Z)
            row += [integral_prob_map.get(e, 0.0) for e in COSETS]
            csv_writer.writerow(row)
            
            # plotter = LatticePlotter(TC, noise_model=(np.zeros(TC.n, dtype=int), C),
            #                          syndromes=(syndX, syndZ_for_plotter))
            
            # # Build a title that displays the optimized phi parameter
            # plot_title = f"{label} Pathology | {phi_mode}: {current_phi:.4f} | L={L}, p={p_phys}"
            # print(f"Plotting lattice for sample {i+1} with {phi_mode}={current_phi:.4f}...")
            # plotter.plot(title=plot_title) # This will still generate the plot object
            # plt.savefig(f"pathology_sample_{i+1}.pdf") # Save the figure
            # plt.close() # Close the figure to free up memory

    if phi_values_for_std:
        std_phi = np.std(phi_values_for_std)
        print(f"\nLambda * std(phi) for '{phi_mode}' mode: {lambda_val * std_phi:.4f}")
    else:
        print("\nNo phi values collected for standard deviation calculation.")