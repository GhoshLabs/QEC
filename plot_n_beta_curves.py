"""
plot_n_beta_curves.py
─────────────────────
Plots ⟨n⟩_{β,E} curves for all four logical cosets of the toric code
under Z-error noise, comparing the single-temperature (ST) decoder 
with the optimal (Opt) free-energy decoder.

INTERFACE ASSUMPTIONS — adapt these five stubs to your codebase:
  - get_stabilizer_matrix(tc)
  - get_min_weight_reps(tc, syndrome)   -> {class_label: binary array}
  - get_logical_class(error, tc)        -> int in {0,1,2,3}
  - sample_syndrome(tc, p)             -> (error, syndrome)
  - run_worm(tc, syndrome, p, n_steps) -> {class_label: int count}

Everything else is self-contained.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg') # Required for remote/headless servers

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from code import ToricCode
from noise import depolarizing_noise
from syndrome import syndrome_from_eX, syndrome_from_eZ
from logical import logical_parity
from decoder import MWPMDecoder, WormDecoder
import utils
from matplotlib.lines import Line2D


# ═══════════════════════════════════════════════════════════════════════════════
# STUBS — replace with your implementations
# ═══════════════════════════════════════════════════════════════════════════════

def get_stabilizer_matrix(tc):
    """
    Returns binary (n_stabilizers × n_qubits) matrix H such that
    H[s] is the support of stabilizer s as a binary vector.
    Applying stabilizer s to error e: e ^ H[s].
    """
    _, HX = tc.stabilizer_matrices()
    return HX


def get_min_weight_reps(tc, syndrome):
    """
    Returns {class_label: binary_array} — the minimum-weight error
    chain in each logical coset consistent with `syndrome`.
    class_label in {0, 1, 2, 3}.
    """
    _, HX = tc.stabilizer_matrices()
    # Basic MWPM rep for Z errors (detectable by X syndrome)
    eZ_base = utils.mwpm_initialize_e_given_syndrome(HX, syndrome).astype(np.int8)
    
    # Toric code logical Z operators (relevant for Z-error classes)
    lZ1 = np.zeros(tc.n, dtype=np.int8)
    lZ1[tc.logical_Z_support()] = 1
    lZ2 = np.zeros(tc.n, dtype=np.int8)
    lZ2[tc.logical_Z_conjugate()] = 1

    # Create the four logical coset representatives
    reps = {
        0: eZ_base,
        1: eZ_base ^ lZ1,
        2: eZ_base ^ lZ2,
        3: eZ_base ^ lZ1 ^ lZ2
    }
    return reps


def sample_syndrome(tc, p, rng=None):
    """
    Sample a random error and its syndrome from the toric code under
    Z-type independent bit-flip noise at rate p.
    Returns (error, syndrome) as binary arrays.
    """
    _, eZ = depolarizing_noise(tc.n, p)
    synd = syndrome_from_eZ(eZ, tc.X_stabilizers)
    return np.array(eZ, dtype=np.int8), synd


def run_worm(tc, syndrome, p, n_steps, error_type='Z', rng=None):
    """
    Run the worm algorithm (Tobias et al.) on the toric code.
    Returns {class_label: count} — number of closed-loop samples
    in each logical coset.
    """
    # The WormDecoder is designed for depolarizing noise, where X and Z errors
    # are treated symmetrically. We need to ensure the correct chain is run
    # based on whether we are interested in X-error logical classes or Z-error logical classes.
    decoder = WormDecoder(tc, p_phys=p, n_samples=n_steps // 100, n_burnin=100)
    
    if error_type == 'X':
        # For X-errors, we need the Z-stabilizer matrix (HZ) to find the reference error.
        # The logical classes are defined by logical Z operators (lZ_vecs in WormDecoder).
        HZ, _ = tc.stabilizer_matrices()
        eX_ref = utils.mwpm_initialize_e_given_syndrome(HZ, syndrome).astype(np.int8)
        log_w = np.log((2*p/3) / (1 - 2*p/3)) # This is for X errors
        log_w_vec = np.full(tc.n, log_w)
        counts_counter = decoder._run_x_chain(eX_ref, log_w_vec)
    elif error_type == 'Z':
        # For Z-errors, we need the X-stabilizer matrix (HX) to find the reference error.
        # The logical classes are defined by logical X operators (lX_vecs in WormDecoder).
        _, HX = tc.stabilizer_matrices()
        eZ_ref = utils.mwpm_initialize_e_given_syndrome(HX, syndrome).astype(np.int8)
        # _run_z_chain returns (z_counts, alpha), we only need z_counts here
        counts_counter, _ = decoder._run_z_chain(eZ_ref) 
    else:
        raise ValueError("error_type must be 'X' or 'Z'")

    
    # Map the tuple (bit1, bit2) keys to integer 0-3
    mapped_counts = {}
    for bits, count in counts_counter.items():
        idx = bits[0] + 2 * bits[1] if len(bits) > 1 else bits[0]
        mapped_counts[idx] = count
    return mapped_counts

def get_logical_class(error, tc):
    lX1 = tc.logical_X_support()
    lX2 = tc.logical_X_conjugate()
    bit1 = logical_parity(error, lX1)
    bit2 = logical_parity(error, lX2)
    return bit1 + 2 * bit2


# ═══════════════════════════════════════════════════════════════════════════════
# CORE: Metropolis within one equivalence class
# ═══════════════════════════════════════════════════════════════════════════════

def metropolis_within_class(
    init_error,
    stabilizer_matrix,
    beta,
    n_warmup=4_000,
    n_samples=12_000,
    rng=None,
):
    """
    Metropolis sampling within one logical equivalence class at inverse
    temperature β.  Moves: randomly apply one stabilizer (XOR with a
    row of stabilizer_matrix), accept with min(1, exp(-β Δn)).

    Returns the mean error weight ⟨n⟩_β estimated from n_samples steps.
    """
    if rng is None:
        rng = np.random.default_rng()

    H = stabilizer_matrix.astype(np.int8)
    n_stab = H.shape[0]
    current = init_error.copy().astype(np.int8)
    current_n = int(current.sum())

    # Warmup
    for _ in range(n_warmup):
        s = int(rng.integers(n_stab))
        proposed = current ^ H[s]
        proposed_n = int(proposed.sum())
        delta_n = proposed_n - current_n
        if delta_n <= 0 or rng.random() < np.exp(-beta * delta_n):
            current = proposed
            current_n = proposed_n

    # Sampling
    acc = 0.0
    for _ in range(n_samples):
        s = int(rng.integers(n_stab))
        proposed = current ^ H[s]
        proposed_n = int(proposed.sum())
        delta_n = proposed_n - current_n
        if delta_n <= 0 or rng.random() < np.exp(-beta * delta_n):
            current = proposed
            current_n = proposed_n
        acc += current_n

    return acc / n_samples


def compute_n_beta_curve(
    init_error,
    stabilizer_matrix,
    beta_values,        # 1-D array, will be traversed high→low (heating schedule)
    n_warmup=4_000,
    n_samples=12_000,
    rng=None,
):
    """
    Compute ⟨n⟩_{β,E} for each β in beta_values.

    Uses a single-chain heating schedule: start from the minimum-weight
    configuration (ground state) and heat upward.  This is more efficient
    than independent chains because the ground state is a valid starting
    point for all temperatures, and higher-T steps carry memory that
    helps mix at the next temperature.

    Returns array of same length as beta_values (matching the input order).
    """
    if rng is None:
        rng = np.random.default_rng()

    H = stabilizer_matrix.astype(np.int8)
    n_stab = H.shape[0]

    # Sort descending (high β = low T = start cold), then sweep upward
    order = np.argsort(beta_values)[::-1]   # indices that sort high→low
    betas_hl = beta_values[order]

    current = init_error.copy().astype(np.int8)
    current_n = int(current.sum())
    n_avg = np.empty(len(beta_values))

    for rank, idx in enumerate(order):
        beta = betas_hl[rank]

        # Warmup at this temperature (fewer steps needed because we carry
        # the chain from the previous, slightly cooler temperature)
        warmup = n_warmup if rank == 0 else n_warmup // 3
        for _ in range(warmup):
            s = int(rng.integers(n_stab))
            proposed = current ^ H[s]
            proposed_n = int(proposed.sum())
            delta_n = proposed_n - current_n
            if delta_n <= 0 or rng.random() < np.exp(-beta * delta_n):
                current = proposed
                current_n = proposed_n

        # Collect samples
        acc = 0.0
        for _ in range(n_samples):
            s = int(rng.integers(n_stab))
            proposed = current ^ H[s]
            proposed_n = int(proposed.sum())
            delta_n = proposed_n - current_n
            if delta_n <= 0 or rng.random() < np.exp(-beta * delta_n):
                current = proposed
                current_n = proposed_n
            acc += current_n

        n_avg[idx] = acc / n_samples

    return n_avg


# ═══════════════════════════════════════════════════════════════════════════════
# DECODER DECISIONS
# ═══════════════════════════════════════════════════════════════════════════════

def hutter_decision(curves, beta_bar):
    """
    Single-temperature rule: compare ⟨n⟩ at β̄ across classes.
    Returns the class with the smallest ⟨n⟩_{β̄,E}.
    curves: {class_label: (beta_values, n_avg_array)}
    """
    vals = {
        cls: np.interp(beta_bar, betas, n_avg)
        for cls, (betas, n_avg) in curves.items()
    }
    return min(vals, key=vals.get), vals


def tobias_decision_integral(curves, beta_bar):
    """
    Integral rule: compare ∫₀^{β̄} ⟨n⟩_{β,E} dβ across classes.
    The class with the SMALLEST integral has the LARGEST log Z_E(β̄).
    Uses Simpson's rule for the integration.
    Returns (winning_class, {class: integral_value}).
    """
    integrals = {}
    for cls, (betas, n_avg) in curves.items():
        # Assumes uniform grid from 0 to beta_bar
        idx = np.argmin(np.abs(betas - beta_bar))
        integrals[cls] = utils.simpson_integral(betas[:idx+1], n_avg[:idx+1])
    return min(integrals, key=integrals.get), integrals


def find_free_energy_threshold(betas, n_a, n_b):
    """
    Finds beta* such that the integral from 0 to beta* of (n_a - n_b) is zero.
    This is the threshold where the two classes have equal probability.
    """
    diff = n_a - n_b
    # Compute cumulative integral using trapezoidal rule
    cum_int = np.zeros_like(betas)
    for i in range(1, len(betas)):
        cum_int[i] = cum_int[i-1] + np.trapezoid(diff[i-1:i+1], betas[i-1:i+1])
    
    # Find the crossing point where cum_int hits 0 (after the initial point)
    # We look for a sign change in the cumulative integral
    for i in range(2, len(cum_int)):
        if np.sign(cum_int[i]) != np.sign(cum_int[i-1]):
            # Linear interpolation for better accuracy
            frac = abs(cum_int[i-1]) / (abs(cum_int[i-1]) + abs(cum_int[i]))
            return betas[i-1] + frac * (betas[i] - betas[i-1])
    return None


def tobias_decision_samples(worm_counts):
    """
    Tobias' actual decision: pick the class with the most worm samples.
    worm_counts: {class_label: int}
    """
    return max(worm_counts, key=worm_counts.get)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

# Four colors for the four logical cosets
CLASS_COLORS  = {0: '#1D9E75', 1: '#D85A30', 2: '#378ADD', 3: '#BA7517'}
CLASS_LABELS  = {0: '$E_0$ (Identity)', 1: '$E_1$ (Logical $Z_1$)',
                 2: '$E_2$ (Logical $Z_2$)', 3: '$E_3$ (Logical $Z_1Z_2$)'}
CLASS_DASHES  = {0: (None, None), 1: (6, 2), 2: (2, 2), 3: (4, 2, 1, 2)}


def _linestyle_from_dash(dash):
    if dash == (None, None):
        return '-'
    return (0, dash)


def shade_between_pair(ax, betas, n_a, n_b, col_a, col_b, alpha=0.22):
    """Shade where n_a < n_b (col_a wins) and where n_b < n_a (col_b wins)."""
    ax.fill_between(betas, n_a, n_b, where=(n_a <= n_b),
                    color=col_a, alpha=alpha, zorder=1)
    ax.fill_between(betas, n_a, n_b, where=(n_b <= n_a),
                    color=col_b, alpha=alpha, zorder=1)


def plot_n_beta_curves(
    tc,                    # your toric code object
    p,                     # physical error rate
    syndrome=None,         # binary syndrome array; if None, sampled randomly
    true_class=None,       # ground-truth class; if None, not annotated
    n_beta=60,             # number of β grid points
    beta_max_factor=1.3,   # plot up to this × β̄
    n_warmup=4_000,
    n_samples=12_000,
    n_worm_steps=500_000,  # steps for worm run (Tobias decision)
    seed=0,
    shade_pair=None,       # (cls_a, cls_b) to shade between; defaults to
                           # (true_class, most-competitive-false) if true_class given
    figsize=(10, 5.5),
    show_threshold=True,   # Toggle for the free-energy crossing line
):
    """
    Main entry point.  Computes and plots ⟨n⟩_{β,E} for all four cosets
    of the toric code, and annotates Hutter vs Tobias decoder decisions.

    Parameters
    ----------
    tc            Your toric code object (passed through to your stubs).
    p             Physical bit-flip error rate.
    syndrome      If None, a random syndrome is drawn with sample_syndrome().
    true_class    Ground-truth logical class (0-3); used only for annotations.
    n_beta        Number of β grid points.
    ...
    """
    rng = np.random.default_rng(seed)
    H = get_stabilizer_matrix(tc)

    # ── 1. Syndrome ────────────────────────────────────────────────────────────
    if syndrome is None:
        error, syndrome = sample_syndrome(tc, p, rng=rng)
        if true_class is None:
            raise ValueError(
                "If syndrome is sampled internally, pass true_class "
                "so the plot can annotate which decoder is correct."
            )

    # ── 2. Minimum-weight representatives ─────────────────────────────────────
    mw_reps = get_min_weight_reps(tc, syndrome)
    all_classes = sorted(mw_reps.keys())

    # ── 3. β grid ─────────────────────────────────────────────────────────────
    bb = -np.log(p / (1.0 - p))
    beta_max = bb * beta_max_factor
    # Simpson's rule requires an odd number of uniform points on [0, bb]
    n_int = n_beta if n_beta % 2 != 0 else n_beta + 1
    betas_decision = np.linspace(0.0, bb, n_int)
    h = betas_decision[1] - betas_decision[0]
    n_extra = int(np.round((bb * (beta_max_factor - 1)) / h))
    if n_extra > 0:
        betas_extra = bb + np.arange(1, n_extra + 1) * h
        betas = np.concatenate([betas_decision, betas_extra])
    else:
        betas = betas_decision

    # ── 4. Compute ⟨n⟩_{β,E} for each class ──────────────────────────────────
    curves = {}   # {cls: n_avg array}
    for cls in all_classes:
        print(f"  Computing ⟨n⟩_β for class {cls}…", flush=True)
        curves[cls] = compute_n_beta_curve(
            mw_reps[cls], H, betas, n_warmup, n_samples, rng=rng
        )

    # ── 5. Decoder decisions ──────────────────────────────────────────────────
    curves_for_decision = {cls: (betas, curves[cls]) for cls in all_classes}

    hutter_cls, hutter_vals     = hutter_decision(curves_for_decision, bb)
    integral_cls, integral_vals = tobias_decision_integral(curves_for_decision, bb)
    worm_counts = {}
    worm_cls = None

    # ── 6. Which pair to shade ────────────────────────────────────────────────
    if shade_pair is None and true_class is not None:
        competitors = [c for c in all_classes if c != true_class]
        false_cls   = min(competitors, key=lambda c: integral_vals[c])
        shade_pair  = (true_class, false_cls)

    # ── 7. Figure ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)

    # Shaded regions
    if shade_pair is not None:
        ca, cb = shade_pair
        shade_between_pair(
            ax, betas, curves[ca], curves[cb],
            CLASS_COLORS[ca], CLASS_COLORS[cb], alpha=0.22
        )
        # Annotate A₁ and A₂ at the midpoint between curves
        for ca2, cb2, label in [(ca, cb, 'A₁'), (cb, ca, 'A₂')]:
            above = betas[curves[ca2] < curves[cb2]]
            if len(above) < 3:
                continue
            b_mid   = above[len(above) // 2]
            n_mid   = (np.interp(b_mid, betas, curves[ca2]) +
                       np.interp(b_mid, betas, curves[cb2])) / 2
            gap     = abs(np.interp(b_mid, betas, curves[ca2]) -
                          np.interp(b_mid, betas, curves[cb2]))
            if gap > 0.4:
                ax.text(b_mid, n_mid, label, ha='center', va='center',
                        fontsize=9, color=CLASS_COLORS[ca2],
                        fontweight='bold', alpha=0.8)

    # Curves for all four classes
    for cls in all_classes:
        lw     = 2.4 if (shade_pair and cls in shade_pair) else 1.2
        ls     = _linestyle_from_dash(CLASS_DASHES[cls])
        zorder = 4 if (shade_pair and cls in shade_pair) else 3
        ax.plot(betas, curves[cls],
                color=CLASS_COLORS[cls], linewidth=lw, linestyle=ls,
                label=CLASS_LABELS[cls], zorder=zorder)

    # β̄ vertical line
    ax.axvline(bb, color='#555', linewidth=1.4, linestyle='--', zorder=5,
               label=f'$\\bar{{\\beta}} = {bb:.2f}$  ($p={p}$)')

    # Dots at β̄ for each class
    for cls in all_classes:
        v = np.interp(bb, betas, curves[cls])
        ax.scatter([bb], [v], color=CLASS_COLORS[cls], s=55, zorder=6,
                   edgecolors='white', linewidths=0.8)

    # ── 8. Annotation box ─────────────────────────────────────────────────────
    def tick(pred, truth):
        if truth is None:
            return ''
        return '  ✓' if pred == truth else '  ✗'

    A1_val, A2_val = 0.0, 0.0
    if shade_pair:
        ca, cb = shade_pair
        A1_val = np.trapezoid(np.maximum(curves[cb] - curves[ca], 0), betas)
        A2_val = np.trapezoid(np.maximum(curves[ca] - curves[cb], 0), betas)

        if show_threshold:
            beta_thresh = find_free_energy_threshold(betas, curves[ca], curves[cb])
            if beta_thresh:
                ax.axvline(beta_thresh, color='purple', alpha=0.6, linestyle=':', 
                           label=f'Threshold $\\beta^*={beta_thresh:.2f}$')

    worm_frac = {cls: worm_counts.get(cls, 0) / max(sum(worm_counts.values()), 1)
                 for cls in all_classes}

    lines = [
        f"ST   ⟨n⟩ at β̄:   E_{hutter_cls}{tick(hutter_cls, true_class)}",
        f"Opt  ∫⟨n⟩dβ:     E_{integral_cls}{tick(integral_cls, true_class)}",
    ]
    if shade_pair:
        lines += [
            "",
            f"A₁ = {A1_val:.3f}   A₂ = {A2_val:.3f}",
            f"Shading: E_{shade_pair[0]} vs E_{shade_pair[1]}",
        ]

    ax.text(0.02, 0.97, '\n'.join(lines),
            transform=ax.transAxes, fontsize=8.5, va='top',
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='#ccc', alpha=0.90))

    # ── 9. Labels, grid, legend ───────────────────────────────────────────────
    ax.set_xlabel(r'$\beta$ (inverse temperature)', fontsize=12)
    ax.set_ylabel(r'$\langle n \rangle_{\beta,E}$', fontsize=12)
    ax.set_title(
        fr'Toric code — $\langle n \rangle_{{\beta,E}}$ curves  '
        fr'($L={getattr(tc,"L","?")}$,  $p={p}$)',
        fontsize=13
    )
    ax.set_xlim(0, beta_max)
    ax.grid(True, linewidth=0.35, alpha=0.5)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # Custom legend: class curves + β̄ line + shaded areas
    legend_handles = [
        Line2D([0], [0], color=CLASS_COLORS[c], linewidth=2,
               linestyle=_linestyle_from_dash(CLASS_DASHES[c]),
               label=CLASS_LABELS[c])
        for c in all_classes
    ]
    legend_handles.append(
        Line2D([0], [0], color='#555', linewidth=1.4, linestyle='--',
               label=f'$\\bar{{\\beta}}={bb:.2f}$')
    )
    if shade_pair:
        from matplotlib.patches import Patch
        ca, cb = shade_pair
        legend_handles += [
            Patch(facecolor=CLASS_COLORS[ca], alpha=0.35,
                  label=f'A₁: $E_{ca}$ advantage'),
            Patch(facecolor=CLASS_COLORS[cb], alpha=0.35,
                  label=f'A₂: $E_{cb}$ advantage'),
        ]
    ax.legend(handles=legend_handles, fontsize=8.5, loc='upper right',
              framealpha=0.9, edgecolor='#ccc')

    fig.tight_layout()
    return fig, ax, {
        'betas': betas,
        'curves': curves,
        'hutter_class': hutter_cls,
        'hutter_vals': hutter_vals,
        'integral_class': integral_cls,
        'integral_vals': integral_vals,
        'worm_class': worm_cls,
        'worm_counts': worm_counts,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SWEEP: plot hutter-vs-tobias *disagreement rate* as a function of p
# ═══════════════════════════════════════════════════════════════════════════════

def sweep_disagreement(
    tc,
    p_values,
    n_trials=40,
    n_warmup=3_000,
    n_samples=8_000,
    n_beta=40,
    n_worm_steps=200_000,
    seed=0,
):
    """
    For each p, estimate the probability that Hutter and Tobias disagree
    and the probability that each decoder is wrong.
    Returns dict of arrays.
    """
    rng = np.random.default_rng(seed)
    results = {
        'p': p_values,
        'disagree_rate':      np.zeros(len(p_values)),
        'hutter_error_rate':  np.zeros(len(p_values)),
        'tobias_error_rate':  np.zeros(len(p_values)),
    }

    for ip, p in enumerate(p_values):
        print(f"\np = {p:.3f}")
        bb = -np.log(p / (1.0 - p))
        beta_max = bb * 1.3
        betas = np.linspace(0.05, beta_max, n_beta)
        H = get_stabilizer_matrix(tc)

        n_disagree = 0
        n_hutter_wrong = 0
        n_tobias_wrong = 0

        for trial in range(n_trials):
            error, syndrome = sample_syndrome(tc, p, rng=rng)
            true_cls = get_logical_class(error, tc)
            mw_reps  = get_min_weight_reps(tc, syndrome)

            curves = {
                cls: compute_n_beta_curve(
                    mw_reps[cls], H, betas, n_warmup, n_samples, rng=rng
                )
                for cls in mw_reps
            }
            curves_d  = {cls: (betas, curves[cls]) for cls in curves}
            hutter_c, _ = hutter_decision(curves_d, bb)
            tobias_c, _ = tobias_decision_integral(curves_d, bb)

            if hutter_c != tobias_c:
                n_disagree += 1
            if hutter_c != true_cls:
                n_hutter_wrong += 1
            if tobias_c != true_cls:
                n_tobias_wrong += 1

        results['disagree_rate'][ip]     = n_disagree / n_trials
        results['hutter_error_rate'][ip] = n_hutter_wrong / n_trials
        results['tobias_error_rate'][ip] = n_tobias_wrong / n_trials
        print(f"  disagree={n_disagree}/{n_trials}  "
              f"hutter_err={n_hutter_wrong}/{n_trials}  "
              f"tobias_err={n_tobias_wrong}/{n_trials}")

    return results


def plot_sweep(results):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    p = results['p']
    ax.plot(p, results['disagree_rate'],     'k-o',  ms=5, label='Hutter ≠ Tobias')
    ax.plot(p, results['hutter_error_rate'], color='#D85A30', marker='s', ms=5,
            label='Hutter wrong')
    ax.plot(p, results['tobias_error_rate'], color='#1D9E75', marker='^', ms=5,
            label='Tobias (integral) wrong')
    ax.set_xlabel('Physical error rate $p$', fontsize=12)
    ax.set_ylabel('Rate', fontsize=12)
    ax.set_title('Hutter vs Tobias disagreement on toric code', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, linewidth=0.35, alpha=0.5)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    tc = ToricCode(L=5)
    p_phys = 0.08
    
    print(f"Generating n-beta curves for L={tc.L}, p={p_phys}...")
    error, syndrome = sample_syndrome(tc, p=p_phys)
    true_class = get_logical_class(error, tc)
    
    fig, ax, data = plot_n_beta_curves(
        tc, p=p_phys,
        syndrome=syndrome,
        true_class=true_class,
        n_beta=40,
        n_warmup=2000,
        n_samples=5000,
        n_worm_steps=100000,
        seed=42,
        show_threshold=False,  # Change to False to toggle off the purple line
    )
    plt.savefig('n_beta_curves_L5_p08.pdf', dpi=150, bbox_inches='tight')
    print("Saved plot to n_beta_curves_L5_p08.pdf")

    # To run the sweep, uncomment below:
    # p_vals = np.linspace(0.05, 0.18, 5)
    # results = sweep_disagreement(tc, p_vals, n_trials=10, seed=0)
    # fig2, ax2 = plot_sweep(results)
    # plt.savefig('sweep_disagreement.pdf', dpi=150, bbox_inches='tight')