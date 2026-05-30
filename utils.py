import numpy as np
from itertools import product
from scipy import sparse
from qiskit.quantum_info import SparsePauliOp
try:
    from scipy.special import logsumexp
except ImportError:
    def logsumexp(a, axis=None, keepdims=False):
        a = np.array(a)
        a_max = np.max(a, axis=axis, keepdims=True)
        if keepdims:
            a_max2 = a_max
        else:
            a_max2 = np.squeeze(a_max, axis=axis)
        summation = np.sum(np.exp(a - a_max), axis=axis)
        return np.log(summation) + a_max2
try:
    import pymatching
    _HAVE_PYMATCHING = True
except Exception:
    _HAVE_PYMATCHING = False

def pauli_list_to_sparseop(pauli_list):
    n = len(pauli_list)
    s = ''.join(reversed(pauli_list))
    return SparsePauliOp.from_list([(s, 1.0)])

def pauli_list_to_matrix(pauli_list):
    return pauli_list_to_sparseop(pauli_list).to_matrix()

def pauli_list_weight(pauli_list):
    return sum(1 for p in pauli_list if p != 'I')

def binary_pair_to_pauli_list(eX, eZ):
    p = []
    for x,z in zip(eX, eZ):
        if x==0 and z==0:
            p.append('I')
        elif x==1 and z==0:
            p.append('X')
        elif x==0 and z==1:
            p.append('Z')
        elif x==1 and z==1:
            p.append('Y')
    return p

def pauli_list_to_binary_pair(plist):
    eX = []
    eZ = []
    for p in plist:
        if p == 'I':
            eX.append(0); eZ.append(0)
        elif p == 'X':
            eX.append(1); eZ.append(0)
        elif p == 'Z':
            eX.append(0); eZ.append(1)
        elif p == 'Y':
            eX.append(1); eZ.append(1)
        else:
            raise ValueError("unknown pauli "+p)
    return np.array(eX, dtype=int), np.array(eZ, dtype=int)

'''def syndrome_from_eX(eX, HZ):
    return (HZ.dot(eX) % 2).astype(int)

def syndrome_from_eZ(eZ, HX):
    return (HX.dot(eZ) % 2).astype(int)'''

def mwpm_initialize_e_given_syndrome(H, syndrome):
    m, n = H.shape
    if _HAVE_PYMATCHING:
        Ms = sparse.csr_matrix(H)
        M = pymatching.Matching(Ms)
        e = M.decode(syndrome.tolist())
        e = np.array(e, dtype=int)
        return e
    else:
        # fallback: simple gaussian elimination mod 2 to find a particular solution
        e= ge_initialize_given_syndrome(H, syndrome)
        return e
    
def ge_initialize_given_syndrome(H, syndrome):
    m, n = H.shape
    # Solve H x = s over GF(2).
    A = np.concatenate([H.copy() % 2, syndrome.reshape(-1,1)], axis=1).astype(int)
    # row reduce (m x (n+1))
    r = 0
    pivots = []
    for c in range(n):
        # find row with 1 in column c at or below row r
        for i in range(r, m):
            if A[i, c] == 1:
                A[[r, i]] = A[[i, r]]
                break
        else:
            continue
        pivots.append(c)
        # eliminate other rows
        for i in range(m):
            if i != r and A[i, c] == 1:
                A[i, :] ^= A[r, :]
        r += 1
        if r == m:
            break
    # now set free variables to 0 and back-substitute to get x
    x = np.zeros(n, dtype=int)
    # for each pivot row, find pivot column
    for i_row in range(min(m, len(pivots))):
        c = pivots[i_row]
        x[c] = A[i_row, -1]  # RHS
    return x

def coset_weight_enum(eX, eZ, code):
    n = code.n
    # Toric codes have one redundant X and Z stabilizer; Planar codes do not.
    is_toric = code.__class__.__name__ == 'ToricCode'
    X_stabs = code.X_stabilizers[:-1] if is_toric else code.X_stabilizers
    Z_stabs = code.Z_stabilizers[:-1] if is_toric else code.Z_stabilizers
    
    # Convert index-based stabilizers to binary vectors
    X_vecs = []
    for stab in X_stabs:
        v = np.zeros(n, dtype=int)
        v[stab] = 1
        X_vecs.append(v)

    Z_vecs = []
    for stab in Z_stabs:
        v = np.zeros(n, dtype=int)
        v[stab] = 1
        Z_vecs.append(v)

    A = np.zeros(n + 1, dtype=np.int64)
    eX_arr, eZ_arr = np.array(eX), np.array(eZ)

    # Iterate through all possible stabilizer combinations
    for xb in product([0, 1], repeat=len(X_vecs)):
        xs = np.zeros(n, dtype=int)
        for b, v in zip(xb, X_vecs):
            if b: xs ^= v
            
        for zb in product([0, 1], repeat=len(Z_vecs)):
            zs = np.zeros(n, dtype=int)
            for b, v in zip(zb, Z_vecs):
                if b: zs ^= v

            # Compute weight using the existing utility function
            w = np.sum(((eX_arr ^ xs) | (eZ_arr ^ zs)))#weight(eX_arr ^ xs, eZ_arr ^ zs)
            A[w] += 1

    return A

def coset_weight_distr(eX, eZ, code, p):
    n = code.n
    A = coset_weight_enum(eX, eZ, code)
    P_coset = 0
    for w, count in enumerate(A):
        P_coset += count * ((p/3)**w) * ((1-p)**(n-w)) 
    return P_coset

def generate_all_sectors(eX, eZ, code):
    """
    Generates error configurations for all logical sectors by applying all 
    combinations of logical X and Z operators to the initial error configuration.
    """
    n = code.n
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

    lX_combinations = []
    for b_bits in product([0, 1], repeat=num_logical_qubits):
        lX = np.zeros(n, dtype=int)
        for i, b in enumerate(b_bits):
            if b: lX ^= log_X_op_vecs[i]
        lX_combinations.append(lX)

    lZ_combinations = []
    for c_bits in product([0, 1], repeat=num_logical_qubits):
        lZ = np.zeros(n, dtype=int)
        for i, c in enumerate(c_bits):
            if c: lZ ^= log_Z_op_vecs[i]
        lZ_combinations.append(lZ)

    eX_arr, eZ_arr = np.array(eX, dtype=int), np.array(eZ, dtype=int)
    return [(eX_arr ^ lX, eZ_arr ^ lZ) for lZ in lZ_combinations for lX in lX_combinations]

def d_kl(probs_1, probs_2):
    # Filter out zero probabilities to avoid division by zero or log(0)
    mask = (probs_1 > 0) & (probs_2 > 0)
    D_KL = np.sum(probs_1[mask] * np.log(probs_1[mask] / probs_2[mask]))
    return D_KL


def simpson_integral(x, y):
    """Compute the integral of y(x) using composite Simpson's rule.

    x and y must be 1D arrays of equal length, with x strictly increasing.
    The number of points must be odd so that there are an even number of intervals.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError('x and y must be one-dimensional arrays.')
    if x.shape != y.shape:
        raise ValueError('x and y must have the same shape.')
    n = x.size
    if n < 2:
        raise ValueError('At least two points are required for Simpson integration.')
    if np.any(np.diff(x) <= 0):
        raise ValueError('x values must be strictly increasing.')
    if n % 2 == 0:
        raise ValueError('Simpson integration requires an odd number of points.')

    h = x[1] - x[0]
    if not np.allclose(np.diff(x), h, atol=1e-9, rtol=1e-8):
        raise ValueError('Simpson integration requires equally spaced x values.')

    return h / 3 * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]))


def mbar(u_kln, N_k=None, max_iter=10000, tol=1e-8):
    """Compute MBAR free energies and weights from reduced potentials.

    Parameters
    ----------
    u_kln : array_like, shape (K, N)
        Reduced potential energies of N samples evaluated in K states.
    N_k : array_like, shape (K,), optional
        Number of samples drawn from each state. If omitted, samples are
        assumed to be equally represented.
    """
    u_kln = np.asarray(u_kln, dtype=float)
    K, N = u_kln.shape
    if N_k is None:
        N_k = np.full(K, N / K, dtype=float)
    else:
        N_k = np.asarray(N_k, dtype=float)
    if np.sum(N_k) <= 0:
        raise ValueError('N_k must contain positive sample counts for each state.')
    N_k = N_k * (N / np.sum(N_k))

    f = np.zeros(K, dtype=float)
    for _ in range(max_iter):
        log_denominator = logsumexp(np.log(N_k)[:, None] - f[:, None] - u_kln, axis=0)
        g_n = -log_denominator
        f_new = -logsumexp(g_n[None, :] - u_kln, axis=1)
        f_new -= f_new[0]
        if np.max(np.abs(f_new - f)) < tol:
            f = f_new
            break
        f = f_new

    log_w_kn = np.log(N_k)[:, None] - f[:, None] - u_kln - log_denominator[None, :]
    weights = np.exp(log_w_kn)
    return {
        'f_k': f,
        'log_weights': log_w_kn,
        'weights': weights
    }


def mbar_reweighted_average(observable, weights):
    """Compute MBAR reweighted expectations for each state."""
    observable = np.asarray(observable, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 2:
        raise ValueError('weights must be 2D with shape (K, N).')
    if observable.shape[-1] != weights.shape[1]:
        raise ValueError('Observable length must match number of samples.')
    numer = np.sum(weights * observable[None, :], axis=1)
    denom = np.sum(weights, axis=1)
    return numer / denom