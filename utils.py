import numpy as np
from scipy import sparse
from qiskit.quantum_info import SparsePauliOp
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