import numpy as np

def syndrome_from_eX(eX, Z_stabilizers):
    return np.array([sum(eX[q] for q in stab) % 2 for stab in Z_stabilizers])

def syndrome_from_eZ(eZ, X_stabilizers):
    return np.array([sum(eZ[q] for q in stab) % 2 for stab in X_stabilizers])
