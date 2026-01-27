from noise import depolarizing_noise
from syndrome import syndrome_from_eX, syndrome_from_eZ
from logical import logical_parity 

def run_trial(code, p, decoder):
    eX, eZ = depolarizing_noise(code.n, p)

    sZ = syndrome_from_eX(eX, code.Z_stabilizers)
    sX = syndrome_from_eZ(eZ, code.X_stabilizers)

    eX_hat, eZ_hat = decoder.decode(sZ, sX)

    rX = [a^b for a,b in zip(eX, eX_hat)]
    rZ = [a^b for a,b in zip(eZ, eZ_hat)]

    fail_Z = logical_parity(rZ, code.logical_Z_support())
    fail_X = logical_parity(rX, code.logical_X_support())

    return fail_Z or fail_X
