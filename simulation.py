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

    # Check X errors against Z logical operators
    fail_X1 = logical_parity(rX, code.logical_Z_support())
    fail_X2 = logical_parity(rX, code.logical_Z_conjugate())
    # Check Z errors against X logical operators
    fail_Z1 = logical_parity(rZ, code.logical_X_support())
    fail_Z2 = logical_parity(rZ, code.logical_X_conjugate())

    return fail_X1 or fail_Z1 or fail_X2 or fail_Z2
