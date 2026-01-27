import utils
import numpy as np  
from MH_sampler import metropolis_hastings_on_stabilizers

class Decoder:
    def decode(self, syndZ, syndX):
        """
        Input:
          syndZ : Z-stabilizer syndrome (detects X errors)
          syndX : X-stabilizer syndrome (detects Z errors)

        Output:
          eX_hat, eZ_hat : estimated Pauli-frame corrections
        """
        raise NotImplementedError

class MWPMDecoder(Decoder):
    def __init__(self, code):
        # Precompute stabilizer matrices once
        self.HZ, self.HX = code.stabilizer_matrices()

    def decode(self, syndZ, syndX):
        """
        Z syndromes -> X error estimate
        X syndromes -> Z error estimate
        """
        eX_hat = utils.mwpm_initialize_e_given_syndrome(self.HZ, syndZ)
        eZ_hat = utils.mwpm_initialize_e_given_syndrome(self.HX, syndX)
        return eX_hat, eZ_hat

class MHDecoder(Decoder):
    def __init__(self, code, q_error, n_samples=2000, burn_in=500):
        self.code = code
        self.q = q_error
        self.n_samples = n_samples
        self.burn_in = burn_in

        # Stabilizer matrices
        self.HZ, self.HX = code.stabilizer_matrices()

        # Precompute stabilizer vectors (for MH moves)
        self.Zstab_vecs = [self.HZ[i] for i in range(self.HZ.shape[0])]
        self.Xstab_vecs = [self.HX[i] for i in range(self.HX.shape[0])]

    def decode(self, syndZ, syndX):
        # Initial solution via MWPM or Gaussian elimination
        eX_init = utils.mwpm_initialize_e_given_syndrome(self.HZ, syndZ)
        eZ_init = utils.mwpm_initialize_e_given_syndrome(self.HX, syndX)

        # MH refinement for X errors
        outX = metropolis_hastings_on_stabilizers(
            self.code,
            self.HZ,
            eX_init.copy(),
            self.Zstab_vecs,
            q_error=self.q,
            n_samples=self.n_samples,
            burn_in=self.burn_in
        )

        # MH refinement for Z errors
        outZ = metropolis_hastings_on_stabilizers(
            self.code,
            self.HX,
            eZ_init.copy(),
            self.Xstab_vecs,
            q_error=self.q,
            n_samples=self.n_samples,
            burn_in=self.burn_in
        )

        return outX['e_map'], outZ['e_map']
    

