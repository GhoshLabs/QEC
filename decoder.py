import utils
import numpy as np  
import itertools
import random
from MH_sampler import metropolis_hastings_on_stabilizers, metropolis_hastings_joint, metropolis_hastings_track_z, metropolis_hastings_avg_weight

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

    def decode(self, syndZ, syndX, init_method='MWPM'):
        # Initial solution via MWPM or Gaussian elimination
        if init_method == 'MWPM':
            eX_init = utils.mwpm_initialize_e_given_syndrome(self.HZ, syndZ)
            eZ_init = utils.mwpm_initialize_e_given_syndrome(self.HX, syndX)
        else:
            eX_init = utils.ge_initialize_given_syndrome(self.HZ, syndZ)
            eZ_init = utils.ge_initialize_given_syndrome(self.HX, syndX)

        # MH refinement for X errors
        outX = metropolis_hastings_on_stabilizers(
            self.code,
            self.HZ,
            eX_init.copy(),
            self.Xstab_vecs,
            q_error=self.q,
            n_samples=self.n_samples,
            burn_in=self.burn_in
        )

        # MH refinement for Z errors
        outZ = metropolis_hastings_on_stabilizers(
            self.code,
            self.HX,
            eZ_init.copy(),
            self.Zstab_vecs,
            q_error=self.q,
            n_samples=self.n_samples,
            burn_in=self.burn_in
        )

        return outX['best_sample'], outZ['best_sample']
    
class MHDecoderSingleChain(Decoder):
    def __init__(self, code, q_error, n_samples=2000, burn_in=500):
        self.code = code
        self.q = q_error
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.HZ, self.HX = code.stabilizer_matrices()
        
        # Precompute stabilizer vectors
        self.Zstab_vecs = [self.HZ[i] for i in range(self.HZ.shape[0])]
        self.Xstab_vecs = [self.HX[i] for i in range(self.HX.shape[0])]
        
        # Combined moves: X-stabs (act on eX) and Z-stabs (act on eZ)
        self.all_stabs = self.Xstab_vecs + self.Zstab_vecs
        self.n_X_stabs = len(self.Xstab_vecs)

    def decode(self, syndZ, syndX, init_method='MWPM'):
        # Initialize
        if init_method == 'MWPM':
            eX = utils.mwpm_initialize_e_given_syndrome(self.HZ, syndZ)
            eZ = utils.mwpm_initialize_e_given_syndrome(self.HX, syndX)
        else:
            eX = utils.ge_initialize_given_syndrome(self.HZ, syndZ)
            eZ = utils.ge_initialize_given_syndrome(self.HX, syndX)
            
        best_eX, best_eZ, _ = metropolis_hastings_joint(
            eX, 
            eZ, 
            self.all_stabs, 
            self.n_X_stabs, 
            self.q, 
            self.n_samples
        )

        return best_eX, best_eZ

class MHDecoderTrackZ(Decoder):
    def __init__(self, code, q_error, n_samples=2000, burn_in=500):
        self.code = code
        self.q = q_error
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.HZ, self.HX = code.stabilizer_matrices()
        
        # Precompute stabilizer vectors
        self.Zstab_vecs = [self.HZ[i] for i in range(self.HZ.shape[0])]
        self.Xstab_vecs = [self.HX[i] for i in range(self.HX.shape[0])]
        
        # Combined moves: X-stabs (act on eX) and Z-stabs (act on eZ)
        self.all_stabs = self.Xstab_vecs + self.Zstab_vecs
        self.n_X_stabs = len(self.Xstab_vecs)

        # Precompute logical operators dynamically
        n = self.code.n
        log_X_supports = [s for s in [self.code.logical_X_support(), self.code.logical_X_conjugate()] if s]
        log_Z_supports = [s for s in [self.code.logical_Z_support(), self.code.logical_Z_conjugate()] if s]

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

        self.logicals_X = []
        self.logicals_Z = []

        lX_combinations = []
        for b_bits in itertools.product([0, 1], repeat=num_logical_qubits):
            lX = np.zeros(n, dtype=int)
            for i, b in enumerate(b_bits):
                if b: lX ^= log_X_op_vecs[i]
            lX_combinations.append(lX)

        lZ_combinations = []
        for c_bits in itertools.product([0, 1], repeat=num_logical_qubits):
            lZ = np.zeros(n, dtype=int)
            for i, c in enumerate(c_bits):
                if c: lZ ^= log_Z_op_vecs[i]
            lZ_combinations.append(lZ)

        for lZ in lZ_combinations:
            for lX in lX_combinations:
                self.logicals_X.append(lX)
                self.logicals_Z.append(lZ)

    def decode(self, syndZ, syndX, init_method='MWPM'):
        # Initialize to trivial logical class
        if init_method == 'MWPM':
            eX = utils.mwpm_initialize_e_given_syndrome(self.HZ, syndZ)
            eZ = utils.mwpm_initialize_e_given_syndrome(self.HX, syndX)
        else:
            eX = utils.ge_initialize_given_syndrome(self.HZ, syndZ)
            eZ = utils.ge_initialize_given_syndrome(self.HX, syndX)
            
        best_eX, best_eZ, Z_ratios = metropolis_hastings_track_z(
            eX, 
            eZ, 
            self.all_stabs, 
            self.n_X_stabs, 
            self.q, 
            self.n_samples, 
            self.burn_in, 
            self.logicals_X, 
            self.logicals_Z
        )
        
        best_class_idx = np.argmax(Z_ratios)
        
        lX_hat, lZ_hat = self.logicals_X[best_class_idx], self.logicals_Z[best_class_idx]
        
        return best_eX ^ lX_hat, best_eZ ^ lZ_hat
    
class MHDecoderParallel(Decoder):
    def __init__(self, code, q_error, n_samples=2000, burn_in=500):
        self.code = code
        self.q = q_error
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.HZ, self.HX = code.stabilizer_matrices()
        
        # Precompute stabilizer vectors
        self.Zstab_vecs = [self.HZ[i] for i in range(self.HZ.shape[0])]
        self.Xstab_vecs = [self.HX[i] for i in range(self.HX.shape[0])]
        
        # Combined moves: X-stabs (act on eX) and Z-stabs (act on eZ)
        self.all_stabs = self.Xstab_vecs + self.Zstab_vecs
        self.n_X_stabs = len(self.Xstab_vecs)

        # Precompute logical operators dynamically
        n = self.code.n
        log_X_supports = [s for s in [self.code.logical_X_support(), self.code.logical_X_conjugate()] if s]
        log_Z_supports = [s for s in [self.code.logical_Z_support(), self.code.logical_Z_conjugate()] if s]

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

        self.logicals_X = []
        self.logicals_Z = []

        lX_combinations = []
        for b_bits in itertools.product([0, 1], repeat=num_logical_qubits):
            lX = np.zeros(n, dtype=int)
            for i, b in enumerate(b_bits):
                if b: lX ^= log_X_op_vecs[i]
            lX_combinations.append(lX)

        lZ_combinations = []
        for c_bits in itertools.product([0, 1], repeat=num_logical_qubits):
            lZ = np.zeros(n, dtype=int)
            for i, c in enumerate(c_bits):
                if c: lZ ^= log_Z_op_vecs[i]
            lZ_combinations.append(lZ)

        for lZ in lZ_combinations:
            for lX in lX_combinations:
                self.logicals_X.append(lX)
                self.logicals_Z.append(lZ)

    def decode(self, syndZ, syndX, init_method='MWPM'):
        # Initialize trivial class representative
        if init_method == 'MWPM':
            eX_trivial = utils.mwpm_initialize_e_given_syndrome(self.HZ, syndZ)
            eZ_trivial = utils.mwpm_initialize_e_given_syndrome(self.HX, syndX)
        else:
            eX_trivial = utils.ge_initialize_given_syndrome(self.HZ, syndZ)
            eZ_trivial = utils.ge_initialize_given_syndrome(self.HX, syndX)
            
        min_avg_weight = np.inf
        overall_best_eX = eX_trivial.copy()
        overall_best_eZ = eZ_trivial.copy()
        
        # Run parallel chains for each logical class
        for k in range(len(self.logicals_X)):
            lX_k, lZ_k = self.logicals_X[k], self.logicals_Z[k]
            
            # Initialize chain in the k-th logical class
            init_eX = eX_trivial ^ lX_k
            init_eZ = eZ_trivial ^ lZ_k
            
            avg_weight_k, best_eX_k, best_eZ_k = metropolis_hastings_avg_weight(
                init_eX, init_eZ, self.all_stabs, self.n_X_stabs, self.q, self.n_samples, self.burn_in
            )
            
            if avg_weight_k < min_avg_weight:
                min_avg_weight = avg_weight_k
                overall_best_eX = best_eX_k.copy()
                overall_best_eZ = best_eZ_k.copy()
                
        return overall_best_eX, overall_best_eZ

class GEDecoder(Decoder):
    def __init__(self, code):
        self.code = code
        self.HZ, self.HX = code.stabilizer_matrices()

    def decode(self, syndZ, syndX):
        eX_hat = utils.ge_initialize_given_syndrome(self.HZ, syndZ)
        eZ_hat = utils.ge_initialize_given_syndrome(self.HX, syndX)
        return eX_hat, eZ_hat