import utils
import numpy as np  
import itertools
import random
from MH_sampler import metropolis_hastings_on_stabilizers, metropolis_hastings_joint, metropolis_hastings_track_z, metropolis_hastings_avg_weight
import ldpc
from collections import Counter

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
            
            avg_weight_k, min_weight_k, min_count_k, best_eX_k, best_eZ_k = metropolis_hastings_avg_weight(
                init_eX, init_eZ, self.all_stabs, self.n_X_stabs, self.q, self.n_samples, self.burn_in
            )
            
            if avg_weight_k < min_avg_weight:
                min_avg_weight = avg_weight_k
                overall_best_eX = best_eX_k.copy()
                overall_best_eZ = best_eZ_k.copy()
                
        return overall_best_eX, overall_best_eZ

class WormDecoder(Decoder):
    """
    Correlated worm decoder for depolarizing noise, supporting all 4^k logical sectors.
    """
    _PAULI = {(0, 0): 'I', (1, 0): 'X', (0, 1): 'Z', (1, 1): 'Y'}

    def __init__(self, code, p_phys, n_samples=1000, n_burnin=100, seed=None):
        if not (0 < p_phys < 0.75):
            raise ValueError("p_phys must be in (0, 0.75)")

        self.code = code
        self.p = p_phys
        self.n_samples = n_samples
        self.n_burnin = n_burnin
        self.n = code.n
        # Use standard random for faster scalar sampling in loops
        import random as py_random
        self._py_rng = py_random.Random(seed)
        self.rng = np.random.default_rng(seed) 

        self.HZ, self.HX = code.stabilizer_matrices()
        self._adj_X, self._valid_X, self._n_vX = self._build_adj(self.HZ)
        self._adj_Z, self._valid_Z, self._n_vZ = self._build_adj(self.HX)

        def _vec(support):
            if not support:  # Handles None and empty lists
                return None
            v = np.zeros(self.n, dtype=np.int8)
            v[list(support)] = 1
            return v

        lZ_candidates = [_vec(code.logical_Z_support()), _vec(code.logical_Z_conjugate())]
        lX_candidates = [_vec(code.logical_X_support()), _vec(code.logical_X_conjugate())]

        self.lZ_vecs = [v for v in lZ_candidates if v is not None and np.any(v)]
        self.lX_vecs = [v for v in lX_candidates if v is not None and np.any(v)]

        if len(self.lZ_vecs) != len(self.lX_vecs):
            raise ValueError("Inconsistent numbers of logical X and Z operators.")
        self.k = len(self.lZ_vecs)

    def _build_adj(self, H):
        n_stabs, n_qubits = H.shape
        boundary, n_verts = n_stabs, n_stabs + 1
        adj = [[] for _ in range(n_verts)]
        for q in range(n_qubits):
            stabs = np.where(H[:, q])[0]
            if len(stabs) == 2:
                u, v = int(stabs[0]), int(stabs[1])
                adj[u].append((q, v)); adj[v].append((q, u))
            elif len(stabs) == 1:
                u = int(stabs[0])
                adj[u].append((q, boundary)); adj[boundary].append((q, u))
        valid_starts = [v for v in range(n_verts) if adj[v]]
        return adj, valid_starts, n_verts

    def _worm_run(self, Sigma, adj, valid_starts, n_verts, log_w_tilde):
        rng = self._py_rng
        i0 = rng.choice(valid_starts)
        i1, i2 = i0, i0
        log_V2 = np.log(n_verts / 2.0)
        # Reduce max_steps: 200k is too high for Python loops. 
        # 100 * N is usually sufficient for the worm to close or fail.
        n_acc, max_steps = 0, 500 * len(Sigma)
        for _ in range(max_steps):
            k_end = rng.getrandbits(1)
            i = i1 if k_end == 0 else i2
            nbrs = adj[i]
            if not nbrs: continue
            q, j = rng.choice(nbrs)
            log_f = -log_w_tilde[q] if Sigma[q] else log_w_tilde[q]
            new_i1, new_i2 = (j, i2) if k_end == 0 else (i1, j)
            if (i1 == i2) and (new_i1 != new_i2): log_psi = -log_V2
            elif (i1 != i2) and (new_i1 == new_i2): log_psi = log_V2
            else: log_psi = 0.0
            if (log_f + log_psi) >= 0.0 or rng.random() < np.exp(min(0.0, log_f + log_psi)):
                Sigma[q] ^= 1; i1, i2 = new_i1, new_i2; n_acc += 1
            if n_acc > 0 and i1 == i2: return

    def _run_z_chain(self, eZ_ref):
        p = self.p
        log_w_Z = np.log((2*p/3) / (1 - 2*p/3))
        log_wt = np.where(eZ_ref.astype(bool), -log_w_Z, log_w_Z)
        offsets = tuple(int(np.dot(lv, eZ_ref) % 2) for lv in self.lX_vecs)
        Sigma, edge_sum, z_counts = np.zeros(self.n, dtype=np.int8), np.zeros(self.n, dtype=np.float64), Counter()
        for _ in range(self.n_burnin):
            self._worm_run(Sigma, self._adj_Z, self._valid_Z, self._n_vZ, log_wt)
        for _ in range(self.n_samples):
            self._worm_run(Sigma, self._adj_Z, self._valid_Z, self._n_vZ, log_wt)
            edge_sum += Sigma ^ eZ_ref
            z_bits = tuple(int((int(np.dot(lv, Sigma)) + offsets[i]) % 2) for i, lv in enumerate(self.lX_vecs))
            z_counts[z_bits] += 1
        alpha = edge_sum / self.n_samples
        return z_counts, alpha

    def _reweight_x_edges(self, alpha):
        p = self.p
        p_x_e = alpha / 2.0 + (1.0 - alpha) * (p / 3.0) / (1.0 - 2*p/3.0)
        p_x_e = np.clip(p_x_e, 1e-10, 1.0 - 1e-10)
        return np.log(p_x_e / (1.0 - p_x_e))

    def _run_x_chain(self, eX_ref, log_w_X):
        log_wt = np.where(eX_ref.astype(bool), -log_w_X, log_w_X)
        offsets = tuple(int(np.dot(lv, eX_ref) % 2) for lv in self.lZ_vecs)
        Sigma, x_counts = np.zeros(self.n, dtype=np.int8), Counter()
        for _ in range(self.n_burnin):
            self._worm_run(Sigma, self._adj_X, self._valid_X, self._n_vX, log_wt)
        for _ in range(self.n_samples):
            self._worm_run(Sigma, self._adj_X, self._valid_X, self._n_vX, log_wt)
            x_bits = tuple(int((int(np.dot(lv, Sigma)) + offsets[i]) % 2) for i, lv in enumerate(self.lZ_vecs))
            x_counts[x_bits] += 1
        return x_counts

    def _apply_correction(self, eX_ref, eZ_ref, best_x_bits, best_z_bits):
        ref_x_bits = tuple(int(np.dot(lv, eX_ref) % 2) for lv in self.lZ_vecs)
        ref_z_bits = tuple(int(np.dot(lv, eZ_ref) % 2) for lv in self.lX_vecs)
        eX, eZ = eX_ref.copy(), eZ_ref.copy()
        for i in range(self.k):
            if best_x_bits[i] != ref_x_bits[i]: eX ^= self.lX_vecs[i]
            if best_z_bits[i] != ref_z_bits[i]: eZ ^= self.lZ_vecs[i]
        return eX, eZ

    def decode(self, syndZ, syndX, init_method='MWPM'):
        eX_ref, eZ_ref = self._init_refs(syndZ, syndX, init_method)
        z_counts, alpha = self._run_z_chain(eZ_ref)
        log_w_X = self._reweight_x_edges(alpha)
        x_counts = self._run_x_chain(eX_ref, log_w_X)
        return self._apply_correction(eX_ref, eZ_ref, max(x_counts, key=x_counts.get), max(z_counts, key=z_counts.get))

    def _init_refs(self, syndZ, syndX, init_method):
        if init_method == 'MWPM':
            eX = utils.mwpm_initialize_e_given_syndrome(self.HZ, syndZ).astype(np.int8)
            eZ = utils.mwpm_initialize_e_given_syndrome(self.HX, syndX).astype(np.int8)
        else:
            eX = utils.ge_initialize_given_syndrome(self.HZ, syndZ).astype(np.int8)
            eZ = utils.ge_initialize_given_syndrome(self.HX, syndX).astype(np.int8)
        return eX, eZ

class GEDecoder(Decoder):
    def __init__(self, code):
        self.code = code
        self.HZ, self.HX = code.stabilizer_matrices()

    def decode(self, syndZ, syndX):
        eX_hat = utils.ge_initialize_given_syndrome(self.HZ, syndZ)
        eZ_hat = utils.ge_initialize_given_syndrome(self.HX, syndX)
        return eX_hat, eZ_hat

class BPDecoder(Decoder):
    def __init__(self, code, p, max_iter=100, bp_method="product_sum"):
        """
        Initializes the Belief Propagation Decoder.

        Args:
            code: The quantum code (e.g., ToricCode, PlanarSurfaceCode) instance.
            p: The physical error rate for depolarizing noise.
            max_iter: Maximum number of iterations for the BP algorithm.
            bp_method: The BP decoding method (e.g., "product_sum", "min_sum").
        """
        self.code = code
        self.p = float(p)
        self.max_iter = max_iter
        self.bp_method = bp_method

        self.HZ, self.HX = code.stabilizer_matrices()

        # For X errors, HZ is the parity check matrix. The error rate for an X error is p/3.
        self.bp_decoder_X = ldpc.bp_decoder(self.HZ, error_rate=self.p / 3, max_iter=self.max_iter, bp_method=self.bp_method)
        # For Z errors, HX is the parity check matrix. The error rate for a Z error is p/3.
        self.bp_decoder_Z = ldpc.bp_decoder(self.HX, error_rate=self.p / 3, max_iter=self.max_iter, bp_method=self.bp_method)

    def decode(self, syndZ, syndX):
        eX_hat = self.bp_decoder_X.decode(syndZ)
        eZ_hat = self.bp_decoder_Z.decode(syndX)
        return eX_hat, eZ_hat