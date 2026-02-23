import numpy as np

class ToricCode:
    def __init__(self, L):
        self.L = L
        self.n = 2 * L * L
        self.hori_offset = 0
        self.vert_offset = L * L
        self.X_stabilizers = self._build_stars()
        self.Z_stabilizers = self._build_faces()

    def _edge_index_hori(self, x, y):
        return self.hori_offset + (y % self.L) * self.L + (x % self.L)

    def _edge_index_vert(self, x, y):
        return self.vert_offset + (y % self.L) * self.L + (x % self.L)

    def _build_faces(self):
        # Faces (Plaquettes) -> Z-type stabilizers (detect X errors)
        faces = []
        for y in range(self.L):
            for x in range(self.L):
                faces.append([
                    self._edge_index_hori(x, y),
                    self._edge_index_vert(x, y),
                    self._edge_index_hori(x, y+1),
                    self._edge_index_vert(x+1, y),
                ])
        return faces

    def _build_stars(self):
        # Stars (Vertices) -> X-type stabilizers (detect Z errors)
        stars = []
        for y in range(self.L):
            for x in range(self.L):
                stars.append([
                    self._edge_index_hori(x, y),
                    self._edge_index_vert(x, y),
                    self._edge_index_hori(x-1, y),
                    self._edge_index_vert(x, y-1),
                ])
        return stars
    
    def logical_Z_support(self):
        # Z1: Primal loop along Y (Vertical edges at x=0)
        return [self._edge_index_vert(0, y) for y in range(self.L)]

    def logical_X_support(self):
        # X1: Dual loop along X (Ladder/Vertical edges at y=0)
        return [self._edge_index_vert(x, 0) for x in range(self.L)]
    
    def logical_X_conjugate(self):
        # X2: Dual loop along Y (Ladder/Horizontal edges at x=0)
        return [self._edge_index_hori(0, y) for y in range(self.L)]

    def logical_Z_conjugate(self):
        # Z2: Primal loop along X (Horizontal edges at y=0)
        return [self._edge_index_hori(x, 0) for x in range(self.L)]
    
    def stabilizer_matrices(self):
        HZ = np.zeros((len(self.Z_stabilizers), self.n), dtype=int)
        HX = np.zeros((len(self.X_stabilizers), self.n), dtype=int)

        for i, stab in enumerate(self.Z_stabilizers):
            HZ[i, stab] = 1

        for i, stab in enumerate(self.X_stabilizers):
            HX[i, stab] = 1

        return HZ, HX

class PlanarSurfaceCode:
    def __init__(self, L):
        self.L = L
        self.num_hori = L * L
        self.num_vert = (L - 1) * (L - 1)
        self.n = self.num_hori + self.num_vert
        self.hori_offset = 0
        self.vert_offset = self.num_hori
        self.X_stabilizers = self._build_stars()
        self.Z_stabilizers = self._build_faces()

    def _edge_index_hori(self, x, y):
        return self.hori_offset + y * self.L + x

    def _edge_index_vert(self, x, y):
        return self.vert_offset + y * (self.L - 1) + (x - 1)

    def _build_faces(self):
        # Faces (Plaquettes) -> Z-type stabilizers (detect X errors)
        faces = []
        for y in range(self.L - 1):
            for x in range(self.L):
                face = []
                face.append(self._edge_index_hori(x, y))
                face.append(self._edge_index_hori(x, y+1))
                if x >= 1:
                    face.append(self._edge_index_vert(x, y))
                if x < self.L - 1:
                    face.append(self._edge_index_vert(x+1, y))
                faces.append(face)
        return faces

    def _build_stars(self):
        # Stars (Vertices) -> X-type stabilizers (detect Z errors)
        stars = []
        for y in range(self.L):
            for x in range(1, self.L):
                star = []
                star.append(self._edge_index_hori(x, y))
                star.append(self._edge_index_hori(x-1, y))
                if y < self.L - 1:
                    star.append(self._edge_index_vert(x, y))
                if y > 0:
                    star.append(self._edge_index_vert(x, y-1))
                stars.append(star)
        return stars

    def logical_Z_support(self):
        return [self._edge_index_hori(x, 0) for x in range(self.L)]

    def logical_X_support(self):
        return [self._edge_index_hori(0, y) for y in range(self.L)]
    
    def logical_X_conjugate(self):
        return []

    def logical_Z_conjugate(self):
        return []
    
    def stabilizer_matrices(self):
        HZ = np.zeros((len(self.Z_stabilizers), self.n), dtype=int)
        HX = np.zeros((len(self.X_stabilizers), self.n), dtype=int)

        for i, stab in enumerate(self.Z_stabilizers):
            HZ[i, stab] = 1

        for i, stab in enumerate(self.X_stabilizers):
            HX[i, stab] = 1

        return HZ, HX