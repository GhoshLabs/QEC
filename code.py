import numpy as np

class ToricCode:
    def __init__(self, L):
        self.L = L
        self.n = 2 * L * L
        self.hori_offset = 0
        self.vert_offset = L * L
        self.Z_stabilizers = self._build_Zs()
        self.X_stabilizers = self._build_Xs()

    def _edge_index_hori(self, x, y):
        return self.hori_offset + (y % self.L) * self.L + (x % self.L)

    def _edge_index_vert(self, x, y):
        return self.vert_offset + (y % self.L) * self.L + (x % self.L)

    def _build_Zs(self):
        Zs = []
        for y in range(self.L):
            for x in range(self.L):
                Zs.append([
                    self._edge_index_hori(x, y),
                    self._edge_index_vert(x, y),
                    self._edge_index_hori(x, y+1),
                    self._edge_index_vert(x+1, y),
                ])
        return Zs

    def _build_Xs(self):
        Xs = []
        for y in range(self.L):
            for x in range(self.L):
                Xs.append([
                    self._edge_index_hori(x, y),
                    self._edge_index_vert(x, y),
                    self._edge_index_hori(x-1, y),
                    self._edge_index_vert(x, y-1),
                ])
        return Xs

    def logical_Z_support(self):
        return [self._edge_index_vert(0, y) for y in range(self.L)]

    def logical_X_support(self):
        return [self._edge_index_hori(x, 0) for x in range(self.L)]
    
    def stabilizer_matrices(self):
        HZ = np.zeros((len(self.Z_stabilizers), self.n), dtype=int)
        HX = np.zeros((len(self.X_stabilizers), self.n), dtype=int)

        for i, stab in enumerate(self.Z_stabilizers):
            HZ[i, stab] = 1

        for i, stab in enumerate(self.X_stabilizers):
            HX[i, stab] = 1

        return HZ, HX

