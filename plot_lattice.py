import matplotlib.pyplot as plt
import numpy as np

class LatticePlotter:
    def __init__(self, toric_code, noise_model, syndromes=None):
        self.toric_code = toric_code
        self.noise_model = noise_model
        self.syndromes = syndromes  # (syndX, syndZ) tuple or None

    @property
    def is_planar(self):
        return self.toric_code.__class__.__name__ == 'PlanarSurfaceCode'

    def _is_valid_edge(self, edge_type, x, y):
        if not self.is_planar:
            return True
        L = self.toric_code.L
        if edge_type == 'hori':
            return True
        else:
            return (x >= 1) and (y < L - 1)

    def _get_primal_edge_coords(self, edge_type, x, y):
        """Get the coordinates of a primal edge given its type and position."""
        if edge_type == 'hori':  # Horizontal edge
            return (x, y), (x + 1, y)
        else:  # Vertical edge
            return (x, y), (x, y + 1)

    def _get_dual_edge_coords(self, edge_type, x, y):
        """
        Get the coordinates of a dual edge. A dual edge crosses a primal edge.
        'hori' refers to a horizontal DUAL edge, which crosses a vertical PRIMAL edge at (x,y).
        'vert' refers to a vertical DUAL edge, which crosses a horizontal PRIMAL edge at (x,y).
        """
        if edge_type == 'hori':
            # Connects dual vertices at centers of faces (x-1, y) and (x, y).
            start = (x - 0.5, y + 0.5)
            end = (x + 0.5, y + 0.5)
            return start, end
        else:  # 'vert'
            # Connects dual vertices at centers of faces (x, y-1) and (x, y).
            start = (x + 0.5, y - 0.5)
            end = (x + 0.5, y + 0.5)
            return start, end

    def plot_corrections(self, eX_hat, eZ_hat, ax, labels_added):
        """Plots the correction chains on the lattice."""
        L = self.toric_code.L

        # Plot X corrections (yellow) on DUAL lattice (Ladder)
        for y in range(L):
            for x in range(L):
                # X correction on horizontal qubit -> dual vertical edge
                if self._is_valid_edge('hori', x, y):
                    hori_idx = self.toric_code._edge_index_hori(x, y)
                    if eX_hat[hori_idx]:
                        start, end = self._get_dual_edge_coords('vert', x, y)
                        label = 'X correction' if 'X correction' not in labels_added else ''
                        ax.plot([start[0], end[0]], [start[1], end[1]], '--', color='yellow', linewidth=2.5, label=label)
                        if label: labels_added.add('X correction')

                # X correction on vertical qubit -> dual horizontal edge
                if self._is_valid_edge('vert', x, y):
                    vert_idx = self.toric_code._edge_index_vert(x, y)
                    if eX_hat[vert_idx]:
                        start, end = self._get_dual_edge_coords('hori', x, y)
                        label = 'X correction' if 'X correction' not in labels_added else ''
                        ax.plot([start[0], end[0]], [start[1], end[1]], '--', color='yellow', linewidth=2.5, label=label)
                        if label: labels_added.add('X correction')

        # Plot Z corrections (cyan) on PRIMAL lattice
        for y in range(L):
            for x in range(L):
                # Z correction on horizontal qubit -> primal horizontal edge
                if self._is_valid_edge('hori', x, y):
                    hori_idx = self.toric_code._edge_index_hori(x, y)
                    if eZ_hat[hori_idx]:
                        start, end = self._get_primal_edge_coords('hori', x, y)
                        label = 'Z correction' if 'Z correction' not in labels_added else ''
                        ax.plot([start[0], end[0]], [start[1], end[1]], '--', color='cyan', linewidth=2.5, label=label)
                        if label: labels_added.add('Z correction')

                # Z correction on vertical qubit -> primal vertical edge
                if self._is_valid_edge('vert', x, y):
                    vert_idx = self.toric_code._edge_index_vert(x, y)
                    if eZ_hat[vert_idx]:
                        start, end = self._get_primal_edge_coords('vert', x, y)
                        label = 'Z correction' if 'Z correction' not in labels_added else ''
                        ax.plot([start[0], end[0]], [start[1], end[1]], '--', color='cyan', linewidth=2.5, label=label)
                        if label: labels_added.add('Z correction')

    def plot(self, corrections=None):
        fig, ax = plt.subplots(figsize=(10, 10))
        L = self.toric_code.L
        # Adjust limits to see dual lattice edges at boundary
        ax.set_xlim(-1, L + 1)
        ax.set_ylim(-1, L + 1)
        ax.set_aspect('equal')
        ax.set_title('Toric Code Lattice with Errors and Syndromes')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_xticks(np.arange(0, L, 1))
        ax.set_yticks(np.arange(0, L, 1))
        ax.grid(True, alpha=0.2)

        eX, eZ = self.noise_model
        
        # --- Plot Lattices ---
        # 1. Plot primal lattice (thicker, black)
        for y in range(L):
            for x in range(L):
                for edge_type in ['hori', 'vert']:
                    if self._is_valid_edge(edge_type, x, y):
                        start, end = self._get_primal_edge_coords(edge_type, x, y)
                        ax.plot([start[0], end[0]], [start[1], end[1]], 'k-', linewidth=2)

        # 2. Plot dual lattice (thick gray)
        for y in range(L):
            for x in range(L):
                for edge_type in ['hori', 'vert']:
                    if self._is_valid_edge(edge_type, x, y):
                        dual_type = 'vert' if edge_type == 'hori' else 'hori'
                        start, end = self._get_dual_edge_coords(dual_type, x, y)
                        ax.plot([start[0], end[0]], [start[1], end[1]], '-', color='gray', linewidth=2, alpha=0.7)

        # --- Plot Errors ---
        # Use sets to add legend labels only once
        labels_added = set()

        # Highlight X errors on DUAL lattice (blue) - Ladders
        for y in range(L):
            for x in range(L):
                # X error on horizontal qubit -> dual vertical edge
                if self._is_valid_edge('hori', x, y):
                    hori_idx = self.toric_code._edge_index_hori(x, y)
                    if eX[hori_idx]:
                        start, end = self._get_dual_edge_coords('vert', x, y)
                        label = 'X error' if 'X error' not in labels_added else ''
                        ax.plot([start[0], end[0]], [start[1], end[1]], 'b-', linewidth=3, label=label)
                        if label: labels_added.add('X error')

                # X error on vertical qubit -> dual horizontal edge
                if self._is_valid_edge('vert', x, y):
                    vert_idx = self.toric_code._edge_index_vert(x, y)
                    if eX[vert_idx]:
                        start, end = self._get_dual_edge_coords('hori', x, y)
                        label = 'X error' if 'X error' not in labels_added else ''
                        ax.plot([start[0], end[0]], [start[1], end[1]], 'b-', linewidth=3, label=label)
                        if label: labels_added.add('X error')

        # 3. Highlight Z error chains on PRIMAL lattice (green)
        for y in range(L):
            for x in range(L):
                # Z error on horizontal qubit -> primal horizontal edge
                if self._is_valid_edge('hori', x, y):
                    hori_idx = self.toric_code._edge_index_hori(x, y)
                    if eZ[hori_idx]:
                        start, end = self._get_primal_edge_coords('hori', x, y)
                        label = 'Z error' if 'Z error' not in labels_added else ''
                        ax.plot([start[0], end[0]], [start[1], end[1]], 'g-', linewidth=3, label=label)
                        if label: labels_added.add('Z error')

                # Z error on vertical qubit -> primal vertical edge
                if self._is_valid_edge('vert', x, y):
                    vert_idx = self.toric_code._edge_index_vert(x, y)
                    if eZ[vert_idx]:
                        start, end = self._get_primal_edge_coords('vert', x, y)
                        label = 'Z error' if 'Z error' not in labels_added else ''
                        ax.plot([start[0], end[0]], [start[1], end[1]], 'g-', linewidth=3, label=label)
                        if label: labels_added.add('Z error')

        # 4. Plot syndrome defects
        if self.syndromes:
            syndX, syndZ = self.syndromes

            # Z stabilizers (Faces) - defects in red on DUAL vertices
            for i, syndrome in enumerate(syndZ):
                if syndrome:
                    y = i // L
                    x = i % L
                    label = 'Z-stabilizer defect' if 'Z-stab' not in labels_added else ''
                    ax.plot(x + 0.5, y + 0.5, 'r*', markersize=15, label=label)
                    if label: labels_added.add('Z-stab')

            # X stabilizers (Stars) - defects in orange on PRIMAL vertices
            for i, syndrome in enumerate(syndX):
                if syndrome:
                    if self.is_planar:
                        y = i // (L - 1)
                        x = (i % (L - 1)) + 1
                    else:
                        y = i // L
                        x = i % L
                    label = 'X-stabilizer defect' if 'X-stab' not in labels_added else ''
                    ax.plot(x, y, 's', color='orange', markersize=10, label=label)
                    if label: labels_added.add('X-stab')

        if corrections:
            eX_hat, eZ_hat = corrections
            self.plot_corrections(eX_hat, eZ_hat, ax, labels_added)

        # Create a legend if there's anything to show
        if labels_added:
            ax.legend()

        plt.show()