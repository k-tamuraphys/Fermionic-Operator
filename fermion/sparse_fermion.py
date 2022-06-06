from scipy.sparse import csr_matrix, kron, spmatrix

identity_matrix = csr_matrix(
    [
        [1.0, 0.0],
        [0.0, 1.0]
    ],
)

pauli_x = csr_matrix(
    [
        [0.0, 1.0], 
        [1.0, 0.0]
    ]
)

pauli_y = csr_matrix(
    [
        [0.0, -1.0j],
        [1.0j, 0.0]
    ]
)

pauli_z = csr_matrix(
    [
        [1.0, 0.0],
        [0.0, -1.0]
    ]
)

raising_op = (1 / 2) * (pauli_x + 1.0j* pauli_y)
lowering_op = (1 / 2) * (pauli_x - 1.0j * pauli_y)

class SparseFermion:
    """Fermionic operators in sparse matrices."""

    def __init__(self, n_modes: int) -> None:
        """
        Args:
            n_modes (int): the number of modes
        """
        self._n_modes = n_modes
        self._dim = 2**n_modes

    def cop(self, index: int, format="csr") -> "spmatrix":
        """Return annihilation operator with label `index`

        Args:
            index (int): Index for the mode
            format (str): format of a sparse matrix

        Raises:
            ValueError: the index `index` have to be between 0 and n_modes

        Returns:
            spmatrix: matrix representation of the annihilation operator
        """
        if not 0 <= index <= self._n_modes - 1:
            raise ValueError(f"`index` have to be between 0 and {self._n_modes}.")
        cop_matrix = 1.0
        for _ in range(index):
            cop_matrix = kron(cop_matrix, -pauli_z, format=format)
        cop_matrix = kron(cop_matrix, lowering_op, format=format)
        for _ in range(index + 1, self._n_modes):
            cop_matrix = kron(cop_matrix, identity_matrix, format=format)
        return cop_matrix

    def cdg(self, index: int) -> "spmatrix":
        """Return creation operator with label `index`

        Args:
            index (int): Index for the mode

        Returns:
            spmatrix: matrix representation of the creation operator
        """
        return self.cop(index).transpose()

    def nop(self, index: int) -> "spmatrix":
        """Return the number operator with label `index`

        Args:
            index (int): Index for the mode

        Returns:
            spmatrix: the number operator
        """
        return self.cdg(index)@self.cop(index)
