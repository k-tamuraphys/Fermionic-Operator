from typing import List

from scipy.sparse import csr_matrix, kron, lil_matrix

identity_matrix = csr_matrix(
    [[1.0, 0.0], [0.0, 1.0]],
)

pauli_x = csr_matrix([[0.0, 1.0], [1.0, 0.0]])

pauli_y = csr_matrix([[0.0, -1.0j], [1.0j, 0.0]])

pauli_z = csr_matrix([[1.0, 0.0], [0.0, -1.0]])

raising_op = (1 / 2) * (pauli_x + 1.0j * pauli_y)
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

    def cop(self, index: int) -> "csr_matrix":
        """Return annihilation operator with label `index`

        Args:
            index (int): Index for the mode
            format (str): format of a sparse matrix

        Raises:
            ValueError: the index `index` have to be between 0 and n_modes

        Returns:
            csr_matrix: matrix representation of the annihilation operator
        """
        if not 0 <= index <= self._n_modes - 1:
            raise ValueError(f"`index` have to be between 0 and {self._n_modes}.")
        format = "csr"
        cop_matrix = 1.0
        for _ in range(index):
            cop_matrix = kron(cop_matrix, -pauli_z, format=format)
        cop_matrix = kron(cop_matrix, lowering_op, format=format)
        for _ in range(index + 1, self._n_modes):
            cop_matrix = kron(cop_matrix, identity_matrix, format=format)
        return cop_matrix

    def cdg(self, index: int) -> "csr_matrix":
        """Return creation operator with label `index`

        Args:
            index (int): Index for the mode

        Returns:
            csr_matrix: matrix representation of the creation operator
        """
        return self.cop(index).transpose()

    def nop(self, index: int) -> "csr_matrix":
        """Return the number operator with label `index`

        Args:
            index (int): Index for the mode

        Returns:
            csr_matrix: the number operator
        """
        return self.cdg(index) @ self.cop(index)

    def vacuum_state(self) -> "lil_matrix":
        """Return the vacuum state annihilated by all the cop

        Returns:
            lil_matrix: vacuum state as lil_matrix object
        """
        vacuum = lil_matrix((1, self._dim))
        vacuum[-1] = 1.0
        return vacuum

    def cop_list(self) -> List[csr_matrix]:
        """Return the list of c operators

        Returns:
            List[csr_matrix]: The list of c operators
        """
        return [self.cop(i) for i in range(self._n_modes)]

    def cdg_list(self) -> List[csr_matrix]:
        """Return the list of c dagger operators

        Returns:
            List[csr_matrix]: The list of c dagger operators
        """
        return [self.cdg(i) for i in range(self._n_modes)]

    def nop_list(self) -> List[csr_matrix]:
        """Return the list of n operators

        Returns:
            List[csr_matrix]: The list of n operators
        """
        return [self.nop(i) for i in range(self._n_modes)]
