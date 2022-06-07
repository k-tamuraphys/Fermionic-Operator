import numpy as np

identity_matrix = np.array([[1.0, 0.0], [0.0, 1.0]])

pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]])

pauli_y = np.array([[0.0, -1.0j], [1.0j, 0.0]])

pauli_z = np.array([[1.0, 0.0j], [0.0, -1.0]])

raising_op = (1 / 2) * (pauli_x + 1.0j * pauli_y)
lowering_op = (1 / 2) * (pauli_x - 1.0j * pauli_y)


class Fermion:
    """Fermionic operators."""

    def __init__(self, n_modes: int) -> None:
        """
        Args:
            n_modes (int): the number of modes
        """
        self._n_modes = n_modes
        self._dim = 2**n_modes

    def cop(self, i: int) -> "np.ndarray":
        """Return annihilation operator with label `i`

        Args:
            i (int): Index for the mode

        Raises:
            ValueError: the index `i` have to be between 0 and n_modes

        Returns:
            np.ndarray: matrix representation of the annihilation operator
        """
        if not 0 <= i <= self._n_modes - 1:
            raise ValueError(f"`i` have to be between 0 and {self._n_modes}.")
        cop_matrix = 1.0
        for _ in range(i):
            cop_matrix = np.kron(cop_matrix, -pauli_z)
        cop_matrix = np.kron(cop_matrix, lowering_op)
        for _ in range(i + 1, self._n_modes):
            cop_matrix = np.kron(cop_matrix, identity_matrix)
        return cop_matrix

    def cdg(self, i: int) -> "np.ndarray":
        """Return creation operator with label `i`

        Args:
            i (int): Index for the mode

        Returns:
            np.ndarray: matrix representation of the creation operator
        """
        return np.transpose(self.cop(i))

    def nop(self, i: int) -> "np.ndarray":
        """Return the number operator with label `i`

        Args:
            i (int): Index for the mode

        Returns:
            np.ndarray: the number operator
        """
        return np.matmul(self.cdg(i), self.cop(i))

    def vacuum_state(self) -> "np.ndarray":
        """Return the vacuum state annihilated by all the cop operators

        Returns:
            np.ndarray: vacuum state
        """
        vacuum = np.zeros(self._dim)
        vacuum[-1] = 1.0
        return vacuum
