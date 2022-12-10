import torch


def ising2d(S: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    """original Hamiltonian of 2D Ising model

    Args:
        S (torch.Tensor): spin. (lattice size x, lattice size y)
        J (torch.Tensor): coefficient of spin interaction.

    Returns:
        torch.Tensor: original Hamiltonian
    """
    Lx, Ly = S.shape
    H = 0
    for i in range(Lx):
        for j in range(Ly):
            prod = neighbor_interact(S, i, j, Lx, Ly)
            H += prod
    H = J / 2 * H
    return H


def neighbor_interact(S: torch.Tensor, i: int, j: int, Lx: int, Ly: int) -> float:
    """calculate interaction with neighbors

    Args:
        S (torch.Tensor): spin. (lattice size x, lattice size y)
        i (int): index of position x.
        j (int): index of position y.
        Lx (int): lattice size of dimension x.
        Ly (int): lattice size of dimension y.

    Returns:
        float: interaction with neighbors.
    """
    return S[i, j] * (
        S[(i + 1) % Lx, j]
        + S[(i - 1) % Lx, j]
        + S[i, (j + 1) % Ly]
        + S[i, (j - 1) % Ly]
    )


def mean_field(S: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """mean field Hamiltonian of 2D Ising model

    Args:
        S (torch.Tensor): spin. (batch size, lattice size x, lattice size y)
        h (torch.Tensor): coeffcients for each spin. (batch size, lattice size x, lattice size y)

    Returns:
        torch.Tensor: mean field Hamiltonian
    """
    H = torch.einsum("bxy,bxy->b", S, h)
    return H
