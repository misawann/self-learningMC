import torch


def ising2d(S, J):
    Lx, Ly = S.shape
    H = 0
    for i in range(Lx):
        for j in range(Ly):
            prod = prod_neighbor(S, i, j, Lx, Ly)
            H += prod
    H = J / 2 * H
    return H


def prod_neighbor(S, i, j, Lx, Ly):
    return S[i, j] * (
        S[(i + 1) % Lx, j]
        + S[(i - 1) % Lx, j]
        + S[i, (j + 1) % Ly]
        + S[i, (j - 1) % Ly]
    )


def mean_field(S, h):
    H = torch.einsum("bxy,bxy->b", S, h)
    return H
