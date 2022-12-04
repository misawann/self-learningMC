import matplotlib.pyplot as plt
import numpy as np
import tqdm


class MCIsing2D:
    def __init__(self, Lx: int, Ly: int, J: float, beta: float, h: float) -> None:
        self.Lx = Lx
        self.Ly = Ly
        self.J = J
        self.beta = beta
        self.h = h

    def transition(self, S, i, j) -> float:
        S_neighbor = (
            S[(i - 1) % self.Lx, j]
            + S[(i + 1) % self.Lx, j]
            + S[i, (j - 1) % self.Ly]
            + S[i, (j + 1) % self.Ly]
        )
        dH = 2 * self.J * S[i, j] * S_neighbor + 2 * self.h * S[i, j]
        W = np.exp(-self.beta * dH)
        return np.minimum(1.0, W)

    def magnetization(self, S) -> float:
        return np.abs(sum(map(sum, S)) / self.Lx / self.Ly)

    def run_iteration(self, S):
        i = np.random.randint(0, self.Lx)
        j = np.random.randint(0, self.Ly)
        T = self.transition(S, i, j)
        flip = 1 - 2 * (np.random.rand() < T)
        S[i][j] *= flip
        return S

    def mean_magnitization(self, n_thermal, n_mc, step_measure) -> float:

        total_steps = n_thermal + n_mc
        S = 1 - 2 * (np.random.rand(self.Lx, self.Ly) < 0.5)

        M = []
        for step in tqdm.tqdm(range(total_steps)):
            S = self.run_iteration(S)

            if step < n_thermal:
                continue

            if step % step_measure == 0:
                M.append(self.magnetization(S))
            if step % 10000 == 0:
                plt.clf()
                plt.imshow(S)
                plt.pause(1e-5)

        return np.mean(M)

    def mean_magnitization_sweep(self, n_thermal, n_mc, step_measure) -> float:

        total_steps = n_thermal + n_mc
        S = 1 - 2 * (np.random.rand(self.Lx, self.Ly) < 0.5)

        M = []
        for step in tqdm.tqdm(range(total_steps)):
            for i in range(self.Lx * self.Ly):
                S = self.run_iteration(S)

            if step < n_thermal:
                continue

            if step % step_measure / 10 == 0:
                M.append(self.magnetization(S))
            if step % 100 == 0:
                plt.clf()
                plt.imshow(S)
                plt.pause(1e-5)

        return np.mean(M)
