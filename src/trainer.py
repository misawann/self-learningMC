import os
from typing import Callable, List, Tuple, Union

import torch
from torch.nn import MSELoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class BaseTrainer:
    def __init__(self) -> None:
        """set basic constants of the system"""
        pass

    def effective_model(self, X: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """effective hamiltonian

        Args:
            X (torch.Tensor): input data.
            params (torch.Tensor): parameters.

        Returns:
            torch.Tensor: energy of effective hamiltonian. (number of data)
        """
        raise NotImplementedError("This method should be overridden by derived class.")

    def original_model(self, X: torch.Tensor) -> torch.Tensor:
        """original hamiltonian

        Args:
            X (torch.Tensor): input data.

        Returns:
            torch.Tensor: energy of original hamiltonian. (number of data)
        """
        raise NotImplementedError("This method should be overridden by derived class.")

    def sample_input(self, n_samples: int) -> torch.Tensor:
        """sample input data

        Args:
            n_samples (int): number of samples.

        Returns:
            torch.Tensor: input data.
        """
        raise NotImplementedError("This method should be overridden by derived class.")

    def init_params(self) -> torch.Tensor:
        """initialize parameters

        Returns:
            torch.Tensor: parameters.
        """
        raise NotImplementedError("This method should be overridden by derived class.")

    def create_dataloader(self, n_samples: int, batch_size: int) -> DataLoader:
        """create dataloader given samples and batch size

        Args:
            n_samples (int): number of data to sample.
            batch_size (int): batch size

        Returns:
            DataLoader: dataloader for training or evaluation
        """
        X = self.sample_input(n_samples)
        Y = torch.FloatTensor(self.original_model(X))
        ds = TensorDataset(X, Y)
        dataloader = DataLoader(ds, batch_size=batch_size)
        return dataloader

    def loop(
        self,
        params: torch.Tensor,
        dataloader: DataLoader,
        loss_fn: Callable,
        optimizer: Optimizer,
        normalize_const: float = 1.0,
        mode: str = "train",
    ) -> Union[float, Tuple[float, List[float], List[float]]]:
        """loop for training, evaluation and testing

        Args:
            params (torch.Tensor): params
            dataloader (DataLoader): dataloader
            loss_fn (Callable): loss function
            optimizer (Optimizer): optimizer
            normalize_const (float, optional): normalize constant for loss. Defaults to 1.0.
            mode (str, optional): "train" or "eval". Defaults to "train".

        Returns:
            Union[float, Tuple[float, List[float], List[float]]]
            loss when mode is "train" or "eval"
            loss, original and effective hamiltonian when mode is "test"
        """

        loss_sum = 0.0
        E_original = []
        E_eff = []
        for X, Y in tqdm(dataloader, leave=False):
            Y_eff = self.effective_model(X, params)
            loss = loss_fn(Y / normalize_const, Y_eff / normalize_const)
            if mode == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            elif mode == "test":
                E_eff += Y_eff.detach().tolist()
                E_original += Y.detach().tolist()
            loss_sum += loss.detach().item()
        loss = loss_sum / len(dataloader)
        if mode == "train" or mode == "eval":
            return loss
        elif mode == "test":
            return loss, E_original, E_eff

    def __call__(
        self,
        output_dir: str,
        lr: float = 0.001,
        train_samples: int = 100,
        eval_samples: int = 100,
        test_samples: int = 100,
        epochs: int = 3,
        batch_size: int = 1,
        optimizer_name: str = "Adam",
        normalize_const: float = 1.0,
        save_model=False,
    ) -> Tuple[List[float], List[float]]:
        """run optimization

        Args:
            output_dir (str): output directory.
            lr (float, optional): learning rate. Defaults to 0.001.
            train_samples (int, optional): number of training samples. Defaults to 100.
            eval_samples (int, optional): number of evaluation samples. Defaults to 100.
            test_samples (int, optional): number of test samples. Defaults to 100.
            epochs (int, optional): number of training epochs. Defaults to 3.
            batch_size (int, optional): batch size. Defaults to 1.
            normalize_const (float, optional): normalize constant for loss. Defaults to 1.0.
            optimizer_name (str, optional): optimizer name. "SGD" & "Adam" can be used. Defaults to "SGD".

        Returns:
            Tuple[List[float], List[float]]: energies of original and effective hamiltonian
        """
        train_dataloader = self.create_dataloader(train_samples, batch_size)
        eval_dataloader = self.create_dataloader(eval_samples, 1)
        test_dataloader = self.create_dataloader(test_samples, 1)

        params = self.init_params()
        loss_fn = MSELoss()

        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam([params], lr=lr)
        elif optimizer_name == "SGD":
            optimizer = torch.optim.SGD([params], lr=lr)
        else:
            raise NotImplementedError

        writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))

        with tqdm(range(epochs)) as pbar_epoch:
            for epoch in range(epochs):
                pbar_epoch.set_description("[Epoch %d]" % (epoch + 1))
                loss = self.loop(
                    params,
                    train_dataloader,
                    loss_fn,
                    optimizer,
                    normalize_const,
                    mode="train",
                )
                writer.add_scalar("train loss", loss, epoch)
                tqdm.write(f"train loss at epoch{epoch+1}: {loss}")

                with torch.no_grad():
                    loss = self.loop(
                        params,
                        eval_dataloader,
                        loss_fn,
                        optimizer,
                        normalize_const,
                        mode="eval",
                    )
                    writer.add_scalar("eval loss", loss, epoch)
                    tqdm.write(f"eval loss at epoch{epoch+1}: {loss}")

        with torch.no_grad():
            loss, E_original, E_eff = self.loop(
                params,
                test_dataloader,
                loss_fn,
                optimizer,
                normalize_const,
                mode="test",
            )
            tqdm.write(f"test loss: {loss}")

        if save_model:
            torch.save(params, os.path.join(output_dir, "model.pth"))

        return E_original, E_eff


class Ising2DTrainer(BaseTrainer):
    def __init__(self, J: float, Lx: int, Ly: int) -> None:
        """set basic constants of the system

        Args:
            J (float): coefficient of spin interaction.
            Lx (int): lattice size of dimension x.
            Ly (int): lattice size of dimension x.
        """
        self.J = J
        self.Lx = Lx
        self.Ly = Ly

    def effective_model(self, X: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """mean field Hamiltonian of 2D Ising model

        Args:
            X (torch.Tensor): spin. (number of data, lattice size x, lattice size y)
            params (torch.Tensor): coefficients for each spin. (lattice size x, lattice size y)

        Returns:
            torch.Tensor: energy of mean field Hamiltonian. (number of data)
        """
        H = -torch.einsum("bxy,xy->b", X, params)
        return H

    def original_model(self, X: torch.Tensor) -> torch.Tensor:
        """original Hamiltonian of 2D Ising model

        Args:
            X (torch.Tensor): spin. (number of data, lattice size x, lattice size y)

        Returns:
            torch.Tensor: energy of original Hamiltonian. (number of data)
        """
        n_data, Lx, Ly = X.shape
        H = torch.zeros(n_data)
        for i in range(Lx):
            for j in range(Ly):
                interact = self.neighbor_interact(X, i, j, Lx, Ly)
                H += -self.J * interact
        H /= 2  # 2 is for double counting
        return H

    def neighbor_interact(
        self, X: torch.Tensor, i: int, j: int, Lx: int, Ly: int
    ) -> torch.Tensor:
        """calculate interaction with neighbors

        Args:
            X (torch.Tensor): spin. (number of data, lattice size x, lattice size y)
            i (int): index of position x.
            j (int): index of position y.
            Lx (int): lattice size of dimension x.
            Ly (int): lattice size of dimension y.

        Returns:
            torch.Tensor: interaction with neighbors. (number of data)
        """
        return X[:, i, j] * (
            X[:, (i + 1) % Lx, j]
            + X[:, (i - 1) % Lx, j]
            + X[:, i, (j + 1) % Ly]
            + X[:, i, (j - 1) % Ly]
        )

    def sample_input(self, n_samples: int) -> torch.Tensor:
        X = 1 - 2 * (torch.rand(n_samples, self.Lx, self.Ly) < 0.5)
        return X.float()

    def init_params(self) -> torch.Tensor:
        h = torch.ones((self.Lx, self.Ly), dtype=torch.float32, requires_grad=True)
        return h


class Spin4InteractionTrainer(BaseTrainer):
    def __init__(self, J: float, K: float, Lx: int, Ly: int) -> None:
        """set basic constants of the system

        Args:
            J (float): coefficient of 2 spin interaction.
            K (float): coefficient of 4 spin interaction.
            Lx (int): lattice size of dimension x.
            Ly (int): lattice size of dimension x.
        """
        self.J = J
        self.K = K
        self.Lx = Lx
        self.Ly = Ly

    def effective_model(self, X: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """effective model

        Args:
            X (torch.Tensor): spin. (number of data, lattice size x, lattice size y)
            params (torch.Tensor): parameters. (number of parameters)

        Returns:
            torch.Tensor: energy of effective Hamiltonian. (number of data)
        """
        batch_size = X.shape[0]
        max_n = params.shape[0]
        interact = torch.zeros((batch_size, max_n - 1))
        for i in range(self.Lx):
            for j in range(self.Ly):
                interact -= torch.stack(
                    [
                        self.neighbor_interact(n, X, i, j, self.Lx, self.Ly)
                        for n in range(1, max_n)
                    ],
                    dim=1,
                )
        interact /= 2  # 2 is for double counting
        interact = torch.concat((torch.ones((batch_size, 1)), interact), dim=1)
        H = torch.einsum("bx,x->b", interact, params)
        return H

    def original_model(self, X: torch.Tensor) -> torch.Tensor:
        """original Hamiltonian

        Args:
            X (torch.Tensor): spin. (number of data, lattice size x, lattice size y)

        Returns:
            torch.Tensor: energy of original Hamiltonian. (number of data)
        """
        n_data, Lx, Ly = X.shape
        H = torch.zeros(n_data)
        for i in range(Lx):
            for j in range(Ly):
                H += (
                    -self.J * self.neighbor_interact(1, X, i, j, self.Lx, self.Ly) / 2
                    - self.K * self.interact_cell(X, i, j, self.Lx, self.Ly) / 4
                )
        return H

    def neighbor_interact(
        self, n: int, X: torch.Tensor, i: int, j: int, Lx: int, Ly: int
    ) -> torch.Tensor:
        """calculate interaction with neighbors

        Args:
            n (int): n th neighbor
            X (torch.Tensor): spin. (number of data, lattice size x, lattice size y)
            i (int): index of position x.
            j (int): index of position y.
            Lx (int): lattice size of dimension x.
            Ly (int): lattice size of dimension y.

        Returns:
            torch.Tensor: interaction with neighbors. (number of data)
        """

        if n != 2:
            return X[:, i, j] * (
                X[:, (i + n) % Lx, j]
                + X[:, (i - n) % Lx, j]
                + X[:, i, (j + n) % Ly]
                + X[:, i, (j - n) % Ly]
            )
        elif n == 2:
            return X[:, i, j] * (
                X[:, (i + 1) % Lx, (j + 1) % Ly]
                + X[:, (i + 1) % Lx, (j - 1) % Ly]
                + X[:, (i - 1) % Lx, (j + 1) % Ly]
                + X[:, (i - 1) % Lx, (j - 1) % Ly]
            )
        else:
            NotImplementedError

    def interact_cell(
        self, X: torch.Tensor, i: int, j: int, Lx: int, Ly: int
    ) -> torch.Tensor:
        """calculate interaction in a cell

        Args:
            X (torch.Tensor): spin. (number of data, lattice size x, lattice size y)
            i (int): index of position x.
            j (int): index of position y.
            Lx (int): lattice size of dimension x.
            Ly (int): lattice size of dimension y.

        Returns:
            torch.Tensor: interaction with neighbors. (number of data)
        """
        return X[:, i, j] * (
            X[:, (i + 1) % Lx, j]
            * X[:, i, (j + 1) % Ly]
            * X[:, (i + 1) % Lx, (j + 1) % Ly]
            + X[:, (i - 1) % Lx, j]
            * X[:, i, (j + 1) % Ly]
            * X[:, (i - 1) % Lx, (j + 1) % Ly]
            + X[:, (i + 1) % Lx, j]
            * X[:, i, (j - 1) % Ly]
            * X[:, (i + 1) % Lx, (j - 1) % Ly]
            + X[:, (i - 1) % Lx, j]
            * X[:, i, (j - 1) % Ly]
            * X[:, (i - 1) % Lx, (j - 1) % Ly]
        )

    def sample_input(self, n_samples: int):
        X = 1 - 2 * (torch.rand(n_samples, self.Lx, self.Ly) < 0.5)
        return X.float()

    def init_params(self):
        h = torch.ones(4, dtype=torch.float32, requires_grad=True)
        return h
