import os
from typing import Callable

import click
import matplotlib.pyplot as plt
import torch
from torch.nn import MSELoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from hamiltonians import ising2d, mean_field


class Ising2D:
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

    def create_dataloader(self, n_samples: int, batch_size: int) -> DataLoader:
        """create dataloader given samples and batch size

        Args:
            n_samples (int): number of data to sample.
            batch_size (int): batch size

        Returns:
            DataLoader: dataloader for training or evaluation
        """
        X = 1 - 2 * (torch.rand(n_samples, self.Lx, self.Ly) < 0.5)
        X = X.float()
        Y = torch.FloatTensor([ising2d(x, self.J) for x in X])
        ds = TensorDataset(X, Y)
        dataloader = DataLoader(ds, batch_size=batch_size)
        return dataloader

    def plot_res(self, h, test_dataloader):
        Y_eff = []
        Y_original = []
        for X, Y in test_dataloader:
            H_eff = mean_field(X, h)
            Y_eff.append(H_eff)
            Y_original.append(Y)
        X = [i for i in range(len(test_dataloader))]
        plt.plot(X, Y_eff, label="H_eff")
        plt.plot(X, Y_original, label="H_original")
        plt.legend()
        plt.show()

    def train_loop(
        self,
        dataloader: DataLoader,
        h: torch.Tensor,
        loss_fn: Callable,
        optimizer: Optimizer,
    ) -> float:
        """training loop

        Args:
            dataloader (DataLoader): dataloader
            h (torch.Tensor): coefficients for each spin. (batch size, lattice size x, lattice size y)
            loss_fn (Callable): loss function
            optimizer (Optimizer): optimizer

        Returns:
            float: final loss value
        """
        for X, Y in tqdm(dataloader, leave=False):
            H_eff = mean_field(X, h)
            loss = loss_fn(Y, H_eff)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss

    def __call__(
        self,
        lr: float = 0.001,
        train_samples: int = 100,
        test_samples: int = 100,
        epochs: int = 3,
        batch_size: int = 1,
        optimizer_name: str = "SGD",
        output_dir: str = None,
    ) -> None:
        """run optimization

        Args:
            lr (float, optional): learning rate. Defaults to 0.001.
            train_samples (int, optional): number of training samples. Defaults to 100.
            test_samples (int, optional): number of test samples. Defaults to 100.
            epochs (int, optional): number of training epochs. Defaults to 3.
            batch_size (int, optional): batch size for both training & evaluation. Defaults to 1.
            optimizer_name (str, optional): optimizer name. "SGD" & "Adam" can be used. Defaults to "SGD".
            output_dir (str, optional): output directory. should be specified when saving parameters. Defaults to None.
        """
        train_dataloader = self.create_dataloader(train_samples, batch_size)
        test_dataloader = self.create_dataloader(test_samples, batch_size)

        h = torch.rand((self.Lx, self.Ly), dtype=torch.float32)
        h = h.unsqueeze(0).repeat(batch_size, 1, 1)
        h.requires_grad_()
        loss_fn = MSELoss()

        if optimizer_name == "SGD":
            optimizer = torch.optim.SGD([h], lr=lr)
        elif optimizer_name == "Adam":
            optimizer = torch.optim.Adam([h], lr=lr)
        else:
            raise NotImplementedError

        with tqdm(range(epochs)) as pbar_epoch:
            for epoch in range(epochs):
                pbar_epoch.set_description("[Epoch %d]" % (epoch + 1))
                loss = self.train_loop(train_dataloader, h, loss_fn, optimizer)
                tqdm.write(f"loss at epoch{epoch+1}: {loss}")

        with torch.no_grad():
            self.plot_res(h, test_dataloader)

        if not output_dir:
            return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(h, os.path.join(output_dir, "model.pth"))


@click.command()
@click.option("--j", default=1)
@click.option("--lx", default=10)
@click.option("--ly", default=10)
@click.option("--lr", default=0.001)
@click.option("--train_samples", default=100)
@click.option("--test_samples", default=100)
@click.option("--epochs", default=3)
@click.option("--batch_size", default=1)
@click.option("--optimizer_name", default="SGD")
@click.option("--output_dir", default=None)
def main(
    j,
    lx,
    ly,
    lr,
    train_samples,
    test_samples,
    epochs,
    batch_size,
    optimizer_name,
    output_dir,
):
    ising = Ising2D(j, lx, ly)
    ising(
        lr=lr,
        train_samples=train_samples,
        test_samples=test_samples,
        epochs=epochs,
        batch_size=batch_size,
        optimizer_name=optimizer_name,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
