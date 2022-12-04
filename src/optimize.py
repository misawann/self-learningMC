import os

import matplotlib.pyplot as plt
import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from hamiltonians import ising2d, mean_field
import click


class Ising2D:
    def __init__(self, J, Lx, Ly):
        self.J = J
        self.Lx = Lx
        self.Ly = Ly

    def create_dataloader(self, n_samples, batch_size):
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

    def train_loop(self, dataloader, h, loss_fn, optimizer):
        for X, Y in tqdm(dataloader, leave=False):
            H_eff = mean_field(X, h)
            loss = loss_fn(Y, H_eff)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss

    def __call__(
        self,
        output_dir,
        lr=0.001,
        train_samples=100,
        test_samples=100,
        epochs=3,
        batch_size=1,
        optimizer_name="SGD",
        save_model=False,
    ):
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

        if not save_model:
            return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(h, os.path.join(output_dir, "model.pth"))


@click.command()
@click.option("--j", default=1)
@click.option("--lx", default=10)
@click.option("--ly", default=10)
@click.option("--output_dir", required=True)
@click.option("--lr", default=0.001)
@click.option("--train_samples", default=100)
@click.option("--test_samples", default=100)
@click.option("--epochs", default=3)
@click.option("--batch_size", default=1)
@click.option("--optimizer_name", default="SGD")
@click.option("--save_model", default=False)
def main(
    j,
    lx,
    ly,
    output_dir,
    lr,
    train_samples,
    test_samples,
    epochs,
    batch_size,
    optimizer_name,
    save_model,
):
    ising = Ising2D(j, lx, ly)
    ising(
        output_dir=output_dir,
        lr=lr,
        train_samples=train_samples,
        test_samples=test_samples,
        epochs=epochs,
        batch_size=batch_size,
        optimizer_name=optimizer_name,
        save_model=save_model,
    )


if __name__ == "__main__":
    main()
