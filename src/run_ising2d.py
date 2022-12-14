import os

import click

from trainer import Ising2DTrainer
from utils import plot_sorted_energy


@click.command()
@click.option("--output_dir", required=True)
@click.option("--j", default=1)
@click.option("--lx", default=10)
@click.option("--ly", default=10)
@click.option("--lr", default=0.001)
@click.option("--train_samples", default=100)
@click.option("--eval_samples", default=100)
@click.option("--test_samples", default=100)
@click.option("--epochs", default=3)
@click.option("--batch_size", default=1)
@click.option("--optimizer_name", default="SGD")
def run(
    output_dir,
    j,
    lx,
    ly,
    lr,
    train_samples,
    eval_samples,
    test_samples,
    epochs,
    batch_size,
    optimizer_name,
):
    trainer = Ising2DTrainer(j, lx, ly)

    E_original, E_eff = trainer(
        output_dir=output_dir,
        lr=lr,
        train_samples=train_samples,
        eval_samples=eval_samples,
        test_samples=test_samples,
        epochs=epochs,
        batch_size=batch_size,
        optimizer_name=optimizer_name,
    )
    plot_sorted_energy(
        E_original,
        E_eff,
        save_path=os.path.join(output_dir, "sorted_energy.png"),
    )


if __name__ == "__main__":
    run()
