import matplotlib.pyplot as plt


def plot_sorted_energy(
    E_original,
    E_eff,
    n_bins=10,
    save_path=None,
):
    X = [i for i in range(len(E_original))]
    sorted_original, sorted_eff = zip(*sorted(zip(E_original, E_eff)))
    plt.plot(X, sorted_original, label="original")
    plt.plot(X, sorted_eff, label="effective")
    plt.xlabel("index")
    plt.ylabel("energy")
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    plt.clf()
    plt.close()


def plot_system_size_error(error, system_size, save_path=None):
    plt.plot(system_size, error)
    plt.xlabel("System Size")
    plt.ylabel("Error")
    if save_path is not None:
        plt.savefig(save_path)
    plt.clf()
    plt.close()
