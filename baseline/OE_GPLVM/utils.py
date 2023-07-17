from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import MinMaxScaler
from matplotlib import ticker, cm
from baseline.OE_GPLVM.train import calculate_elbo


from dataclasses import dataclass


@dataclass
class Experiment:
    X_train: torch.tensor
    X_test: torch.tensor
    lb_train: torch.tensor
    lb_test: torch.tensor
    N: int
    data_dim: int
    batch_size: int
    latent_dim: int
    n_inducing: int
    n_epochs: int
    nn_layers: tuple
    lr: float
    elbo: str
    method: str


def results_plot(results: dict, save=False):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f'GPLVM - ELBO {results["elbo"]["type"]} - Method {results["elbo"]["method"]}'
    )
    for idx, result in enumerate(results.keys()):
        if result == "elbo":
            break
        value = 320 + 1 + idx
        plt.subplot(value)
        Y = results[result][0]
        idx_n = np.where(results[result][1] == 0)
        idx_a = np.where(results[result][1] == 1)
        plt.scatter(
            Y[:, 0][idx_n].detach().numpy(),
            Y[:, 1][idx_n].detach().numpy(),
            label="Normal",
        )
        plt.scatter(
            Y[:, 0][idx_a].detach().numpy(),
            Y[:, 1][idx_a].detach().numpy(),
            label="Anomaly",
            alpha=0.5,
        )
        plt.xlim([-2.5, 2.5])
        plt.ylim([-2.5, 2.5])
        plt.legend()
        plt.title(result)

    plt.subplot(value + 1)
    plt.plot(results["loss"], c="#348ABD")
    plt.legend()
    plt.title("ELBO")

    ax6 = plt.subplot(value + 2)
    RocCurveDisplay.from_predictions(
        results["test"][1], results["elbo"]["score"], ax=ax6
    )
    plt.legend()
    plt.title("ROC")

    if save:
        plt.savefig(
            f'results/figures/elbo_{results["elbo"]["type"]}_method_{results["elbo"]["method"]}'
        )


class Reporter:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment
        self.X_train = experiment.X_train
        self.y_train = experiment.lb_train
        self.X_test = experiment.X_test
        self.y_test = experiment.lb_test
        self.indices_n = []
        self.indices_a = []
        self.scores = []
        self.epochs = []
        self.terms = []

    def save_batch_info(self, epoch, idx_n, idx_a, score):
        self.indices_n.append(idx_n)
        self.indices_a.append(idx_a)
        self.scores.append(score)
        self.epochs.append(epoch)

    def save_elbo_terms(self, ll_n, kl_n, ll_a, kl_a):
        batch_result = {
            "ll_n": ll_n,
            "kl_n": kl_n,
            "ll_a": ll_a,
            "kl_a": kl_a,
        }
        self.terms.append(batch_result)

    def plot_train_evolution(self):
        fig, axs = plt.subplots(2, 3)
        fig.set_figheight(10)
        fig.set_figwidth(20)
        plt.suptitle(
            f"Experiment Attrs: elbo: {self.experiment.elbo}, method: {self.experiment.method}, lr: {self.experiment.lr}, nn_layers: {self.experiment.nn_layers}, batch_size: {self.experiment.batch_size}"
        )
        k = 0
        for i in range(2):
            for j in range(3):
                if i == 0 and j == 0:
                    title = f"Dados de Treino"
                    idx_n = np.where(self.y_train == 0)[0]
                    idx_a = np.where(self.y_train == 1)[0]
                    score = 0
                    epoch = 0
                else:
                    title = f"Epoch {epoch} , ELBO = {score:.2f}"
                    idx_n = self.indices_n[k]
                    idx_a = self.indices_a[k]
                    score = self.scores[k]
                    epoch = self.epochs[k]

                axs[i, j].scatter(
                    self.X_train[:, 0][idx_a],
                    self.X_train[:, 1][idx_a],
                    label="Anomaly",
                    alpha=0.7,
                )
                axs[i, j].scatter(
                    self.X_train[:, 0][idx_n],
                    self.X_train[:, 1][idx_n],
                    label="Normal",
                    alpha=0.7,
                )
                axs[i, j].set_xlim([-0.5, 1.5])
                axs[i, j].set_ylim([-0.5, 1.5])
                axs[i, j].legend()
                axs[i, j].set_title(title)
                k += 1

    def plot_test(self, model, likelihood):
        feature_1, feature_2 = np.meshgrid(
            np.linspace(0, 1, 30),
            np.linspace(0, 1, 30),
        )
        grid = np.vstack([feature_1.ravel(), feature_2.ravel()]).T
        grid = torch.tensor(grid, dtype=torch.float32)
        ll_test, kl_test = calculate_elbo(
            model,
            likelihood,
            target=grid,
            num_data=len(grid),
            batch_size=len(grid),
            elbo_shape="loe",
        )
        elbo_test = -(ll_test - kl_test).detach().numpy()
        zz = np.reshape(elbo_test, feature_1.shape)
        zz = MinMaxScaler().fit_transform(zz)
        fig, ax = plt.subplots()
        cs = ax.contourf(feature_1, feature_2, zz, levels=10)
