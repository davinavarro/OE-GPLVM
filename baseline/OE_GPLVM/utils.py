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
    dataset: str
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
        self.countour = []

        self.grid_x, self.grid_y = np.meshgrid(
            np.linspace(-5, 5, 50),
            np.linspace(-5, 5, 50),
        )
        grid = np.vstack([self.grid_x.ravel(), self.grid_y.ravel()]).T

        self.grid = torch.tensor(grid, dtype=torch.float32)

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

    def plot_train_evolution(self,save = False):
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
        if save:
            plt.savefig(f"results/scatter_dataset_{self.experiment.dataset}_{self.experiment.elbo}_method_{self.experiment.method}_lr_{self.experiment.lr}_batch_size_{self.experiment.batch_size}.png")

    def save_countour_evolution(self, model, likelihood):
        ll_test, klu_test ,klx_test = calculate_elbo(
            model,
            likelihood,
            target=self.grid,
            num_data=len(self.grid),
            batch_size=len(self.grid),
            elbo_shape="loe",
        )
        elbo_test = -(ll_test - klu_test - klx_test).detach().numpy()
        zz = np.reshape(elbo_test, self.grid_x.shape)
        zz = MinMaxScaler().fit_transform(zz)
        self.countour.append(zz)

    def plot_countour_evolution(self, save  = False):
        fig, axs = plt.subplots(2, 3)
        fig.set_figheight(10)
        fig.set_figwidth(20)
        plt.suptitle(
            f"Experiment Attrs: elbo: {self.experiment.elbo}, method: {self.experiment.method}, lr: {self.experiment.lr}, nn_layers: {self.experiment.nn_layers}, batch_size: {self.experiment.batch_size}"
        )
        k = 0
        for i in range(2):
            for j in range(3):
                score = self.scores[k]
                epoch = self.epochs[k]
                title = f"Epoch {epoch} , ELBO = {score:.2f}"
                axs[i, j].contourf(self.grid_x, self.grid_y, self.countour[k], levels=10)
                axs[i, j].set_xlim([-5, 5])
                axs[i, j].set_ylim([-5, 5])
                axs[i, j].set_title(title)
                k += 1
        
        if save:
            plt.savefig(f"results/countour_dataset_{self.experiment.dataset}_{self.experiment.elbo}_method_{self.experiment.method}_lr_{self.experiment.lr}_batch_size_{self.experiment.batch_size}.png")