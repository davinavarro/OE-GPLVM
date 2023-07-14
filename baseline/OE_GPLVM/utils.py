from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
import numpy as np
import torch
def results_plot(results: dict , save = False):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'GPLVM - ELBO {results["elbo"]["type"]} - Method {results["elbo"]["method"]}')
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
            alpha=0.5
        )
        plt.xlim([-2.5, 2.5])
        plt.ylim([-2.5, 2.5])
        plt.legend()
        plt.title(result)

    

    plt.subplot(value + 1)
    plt.plot(results["loss"] ,c = "#348ABD")
    plt.legend()
    plt.title("ELBO")

    ax6 = plt.subplot(value + 2)
    RocCurveDisplay.from_predictions(results["test"][1], results["elbo"]["score"] ,ax = ax6)
    plt.legend()
    plt.title("ROC")
    
    if save:
        plt.savefig(f'results/figures/elbo_{results["elbo"]["type"]}_method_{results["elbo"]["method"]}')

