class Trainer_OE_GPLVM:
    def __init__(self, args, X_train, y_train, device) -> None:
        pass

def evaluate(model, data, device):
    pass

class OE_GPLVM():
    def __init__(self, seed, model_name='OE_GPLVM', tune=False,
                 num_epochs=200, patience=50, lr=1e-4, lr_milestones=[50], batch_size=256,
                 latent_dim=1, n_gmm=4, lambda_energy=0.1, lambda_cov=0.005):

        self.utils = Utils()
        self.device = self.utils.get_device()  # get device
        self.seed = seed
        self.tune = tune

        # hyper-parameter
        class Args:
            pass

        self.args = Args()
        self.args.num_epochs = num_epochs
        self.args.patience = patience
        self.args.lr = lr
        self.args.lr_milestones = lr_milestones
        self.args.batch_size = batch_size
        self.args.latent_dim = latent_dim
        self.args.n_gmm = n_gmm
        self.args.lambda_energy = lambda_energy
        self.args.lambda_cov = lambda_cov

    def grid_search(self, X_train, y_train, ratio):
        return self

    def fit(self, X_train, y_train=None, ratio=None):
        self.utils.set_seed(self.seed)

        if  self.tune:
            self.grid_search(X_train, y_train, ratio)
        else:
            pass

        self.model = Trainer_OE_GPLVM(self.args, X_train, y_train, self.device)
        self.model.train()

        return self

    def predict_score(self, X_train, X_test):
        data = {'X_train': X_train, 'X_test': X_test}
        score = evaluate(self.model.model, data, self.device)
        return score