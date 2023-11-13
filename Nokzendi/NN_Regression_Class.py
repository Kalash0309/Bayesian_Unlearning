# Imports
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import jax.tree_util as jtu
from functools import partial
from numpy import linalg as la
import numpy as np
import hamiltorch
from tqdm import tqdm
import torch.distributions as dist
import copy

# https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd


# Create a class for the neural network
class Neural_Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_sizes=[]):
        # Define the parameters
        super(Neural_Network, self).__init__()
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.no_hidden_layers = len(hidden_layer_sizes)

        self.fcs = []
        self.fcs.append(nn.Linear(input_size, hidden_layer_sizes[0]))
        for i in range(1, self.no_hidden_layers):
            self.fcs.append(nn.Linear(hidden_layer_sizes[i - 1], hidden_layer_sizes[i]))
        self.fcs.append(nn.Linear(hidden_layer_sizes[-1], output_size))
        self.model_parameters = nn.Sequential(*self.fcs)

    def negative_log_prior(self, params, model, prior_var=1.0):
        log_prior = 0.0
        for param in model.parameters():
            log_prior += torch.distributions.Normal(0, prior_var).log_prob(param).sum()
        return -log_prior

    def negative_log_likelihood(self, params, model, X, y):
        y_pred = model(X).squeeze()
        loss = nn.MSELoss()
        y_pred = y_pred.unsqueeze(1) 
        assert(y_pred.shape == y.shape)
        return loss(y_pred, y)
    
    def negative_log_likelihood1(self, params, model, X, y):
        y_pred = model(X).squeeze()
        value = dist.Normal(y_pred, 0.1).log_prob(y).sum()
        return -value

    def negative_log_joint(self, params, model, X, y, prior_var):
        return self.negative_log_likelihood(
            params, model, X, y
        ) + self.negative_log_prior(params, model, prior_var)

    def negative_log_unlearned_joint(self, params, model, X, y):
        flattened_params = hamiltorch.util.flatten(model)
        return -self.posterior.log_prob(
            flattened_params
        ) - self.negative_log_likelihood(params, model, X, y)

    def functional_negative_log_prior(self, params, model, prior_var=1.0):
        partial_params_leaves = jtu.tree_leaves(params)
        log_prob = 0.0
        for param in partial_params_leaves:
            data = param.data.flatten()
            for i in range(len(data)):
                log_prob += torch.distributions.Normal(0, prior_var).log_prob(data[i])
        return -log_prob

    def functional_negative_log_likelihood(self, params, model, X, y):
        y_pred = torch.func.functional_call(model, params, X).squeeze()
        loss = nn.MSELoss()
        y_pred = y_pred.unsqueeze(1)
        assert(y_pred.shape == y.shape)
        return loss(y_pred, y)

    def functional_negative_log_joint(self, params, model, X, y, prior_var):
        return self.functional_negative_log_likelihood(
            params, model, X, y
        ) + self.functional_negative_log_prior(params, model, prior_var)

    def functional_negative_log_unlearned_joint(self, params, model, X, y):
        params_flat = jtu.tree_leaves(params)
        flattened_params = torch.cat([param.data.flatten() for param in params_flat])
        return -self.posterior.log_prob(
            flattened_params
        ) - self.functional_negative_log_likelihood(params, model, X, y)

    def train(self, X, Y, lr = 0.01, epochs = 1000, mode = 'optim', plot = True, save_plot = True, save_name = 'Sine_MAP.png'):
        if save_plot is True and plot is False:
            raise ValueError("Cannot save without plotting. Set plot = True")
        self.X = X
        self.Y = Y
        # Theta MAP
        model = []
        for layer in self.fcs[:-1]:
            model.append(layer)
            model.append(nn.Tanh())
        model.append(self.fcs[-1])
        model = nn.Sequential(*model)

        self.no_parameters = sum(p.numel() for p in model.parameters())

        # Train the model based on the method of choice
        if mode == 'optim':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            params = {"nn_params": list(model.parameters())}
            for i in range(epochs):
                optimizer.zero_grad()
                loss = self.negative_log_joint(params, model, X, Y, 10)
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                    print(f"Loss at iteration {i}: {loss.item()}")

            self.model = model
            # Hessian
            params = dict(model.named_parameters())

            partial_func = partial(
                self.functional_negative_log_joint, model=model, X=X, y=Y, prior_var=10
            )

            H = torch.func.hessian(partial_func)(params)

            H_mat = self.hessian_dict_to_matrix(H)

            # cov = torch.eye(H_mat.shape[0])*0.01 
            cov = torch.inverse(H_mat)
            # Keep only iag terms
            # cov = torch.diag(torch.diag(cov))
            # # cov = torch.inverse(H_mat + 1e-3 * torch.eye(H_mat.shape[0]))
            cov = cov.detach().numpy()
            cov = self.nearestPD(cov)
            cov = torch.tensor(cov)
            # cov = torch.diag(torch.diag(cov))*0.01

            theta_map = hamiltorch.util.flatten(model)

            laplace_posterior = torch.distributions.MultivariateNormal(theta_map, cov)

            self.posterior = laplace_posterior
        else:
            # MCMC Sampling
            params_init = hamiltorch.util.flatten(model)
            num_samples = epochs
            step_size = lr
            num_steps_per_sample = 5
            hamiltorch.set_random_seed(123)

            params_hmc = hamiltorch.sample(log_prob_func=self.log_prob_posterior, params_init=params_init,  
                                num_samples=num_samples, step_size=step_size, 
                                num_steps_per_sample=num_steps_per_sample, pass_grad=torch.tensor)
            params_hmc_stack = torch.stack(params_hmc)
            params_map = torch.mean(params_hmc_stack, dim=0)

            self.model = model

            params_hmc_stack = torch.stack(params_hmc)
            params_hmc_stack_t = params_hmc_stack.t()
            cov = torch.cov(params_hmc_stack_t)
            cov = cov.detach().numpy()
            cov = self.nearestPD(cov)
            cov = torch.tensor(cov)

            laplace_posterior = torch.distributions.MultivariateNormal(params_map, cov)

            self.posterior = laplace_posterior

        if plot:
            y_pred = model(X)
            X_ = X.detach().numpy()
            y_ = Y.detach().numpy()
            y_pred_ = y_pred.detach().numpy()
            assert(X_.shape == y_.shape == y_pred_.shape)
            plt.plot(X_, y_, 'ro', label = 'Actual Data')
            plt.plot(X_, y_pred_, 'b', label = 'Predictions')
            plt.legend()
            plt.title("MAP Estimate")
            if save_plot:
                plt.savefig(f'{save_name}.png')
            plt.show()

    def unlearn(self, X_del, Y_del, X_ret, y_ret, lr = 0.01, epochs = 1000, mode = 'optim', lambda_L = 1, plot = True, save_plot = True, save_name = 'Sine_Unlearned_MAP.png'):
        if save_plot is True and plot is False:
            raise ValueError("Cannot save without plotting. Set plot = True")
        
        self.X_del = X_del
        self.Y_del = Y_del
        
        if mode == 'optim':
            # Theta MAP Unlearned
            model = self.model
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            params = {"nn_params": list(model.parameters())}
            for i in range(epochs):
                optimizer.zero_grad()
                loss = self.negative_log_unlearned_joint(params, model, X_del, Y_del)
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                    print(f"Loss at iteration {i}: {loss.item()}")
            
            self.model = model

            # Hessian
            params = dict(model.named_parameters())

            partial_func = partial(
                self.functional_negative_log_unlearned_joint, model=model, X=X_del, y=Y_del
            )

            H = torch.func.hessian(partial_func)(params)

            H_mat = self.hessian_dict_to_matrix(H)

            cov = torch.inverse(H_mat + 1e-1 * torch.eye(H_mat.shape[0]))

            cov = cov.detach().numpy()
            cov = self.nearestPD(cov)
            cov = torch.tensor(cov)

            theta_map_del = hamiltorch.util.flatten(model)

            del_posterior = torch.distributions.MultivariateNormal(theta_map_del, cov)

            self.unlearned_posterior = del_posterior
        else:
            # MCMC Sampling
            params_init = hamiltorch.util.flatten(model)
            num_samples = epochs
            step_size = lr
            num_steps_per_sample = 5
            hamiltorch.set_random_seed(123)

            params_hmc = hamiltorch.sample(log_prob_func=self.log_prob_unlearned_posterior, params_init=params_init,  
                                num_samples=num_samples, step_size=step_size, 
                                num_steps_per_sample=num_steps_per_sample, pass_grad=torch.tensor)
            params_hmc_stack = torch.stack(params_hmc)
            params_map = torch.mean(params_hmc_stack, dim=0)

            self.model = model

            params_hmc_stack = torch.stack(params_hmc)
            params_hmc_stack_t = params_hmc_stack.t()
            cov = torch.cov(params_hmc_stack_t)
            cov = cov.detach().numpy()
            cov = self.nearestPD(cov)
            cov = torch.tensor(cov)

            laplace_posterior = torch.distributions.MultivariateNormal(params_map, cov)

            self.unlearned_posterior = laplace_posterior

        if plot:
            model = self.model

            X_ = X_del.detach().numpy()
            y_ = Y_del.detach().numpy()
            assert(X_.shape == y_.shape)

            X_ret = X_ret.detach().numpy()
            y_ret = y_ret.detach().numpy()
            assert(X_ret.shape == y_ret.shape)

            X = self.X.detach().numpy()
            y = self.Y.detach().numpy()
            assert(X.shape == y.shape)

            y_pred = model(self.X)
            y_pred = y_pred.detach().numpy()
            assert(y_pred.shape == y.shape)

            plt.plot(X_ret, y_ret, 'ro', label = 'Retained Data')
            plt.plot(X_, y_, 'gx', label = 'Deleted Data')
            plt.plot(X, y_pred, 'b', label = 'Unleared Predictions')
            plt.legend()
            plt.title("Unlearned MAP Estimate")
            if save_plot:
                plt.savefig(f'{save_name}.png')
            plt.show()

    def log_prob_posterior(self, params):
        model = []
        for layer in self.fcs[:-1]:
            model.append(layer)
            model.append(nn.Tanh())
        model.append(self.fcs[-1])
        model = nn.Sequential(*model)

        # log_prior
        prior_var = 1.0
        log_prior = -self.negative_log_prior(params, model, prior_var)
 
        # log_likelihood
        log_likelihood = -self.negative_log_likelihood1(params, model, self.X, self.Y)

        # log_posterior
        log_posterior = log_prior + log_likelihood
        return log_posterior
    
    def log_prob_unlearned_posterior(self, params):
        model = []
        for layer in self.fcs[:-1]:
            model.append(layer)
            model.append(nn.Tanh())
        model.append(self.fcs[-1])
        model = nn.Sequential(*model)

        flattened_params = hamiltorch.util.flatten(model)

        # log_prior. In this case, it is the posterior 
        log_prior = self.posterior.log_prob(flattened_params)

        # log_likelihood
        log_likelihood = self.negative_log_likelihood1(params, model, self.X_del, self.Y_del)

        # log_posterior
        log_posterior = log_prior + log_likelihood
        return log_posterior

    def kl_divergence(self, approx_posterior, X_del, y_del):
        X = self.X
        Y = self.Y

        # Remove X_del and y_del from X and Y
        for i in range(len(X_del)):
            for j in range(len(X)):
                if torch.all(torch.eq(X_del[i], X[j])):
                    X = torch.cat((X[:j], X[j + 1 :]))
                    Y = torch.cat((Y[:j], Y[j + 1 :]))
                    break

        model = []
        for layer in self.fcs[:-1]:
            model.append(layer)
            model.append(nn.ReLU())
        model.append(self.fcs[-1])
        model.append(nn.Sigmoid())
        model = nn.Sequential(*model)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        params = {"nn_params": list(model.parameters())}
        for i in range(1000):
            optimizer.zero_grad()
            loss = self.negative_log_joint(params, model, X, Y, 10)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"Loss at iteration {i}: {loss.item()}")

        min1, max1 = X[:, 0].min() - 1, X[:, 0].max() + 1
        min2, max2 = X[:, 1].min() - 1, X[:, 1].max() + 1

        x1grid = np.arange(min1, max1, 0.01)
        x2grid = np.arange(min2, max2, 0.01)

        xx, yy = np.meshgrid(x1grid, x2grid)

        r1, r2 = xx.flatten(), yy.flatten()
        r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

        grid = np.hstack((r1, r2))

        yhat = model(torch.tensor(grid).float()).detach().numpy()

        zz = yhat.reshape(xx.shape)

        plt.contourf(xx, yy, zz, cmap="bwr", alpha=0.5, levels=100)
        plt.scatter(X[:, 0], X[:, 1], c=Y, cmap="bwr")
        plt.title("Theta MAP Retained Boundary")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.colorbar()
        plt.savefig("theta_map_retained_boundary.png")
        plt.show()

        # Hessian
        params = dict(model.named_parameters())

        partial_func = partial(
            self.functional_negative_log_joint, model=model, X=X, y=Y, prior_var=10
        )

        H = torch.func.hessian(partial_func)(params)

        H_mat = self.hessian_dict_to_matrix(H)

        cov = torch.inverse(H_mat + 1e-1 * torch.eye(H_mat.shape[0]))
        # cov = torch.inverse(H_mat + 1e-3 * torch.eye(H_mat.shape[0]))
        cov = cov.detach().numpy()
        cov = self.nearestPD(cov)
        cov = torch.tensor(cov)

        theta_map = hamiltorch.util.flatten(model)

        retained_poserior = torch.distributions.MultivariateNormal(theta_map, cov)

        kl = torch.distributions.kl.kl_divergence(self.posterior, approx_posterior)
        return kl

    def predict_mc(self, x, n_samples, approx_posterior):
        model = copy.deepcopy(self.model)
        samples = approx_posterior.sample((n_samples,))
        y = []
        for theta in samples:
            params_list = hamiltorch.util.unflatten(model, theta)
            hamiltorch.util.update_model_params_in_place(model, params_list)
            y.append(model(x).detach().numpy())

        mean = np.mean(y)
        std = np.std(y)
        return mean, std

    def plot_regression_boundary(self, X, Y, X_del = None, y_del = None, X_ret = None, y_ret = None, samples = 1000, post = 'l', save = False, save_name = 'MC_Laplace_prediction.png'):
        if post == 'u':
            if X_del is None or y_del is None or X_ret is None or y_ret is None:
                raise ValueError("Must provide X_del, y_del, X_ret, y_ret when plotting unlearned posterior")
        elif post == 'r':
            if X_ret is None or y_ret is None:
                raise ValueError("Must provide X_ret, y_ret when plotting retained posterior")

        means = []
        stds = []

        if post == 'l' or post == 'r':
            dist = self.posterior
        else:
            dist = self.unlearned_posterior

        for x in tqdm(X):
            mean, std = self.predict_mc(x, samples, dist)
            means.append(mean)
            stds.append(std)

        means = np.array(means)
        stds = np.array(stds)
        
        if post == 'l':
            plt.plot(X, means, "b", label="Predictions")
            plt.fill_between(
                X.squeeze(), means.squeeze() - stds.squeeze(), means.squeeze() + stds.squeeze(), alpha=0.5
            )
            plt.plot(X, Y, "ro", label="Actual Data")
        elif post == 'r':
            plt.plot(X, means, "b", label="Retained Predictions")
            plt.fill_between(
                X.squeeze(), means.squeeze() - stds.squeeze(), means.squeeze() + stds.squeeze(), alpha=0.5
            )
            plt.plot(X_ret, y_ret, "ro", label="Retained Data")
        else:
            plt.plot(X, means, "b", label="Unlearned Predictions")
            plt.fill_between(
                X.squeeze(), means.squeeze() - stds.squeeze(), means.squeeze() + stds.squeeze(), alpha=0.5
            )
            plt.plot(X_ret, y_ret, "ro", label="Retained Data")
            plt.plot(X_del, y_del, "gx", label="Deleted Data")

        plt.title(save_name)
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig(f'{save_name}.png')
        plt.show()

        return means, stds
        

    def plot_decision_boundary(
        self,
        X,
        Y,
        approx_posterior,
        title,
        save_file="decision_boundary.png",
        unlearned=False,
        X_unlearned=None,
        Y_unlearned=None,
    ):
        min1, max1 = X[:, 0].min() - 1, X[:, 0].max() + 1
        min2, max2 = X[:, 1].min() - 1, X[:, 1].max() + 1

        x1grid = np.arange(min1, max1, 0.8)
        x2grid = np.arange(min2, max2, 0.8)

        xx, yy = np.meshgrid(x1grid, x2grid)

        # mean = np.zeros((len(x1grid), len(x2grid)))
        mean = np.zeros((len(x2grid), len(x1grid)))
        std = np.zeros((len(x2grid), len(x1grid)))

        for i in tqdm(range(len(x1grid)), desc="Plotting Decision Boundary"):
            for j in range(len(x2grid)):
                x = torch.tensor([x1grid[i], x2grid[j]]).float()
                mean[j, i], std[j, i] = self.predict_mc(x, 1000, approx_posterior)

        # Plot the mean
        plt.contourf(xx, yy, mean, cmap="bwr", alpha=0.5, levels=100)
        plt.scatter(X[:, 0], X[:, 1], c=Y, cmap="bwr")
        if unlearned:
            plt.scatter(
                X_unlearned[:, 0],
                X_unlearned[:, 1],
                c=Y_unlearned,
                cmap="bwr",
                marker="x",
                s=100,
            )
        plt.title(title)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.colorbar()
        plt.savefig(save_file)
        plt.show()

    def hessian_dict_to_matrix(self, H):
        H_matrix = torch.zeros((self.no_parameters, self.no_parameters))
        named_parameters = dict(self.model.named_parameters())
        size_params = []
        for key in named_parameters:
            temp = []
            if len(named_parameters[key].shape) == 2:
                temp.append(named_parameters[key].shape[0])
                temp.append(named_parameters[key].shape[1])
            else:
                temp.append(named_parameters[key].shape[0])
            size_params.append(temp)

        i = 0
        x = 0
        for key1 in H:
            if len(size_params[i]) == 2:
                # Weight matrix
                rows = size_params[i][0]
                cols = size_params[i][1]
                j = x
                y = 0
                for key2 in H[key1]:
                    no_params = H[key1][key2][0][0].flatten().shape[0]
                    for r in range(0, rows):
                        for c in range(0, cols):
                            flat_params = H[key1][key2][r][c].flatten()
                            H_matrix[j, y : y + no_params] = flat_params
                            j += 1
                    y += no_params
                    j = x
                x += rows * cols
            else:
                rows = size_params[i][0]
                j = x
                y = 0
                for key2 in H[key1]:
                    no_params = H[key1][key2][0].flatten().shape[0]
                    for r in range(0, rows):
                        flat_params = H[key1][key2][r].flatten()
                        H_matrix[j, y : y + no_params] = flat_params
                        j += 1
                    y += no_params
                    j = x
                x += rows
            i += 1

        return H_matrix

    def nearestPD(self, A):
        B = (A + A.T) / 2
        _, s, V = la.svd(B)

        H = np.dot(V.T, np.dot(np.diag(s), V))

        A2 = (B + H) / 2

        A3 = (A2 + A2.T) / 2

        if self.isPD(A3):
            return A3

        spacing = np.spacing(la.norm(A))
        I = np.eye(A.shape[0])
        k = 1
        while not self.isPD(A3):
            mineig = np.min(np.real(la.eigvals(A3)))
            A3 += I * (-mineig * k**2 + spacing)
            k += 1

        return A3

    def isPD(self, B):
        try:
            _ = la.cholesky(B)
            return True
        except la.LinAlgError:
            return False

    def forward(self, x):
        assert x.shape[1] == self.input_size
        return self.model(x)

    def print_parameters(self):
        print(self.model.state_dict())
