# Imports
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import jax.tree_util as jtu
from functools import partial


# Create a class for the neural network
class Neural_Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_sizes=[]):
        # Define the parameters
        super(Neural_Network, self).__init__()
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.no_hidden_layers = len(hidden_layer_sizes)

        self.no_parameters = 0
        self.fcs = []
        self.fcs.append(nn.Linear(input_size, hidden_layer_sizes[0]))
        self.no_parameters += input_size * hidden_layer_sizes[0] + hidden_layer_sizes[0]
        for i in range(1, self.no_hidden_layers):
            self.fcs.append(nn.Linear(hidden_layer_sizes[i - 1], hidden_layer_sizes[i]))
            self.no_parameters += (
                hidden_layer_sizes[i - 1] * hidden_layer_sizes[i]
                + hidden_layer_sizes[i]
            )
        self.fcs.append(nn.Linear(hidden_layer_sizes[-1], output_size))
        self.no_parameters += hidden_layer_sizes[-1] * output_size + output_size
        self.model = []
        for layer in self.fcs[:-1]:
            self.model.append(layer)
            self.model.append(nn.ReLU())
        self.model.append(self.fcs[-1])
        self.model.append(nn.Softmax(dim=0))
        self.model = nn.Sequential(*self.model)
        self.model_parameters = nn.Sequential(*self.fcs)

    ### Defining a function for prior (cannot be tweaked by users rn)
    def prior(self):
        prior_mean = torch.zeros(self.no_parameters)
        prior_cov = (
            torch.eye(self.no_parameters) * 100
        )  # Factor to reduce the afffect of prior
        return torch.distributions.MultivariateNormal(prior_mean, prior_cov)

    def unnormalized_log_posterior(self, X, Y):
        log_likelihood = torch.tensor(0.0, requires_grad=True)
        for i in range(len(X)):
            y_pred = self.model(X[i])
            log_prob = torch.distributions.Categorical(probs=y_pred).log_prob(Y[i])
            temp = log_likelihood.clone()
            temp += log_prob
            log_likelihood = temp

        flattened_parameter = torch.cat(
            [
                torch.cat([layer.weight.flatten(), layer.bias.flatten()])
                for layer in self.model
                if isinstance(layer, nn.Linear)
            ]
        )

        assert len(flattened_parameter) == self.no_parameters

        log_prior = self.prior().log_prob(flattened_parameter)
        return log_likelihood + log_prior

    def unnormalized_log_posterior_functional(self, theta):
        X = self.X
        Y = self.Y
        model = []
        # Create a model given the theta
        start = 0
        weight_end = self.input_size * self.hidden_layer_sizes[0]
        bias_end = (
            self.input_size * self.hidden_layer_sizes[0] + self.hidden_layer_sizes[0]
        )

        for i in range(self.no_hidden_layers):
            if i == 0:
                layer = nn.Linear(self.input_size, self.hidden_layer_sizes[i])
                layer.weight = nn.Parameter(
                    theta[start:weight_end].reshape(
                        self.hidden_layer_sizes[i], self.input_size
                    )
                )
                layer.bias = nn.Parameter(theta[weight_end:bias_end])
                model.append(layer)
                model.append(nn.ReLU())

                if i != self.no_hidden_layers - 1:
                    start = bias_end
                    weight_end = (
                        bias_end
                        + self.hidden_layer_sizes[i] * self.hidden_layer_sizes[i + 1]
                    )
                    bias_end = weight_end + self.hidden_layer_sizes[i + 1]

                if i == self.no_hidden_layers - 1:
                    layer = nn.Linear(self.hidden_layer_sizes[i], self.output_size)
                    layer.weight = nn.Parameter(
                        theta[
                            bias_end : bias_end
                            + self.hidden_layer_sizes[i] * self.output_size
                        ].reshape(self.output_size, self.hidden_layer_sizes[i])
                    )
                    layer.bias = nn.Parameter(
                        theta[
                            bias_end
                            + self.hidden_layer_sizes[i]
                            * self.output_size : self.no_parameters
                        ]
                    )
                    model.append(layer)
                    model.append(nn.Softmax(dim=0))
            else:
                layer = nn.Linear(
                    self.hidden_layer_sizes[i - 1], self.hidden_layer_sizes[i]
                )
                layer.weight = nn.Parameter(
                    theta[start:weight_end].reshape(
                        self.hidden_layer_sizes[i], self.input_size
                    )
                )
                layer.bias = nn.Parameter(theta[weight_end:bias_end])
                model.append(layer)
                model.append(nn.ReLU())

                if i != self.no_hidden_layers - 1:
                    start = bias_end
                    weight_end = (
                        bias_end
                        + self.hidden_layer_sizes[i] * self.hidden_layer_sizes[i + 1]
                    )
                    bias_end = weight_end + self.hidden_layer_sizes[i + 1]

                if i == self.no_hidden_layers - 1:
                    layer = nn.Linear(self.hidden_layer_sizes[i], self.output_size)
                    layer.weight = nn.Parameter(
                        theta[
                            bias_end : bias_end
                            + self.hidden_layer_sizes[i] * self.output_size
                        ].reshape(self.output_size, self.hidden_layer_sizes[i])
                    )
                    layer.bias = nn.Parameter(
                        theta[
                            bias_end
                            + self.hidden_layer_sizes[i]
                            * self.output_size : self.no_parameters
                        ]
                    )
                    model.append(layer)
                    model.append(nn.Softmax(dim=0))

        model = nn.Sequential(*model)

        log_likelihood = torch.tensor(0.0, requires_grad=True)
        for i in range(len(X)):
            y_pred = model(X[i])
            log_prob = torch.distributions.Categorical(probs=y_pred).log_prob(Y[i])
            temp = log_likelihood.clone()
            temp += log_prob
            log_likelihood = temp

        flattened_parameter = torch.cat(
            [
                torch.cat([layer.weight.flatten(), layer.bias.flatten()])
                for layer in model
                if isinstance(layer, nn.Linear)
            ]
        )

        log_prior = self.prior().log_prob(flattened_parameter)
        return log_likelihood + log_prior

    def train(self, X, Y):
        # Find theta map of unnormalized log posterior
        self.X = X
        self.Y = Y
        params = {"nn_params": list(self.model.parameters())}
        parameter_leaves = jtu.tree_leaves(params)
        optimizer = torch.optim.Adam(parameter_leaves, lr=0.01)
        for i in range(1000):
            if i % 100 == 0:
                print(i / 10, "%")
            optimizer.zero_grad()
            loss = -self.unnormalized_log_posterior(X, Y)
            loss.backward()
            optimizer.step()

        theta_map = torch.cat(
            [
                torch.cat([layer.weight.flatten(), layer.bias.flatten()])
                for layer in self.model
                if isinstance(layer, nn.Linear)
            ]
        )

        partial_func = partial(self.unnormalized_log_posterior_functional)

        H = torch.func.hessian(partial_func)(theta_map)
        print(H)

    def forward(self, x):
        assert x.shape[0] == self.input_size
        return self.model(x)

    def print_parameters(self):
        print(self.model.state_dict())
