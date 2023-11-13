from NN_Regression_Class import Neural_Network
import numpy as np
import torch
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Creating a simple dataset
# Generating data
N = 100
X = torch.linspace(0, 2, N).reshape(-1, 1)
y = X + 0.3 * torch.randn(N, 1)

#Genreate sine function
# X = torch.linspace(0, 2*np.pi, N).reshape(-1, 1)
# y = torch.sin(X) + 0.1 * torch.randn(N, 1)

# Plotting the results
# def plot_predictions(X, y, y_pred):
#     X_ = X.detach().numpy()
#     y_ = y.detach().numpy()
#     y_pred_ = y_pred.detach().numpy()
#     assert(X_.shape == y_.shape == y_pred_.shape)
#     plt.plot(X_, y_, 'ro', label = 'Actual Data')
#     plt.plot(X_, y_pred_, 'b', label = 'Predictions')
#     plt.legend()
#     plt.show()


def laplace():
    # Creating a simple model
    simple_model = Neural_Network(input_size = 1, hidden_layer_sizes=[4], output_size=1)

    # Training the model
    simple_model.train(
        X, 
        y, 
        plot = True, 
        save_plot = True,
        save_name = 'Line_MAP')

    # Plot the posterior predictive
    Laplace_sample_means, Laplace_sample_stds = simple_model.plot_regression_boundary(
        X, 
        y, 
        post = 'l', 
        save = True,
        save_name = 'line_MC_Laplace_predictions')

    # Separating the dataset into retained and deleted subsets
    # Keep only the first 50 and last 50 datapoints of the dataset
    X_subset = X[0:30]
    y_subset = y[0:30]

    X_retained = torch.cat((X_subset, X[60:100]), 0)
    y_retained = torch.cat((y_subset, y[60:100]), 0)

    # Saving the Deleted data for later use
    X_deleted = X[30:60]
    y_deleted = y[30:60]


    # Unlearning the model
    simple_model.unlearn(
        X_deleted, 
        y_deleted, 
        X_retained, 
        y_retained,  
        plot = True, 
        save_plot = True,
        save_name = 'Line_Unlearned_MAP')

    # Plot the unlearned posterior predictive
    Laplace_unlearned_sample_means, Laplace_unlearned_sample_stds = simple_model.plot_regression_boundary(
        X, 
        y, 
        X_deleted, 
        y_deleted, 
        X_retained, 
        y_retained, 
        post = 'u', 
        save = True,
        save_name = 'line_MC_Unlearned_Laplace_predictions')
    
    # Create and train another model on the retained dataset
    simple_model_ret = Neural_Network(input_size = 1, hidden_layer_sizes=[4], output_size=1)

    # Training the model
    simple_model_ret.train(
        X_retained, 
        y_retained, 
        plot = True, 
        save_plot = True,
        save_name = 'Line_Retained_MAP')
    
    # Plot the posterior predictive
    Laplace_sample_means_ret, Laplace_sample_stds_ret = simple_model_ret.plot_regression_boundary(
        X, 
        y, 
        X_deleted, 
        y_deleted, 
        X_retained, 
        y_retained, 
        post = 'r',
        save_name = 'line_MC_Retained_Laplace_predictions')
    
    # Create a subplots to compare the Laplace, Unlearned Laplace, and the retained Laplace preditive distributions
    fig, ax = plt.subplots(1, 3, figsize = (15, 5))
    ax[0].plot(X, y, 'ro', label = 'Actual Data')
    ax[0].plot(X, Laplace_sample_means, 'b', label = 'Laplace')
    ax[0].fill_between(X.reshape(-1), Laplace_sample_means - 1.96 * Laplace_sample_stds, 1.96 * Laplace_sample_means + Laplace_sample_stds, alpha = 0.5)
    ax[0].set_title('Laplace Predictive Posterior')
    ax[0].legend()
    ax[1].plot(X_retained, y_retained, 'ro', label = 'Retained Data')
    ax[1].plot(X_deleted, y_deleted, 'gx', label = 'Deleted Data')
    ax[1].plot(X, Laplace_unlearned_sample_means, 'b', label = 'Unlearned Laplace')
    ax[1].fill_between(X.reshape(-1), Laplace_unlearned_sample_means - 1.96 * Laplace_unlearned_sample_stds, Laplace_unlearned_sample_means + 1.96 * Laplace_unlearned_sample_stds, alpha = 0.5)
    ax[1].set_title('Unlearned Laplace Predictive Posterior')
    ax[1].legend()
    ax[2].plot(X_retained, y_retained, 'ro', label = 'Retained Data')
    ax[2].plot(X, Laplace_sample_means_ret, 'b', label = 'Retained Laplace')
    ax[2].fill_between(X.reshape(-1), Laplace_sample_means_ret - 1.96 * Laplace_sample_stds_ret, 1.96 * Laplace_sample_means_ret + Laplace_sample_stds_ret, alpha = 0.5)
    ax[2].set_title('Retained Laplace Predictive Posterior')
    ax[2].legend()

    ax[0].set_ylim([-90, 90])
    ax[1].set_ylim([-90, 90])
    ax[2].set_ylim([-90, 90])

    fig.text(0.5, 0.04, 'x', ha='center')
    fig.text(0.04, 0.5, 'y', va='center', rotation='vertical')

    plt.savefig('Laplace_Comparison.png')
    plt.show()

def mc_normal_posterior():
    # Creating a simple model
    simple_model = Neural_Network(input_size = 1, hidden_layer_sizes=[4], output_size=1)

    # Training the model
    simple_model.train(
        X, 
        y, 
        mode = 'MCMC',
        plot = True, 
        save_plot = True,
        save_name = 'Line_MAP_MC_normal')

    # Plot the posterior predictive
    MC_normal_sample_means, MC_normal_sample_stds = simple_model.plot_regression_boundary(
        X, 
        y, 
        post = 'l', 
        save = True,
        save_name = 'line_MC_MC_normal_Posterior_predictions')

    # Separating the dataset into retained and deleted subsets
    # Keep only the first 50 and last 50 datapoints of the dataset
    X_subset = X[0:30]
    y_subset = y[0:30]

    X_retained = torch.cat((X_subset, X[60:100]), 0)
    y_retained = torch.cat((y_subset, y[60:100]), 0)

    # Saving the Deleted data for later use
    X_deleted = X[30:60]
    y_deleted = y[30:60]


    # Unlearning the model
    simple_model.unlearn(
        X_deleted, 
        y_deleted, 
        X_retained, 
        y_retained,  
        plot = True, 
        save_plot = True,
        save_name = 'Line_Unlearned_MAP_MC_normal')

    # Plot the unlearned posterior predictive
    MC_normal_unlearned_sample_means, MC_normal_unlearned_sample_stds = simple_model.plot_regression_boundary(
        X, 
        y, 
        X_deleted, 
        y_deleted, 
        X_retained, 
        y_retained, 
        post = 'u', 
        save = True,
        save_name = 'line_MC_Unlearned_MC_normal_Posterior_predictions')
    
    # Create and train another model on the retained dataset
    simple_model_ret = Neural_Network(input_size = 1, hidden_layer_sizes=[4], output_size=1)

    # Training the model
    simple_model_ret.train(
        X_retained, 
        y_retained, 
        plot = True, 
        save_plot = True,
        save_name = 'Line_Retained_MAP_MC_normal')
    
    # Plot the posterior predictive
    MC_normal_sample_means_ret, MC_normal_sample_stds_ret = simple_model_ret.plot_regression_boundary(
        X, 
        y, 
        X_deleted, 
        y_deleted, 
        X_retained, 
        y_retained, 
        post = 'r',
        save_name = 'line_MC_Retained_MC_normal_Posterior_predictions')
    
    # Create a subplots to compare the Laplace, Unlearned Laplace, and the retained Laplace preditive distributions
    fig, ax = plt.subplots(1, 3, figsize = (15, 5))
    ax[0].plot(X, y, 'ro', label = 'Actual Data')
    ax[0].plot(X, MC_normal_sample_means, 'b', label = 'MC Normal Sample Posterior')
    ax[0].fill_between(X.reshape(-1), MC_normal_sample_means - 1.96 * MC_normal_sample_stds, 1.96 * MC_normal_sample_means + MC_normal_sample_stds, alpha = 0.5)
    ax[0].set_title('MC Normal Sample Predictive Posterior')
    ax[0].legend()
    ax[1].plot(X_retained, y_retained, 'ro', label = 'Retained Data')
    ax[1].plot(X_deleted, y_deleted, 'gx', label = 'Deleted Data')
    ax[1].plot(X, MC_normal_unlearned_sample_means, 'b', label = 'Unlearned MC Normal Sample Posterior')
    ax[1].fill_between(X.reshape(-1), MC_normal_unlearned_sample_means - 1.96 * MC_normal_unlearned_sample_stds, MC_normal_unlearned_sample_means + 1.96 * MC_normal_unlearned_sample_stds, alpha = 0.5)
    ax[1].set_title('Unlearned MC Normal Sample Predictive Posterior')
    ax[1].legend()
    ax[2].plot(X_retained, y_retained, 'ro', label = 'Retained Data')
    ax[2].plot(X, MC_normal_sample_means_ret, 'b', label = 'Retained MC Normal Sample Posterior')
    ax[2].fill_between(X.reshape(-1), MC_normal_sample_means_ret - 1.96 * MC_normal_sample_stds_ret, 1.96 * MC_normal_sample_means_ret + MC_normal_sample_stds_ret, alpha = 0.5)
    ax[2].set_title('Retained MC Normal Sample Predictive Posterior')
    ax[2].legend()

    ax[0].set_ylim([-90, 90])
    ax[1].set_ylim([-90, 90])
    ax[2].set_ylim([-90, 90])

    fig.text(0.5, 0.04, 'x', ha='center')
    fig.text(0.04, 0.5, 'y', va='center', rotation='vertical')

    plt.savefig('MC_Comparison.png')
    plt.show()


# laplace()
# mc_normal_posterior()

