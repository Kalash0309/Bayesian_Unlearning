from NN_classification_sampling import Neural_Network
import torch
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

model = Neural_Network(input_size=2, hidden_layer_sizes=[5], output_size=1)

# Swiss Roll
X, y = make_blobs(n_samples=50, centers=2, n_features=2, random_state=42)
X = torch.tensor(X).float()
y = torch.tensor(y).float()
X = torch.cat((X, torch.tensor([[0, 5]]).float()))
y = torch.cat((y, torch.tensor([1]).float()))
X = torch.cat((X, torch.tensor([[-1, 5]]).float()))
y = torch.cat((y, torch.tensor([1]).float()))

X_del = torch.tensor([[0, 5]]).float()
y_del = torch.tensor([1]).float()
X_del = torch.cat((X_del, torch.tensor([[-1, 5]]).float()))
y_del = torch.cat((y_del, torch.tensor([1]).float()))


# Circles
# X, y = make_circles(n_samples=100, noise=0.02, random_state=42)
# X = torch.tensor(X).float()
# y = torch.tensor(y).float()

# # # pick some points to delete
# X_del = []
# y_del = []

# for i in range(100):
#     if y[i] == 1 and X[i, 0] > 0.5:
#         X_del.append([X[i,0],X[i,1]])
#         y_del.append(y[i])


# y = y.view(-1)
# y_del = torch.tensor(y_del).view(-1)
# X_del = torch.tensor(X_del)
# X_del = X_del.float()
# y_del = y_del.float()

# Moons
# X, y = make_moons(n_samples=50, noise=0.1)
# X = torch.tensor(X, dtype=torch.float)
# y = torch.tensor(y).float()

# # pick some points to delete
# X_del = []
# y_del = []

# for i in range(50):
#     if y[i] == 1 and X[i, 0] < 0 and X[i, 1] > 0:
#         X_del.append([X[i,0],X[i,1]])
#         y_del.append(y[i])


# y = y.view(-1)
# y_del = torch.tensor(y_del).view(-1)
# X_del = torch.tensor(X_del)
# X_del = X_del.float()
# y_del = y_del.float()


# Data of all the points with label 0
# X0 = X[y == 0]
# Y0 = y[y == 0]

# Create a circle of points with label 0
# X = torch.zeros(100, 2)
# y = torch.zeros(100, 1)
# X_del = []
# y_del = []
# for i in range(100):
#     if i >= 0 and i <= 50:
#         if i > 22 and i < 28:
#             X[i, 0] = torch.cos(torch.tensor(i * 2 * 3.14159 / 100))
#             X[i, 1] = torch.sin(torch.tensor(i * 2 * 3.14159 / 100))
#             y[i, 0] = 1
#             X_del.append(
#                 [
#                     torch.cos(torch.tensor(i * 2 * 3.14159 / 100)),
#                     torch.sin(torch.tensor(i * 2 * 3.14159 / 100)),
#                 ]
#             )
#             y_del.append(1)

#         else:
#             X[i, 0] = torch.cos(torch.tensor(i * 2 * 3.14159 / 100))
#             X[i, 1] = torch.sin(torch.tensor(i * 2 * 3.14159 / 100))
#             y[i, 0] = 0
#     else:
#         X[i, 0] = torch.cos(torch.tensor(i * 2 * 3.14159 / 100))
#         X[i, 1] = torch.sin(torch.tensor(i * 2 * 3.14159 / 100))
#         y[i, 0] = 1

# y = y.view(-1)
# y_del = torch.tensor(y_del).view(-1)
# X_del = torch.tensor(X_del)
# X_del = X_del.float()
# y_del = y_del.float()

# plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr")
# plt.show()

fig_theta, axes_theta = plt.subplots(1, 2, figsize=(12, 4)) 
fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # 1 row and 2 columns
fig_unlearned_theta, axes_unlearned_theta = plt.subplots(1, 2, figsize=(12, 4))
fig_unlearned_posterior, axes_unlearned_posterior = plt.subplots(1, 2, figsize=(12, 4))

posterior_sampling = model.train_sampling(X, y, subplot=axes_theta[0])
fig_sampling = model.plot_decision_boundary(X, y, approx_posterior=posterior_sampling, title="Predictive Posterior Distribution - Sampling", subplot=axes[0])
model.unlearn_sampling(X_del, y_del, subplot_theta=axes_unlearned_theta[0], subplot_posterior=axes_unlearned_posterior[0], type="sampling")
# kl_sampling = model.kl_divergence(X_del, y_del,posterior_sampling)


posterior_laplace = model.train_laplace(X, y, subplot=axes_theta[1])
fig_laplace = model.plot_decision_boundary(X, y, approx_posterior=posterior_laplace, title="Predictive Posterior Distribution - Laplace", subplot=axes[1])
model.unlearn_laplace(X_del, y_del, subplot_theta=axes_unlearned_theta[1], subplot_posterior=axes_unlearned_posterior[1], type="laplace")
# kl_laplace = model.kl_divergence(X_del, y_del,posterior_laplace)

fig_theta.savefig("swiss_theta_map_sampling_vs_laplace.png")
fig.savefig("swiss_predictive_posterior_sampling_vs_laplace.png")
fig_unlearned_theta.savefig("swiss_unlearned_theta_sampling_vs_laplace.png")
fig_unlearned_posterior.savefig("swiss_unlearned_posterior_sampling_vs_laplace.png")
# plt.show()


# print("KL Divergence - Sampling: ", kl_sampling)
# print("KL Divergence - Laplace: ", kl_laplace)

# plt.show()
