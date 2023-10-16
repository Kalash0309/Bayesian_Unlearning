from NN_classification_sampling import Neural_Network
import torch
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

model = Neural_Network(input_size=2, hidden_layer_sizes=[8], output_size=1)

# Swiss Roll
# X, y = make_blobs(n_samples=20, centers=2, n_features=2, random_state=42)
# X = torch.tensor(X).float()
# y = torch.tensor(y).float()
# X = torch.cat((X, torch.tensor([[0, 5]]).float()))
# y = torch.cat((y, torch.tensor([1]).float()))
# X = torch.cat((X, torch.tensor([[-1, 5]]).float()))
# y = torch.cat((y, torch.tensor([1]).float()))

# X_del = torch.tensor([[0, 5]]).float()
# y_del = torch.tensor([1]).float()
# X_del = torch.cat((X_del, torch.tensor([[-1, 5]]).float()))
# y_del = torch.cat((y_del, torch.tensor([1]).float()))


# Circles
# X, y = make_circles(n_samples=100, noise=0.02, random_state=42)
# X = torch.tensor(X).float()
# y = torch.tensor(y).float()

# Moons
X, y = make_moons(n_samples=50, noise=0.1)
X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y).float()

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

model.train(X, y)
# model.unlearn(X_del, y_del)
