from NN_Classification_Class import Neural_Network
import torch

model = Neural_Network(input_size=2, hidden_layer_sizes=[2], output_size=2)
model.print_parameters()
print("#######################")
X = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)
Y = torch.tensor([0, 0, 1], dtype=torch.long)
model.train(X, Y)
print("#######################")
model.print_parameters()
print("#######################")
