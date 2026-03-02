import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from model import GCN

# Device setup
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Load dataset
dataset = Planetoid(root='.\data', name='Cora')
data = dataset[0].to(device)

# Initialize Model
model = GCN(num_features=dataset.num_node_features,
            hidden_channels=16,
            num_classes=dataset.num_classes).to(device)

# Optimizer (Adam)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()

    # Forward Pass
    out = model(data.x, data.edge_index)

    # Calculate Loss (Only on nodes we are allowed to see: the train_mask)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

    loss.backward() # Backpropagation
    optimizer.step() # Update weights
    return loss.item()

@torch.no_grad() # Disable gradient tracking for speed during evalution
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    # Calculate accuracy for test set
    correct = pred[data.test_mask] == data.y[data.test_mask]
    acc = int(correct.sum()) / int(data.test_mask.sum())
    return acc

for epoch in range(1, 101):
    loss = train()
    if epoch % 10 == 0:
        test_acc = test()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Acc: {test_acc:.4f}') 
