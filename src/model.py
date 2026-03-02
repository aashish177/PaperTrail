import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCN, self).__init__()

        # Layer 1: Takes the 1433 words and compresses them down to 'hidden_channels' (e.g., 16)
        self.conv1 = GCNConv(num_features, hidden_channels)
        
        # Layer 2: Takes the hidden channels and outputs 7 scores (one for each class)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        # 1. First Message Passing step
        x = self.conv1(x, edge_index)

        # 2. Activation function (keeps only positive signals)
        x = F.relu(x)

        # 3. Dropout (randomly turns off neurons to prevent overfitting)
        x = F.dropout(x, p=0.5, training=self.training)

        # 4. Second message Passing step
        x = self.conv2(x, edge_index)

        return x