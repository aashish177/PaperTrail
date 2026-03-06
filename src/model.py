import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

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
    
class GAT(torch.nn.Module):
    """Graph Attention Network"""
    def __init__(self, num_features, hidden_channels, num_classes, heads=8, dropout=0.6):
        super(GAT, self).__init__()
        self.dropout = dropout
        
        # First GAT layer with multi-head attention
        # heads=8 means 8 attention heads, output is concatenated
        self.conv1 = GATConv(
            num_features, 
            hidden_channels, 
            heads=heads, 
            dropout=dropout
        )
        
        # Second GAT layer
        # Input size is hidden_channels * heads (concatenated from previous layer)
        # Output single head, concat=False means average instead of concatenate
        self.conv2 = GATConv(
            hidden_channels * heads, 
            num_classes, 
            heads=1, 
            concat=False, 
            dropout=dropout
        )
        
    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GraphSAGE(torch.nn.Module):
    """GraphSAGE Network"""
    def __init__(self, num_features, hidden_channels, num_classes, dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(num_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, num_classes)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x