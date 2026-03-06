import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from model import GCN, GAT, GraphSAGE
import matplotlib.pyplot as plt
import numpy as np
import time
import json

# Set device
device = torch.device('cpu')

# Load data
print("Loading Cora dataset...")
dataset = Planetoid(root='./data', name='Cora')
data = dataset[0].to(device)

print(f"Dataset: {dataset}")
print(f"Number of nodes: {data.num_nodes}")
print(f"Number of edges: {data.num_edges}")
print(f"Number of features: {data.num_node_features}")
print(f"Number of classes: {dataset.num_classes}")
print(f"Training nodes: {data.train_mask.sum()}")
print(f"Validation nodes: {data.val_mask.sum()}")
print(f"Test nodes: {data.test_mask.sum()}\n")


def train(model, optimizer, data):
    """Single training step"""
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, data):
    """Evaluate model on train/val/test sets"""
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        
        # Calculate accuracy for each split
        train_correct = pred[data.train_mask] == data.y[data.train_mask]
        train_acc = int(train_correct.sum()) / int(data.train_mask.sum())
        
        val_correct = pred[data.val_mask] == data.y[data.val_mask]
        val_acc = int(val_correct.sum()) / int(data.val_mask.sum())
        
        test_correct = pred[data.test_mask] == data.y[data.test_mask]
        test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
        
    return train_acc, val_acc, test_acc


def train_model(model_class, model_name, num_epochs=200, lr=0.01, weight_decay=5e-4):
    """Train a model and return training history"""
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"{'='*50}")
    
    # Initialize model
    if model_name == "GAT":
        model = model_class(
            num_features=dataset.num_node_features,
            hidden_channels=8,  # Lower for GAT due to multi-head
            num_classes=dataset.num_classes,
            heads=8
        ).to(device)
    else:
        model = model_class(
            num_features=dataset.num_node_features,
            hidden_channels=16,
            num_classes=dataset.num_classes
        ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'test_acc': []
    }
    
    # Timing
    start_time = time.time()
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        loss = train(model, optimizer, data)
        train_acc, val_acc, test_acc = test(model, data)
        
        history['train_loss'].append(loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['test_acc'].append(test_acc)
        
        if epoch % 20 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                  f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    
    training_time = time.time() - start_time
    
    # Final evaluation
    final_train_acc, final_val_acc, final_test_acc = test(model, data)
    
    print(f'\n{model_name} Results:')
    print(f'Training time: {training_time:.2f}s')
    print(f'Final Train Accuracy: {final_train_acc:.4f}')
    print(f'Final Val Accuracy: {final_val_acc:.4f}')
    print(f'Final Test Accuracy: {final_test_acc:.4f}')
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {num_params:,}')
    
    return {
        'model': model,
        'history': history,
        'final_train_acc': final_train_acc,
        'final_val_acc': final_val_acc,
        'final_test_acc': final_test_acc,
        'training_time': training_time,
        'num_params': num_params
    }


# Train all models
results = {}

models_to_compare = [
    (GCN, "GCN"),
    (GAT, "GAT"),
    (GraphSAGE, "GraphSAGE")
]

for model_class, model_name in models_to_compare:
    results[model_name] = train_model(model_class, model_name)


# Create comparison visualizations
print(f"\n{'='*50}")
print("Creating comparison visualizations...")
print(f"{'='*50}")

# 1. Training curves comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Loss curves
ax = axes[0, 0]
for model_name in results.keys():
    ax.plot(results[model_name]['history']['train_loss'], label=model_name, linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Training Loss', fontsize=12)
ax.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Train accuracy
ax = axes[0, 1]
for model_name in results.keys():
    ax.plot(results[model_name]['history']['train_acc'], label=model_name, linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Training Accuracy', fontsize=12)
ax.set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Validation accuracy
ax = axes[1, 0]
for model_name in results.keys():
    ax.plot(results[model_name]['history']['val_acc'], label=model_name, linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Validation Accuracy', fontsize=12)
ax.set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Test accuracy
ax = axes[1, 1]
for model_name in results.keys():
    ax.plot(results[model_name]['history']['test_acc'], label=model_name, linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Test Accuracy', fontsize=12)
ax.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison_curves.png', dpi=300, bbox_inches='tight')
print("Saved: model_comparison_curves.png")

# 2. Bar chart comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

model_names = list(results.keys())
test_accs = [results[m]['final_test_acc'] for m in model_names]
train_times = [results[m]['training_time'] for m in model_names]
num_params = [results[m]['num_params'] for m in model_names]

# Test accuracy comparison
ax = axes[0]
bars = ax.bar(model_names, test_accs, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
ax.set_ylabel('Test Accuracy', fontsize=12)
ax.set_title('Final Test Accuracy', fontsize=14, fontweight='bold')
ax.set_ylim([0.7, 0.85])
ax.grid(True, alpha=0.3, axis='y')
# Add value labels on bars
for bar, acc in zip(bars, test_accs):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Training time comparison
ax = axes[1]
bars = ax.bar(model_names, train_times, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
ax.set_ylabel('Training Time (seconds)', fontsize=12)
ax.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar, time in zip(bars, train_times):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{time:.2f}s',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Number of parameters
ax = axes[2]
bars = ax.bar(model_names, num_params, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
ax.set_ylabel('Number of Parameters', fontsize=12)
ax.set_title('Model Complexity', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar, params in zip(bars, num_params):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{params:,}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('model_comparison_bars.png', dpi=300, bbox_inches='tight')
print("Saved: model_comparison_bars.png")

# 3. Create summary table
print(f"\n{'='*50}")
print("FINAL COMPARISON SUMMARY")
print(f"{'='*50}\n")

print(f"{'Model':<15} {'Test Acc':<12} {'Val Acc':<12} {'Train Time':<15} {'Parameters':<15}")
print("-" * 70)
for model_name in model_names:
    r = results[model_name]
    print(f"{model_name:<15} {r['final_test_acc']:<12.4f} {r['final_val_acc']:<12.4f} "
          f"{r['training_time']:<15.2f} {r['num_params']:<15,}")

# Save results to JSON
results_summary = {}
for model_name in model_names:
    results_summary[model_name] = {
        'test_accuracy': results[model_name]['final_test_acc'],
        'val_accuracy': results[model_name]['final_val_acc'],
        'train_accuracy': results[model_name]['final_train_acc'],
        'training_time': results[model_name]['training_time'],
        'num_parameters': results[model_name]['num_params']
    }

with open('comparison_results.json', 'w') as f:
    json.dump(results_summary, f, indent=4)
print("\nSaved: comparison_results.json")

print("\n Comparison complete! Check the generated files:")
print("   - model_comparison_curves.png")
print("   - model_comparison_bars.png")
print("   - comparison_results.json")