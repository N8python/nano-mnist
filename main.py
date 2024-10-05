import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

def modify_parameters(param, c):
    return 0.5 + 0.5 * torch.tanh(c * (param - 1)) - 0.5 + 0.5 * torch.tanh(c * (param + 1))

def quantize_parameters(param, epsilon=0.001):
    return torch.where(torch.abs(param) < epsilon, torch.zeros_like(param), torch.sign(param))

class ModifiedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModifiedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)

    def forward(self, input, c):
        if c > 0:
            modified_weight = modify_parameters(self.weight, c)
        else:
            modified_weight = self.weight
        return nn.functional.linear(input, modified_weight)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = ModifiedLinear(784, 2048)
        self.relu = nn.ReLU()
        self.fc2 = ModifiedLinear(2048, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, c):
        x = self.flatten(x)
        x = self.fc1(x, c)
        x = self.relu(x)
        x = self.fc2(x, c)
        x = self.sigmoid(x)
        return x

# Set device
device = torch.device("mps")

# Hyperparameters
batch_size = 64
learning_rate = 0.01
num_epochs = 100
c_initial = 1.0
c_increment = 0.01
epsilon = 0.001  # Threshold for quantizing to zero

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function, and optimizer
model = MLP().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
c = c_initial
for epoch in range(num_epochs):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        data = torch.round(data)
        target_one_hot = nn.functional.one_hot(target, num_classes=10).float()
        optimizer.zero_grad()
        output = model(data, c)
        
        loss = criterion(output, target_one_hot)
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(output.data, 1)
        correct = (predicted == target).sum().item()
        accuracy = correct / batch_size * 100
        
        pbar.set_postfix({
            'Loss': f"{loss.item():.4f}",
            'Accuracy': f"{accuracy:.2f}%",
            'c': f"{c:.2f}"
        })
        
        c += c_increment

# Quantize parameters after training
with torch.no_grad():
    model.fc1.weight.data = quantize_parameters(modify_parameters(model.fc1.weight.data, c), epsilon)
    model.fc2.weight.data = quantize_parameters(modify_parameters(model.fc2.weight.data, c), epsilon)

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in tqdm(test_loader, desc="Evaluating"):
        data, target = data.to(device), target.to(device)
        data = torch.round(data)
        output = model(data, 0)  # Use the final c value
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f"Accuracy on test set: {100 * correct / total:.2f}%")
print(f"Final c value: {c:.2f}")

# Log out model parameters
print("\nModel Parameters:")
for name, param in model.named_parameters():
    #quantized_param = quantize_parameters(modify_parameters(param.data, c), epsilon)
    quantized_param = param.data
    print(f"{name} (quantized):")
    print(quantized_param)
    print(f"Shape: {quantized_param.shape}")
    unique_values = torch.unique(quantized_param)
    print(f"Unique values: {unique_values}")
    print(f"Number of -1s: {torch.sum(quantized_param == -1).item()}")
    print(f"Number of 0s: {torch.sum(quantized_param == 0).item()}")
    print(f"Number of 1s: {torch.sum(quantized_param == 1).item()}")
    print(f"Mean: {quantized_param.mean().item():.4f}")
    print(f"Std: {quantized_param.std().item():.4f}")
    print()

# Save the model
torch.save(model.state_dict(), "model.pth")
