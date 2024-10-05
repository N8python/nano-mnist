import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import numpy as np

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
        modified_weight = modify_parameters(self.weight, c)
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

model = MLP()
model.load_state_dict(torch.load("model.pth"))
import numpy as np

# Extract quantized weights from the model
fc1_weight = model.fc1.weight.data.cpu().numpy().astype(np.int8)
fc2_weight = model.fc2.weight.data.cpu().numpy().astype(np.int8)

# Function to determine which neurons and inputs are used
def analyze_network():
    # Determine which neurons in the first layer are used
    used_first_layer_neurons = set()
    used_output_neurons = set()
    for output_idx, weights in enumerate(fc2_weight):
        # Check if the output neuron is connected to any first layer neuron
        connected_neurons = np.nonzero(weights)[0]
        # Keep only connected neurons that are in used_first_layer_neurons
        connected_used_neurons = set(connected_neurons)
        if connected_used_neurons:
            used_first_layer_neurons.update(connected_used_neurons)
            used_output_neurons.add(output_idx)
    # Determine which input variables are used
    used_input_indices = set()
    for neuron_idx in used_first_layer_neurons:
        weights = fc1_weight[neuron_idx]
        input_indices = np.nonzero(weights)[0]
        used_input_indices.update(input_indices)
    return sorted(used_first_layer_neurons), sorted(used_input_indices), sorted(used_output_neurons)
# Function to generate code for the first layer
no_nos = []
def generate_first_layer_code(used_neurons, used_inputs):
    global i_used
    code = ""
    for neuron_idx in used_neurons:
        weights = fc1_weight[neuron_idx]
        terms = []
        i = 0
        for input_idx in used_inputs:
            weight = weights[input_idx]
            """if weight == 1:
                terms.append(f"i[{input_idx}]")
            elif weight == -1:
                terms.append(f"-i[{input_idx}]")"""
            if weight == 1:
                terms.append(f"i[{input_idx}]" if i == 0 else f"+i[{input_idx}]")
            elif weight == -1:
                terms.append(f"-i[{input_idx}]" if i == 0 else f"-i[{input_idx}]")
            i += 1
            # Ignore weights that are 0
        if not terms:
            # Remove the neuron from the used_neurons list
            no_nos.append(neuron_idx)
            continue  # Skip if the neuron has no non-zero weights
        # Build the accumulator as a single expression
        #accumulator_expr = "+".join(terms)
        accumulator_expr = "".join(terms)
        #code += f"    # Neuron {neuron_idx} in the first layer\n"
        code += f"    a{neuron_idx}=max(0,{accumulator_expr})\n"
    return code

# Function to generate code for the second layer
def generate_second_layer_code(used_neurons, used_output_neurons):
    code = ""
    for neuron_idx in used_output_neurons:
        weights = fc2_weight[neuron_idx]
        terms = []
        i = 0
        for input_idx in used_neurons:
            if input_idx in no_nos:
                continue
            weight = weights[input_idx]
            if weight == 1:
                terms.append(f"a{input_idx}" if i == 0 else f"+a{input_idx}")
            elif weight == -1:
                terms.append(f"-a{input_idx}" if i == 0 else f"-a{input_idx}")
            i += 1
            # Ignore weights that are 0 or if activation_{input_idx} is not defined
        if not terms:
            continue  # Skip if the output neuron has no non-zero weights from used neurons
        # Build the output as a single expression
        output_expr = "".join(terms)
        #code += f"    # Output neuron {neuron_idx}\n"
        code += f"    o{neuron_idx}={output_expr}\n"
    return code

# Function to generate code for finding the predicted class
def generate_output_code(used_output_neurons):
    code = "    outputs = [" + ",".join([f"o{neuron_idx}" for neuron_idx in used_output_neurons]) + "]\n"
    code += "    return outputs.index(max(outputs))\n"
    return code

# Main function to generate the entire predict function code
def generate_predict_function():
    used_neurons, used_inputs, used_output_neurons = analyze_network()
    first_layer_code = generate_first_layer_code(used_neurons, used_inputs)
    second_layer_code = generate_second_layer_code(used_neurons, used_output_neurons)
    output_code = generate_output_code(used_output_neurons)

    # Generate the entire function code
    function_code = 'def predict(i):\n'
    function_code += first_layer_code
    function_code += second_layer_code
    function_code += output_code
    return function_code

# Generate the predict function code
predict_function_code = generate_predict_function()

# Write the function to a file
with open('predict_function.py', 'w') as file:
    file.write(predict_function_code)
