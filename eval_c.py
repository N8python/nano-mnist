import torch
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
import ctypes

# Load the shared library
lib = ctypes.CDLL('./libpredict.so')  # Use './predict.dll' on Windows

# Define the argument and return types for the predict function
lib.predict.argtypes = [ctypes.POINTER(ctypes.c_int)]
lib.predict.restype = ctypes.c_int

# Function to call the predict function from the C library
def predict(input_array):
    input_ctypes = (ctypes.c_int * len(input_array))(*np.array(input_array, dtype=np.int32))

    return lib.predict(input_ctypes)

# Load MNIST test dataset
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Prepare the test data loader
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# Initialize counters for accuracy calculation
correct = 0
total = 0

# Iterate over the test dataset
for data, target in tqdm(test_loader, desc="Testing"):
    # Flatten the input image and convert it to a list
    input_array = data.view(-1).numpy().tolist()
    # Ensure the input is a list of length 784 and round the values
    input_array = [round(x) for x in input_array]  # Scale from [0, 1] to [0, 255]
    # Use the predict function to get the predicted class
    predicted_class = predict(input_array)
    
    # Update the counters
    total += 1
    correct += (predicted_class == target.item())

# Compute and print the test accuracy
accuracy = 100 * correct / total
print(f"Accuracy of the generated predict function on the MNIST test set: {accuracy:.2f}%")
