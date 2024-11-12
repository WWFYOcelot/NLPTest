import torch

print(torch.version.cuda)  # Check CUDA version in PyTorch
print(torch.cuda.is_available())  # Check CUDA availability

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print("CUDA is available:", cuda_available)

# If available, print the CUDA device name
if cuda_available:
    print("CUDA device name:", torch.cuda.get_device_name(0))
else:
    print("No CUDA devices found.")
