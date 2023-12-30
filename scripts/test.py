import os 
import time

import torch
# os.environ['CUDA_VISIBLE_DEVICES'] = "6"
# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available")
    device = torch.device('cuda')  # Use 'cuda:1', 'cuda:2', etc. for other GPUs
    tensor = torch.randn(10000, 10000, device=device)  # An example large tensor directly on GPU
    print(device)
    time.sleep(10) 
else:
    print("CUDA not available")