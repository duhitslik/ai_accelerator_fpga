import sys
import os
sys.path.append(os.path.abspath(".."))
import torch
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model_training')))
from train_mlp_mnist import MLP
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

model = MLP()
model.load_state_dict(torch.load('mlp_mnist.pth'))
model.eval()

# Extract weights and biases
with torch.no_grad():
    w1 = model.fc1.weight.numpy()
    b1 = model.fc1.bias.numpy()
    w2 = model.fc2.weight.numpy()
    b2 = model.fc2.bias.numpy()

# Save as INT8 .txt for Verilog
def save_txt(name, arr):
    arr_int = np.round(arr * 128).astype(np.int8)
    np.savetxt(name, arr_int, fmt='%d')

import os
os.makedirs("../inputs_outputs", exist_ok=True)

save_txt("../inputs_outputs/w1.txt", w1)
save_txt("../inputs_outputs/b1.txt", b1)
save_txt("../inputs_outputs/w2.txt", w2)
save_txt("../inputs_outputs/b2.txt", b2)

# Export one MNIST input image
transform = transforms.Compose([transforms.ToTensor()])
test_data = MNIST(root='./data', train=False, download=True, transform=transform)
img, label = test_data[0]
img_np = img.view(-1).numpy()
img_int = np.round(img_np * 255).astype(np.uint8)
np.savetxt("../inputs_outputs/image_input.txt", img_int, fmt='%d')

output = model(img.view(1, 1, 28, 28)).detach().numpy()  # add .detach()
np.savetxt("../inputs_outputs/expected_output.txt", output, fmt='%.4f')
print("✅ Export complete.")
