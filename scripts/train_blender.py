import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import argparse
from PIL import Image
import glob
import re

def parse_hpp_array(file_path, var_name):
    with open(file_path, 'r') as f:
        content = f.read()
        # Use raw string r"" to avoid invalid escape sequence warnings
        pattern = rf"std::array<int, \d+> {var_name} = \{{ (.*?) \}};"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return np.fromstring(match.group(1).replace(',', ' '), sep=' ', dtype=np.float32)
    return None

class BlenderNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 3) 
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def med_predict(a, b, c):
    res = torch.zeros_like(a)
    mask1 = (c >= torch.max(a, b))
    res[mask1] = torch.min(a, b)[mask1]
    mask2 = (c <= torch.min(a, b))
    res[mask2] = torch.max(a, b)[mask2]
    mask3 = ~(mask1 | mask2)
    res[mask3] = (a + b - c)[mask3]
    return res

def mlp_predict_torch(a, b, c, w1, b1, w2, b2, scale):
    in0, in1, in2 = a - 128, b - a, c - a
    h = torch.matmul(torch.stack([in0, in1, in2], dim=1), w1.T) + b1
    h = torch.clamp(h, min=0)
    out = torch.matmul(h, w2.T) + b2 * scale
    delta = torch.round(out / scale).squeeze()
    return torch.clamp(a + delta, 0, 255)

def load_data(data_dir):
    inputs, targets = [], []
    image_paths = glob.glob(os.path.join(data_dir, "*.[jp][pn]g"))
    for path in image_paths:
        img = Image.open(path).convert('L')
        arr = np.array(img, dtype=np.float32)
        a, b, c, p = arr[1:, :-1].flatten(), arr[:-1, 1:].flatten(), arr[:-1, :-1].flatten(), arr[1:, 1:].flatten()
        inputs.append(np.stack([a, b, c], axis=1))
        targets.append(p.reshape(-1, 1))
    return torch.tensor(np.vstack(inputs)), torch.tensor(np.vstack(targets))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    w1_raw = parse_hpp_array("include/predictors/mlp_weights.hpp", "w1")
    hidden_size = len(w1_raw) // 3
    print(f"Detected MLP Hidden Size: {hidden_size}")

    mlp_w1 = torch.tensor(w1_raw).reshape(hidden_size, 3)
    mlp_b1 = torch.tensor(parse_hpp_array("include/predictors/mlp_weights.hpp", "b1"))
    mlp_w2 = torch.tensor(parse_hpp_array("include/predictors/mlp_weights.hpp", "w2")).reshape(1, hidden_size)
    mlp_b2 = torch.tensor(parse_hpp_array("include/predictors/mlp_weights.hpp", "b2"))
    mlp_scale = 4096.0

    inputs, targets = load_data(args.data_dir)
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=4096, shuffle=True)

    model = BlenderNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.L1Loss()
    softmax = nn.Softmax(dim=1)

    mlp_w1, mlp_b1, mlp_w2, mlp_b2 = mlp_w1.to(device), mlp_b1.to(device), mlp_w2.to(device), mlp_b2.to(device)

    for epoch in range(15):
        total_loss = 0
        for batch_in, batch_tar in loader:
            batch_in, batch_tar = batch_in.to(device), batch_tar.to(device)
            a, b, c = batch_in[:, 0], batch_in[:, 1], batch_in[:, 2]
            
            with torch.no_grad():
                p0 = med_predict(a, b, c).unsqueeze(1)
                p1 = mlp_predict_torch(a, b, c, mlp_w1, mlp_b1, mlp_w2, mlp_b2, mlp_scale).unsqueeze(1)
                p2 = torch.clamp(a + b - c, 0, 255).unsqueeze(1)
            
            optimizer.zero_grad()
            logits = model(torch.stack([a-128, b-a, c-a], dim=1))
            weights = softmax(logits)
            final_p = weights[:, 0:1] * p0 + weights[:, 1:2] * p1 + weights[:, 2:3] * p2
            
            loss = criterion(final_p, batch_tar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

    model.to("cpu")
    SCALE = 4096
    w1, b1 = model.fc1.weight.detach().numpy(), model.fc1.bias.detach().numpy()
    w2, b2 = model.fc2.weight.detach().numpy(), model.fc2.bias.detach().numpy()
    
    with open("include/predictors/blender_weights.hpp", "w") as f:
        f.write("#ifndef BLENDER_WEIGHTS_HPP\n#define BLENDER_WEIGHTS_HPP\n#include <array>\nnamespace BlenderWeights {\n")
        f.write(f"  static constexpr int SCALE = {SCALE};\n")
        f.write(f"  static constexpr std::array<int, {w1.size}> w1 = {{ {', '.join(map(str, (w1*SCALE).astype(int).flatten()))} }};\n")
        f.write(f"  static constexpr std::array<int, {b1.size}> b1 = {{ {', '.join(map(str, (b1*SCALE).astype(int).flatten()))} }};\n")
        f.write(f"  static constexpr std::array<int, {w2.size}> w2 = {{ {', '.join(map(str, (w2*SCALE).astype(int).flatten()))} }};\n")
        f.write(f"  static constexpr std::array<int, {b2.size}> b2 = {{ {', '.join(map(str, (b2*SCALE).astype(int).flatten()))} }};\n")
        f.write("}\n#endif\n")
