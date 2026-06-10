import torch
import torch.nn as nn
import torch.nn.functional as F
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
        pattern = rf"std::array<int, \d+> {var_name} = \{{ (.*?) \}};"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return np.fromstring(match.group(1).replace(',', ' '), sep=' ', dtype=np.float32)
    return None

class BlenderNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 16)
        self.relu = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x):
        # x is [(a-128)/128, (b-a)/128, (c-a)/128]
        h = self.relu(self.fc1(x))
        w = self.fc2(h)
        w = torch.relu(w) # Match C++: if (w[i] < 0) w[i] = 0;
        # Normalize weights to sum to 1
        return w / (w.sum(dim=1, keepdim=True) + 1.0/4096.0) # Match C++: total_w = w[0]+w[1]+w[2]+1

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
    # Simulate C++ MLPPredictor::predict logic
    # long long in[3] = { a - 128, b - a, c - a };
    in_raw = torch.stack([a - 128, b - a, c - a], dim=1)
    
    # sum = in[0]*w1 + in[1]*w1 + in[2]*w1 + b1
    h = torch.matmul(in_raw, w1.T) + b1
    
    # h[i] = (sum > 0) ? sum : sum / 10;
    h = F.leaky_relu(h, 0.1)
    
    # out = h * w2 + b2 * SCALE
    out = torch.matmul(h, w2.T) + b2.unsqueeze(0) * scale
    
    # delta = (out + scale2/2) / scale2
    delta = out / (scale * scale)
    
    return a + delta.squeeze()

def load_data(data_dir):
    inputs, targets = [], []
    image_paths = glob.glob(os.path.join(data_dir, "*.[jp][pn]g"))
    for path in image_paths:
        print(f"Loading {os.path.basename(path)}...")
        img = Image.open(path).convert('L')
        arr = np.array(img, dtype=np.float32)
        a, b, c, p = arr[1:, :-1].flatten(), arr[:-1, 1:].flatten(), arr[:-1, :-1].flatten(), arr[1:, 1:].flatten()
        c_in = np.stack([a, b, c], axis=1)
        c_tar = p.reshape(-1, 1)

        if c_in.shape[0] > 100000:
            stride = c_in.shape[0] // 100000
            c_in = c_in[::stride]
            c_tar = c_tar[::stride]

        inputs.append(c_in)
        targets.append(c_tar)
    return torch.tensor(np.vstack(inputs)), torch.tensor(np.vstack(targets))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Blender using device: {device}")

    # Load fixed MLP weights
    mlp_w1 = torch.tensor(parse_hpp_array("include/predictors/mlp_weights.hpp", "w1")).reshape(-1, 3)
    mlp_b1 = torch.tensor(parse_hpp_array("include/predictors/mlp_weights.hpp", "b1"))
    mlp_w2 = torch.tensor(parse_hpp_array("include/predictors/mlp_weights.hpp", "w2")).reshape(1, -1)
    mlp_b2 = torch.tensor(parse_hpp_array("include/predictors/mlp_weights.hpp", "b2"))
    mlp_scale = 4096.0

    inputs, targets = load_data(args.data_dir)
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=4096, shuffle=True)

    model = BlenderNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.L1Loss()
    
    mlp_w1, mlp_b1, mlp_w2, mlp_b2 = mlp_w1.to(device), mlp_b1.to(device), mlp_w2.to(device), mlp_b2.to(device)

    for epoch in range(10):
        total_loss = 0
        for batch_in, batch_tar in loader:
            batch_in, batch_tar = batch_in.to(device), batch_tar.to(device)
            a, b, c = batch_in[:, 0], batch_in[:, 1], batch_in[:, 2]
            
            with torch.no_grad():
                p0 = med_predict(a, b, c)
                p1 = mlp_predict_torch(a, b, c, mlp_w1, mlp_b1, mlp_w2, mlp_b2, mlp_scale)
                p2 = torch.clamp(a + b - c, 0, 255)
            
            optimizer.zero_grad()
            # Normalize blender inputs to match C++ 'in' logic if we were using it there
            # But C++ uses raw 'in' for blender too.
            norm_in = torch.stack([(a-128)/128.0, (b-a)/128.0, (c-a)/128.0], dim=1)
            weights = model(norm_in)
            
            # Weighted average
            final_p = weights[:, 0] * p0 + weights[:, 1] * p1 + weights[:, 2] * p2
            final_p = final_p.unsqueeze(1)
            
            # Loss on normalized scale (relative to A)
            loss = criterion((final_p - a.unsqueeze(1))/128.0, (batch_tar - a.unsqueeze(1))/128.0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.6f}")

    model.to("cpu")
    SCALE = 4096
    # Export with correct scaling for C++
    w1 = (model.fc1.weight.detach().numpy() / 128.0) * SCALE
    b1 = model.fc1.bias.detach().numpy() * SCALE
    w2 = model.fc2.weight.detach().numpy() * SCALE
    b2 = model.fc2.bias.detach().numpy() * SCALE
    
    with open("include/predictors/blender_weights.hpp", "w") as f:
        f.write("#ifndef BLENDER_WEIGHTS_HPP\n#define BLENDER_WEIGHTS_HPP\n#include <array>\nnamespace BlenderWeights {\n")
        f.write(f"  static constexpr int SCALE = {SCALE};\n")
        f.write(f"  static constexpr std::array<int, {w1.size}> w1 = {{ {', '.join(map(str, w1.astype(int).flatten()))} }};\n")
        f.write(f"  static constexpr std::array<int, {b1.size}> b1 = {{ {', '.join(map(str, b1.astype(int).flatten()))} }};\n")
        f.write(f"  static constexpr std::array<int, {w2.size}> w2 = {{ {', '.join(map(str, w2.astype(int).flatten()))} }};\n")
        f.write(f"  static constexpr std::array<int, {b2.size}> b2 = {{ {', '.join(map(str, b2.astype(int).flatten()))} }};\n")
        f.write("}\n#endif\n")
