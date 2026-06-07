import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import argparse
from PIL import Image
import glob

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def load_data(data_dir, mode='3'):
    inputs, targets = [], []
    image_paths = glob.glob(os.path.join(data_dir, "*.[jp][pn]g"))
    
    for path in image_paths:
        img = Image.open(path).convert('L')
        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape
        
        if mode == '3':
            a, b, c, p = arr[1:, :-1].flatten(), arr[:-1, 1:].flatten(), arr[:-1, :-1].flatten(), arr[1:, 1:].flatten()
            chunk_in = np.stack([a - 128, b - a, c - a], axis=1)
            chunk_tar = (p - a).reshape(-1, 1)
        else:
            num_samples = 100000
            ys = np.random.randint(2, h, num_samples)
            xs = np.random.randint(2, w - 2, num_samples)
            chunk_in, chunk_tar = [], []
            for i in range(num_samples):
                y, x = ys[i], xs[i]
                ref = arr[y, x-1]
                ctx = np.concatenate([arr[y-2, x-2:x+3], arr[y-1, x-2:x+3], arr[y, x-2:x]])
                diffs = ctx - ref
                diffs[-1] = ref - 128
                chunk_in.append(diffs)
                chunk_tar.append([arr[y, x] - ref])
            chunk_in = np.array(chunk_in)
            chunk_tar = np.array(chunk_tar)

        inputs.append(chunk_in)
        targets.append(chunk_tar)
            
    return torch.tensor(np.vstack(inputs)), torch.tensor(np.vstack(targets))

def export_weights(model, path, namespace):
    SCALE = 4096
    w1, b1 = model.fc1.weight.detach().numpy(), model.fc1.bias.detach().numpy()
    w2, b2 = model.fc2.weight.detach().numpy(), model.fc2.bias.detach().numpy()
    with open(path, 'w') as f:
        f.write(f"#ifndef {namespace.upper()}_HPP\n#define {namespace.upper()}_HPP\n#include <array>\n\n")
        f.write(f"namespace {namespace} {{\n")
        f.write(f"    static constexpr int SCALE = {SCALE};\n")
        f.write(f"    static constexpr std::array<int, {w1.size}> w1 = {{ {', '.join(map(str, (w1*SCALE).astype(int).flatten()))} }};\n")
        f.write(f"    static constexpr std::array<int, {b1.size}> b1 = {{ {', '.join(map(str, (b1*SCALE).astype(int).flatten()))} }};\n")
        f.write(f"    static constexpr std::array<int, {w2.size}> w2 = {{ {', '.join(map(str, (w2*SCALE).astype(int).flatten()))} }};\n")
        f.write(f"    static constexpr std::array<int, {b2.size}> b2 = {{ {', '.join(map(str, (b2*SCALE).astype(int).flatten()))} }};\n")
        f.write("}\n#endif\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--mode", choices=['3', '12'], default='3')
    args = parser.parse_args()

    inputs, targets = load_data(args.data_dir, args.mode)
    dataset = TensorDataset(inputs.float(), targets.float())
    loader = DataLoader(dataset, batch_size=4096, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_size = 3 if args.mode == '3' else 12
    model = SimpleMLP(input_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    criterion = nn.L1Loss()

    for epoch in range(15):
        total_loss = 0
        for batch_in, batch_tar in loader:
            batch_in, batch_tar = batch_in.to(device), batch_tar.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_in), batch_tar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
            
    model.to("cpu")
    out_path = "include/predictors/mlp_weights.hpp" if args.mode == '3' else "include/predictors/mlp_5x5_weights.hpp"
    namespace = "MLPWeights" if args.mode == '3' else "MLP5x5Weights"
    export_weights(model, out_path, namespace)
