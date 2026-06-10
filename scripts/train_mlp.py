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
        self.relu = nn.LeakyReLU(0.1) 
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
def load_data(data_dir, mode='3'):
    inputs, targets = [], []
    image_paths = glob.glob(os.path.join(data_dir, "*.[jp][pn]g"))

    for path in image_paths:
        print(f"Loading {os.path.basename(path)}...")
        img = Image.open(path).convert('L')
        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape

        if mode == '3':
            a, b, c, p = arr[1:, :-1].flatten(), arr[:-1, 1:].flatten(), arr[:-1, :-1].flatten(), arr[1:, 1:].flatten()
            c_in = np.stack([(a - 128)/128.0, (b - a)/128.0, (c - a)/128.0], axis=1)
            c_tar = ((p - a) / 128.0).reshape(-1, 1)
        else:
            ref = np.zeros_like(arr)
            ref[:, 1:] = arr[:, :-1]
            ref[1:, 0] = arr[:-1, 0]
            ref[0, 0] = 0

            padded = np.pad(arr, ((2, 0), (2, 2)), mode='constant', constant_values=0)

            ctx = []
            for dy in range(2): 
                for dx in range(5):
                    ctx.append((padded[dy:dy+h, dx:dx+w] - ref).flatten())

            ctx.append((padded[2:2+h, 0:w] - ref).flatten())
            ctx.append(np.zeros(h*w, dtype=np.float32))
            ctx.append((ref - 128).flatten())

            c_in = np.stack(ctx, axis=-1) / 128.0
            c_tar = ((arr - ref) / 128.0).reshape(-1, 1)

        if c_in.shape[0] > 100000:
            stride = c_in.shape[0] // 100000
            c_in = c_in[::stride]
            c_tar = c_tar[::stride]

        inputs.append(c_in)
        targets.append(c_tar)
            
    return torch.tensor(np.vstack(inputs)), torch.tensor(np.vstack(targets))

def export_weights(model, path, namespace):
    SCALE = 4096
    w1 = model.fc1.weight.detach().numpy() / 128.0
    b1 = model.fc1.bias.detach().numpy()
    w2 = model.fc2.weight.detach().numpy() * 128.0
    b2 = model.fc2.bias.detach().numpy() * 128.0
    
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

    input_size = 3 if args.mode == '3' else 13
    hidden_size = 64 if args.mode == '3' else 128
    model = SimpleMLP(input_size, hidden_size).to(device)
    lr = 0.01 if args.mode == '3' else 0.005
    epochs = 15 if args.mode == '3' else 20
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    criterion = nn.L1Loss()

    for epoch in range(epochs):
        total_loss = 0
        all_preds, all_tars = [], []
        for batch_in, batch_tar in loader:
            batch_in, batch_tar = batch_in.to(device), batch_tar.to(device)
            optimizer.zero_grad()
            preds = model(batch_in)
            loss = criterion(preds, batch_tar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            all_preds.append(preds.detach().cpu())
            all_tars.append(batch_tar.detach().cpu())
            
        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)
        p_std = torch.cat(all_preds).std().item()
        t_std = torch.cat(all_tars).std().item()
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f} | Pred StdDev: {p_std:.4f} (Target StdDev: {t_std:.4f})")
            
    model.to("cpu")
    out_path = "include/predictors/mlp_weights.hpp" if args.mode == '3' else "include/predictors/mlp_5x5_weights.hpp"
    namespace = "MLPWeights" if args.mode == '3' else "MLP5x5Weights"
    export_weights(model, out_path, namespace)
