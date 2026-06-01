import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
from PIL import Image
import glob

class SimpleMLP(nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.fc1 = nn.Linear(3, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def load_data(data_dir):
    inputs, targets = [], []
    image_paths = glob.glob(os.path.join(data_dir, "*.[jp][pn]g"))
    for path in image_paths:
        img = Image.open(path).convert('L')
        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape
        for y in range(1, h):
            for x in range(1, w):
                inputs.append([arr[y, x-1], arr[y-1, x], arr[y-1, x-1]])
                targets.append([arr[y, x]])
    return torch.tensor(inputs), torch.tensor(targets)

def export_weights(model, path):
    SCALE = 1024
    w1, b1 = model.fc1.weight.detach().numpy(), model.fc1.bias.detach().numpy()
    w2, b2 = model.fc2.weight.detach().numpy(), model.fc2.bias.detach().numpy()
    
    with open(path, 'w') as f:
        f.write("#ifndef MLP_WEIGHTS_HPP\n#define MLP_WEIGHTS_HPP\n#include <array>\n\n")
        f.write("namespace MLPWeights {\n")
        f.write(f"    static constexpr int SCALE = {SCALE};\n")
        f.write(f"    static constexpr std::array<int, {w1.size}> w1 = {{ {', '.join(map(str, (w1*SCALE).astype(int).flatten()))} }};\n")
        f.write(f"    static constexpr std::array<int, {b1.size}> b1 = {{ {', '.join(map(str, (b1*SCALE).astype(int).flatten()))} }};\n")
        f.write(f"    static constexpr std::array<int, {w2.size}> w2 = {{ {', '.join(map(str, (w2*SCALE).astype(int).flatten()))} }};\n")
        f.write(f"    static constexpr std::array<int, {b2.size}> b2 = {{ {', '.join(map(str, (b2*SCALE).astype(int).flatten()))} }};\n")
        f.write("}\n#endif\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()

    inputs, targets = load_data(args.data_dir)
    model = SimpleMLP()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(100):
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    export_weights(model, "include/predictors/mlp_weights.hpp")
