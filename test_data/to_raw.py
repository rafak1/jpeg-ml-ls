import sys
from PIL import Image
import os

if len(sys.argv) < 2:
    print("Usage: python to_raw.py <image_path>")
    sys.exit(1)

path = sys.argv[1]
img = Image.open(path).convert('L')
name = os.path.splitext(path)[0] + ".raw"

with open(name, "wb") as f:
    f.write(img.tobytes())

print(f"File: {name}")
print(f"Dimensions: {img.width} {img.height}")
print(f"Command: ./jpegls_app.exe {name} {img.width} {img.height}")
