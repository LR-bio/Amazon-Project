import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
import rasterio
from scipy.ndimage import gaussian_filter
from typing import Tuple

# === Preprocessing Functions ===

def load_dem(filepath: str) -> Tuple[np.ndarray, object]:
    with rasterio.open(filepath) as src:
        return src.read(1), src.transform

def normalize(tile: np.ndarray) -> np.ndarray:
    return (tile - tile.min()) / (tile.ptp() + 1e-8)

def compute_slope(dem: np.ndarray, scale=1) -> np.ndarray:
    gx, gy = np.gradient(dem, scale)
    return np.sqrt(gx**2 + gy**2)

def preprocess_tile(path: str) -> np.ndarray:
    dem, _ = load_dem(path)
    dem_smoothed = gaussian_filter(dem, sigma=1)
    slope = compute_slope(dem_smoothed)
    norm_dem = normalize(dem_smoothed)
    norm_slope = normalize(slope)
    return np.stack([norm_dem, norm_slope], axis=0)  # Shape: (2, H, W)

# === Dataset ===

class EarthworkDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.samples = [f for f in os.listdir(data_dir) if f.endswith('.npz')]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.data_dir, self.samples[idx]))
        x = torch.tensor(data['x'], dtype=torch.float32)
        y = torch.tensor(data['y'], dtype=torch.long)
        return x, y

# === Model ===

class EarthworkCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 64 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Earthwork / No Earthwork
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# === Training ===

def train_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EarthworkCNN().to(device)
    dataset = EarthworkDataset('data/processed_tiles')
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = criterion(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss: {total_loss/len(loader):.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/cnn_model.pth")

# === Inference ===

def run_inference(tile_path: str):
    model = EarthworkCNN()
    model.load_state_dict(torch.load("models/cnn_model.pth", map_location='cpu'))
    model.eval()

    tile = preprocess_tile(tile_path)
    tile_tensor = torch.tensor(tile).unsqueeze(0).float()  # Shape: (1, 2, H, W)
    with torch.no_grad():
        output = model(tile_tensor)
        prediction = output.argmax(dim=1).item()

    print(f"{tile_path}: {'Earthwork Detected' if prediction == 1 else 'No Earthwork'}")

# === Synthetic Data Generator ===

def generate_fake_mound_tile(size=256) -> np.ndarray:
    base = np.random.rand(size, size) * 0.1
    cx, cy = np.random.randint(64, 192, size=2)
    for i in range(size):
        for j in range(size):
            dist = np.sqrt((i - cx)**2 + (j - cy)**2)
            base[i, j] += np.exp(-dist**2 / 500)
    return base

def generate_dataset(n=200, out_dir='data/processed_tiles'):
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n):
        label = np.random.randint(0, 2)
        if label == 1:
            dem = generate_fake_mound_tile()
        else:
            dem = np.random.rand(256, 256) * 0.2
        slope = compute_slope(dem)
        x = np.stack([normalize(dem), normalize(slope)])
        np.savez_compressed(f"{out_dir}/tile_{i}.npz", x=x, y=label)
    print(f"Generated {n} samples in {out_dir}/")

# === Main CLI ===

def print_usage():
    print("\nUsage:")
    print("  python amazon_earthworks_detector.py generate     # Generate synthetic dataset")
    print("  python amazon_earthworks_detector.py train        # Train the CNN model")
    print("  python amazon_earthworks_detector.py infer <file> # Run inference on DEM tile\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1]

    if command == "generate":
        generate_dataset()
    elif command == "train":
        train_model()
    elif command == "infer":
        if len(sys.argv) < 3:
            print("Please provide a path to the DEM tile.")
        else:
            run_inference(sys.argv[2])
    else:
        print_usage()
