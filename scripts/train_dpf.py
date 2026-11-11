from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SEED = 42


class DPFFusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(360, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
        )
        self.weight_estimator = nn.Sequential(
            nn.Linear(16 + 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus(),
        )

    def forward(self, scan, particles, prior_state):
        feat = self.feature_extractor(scan)
        feat_expanded = feat.unsqueeze(0).repeat(particles.shape[0], 1)
        state_residual = particles - prior_state.unsqueeze(0)
        fusion_input = torch.cat([feat_expanded, state_residual], dim=1)
        raw_weights = self.weight_estimator(fusion_input).squeeze()
        return raw_weights / (torch.sum(raw_weights) + 1e-8)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    dataset_path = DATA_DIR / "dpf_dataset.pt"
    if not dataset_path.exists():
        raise SystemExit(
            f"Missing {dataset_path}. This file is a synthetic 2D demo dataset and "
            "is not the real robot bag. See the repository notes before using it."
        )
    dataset = torch.load(dataset_path, weights_only=False)
    poses, scans = dataset["poses"], dataset["scans"]

    model = DPFFusionNet()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    num_particles = 100
    epochs = 10

    print("Training the 2D DPF demo network.")
    print("This is not the 6-DoF SAPF described in the paper.")
    for epoch in range(epochs):
        total_loss = 0.0
        for t in range(1, 300):
            true_pose = poses[t]
            prev_pose = poses[t - 1]
            scan = scans[t]
            particles = prev_pose + torch.randn(num_particles, 2) * 0.3
            weights = model(scan, particles, prev_pose)
            estimated_pose = torch.sum(weights.unsqueeze(1) * particles, dim=0)
            loss = torch.mean((estimated_pose - true_pose) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1:02d}/{epochs} | MSE: {total_loss / 299:.6f}")

    output_path = DATA_DIR / "dpf.pth"
    torch.save(model.state_dict(), output_path)
    print(f"Saved weights to {output_path}")


if __name__ == "__main__":
    main()
