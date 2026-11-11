#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures"

np.random.seed(42)
torch.manual_seed(42)

print("🐾 启动真实 MSRCv1 数据集的自适应分析与 AHDD 蒸馏流水线...")
mat_file = str(DATA_DIR / "MSRCv1_real.mat")

try:
    mat_data = sio.loadmat(mat_file)
except Exception as e:
    print(f"❌ 无法读取 {mat_file}。请确保文件存在且是有效的 .mat 文件。\n错误信息: {e}")
    sys.exit(1)

# ==========================================
# 1. 自动探针反射机制：寻找数据和标签
# ==========================================
data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
print(f"🔍 在 .mat 文件中找到的数据键: {data_keys}")

X_data = None
Y_data = None

for key in data_keys:
    val = mat_data[key]
    if isinstance(val, np.ndarray):
        if val.dtype == 'O' and val.size >= 2: 
            X_data = val
            print(f"  -> 选定 '{key}' 作为特征矩阵 X")
        elif val.size > 1 and (val.shape[0] == 1 or val.shape[1] == 1 or len(val.shape) == 1):
            Y_data = val.flatten()
            print(f"  -> 选定 '{key}' 作为标签 Y")

if X_data is None and 'data' in data_keys: X_data = mat_data['data']
if Y_data is None and 'labels' in data_keys: Y_data = mat_data['labels'].flatten()
     
if X_data is None or Y_data is None:
    print("❌ 自动探测数据结构失败，请手动检查 .mat 文件的 keys!")
    sys.exit(1)

Y = Y_data
if Y.min() == 1: Y = Y - 1
labels = torch.tensor(Y, dtype=torch.long)
num_classes = len(np.unique(Y))

try:
    if X_data.dtype == 'O': 
        v1_raw = X_data[0,0] if len(X_data.shape) > 1 else X_data[0]
        v2_raw = X_data[0,1] if len(X_data.shape) > 1 else X_data[1]
    else:
        print("❌ 警告：特征结构不是标准多视图，无法直接切分。")
        sys.exit(1)

    if v1_raw.shape[1] == len(Y): v1_raw = v1_raw.T
    if v2_raw.shape[1] == len(Y): v2_raw = v2_raw.T

    def adapt_dim(data_np, target_dim):
        n, d = data_np.shape
        if d >= target_dim: return data_np[:, :target_dim]
        else:
            padded = np.zeros((n, target_dim))
            padded[:, :d] = data_np
            return padded

    v1_np = adapt_dim(v1_raw, 24)
    v2_np = adapt_dim(v2_raw, 576)

    v1_np = (v1_np - v1_np.mean(axis=0)) / (v1_np.std(axis=0) + 1e-8)
    v2_np = (v2_np - v2_np.mean(axis=0)) / (v2_np.std(axis=0) + 1e-8)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    v1_data = torch.tensor(v1_np, dtype=torch.float32).to(device)
    v2_data = torch.tensor(v2_np, dtype=torch.float32).to(device)
except Exception as e:
    print(f"❌ 提取和适配数据时出错: {e}")
    sys.exit(1)

print(f"✅ 数据适配完毕！V1: {v1_data.shape}, V2: {v2_data.shape}, 类别数: {num_classes}")

# ==========================================
# 2. 联合训练与可视化
# ==========================================
class ViewAutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=128, dict_size=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, input_dim))
        self.sparse_linear = nn.Linear(latent_dim, dict_size)
        self.dictionary = nn.Parameter(torch.randn(dict_size, latent_dim))

    def forward(self, x):
        z = self.encoder(x)
        l_v = torch.sigmoid(self.sparse_linear(z))
        z_dict = torch.matmul(l_v, self.dictionary)
        return z, self.decoder(z), l_v, z_dict

ae1 = ViewAutoEncoder(input_dim=24).to(device)
ae2 = ViewAutoEncoder(input_dim=576).to(device)
optimizer = optim.Adam(list(ae1.parameters()) + list(ae2.parameters()), lr=1e-3)

print("🚀 正在执行跨模态特征蒸馏...")
epochs = 50
ae1.train(); ae2.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    z1, recon1, l1, _ = ae1(v1_data)
    z2, recon2, l2, _ = ae2(v2_data)
    
    loss_fr = nn.MSELoss()(recon1, v1_data) + nn.MSELoss()(recon2, v2_data)
    alpha = 0.65 if epoch < 25 else 0.45 
    
    z_1to2 = torch.matmul(l1, ae2.dictionary)
    z_2to1 = torch.matmul(l2, ae1.dictionary)
    
    loss_distill = alpha * nn.MSELoss()(z1, z_1to2) + (1 - alpha) * nn.MSELoss()(z2, z_2to1)
    total_loss = loss_fr + 0.1 * loss_distill
    total_loss.backward()
    optimizer.step()

ae1.eval(); ae2.eval()
with torch.no_grad():
    z1, _, _, _ = ae1(v1_data)
    z2, _, _, _ = ae2(v2_data)
    z_fused = ((z1 + z2) / 2.0).cpu().numpy()

perp = min(15, v1_data.shape[0] - 1)
try:
    tsne = TSNE(n_components=2, perplexity=perp, max_iter=1500, random_state=42)
except TypeError:
    tsne = TSNE(n_components=2, perplexity=perp, n_iter=1500, random_state=42)
z_2d = tsne.fit_transform(z_fused)

plt.figure(figsize=(10, 8))
colors = plt.cm.tab20(np.linspace(0, 1, num_classes))

for i in range(num_classes):
    idx = (labels.numpy() == i)
    plt.scatter(z_2d[idx, 0], z_2d[idx, 1], c=[colors[i]], label=f'Class {i}', alpha=0.8, edgecolors='w', s=65)

plt.title('t-SNE of Real MSRCv1 Features ($Z^f$) After AHDD Fusion', fontsize=15)
plt.xlabel('t-SNE Dimension 1', fontsize=12)
plt.ylabel('t-SNE Dimension 2', fontsize=12)
if num_classes <= 10:
    plt.legend(loc='best', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)

FIG_DIR.mkdir(parents=True, exist_ok=True)
save_path = FIG_DIR / "msrc_real_tsne.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"🎉 真实公开数据集打榜可视化完成！图像已保存至: {save_path}")
