import torch
import torch.nn as nn
import torch.nn.functional as F

print("🐾 启动 ADAPT 自适应硬度驱动字典蒸馏 (AHDD) 模块验证...")

def compute_ahdd_loss(Z_b, Z_v, L_b, L_v, D_b, D_v, alpha):
    """
    实现论文中的 Adaptive Hardness-Driven Dictionary Distillation Loss
    Z_b, Z_v: Teacher(b) 和 Student(v) 的潜在特征 [N, d]
    L_b, L_v: 稀疏表示 [N, d_b]
    D_b, D_v: 学习字典 [d_b, d]
    alpha: 由轮廓系数计算出的自适应调节因子[cite: 187]
    """
    # 1. 特征交叉重建 (Cross-Reconstruction)[cite: 186]
    # Z_b2v: 用 Teacher 的稀疏表示乘 Student 的字典
    Z_b2v = torch.matmul(L_b, D_v) 
    # Z_v2b: 用 Student 的稀疏表示乘 Teacher 的字典
    Z_v2b = torch.matmul(L_v, D_b)
    
    # 2. 计算融合特征 Z^f[cite: 186]
    Z_f = (Z_b2v + Z_v2b) / 2.0
    
    # 3. 计算相似度分布 Q^b 和 Q^v[cite: 186]
    # 为简化演示，这里使用余弦相似度并做 Softmax 归一化
    sim_b2v_f = F.cosine_similarity(Z_b2v.unsqueeze(1), Z_f.unsqueeze(0), dim=-1)
    sim_b_v = F.cosine_similarity(Z_b.unsqueeze(1), Z_v.unsqueeze(0), dim=-1)
    
    # 公式 (10): 计算概率分布[cite: 186]
    Q_b = torch.softmax(sim_b2v_f / (sim_b2v_f + sim_b_v + 1e-8), dim=1)
    
    sim_v2b_f = F.cosine_similarity(Z_v2b.unsqueeze(1), Z_f.unsqueeze(0), dim=-1)
    Q_v = torch.softmax(sim_v2b_f / (sim_v2b_f + sim_b_v + 1e-8), dim=1)
    
    # 4. 公式 (14): 自适应硬度驱动字典蒸馏损失 L_d[cite: 187]
    # 限制 alpha 避免分母为 0
    alpha = torch.clamp(torch.tensor(alpha), 1e-4, 1.0 - 1e-4)
    
    # \sum (Q_b^\alpha * Q_v^{1-\alpha})
    fused_Q = torch.sum(torch.pow(Q_b, alpha) * torch.pow(Q_v, 1.0 - alpha), dim=1)
    
    # L_d = 1 / (alpha * (alpha - 1)) * log(...)[cite: 187]
    loss_d = (1.0 / (alpha * (alpha - 1.0))) * torch.log(fused_Q.mean() + 1e-8)
    
    return loss_d

# ==========================================
# 模拟导盲犬场景：Teacher(RGB视觉已对齐) vs Student(激光雷达刚接入)
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_samples = 210
latent_dim = 128
dict_size = 64

print(f"📡 将张量挂载至 {device}...")
# 模拟上一节代码跑出来的特征和字典
Z_teacher = torch.rand(num_samples, latent_dim).to(device)
L_teacher = torch.rand(num_samples, dict_size).to(device)
D_teacher = torch.rand(dict_size, latent_dim).to(device)

Z_student = torch.rand(num_samples, latent_dim).to(device)
L_student = torch.rand(num_samples, dict_size).to(device)
D_student = torch.rand(dict_size, latent_dim).to(device)

# ==========================================
# 模拟轮廓系数 (Silhouette Coefficient) 与 Alpha 计算[cite: 187]
# ==========================================
# 假设晴天，Teacher (视觉) 聚类质量极好，SI_b 很高
SI_b = 0.85 
# 假设有雾，Student (雷达) 聚类质量差，SI_v 很低
SI_v = 0.35 

# 公式 (13): alpha = SI^b / (SI^b + SI^v)[cite: 187]
alpha = SI_b / (SI_b + SI_v)
print(f"🌡️  环境质量评估: Teacher SI={SI_b}, Student SI={SI_v}")
print(f"⚖️  计算得到自适应蒸馏因子 Alpha = {alpha:.4f}")

# 计算 AHDD Loss
loss_d = compute_ahdd_loss(Z_teacher, Z_student, L_teacher, L_student, D_teacher, D_student, alpha)

print(f"🎯 成功计算 AHDD 蒸馏损失 (Loss_d): {loss_d.item():.4f}")
print("🎉 ADAPT 论文的核心多视图动态知识蒸馏引擎已成功驱动！")
