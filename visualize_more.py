import os, sys, re, torch, numpy as np
import matplotlib
matplotlib.use('Agg')  # 服务器无显示器用Agg
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
import torch.nn as nn

# ============================================================
# 路径配置
# ============================================================
DATA_ROOT  = '/root/TransReID-main/data/MoRe-Final-Version'
CKPT_PATH  = '/root/TransReID-main/output/more/best_model.pth'
SAVE_DIR   = '/root/TransReID-main/output/visualizations'
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_H, IMG_W = 256, 128
# ============================================================

os.makedirs(SAVE_DIR, exist_ok=True)
print(f"设备: {DEVICE}")
print(f"保存目录: {SAVE_DIR}")

# ============================================================
# 1. 数据集解析（完全复用train_more.py的逻辑）
# ============================================================
def parse_files(txt_path, data_root):
    """解析train_files.txt / test_files.txt"""
    data = []
    pid_set = set()
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        rel_path = parts[0]
        abs_path = os.path.join(data_root, rel_path)
        if not os.path.exists(abs_path):
            abs_path = rel_path
        # 从文件名解析 pid 和 camid
        fname = os.path.basename(rel_path)
        # 格式: pairXX-camA_id_XXXXX_num_XXX.png
        m_id  = re.search(r'id_(\d+)', fname)
        m_cam = re.search(r'cam([AB])',  fname, re.IGNORECASE)
        pid   = int(m_id.group(1))  if m_id  else 0
        cam_char = m_cam.group(1).upper() if m_cam else 'A'
        camid = 0 if cam_char == 'A' else 1
        data.append((abs_path, pid, camid))
        pid_set.add(pid)
    return data, sorted(pid_set)

def build_query_gallery(test_data):
    """同train_more.py：camA做query，camB做gallery"""
    pid_cam = defaultdict(lambda: defaultdict(list))
    for path, pid, camid in test_data:
        pid_cam[pid][camid].append(path)

    query, gallery = [], []
    for pid, cam_dict in pid_cam.items():
        cams = sorted(cam_dict.keys())
        if len(cams) >= 2:
            # camA(0) → query, camB(1) → gallery
            for p in cam_dict[cams[0]]:
                query.append((p, pid, cams[0]))
            for p in cam_dict[cams[1]]:
                gallery.append((p, pid, cams[1]))
        else:
            for p in cam_dict[cams[0]]:
                gallery.append((p, pid, cams[0]))
    return query, gallery

print("\n解析数据集...")
train_data, train_pids = parse_files(
    os.path.join(DATA_ROOT, 'train_files.txt'), DATA_ROOT)
test_data,  test_pids  = parse_files(
    os.path.join(DATA_ROOT, 'test_files.txt'),  DATA_ROOT)
query, gallery = build_query_gallery(test_data)

num_classes = len(train_pids)
print(f"  train: {len(train_data)} imgs, {num_classes} IDs")
print(f"  query: {len(query)} imgs")
print(f"  gallery: {len(gallery)} imgs")

# ============================================================
# 2. 数据集 & 变换
# ============================================================
class ReIDDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        path, pid, camid = self.data[idx]
        try:
            img = Image.open(path).convert('RGB')
        except:
            img = Image.new('RGB', (IMG_W, IMG_H), (128,128,128))
        return self.transform(img), pid, camid, path

def get_test_tf():
    return transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])

# ============================================================
# 3. 加载模型（完全匹配train_more.py结构）
# ============================================================
print("\n加载模型...")
import timm

backbone = timm.create_model(
    'vit_base_patch16_224',
    pretrained=False,
    num_classes=0,
    img_size=(IMG_H, IMG_W)
)
embed_dim = backbone.embed_dim  # 768

class ReIDModel(nn.Module):
    def __init__(self, backbone, embed_dim, num_classes):
        super().__init__()
        self.backbone   = backbone
        self.bottleneck = nn.BatchNorm1d(embed_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(embed_dim, num_classes, bias=False)
    def forward(self, x):
        feat    = self.backbone(x)
        feat_bn = self.bottleneck(feat)
        return feat_bn

model = ReIDModel(backbone, embed_dim, num_classes).to(DEVICE)

ckpt = torch.load(CKPT_PATH, map_location='cpu')
sd   = ckpt['model']
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"  embed_dim={embed_dim}, num_classes={num_classes}")
print(f"  missing={len(missing)}, unexpected={len(unexpected)}")
model.eval()
print("  ✓ 模型加载成功")

# ============================================================
# 4. 特征提取
# ============================================================
@torch.no_grad()
def extract(data_list, desc=''):
    tf  = get_test_tf()
    ds  = ReIDDataset(data_list, tf)
    dl  = DataLoader(ds, batch_size=64, shuffle=False,
                     num_workers=2, pin_memory=True)
    feats, pids, camids, paths = [], [], [], []
    for imgs, pid, camid, path in dl:
        f = model(imgs.to(DEVICE)).cpu()
        f = torch.nn.functional.normalize(f, dim=1)
        feats.append(f)
        pids.extend(pid.tolist())
        camids.extend(camid.tolist())
        paths.extend(path)
    print(f"  {desc}: {len(pids)} 张")
    return torch.cat(feats), np.array(pids), np.array(camids), paths

print("\n提取特征...")
q_feat, q_pids, q_cams, q_paths = extract(query,   'query')
g_feat, g_pids, g_cams, g_paths = extract(gallery, 'gallery')

# 距离矩阵
dist = (1 - torch.mm(q_feat, g_feat.t())).numpy()
print(f"  距离矩阵: {dist.shape}")

# ============================================================
# 5. 计算CMC & mAP
# ============================================================
def compute_metrics(dist, q_pids, g_pids, q_cams, g_cams, max_rank=50):
    indices  = np.argsort(dist, axis=1)
    all_cmc, all_AP = [], []
    for qi in range(len(q_pids)):
        qpid, qcam = q_pids[qi], q_cams[qi]
        order = indices[qi]
        junk  = (g_pids[order] == qpid) & (g_cams[order] == qcam)
        keep  = ~junk
        ranked = (g_pids[order[keep]] == qpid).astype(int)
        if ranked.sum() == 0:
            continue
        cmc = ranked.cumsum(); cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_rel = ranked.sum()
        tmp = ranked.cumsum()
        prec = [tmp[i]/(i+1) for i in range(len(tmp))]
        AP = (np.array(prec) * ranked).sum() / num_rel
        all_AP.append(AP)
    cmc_arr = np.mean(all_cmc, axis=0)
    mAP     = float(np.mean(all_AP))
    return cmc_arr, mAP

cmc, mAP = compute_metrics(dist, q_pids, g_pids, q_cams, g_cams)
print(f"\n{'='*45}")
print(f"  Rank-1  : {cmc[0]*100:.2f}%")
print(f"  Rank-5  : {cmc[4]*100:.2f}%")
print(f"  Rank-10 : {cmc[9]*100:.2f}%")
print(f"  mAP     : {mAP*100:.2f}%")
print(f"{'='*45}")

# ============================================================
# 可视化A：CMC曲线
# ============================================================
print("\n[1/4] CMC曲线...")
fig, ax = plt.subplots(figsize=(9,6))
ranks = np.arange(1, len(cmc)+1)
ax.plot(ranks, cmc*100, 'b-o', lw=2.5, ms=4)
ax.fill_between(ranks, cmc*100, alpha=0.12, color='blue')
for r, col in zip([1,5,10,20],['#e74c3c','#e67e22','#27ae60','#8e44ad']):
    if r <= len(cmc):
        ax.axvline(r, color=col, ls='--', alpha=0.7, lw=1.5)
        ax.text(r+0.3, cmc[r-1]*100+1.5,
                f'R{r}:{cmc[r-1]*100:.1f}%',
                fontsize=9, color=col, fontweight='bold')
ax.set_xlabel('Rank', fontsize=13)
ax.set_ylabel('Recognition Rate (%)', fontsize=13)
ax.set_title(
    f'CMC Curve — MoRe Motorcycle Re-ID (ViT-Base)\n'
    f'Rank-1={cmc[0]*100:.2f}%  Rank-5={cmc[4]*100:.2f}%  '
    f'Rank-10={cmc[9]*100:.2f}%  mAP={mAP*100:.2f}%',
    fontsize=11, fontweight='bold')
ax.set_xlim(0, 50); ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3)
plt.tight_layout()
p = os.path.join(SAVE_DIR, '1_cmc_curve.png')
plt.savefig(p, dpi=150); plt.close()
print(f"  ✓ {p}")

# ============================================================
# 可视化B：Ranking结果
# ============================================================
print("[2/4] Ranking结果...")
def load_img(path):
    try:
        return np.array(
            Image.open(path).convert('RGB').resize((IMG_W, IMG_H)))
    except:
        return np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

indices_sorted = np.argsort(dist, axis=1)
TOP_K   = 10
N_QUERY = 8

# 选有正样本的query
selected = []
for qi in range(len(q_pids)):
    top_gpids = g_pids[indices_sorted[qi, :TOP_K]]
    top_gcams = g_cams[indices_sorted[qi, :TOP_K]]
    has_pos = any((top_gpids[k]==q_pids[qi]) and
                  (top_gcams[k]!=q_cams[qi])
                  for k in range(TOP_K))
    if has_pos:
        selected.append(qi)
    if len(selected) == N_QUERY:
        break
if len(selected) < N_QUERY:
    selected = list(range(min(N_QUERY, len(q_pids))))

fig, axes = plt.subplots(N_QUERY, TOP_K+1,
                         figsize=(2.3*(TOP_K+1), 3.0*N_QUERY))
fig.suptitle('MoRe Re-ID Ranking Results  '
             '(Green ✓ = Match,  Red ✗ = Wrong)',
             fontsize=15, fontweight='bold', y=1.005)

for row, qi in enumerate(selected):
    # Query列
    ax = axes[row, 0]
    ax.imshow(load_img(q_paths[qi]))
    ax.set_title(f'QUERY\nID:{q_pids[qi]}\nCam:{q_cams[qi]}',
                 fontsize=7, color='blue', fontweight='bold')
    for sp in ax.spines.values():
        sp.set_edgecolor('blue'); sp.set_linewidth(4)
    ax.set_xticks([]); ax.set_yticks([])

    # Top-K gallery列
    for col in range(TOP_K):
        gi  = indices_sorted[qi, col]
        ax  = axes[row, col+1]
        ax.imshow(load_img(g_paths[gi]))
        match = (g_pids[gi]==q_pids[qi]) and (g_cams[gi]!=q_cams[qi])
        color = '#00bb44' if match else '#dd2222'
        label = '✓' if match else '✗'
        ax.set_title(
            f'#{col+1} {label}\nID:{g_pids[gi]}\n'
            f'd={dist[qi,gi]:.3f}',
            fontsize=6.5, color=color)
        for sp in ax.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(3)
        ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
p = os.path.join(SAVE_DIR, '2_ranking_results.png')
plt.savefig(p, dpi=120, bbox_inches='tight'); plt.close()
print(f"  ✓ {p}")

# ============================================================
# 可视化C：t-SNE特征分布
# ============================================================
print("[3/4] t-SNE特征分布...")
try:
    from sklearn.manifold import TSNE
    MAX_IDS = 20
    uniq_pids = np.unique(q_pids)[:MAX_IDS]
    mask = np.isin(q_pids, uniq_pids)
    feat_sub = q_feat[mask].numpy()
    pids_sub = q_pids[mask]

    tsne = TSNE(n_components=2, random_state=42,
                perplexity=min(30, len(feat_sub)-1),
                n_iter=1000, verbose=0)
    emb = tsne.fit_transform(feat_sub)

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.get_cmap('tab20', MAX_IDS)
    for i, pid in enumerate(uniq_pids):
        idx = pids_sub == pid
        ax.scatter(emb[idx,0], emb[idx,1],
                   color=cmap(i), label=f'ID {pid}',
                   s=70, alpha=0.85, edgecolors='white', lw=0.5)
    ax.set_title(
        f't-SNE Feature Visualization\n'
        f'(Query set, top-{MAX_IDS} identities)',
        fontsize=13, fontweight='bold')
    ax.set_xlabel('Dim 1', fontsize=11)
    ax.set_ylabel('Dim 2', fontsize=11)
    ax.legend(bbox_to_anchor=(1.02,1), loc='upper left',
              fontsize=7, ncol=2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    p = os.path.join(SAVE_DIR, '3_tsne.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  ✓ {p}")
except ImportError:
    print("  跳过 (pip install scikit-learn)")

# ============================================================
# 可视化D：正负样本距离分布
# ============================================================
print("[4/4] 正负样本距离分布...")
pos_d, neg_d = [], []
for qi in range(len(q_pids)):
    qpid, qcam = q_pids[qi], q_cams[qi]
    for gi in range(len(g_pids)):
        if g_pids[gi]==qpid and g_cams[gi]==qcam:
            continue
        d = dist[qi, gi]
        if g_pids[gi] == qpid:
            pos_d.append(d)
        elif len(neg_d) < 80000:
            neg_d.append(d)

pos_d = np.array(pos_d)
neg_d = np.array(neg_d)
sep   = (neg_d.mean()-pos_d.mean()) / \
        np.sqrt((pos_d.std()**2+neg_d.std()**2)/2+1e-8)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# 直方图
ax = axes[0]
ax.hist(pos_d, bins=60, alpha=0.7, color='#2ecc71', density=True,
        label=f'Positive (n={len(pos_d):,})')
ax.hist(neg_d, bins=60, alpha=0.7, color='#e74c3c', density=True,
        label=f'Negative (n={len(neg_d):,})')
ax.axvline(pos_d.mean(), color='green', ls='--', lw=2,
           label=f'Pos μ={pos_d.mean():.3f}')
ax.axvline(neg_d.mean(), color='red',   ls='--', lw=2,
           label=f'Neg μ={neg_d.mean():.3f}')
ax.set_xlabel('Cosine Distance', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Distance Distribution', fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
# Box plot
ax = axes[1]
bp = ax.boxplot([pos_d, neg_d],
                labels=['Positive\n(same ID)','Negative\n(diff ID)'],
                patch_artist=True,
                medianprops=dict(color='black', lw=2.5))
bp['boxes'][0].set_facecolor('#2ecc71')
bp['boxes'][1].set_facecolor('#e74c3c')
ax.set_ylabel('Cosine Distance', fontsize=12)
ax.set_title('Box Plot', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
fig.suptitle(
    f'Feature Discriminability  |  d-prime={sep:.3f}  (higher=better)',
    fontsize=13, fontweight='bold')
plt.tight_layout()
p = os.path.join(SAVE_DIR, '4_distance_dist.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"  ✓ {p}")
print(f"  正样本距离均值: {pos_d.mean():.4f}")
print(f"  负样本距离均值: {neg_d.mean():.4f}")
print(f"  分离度 d-prime: {sep:.4f}")

# ============================================================
# 最终汇总
# ============================================================
print(f"\n{'='*45}")
print("  全部可视化完成！")
print(f"  Rank-1  : {cmc[0]*100:.2f}%")
print(f"  Rank-5  : {cmc[4]*100:.2f}%")
print(f"  Rank-10 : {cmc[9]*100:.2f}%")
print(f"  mAP     : {mAP*100:.2f}%")
print(f"  图片目录: {SAVE_DIR}")
print(f"{'='*45}")
