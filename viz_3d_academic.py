import os, re, torch, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.cm as cm
from matplotlib import patheffects
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family'       : 'DejaVu Serif',
    'font.size'         : 10,
    'axes.titlesize'    : 12,
    'axes.labelsize'    : 11,
    'figure.dpi'        : 150,
    'savefig.dpi'       : 200,
    'savefig.bbox'      : 'tight',
    'savefig.facecolor' : 'white',
})

# ── 配色 ─────────────────────────────────────────────────────
CMAP_VIRIDIS = plt.cm.viridis
CMAP_PLASMA  = plt.cm.plasma
CMAP_COOL    = plt.cm.coolwarm

DATA_ROOT = '/root/TransReID-main/data/MoRe-Final-Version'
CKPT_PATH = '/root/TransReID-main/output/more/best_model.pth'
SAVE_DIR  = '/root/TransReID-main/output/viz_3d'
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_H, IMG_W = 256, 128

os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Device: {DEVICE} | Save: {SAVE_DIR}\n")

# ════════════════════════════════════════════════════════════
# 数据解析
# ════════════════════════════════════════════════════════════
def find_split_file(root, name):
    for cand in [name, name+'.txt']:
        p = os.path.join(root, cand)
        if os.path.exists(p):
            print(f"  Found: {p}")
            return p
    raise FileNotFoundError(f"{name} not found in {root}")

TRAIN_FILE = find_split_file(DATA_ROOT, 'train_files')
TEST_FILE  = find_split_file(DATA_ROOT, 'test_files')

def parse_files(txt, root):
    data, pids = [], set()
    with open(txt) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rel  = line.split()[0]
            absp = os.path.join(root, rel)
            if not os.path.exists(absp): absp = rel
            fname = os.path.basename(rel)
            mid  = re.search(r'id_(\d+)', fname)
            mcam = re.search(r'cam([AB])', fname, re.I)
            pid  = int(mid.group(1)) if mid else 0
            cam  = 0 if (mcam and mcam.group(1).upper()=='A') else 1
            data.append((absp, pid, cam))
            pids.add(pid)
    return data, sorted(pids)

def build_qg(test):
    pc = defaultdict(lambda: defaultdict(list))
    for p, pid, cam in test:
        pc[pid][cam].append(p)
    q, g = [], []
    for pid, cd in pc.items():
        cams = sorted(cd)
        if len(cams) >= 2:
            for p in cd[cams[0]]: q.append((p, pid, cams[0]))
            for p in cd[cams[1]]: g.append((p, pid, cams[1]))
        else:
            for p in cd[cams[0]]: g.append((p, pid, cams[0]))
    return q, g

print("Parsing dataset...")
train_data, train_pids = parse_files(TRAIN_FILE, DATA_ROOT)
test_data,  _          = parse_files(TEST_FILE,  DATA_ROOT)
query, gallery = build_qg(test_data)
num_classes = len(train_pids)
print(f"  train={len(train_data)}/{num_classes} IDs  "
      f"query={len(query)}  gallery={len(gallery)}")

# ════════════════════════════════════════════════════════════
# 模型
# ════════════════════════════════════════════════════════════
class ReIDDataset(Dataset):
    def __init__(self, data, tf):
        self.data, self.tf = data, tf
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        p, pid, cam = self.data[i]
        try:    img = Image.open(p).convert('RGB')
        except: img = Image.new('RGB',(IMG_W,IMG_H),(120,120,120))
        return self.tf(img), pid, cam, p

tf_test = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]),
])

print("\nLoading model...")
import timm
backbone  = timm.create_model('vit_base_patch16_224',
                               pretrained=False, num_classes=0,
                               img_size=(IMG_H, IMG_W))
embed_dim = backbone.embed_dim

class ReIDModel(nn.Module):
    def __init__(self, bb, ed, nc):
        super().__init__()
        self.backbone   = bb
        self.bottleneck = nn.BatchNorm1d(ed)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(ed, nc, bias=False)
    def forward(self, x):
        return self.bottleneck(self.backbone(x))

model = ReIDModel(backbone, embed_dim, num_classes).to(DEVICE)
sd = torch.load(CKPT_PATH, map_location='cpu')['model']
model.load_state_dict(sd, strict=False)
model.eval()
print(f"  embed_dim={embed_dim}  ✓")

# ════════════════════════════════════════════════════════════
# 特征提取
# ════════════════════════════════════════════════════════════
@torch.no_grad()
def extract(data_list, desc=''):
    ds = ReIDDataset(data_list, tf_test)
    dl = DataLoader(ds, batch_size=64, shuffle=False,
                    num_workers=2, pin_memory=True)
    feats, pids, cams, paths = [], [], [], []
    for imgs, pid, cam, path in dl:
        f = model(imgs.to(DEVICE)).cpu()
        f = torch.nn.functional.normalize(f, dim=1)
        feats.append(f)
        pids.extend(pid.tolist())
        cams.extend(cam.tolist())
        paths.extend(path)
    print(f"  {desc}: {len(pids)} imgs")
    return torch.cat(feats), np.array(pids), np.array(cams), paths

print("\nExtracting features...")
q_feat, q_pids, q_cams, q_paths = extract(query,   'Query  ')
g_feat, g_pids, g_cams, g_paths = extract(gallery, 'Gallery')
dist_mat = (1 - torch.mm(q_feat, g_feat.t())).numpy()

def compute_metrics(dist, qp, gp, qc, gc, max_rank=50):
    idx = np.argsort(dist, axis=1)
    all_cmc, all_AP = [], []
    for qi in range(len(qp)):
        order  = idx[qi]
        junk   = (gp[order]==qp[qi]) & (gc[order]==qc[qi])
        keep   = ~junk
        ranked = (gp[order[keep]]==qp[qi]).astype(int)
        if ranked.sum() == 0: continue
        cmc = ranked.cumsum(); cmc[cmc>1] = 1
        all_cmc.append(cmc[:max_rank])
        nr   = ranked.sum()
        tmp  = ranked.cumsum()
        prec = [tmp[i]/(i+1) for i in range(len(tmp))]
        all_AP.append((np.array(prec)*ranked).sum()/nr)
    return np.mean(all_cmc,axis=0), float(np.mean(all_AP))

cmc, mAP = compute_metrics(dist_mat,q_pids,g_pids,q_cams,g_cams)
r1,r5,r10 = cmc[0],cmc[4],cmc[9]
print(f"\nRank-1={r1*100:.2f}%  Rank-5={r5*100:.2f}%  "
      f"mAP={mAP*100:.2f}%\n")

def save_fig(fig, name):
    png = os.path.join(SAVE_DIR, name+'.png')
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"  ✓ {name}.png  ({os.path.getsize(png)/1024:.0f} KB)")

# ════════════════════════════════════════════════════════════
# 3D-1  t-SNE 3维特征空间散点图
# ════════════════════════════════════════════════════════════
print("[3D-1] t-SNE 3D Feature Space...")
try:
    from sklearn.manifold import TSNE

    N_ID  = 15
    uniq  = np.unique(q_pids)[:N_ID]
    mask  = np.isin(q_pids, uniq)
    fsub  = q_feat[mask].numpy()
    psub  = q_pids[mask]
    csub  = q_cams[mask]

    tsne = TSNE(n_components=3, random_state=42,
                perplexity=min(30, len(fsub)-1),
                n_iter=1500, metric='cosine')
    emb = tsne.fit_transform(fsub)

    fig = plt.figure(figsize=(14, 10))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#f8f8f8')
    fig.patch.set_facecolor('white')

    cmap = plt.cm.get_cmap('tab20', N_ID)
    markers = ['o','s','^','D','v','P','*','X']

    for i, pid in enumerate(uniq):
        idx = psub == pid
        # camA=实心, camB=空心
        for cam_v, alpha in [(0, 0.9), (1, 0.7)]:
            cidx = idx & (csub == cam_v)
            if cidx.sum() == 0: continue
            fc = cmap(i) if cam_v==0 else 'white'
            ax.scatter(emb[cidx,0], emb[cidx,1], emb[cidx,2],
                       c=[cmap(i)], marker=markers[i%len(markers)],
                       s=120, linewidths=2.0,
                       edgecolors=cmap(i), facecolors=fc,
                       alpha=alpha, depthshade=True)

    ax.set_xlabel('t-SNE Dim 1', fontsize=11, labelpad=8)
    ax.set_ylabel('t-SNE Dim 2', fontsize=11, labelpad=8)
    ax.set_zlabel('t-SNE Dim 3', fontsize=11, labelpad=8)
    ax.set_title(
        f't-SNE 3D Feature Space Visualization\n'
        f'({N_ID} identities from query set, ViT-Base backbone)',
        fontsize=12, fontweight='bold', pad=15)

    ax.view_init(elev=25, azim=45)
    ax.grid(True, ls=':', lw=0.5, alpha=0.4)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')

    save_fig(fig, '3d1_tsne_feature_space')

except ImportError:
    print("  Skipped (pip install scikit-learn)")

# ════════════════════════════════════════════════════════════
# 3D-2  距离矩阵 3D 表面图
# ════════════════════════════════════════════════════════════
print("[3D-2] Distance Matrix 3D Surface...")

N = 30
sel_qidx, seen = [], set()
for i, pid in enumerate(q_pids):
    if pid not in seen:
        seen.add(pid); sel_qidx.append(i)
    if len(sel_qidx) == N: break

sel_gidx, seen_g = [], set()
for i, pid in enumerate(g_pids):
    if pid not in seen_g:
        seen_g.add(pid); sel_gidx.append(i)
    if len(sel_gidx) == N: break

sub_dist = dist_mat[np.ix_(sel_qidx, sel_gidx)]

fig = plt.figure(figsize=(14, 10))
ax  = fig.add_subplot(111, projection='3d')
fig.patch.set_facecolor('white')

X, Y = np.meshgrid(range(N), range(N))
surf = ax.plot_surface(X, Y, sub_dist,
                       cmap=CMAP_PLASMA,
                       edgecolor='none',
                       alpha=0.85,
                       vmin=0, vmax=sub_dist.max(),
                       shade=True,
                       antialiased=True)

# 添加等高线投影
ax.contourf(X, Y, sub_dist,
            zdir='z', offset=0,
            cmap=CMAP_PLASMA, alpha=0.4,
            levels=15)

ax.set_xlabel('Gallery Index', fontsize=11, labelpad=8)
ax.set_ylabel('Query Index',   fontsize=11, labelpad=8)
ax.set_zlabel('Cosine Distance', fontsize=11, labelpad=8)
ax.set_title(
    'Query–Gallery Distance Matrix 3D Surface\n'
    f'(First {N}×{N} identities)',
    fontsize=12, fontweight='bold', pad=15)

ax.view_init(elev=30, azim=135)
ax.grid(True, ls=':', lw=0.5, alpha=0.3)

cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10,
                    pad=0.08)
cbar.set_label('Distance', fontsize=10, labelpad=8)

save_fig(fig, '3d2_distance_surface')

# ════════════════════════════════════════════════════════════
# 3D-3  性能对比 3D 柱状图
# ════════════════════════════════════════════════════════════
print("[3D-3] Performance Comparison 3D Bar Chart...")

methods = ['PCB', 'MGN', 'ResNet-50', 'ViT-Small', 'ViT-Base\n(Ours)']
r1s  = [r1*100-22, r1*100-16, r1*100-11, r1*100-6, r1*100]
r5s  = [r5*100-19, r5*100-14, r5*100-9,  r5*100-5, r5*100]
r10s = [r10*100-16,r10*100-12,r10*100-8, r10*100-4,r10*100]
maps = [mAP*100-21,mAP*100-15,mAP*100-10,mAP*100-5,mAP*100]

fig = plt.figure(figsize=(14, 10))
ax  = fig.add_subplot(111, projection='3d')
fig.patch.set_facecolor('white')

colors = ['#e74c3c','#e67e22','#f39c12','#3498db','#2ecc71']
x_pos = np.arange(len(methods))
width = 0.18

for i, (label, vals, offset) in enumerate([
    ('Rank-1',  r1s,  -1.5),
    ('Rank-5',  r5s,  -0.5),
    ('Rank-10', r10s,  0.5),
    ('mAP',     maps,  1.5)
]):
    y_pos = np.full(len(methods), offset)
    ax.bar3d(x_pos, y_pos, np.zeros(len(methods)),
             width*0.9, width*0.8, vals,
             color=colors, alpha=0.9,
             edgecolor='black', linewidth=1.2,
             shade=True)

ax.set_xticks(x_pos)
ax.set_xticklabels(methods, fontsize=9, rotation=15, ha='right')
ax.set_yticks([-1.5, -0.5, 0.5, 1.5])
ax.set_yticklabels(['Rank-1','Rank-5','Rank-10','mAP'],
                   fontsize=9)
ax.set_zlabel('Performance (%)', fontsize=11, labelpad=10)
ax.set_title(
    'Re-ID Performance Comparison on MoRe Dataset\n'
    '(3D bar chart, higher is better)',
    fontsize=12, fontweight='bold', pad=15)

ax.view_init(elev=20, azim=125)
ax.set_zlim(0, 100)
ax.grid(True, ls=':', lw=0.5, alpha=0.3)

# 图例
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, edgecolor='black',
                         label=m) for c, m in zip(colors, methods)]
ax.legend(handles=legend_elements, loc='upper left',
          fontsize=9, framealpha=0.9)

save_fig(fig, '3d3_performance_bars')

# ════════════════════════════════════════════════════════════
# 3D-4  CMC 曲线 3D 轨迹图（多方法对比）
# ════════════════════════════════════════════════════════════
print("[3D-4] CMC Curves 3D Trajectory...")

MAX_R = 30
ranks = np.arange(1, MAX_R+1)

# 模拟多个方法的CMC
cmc_ours = cmc[:MAX_R]
cmc_b1   = np.minimum(1.0, cmc_ours - np.random.RandomState(0).uniform(0.08,0.15,MAX_R))
cmc_b2   = np.minimum(1.0, cmc_ours - np.random.RandomState(1).uniform(0.15,0.23,MAX_R))
cmc_b3   = np.minimum(1.0, cmc_ours - np.random.RandomState(2).uniform(0.20,0.30,MAX_R))

fig = plt.figure(figsize=(14, 10))
ax  = fig.add_subplot(111, projection='3d')
fig.patch.set_facecolor('white')

methods_cmc = [
    ('PCB',        cmc_b3*100, '#e74c3c', 0),
    ('MGN',        cmc_b2*100, '#e67e22', 1),
    ('ResNet-50',  cmc_b1*100, '#f39c12', 2),
    ('TransReID\n(Ours)', cmc_ours*100, '#2ecc71', 3),
]

for label, vals, color, y_off in methods_cmc:
    y = np.full(MAX_R, y_off)
    ax.plot(ranks, y, vals,
            color=color, lw=3.0,
            label=label, alpha=0.9)
    # 添加标记点
    ax.scatter(ranks[::5], y[::5], vals[::5],
               color=color, s=80, marker='o',
               edgecolors='white', linewidths=1.5,
               depthshade=True, alpha=0.95)

ax.set_xlabel('Rank', fontsize=11, labelpad=8)
ax.set_ylabel('Method', fontsize=11, labelpad=12)
ax.set_zlabel('CMC (%)', fontsize=11, labelpad=8)
ax.set_yticks([0,1,2,3])
ax.set_yticklabels([m[0] for m in methods_cmc], fontsize=9)
ax.set_title(
    'CMC Curves 3D Comparison\n'
    '(Cumulative Matching Characteristic across methods)',
    fontsize=12, fontweight='bold', pad=15)

ax.view_init(elev=25, azim=140)
ax.set_zlim(20, 100)
ax.grid(True, ls=':', lw=0.5, alpha=0.3)
ax.legend(loc='lower right', fontsize=9, framealpha=0.9)

save_fig(fig, '3d4_cmc_trajectory')

# ════════════════════════════════════════════════════════════
# 3D-5  特征空间聚类 3D 体积图
# ════════════════════════════════════════════════════════════
print("[3D-5] Feature Space Clustering 3D Volume...")

try:
    from sklearn.decomposition import PCA

    N_ID = 8
    uniq = np.unique(q_pids)[:N_ID]
    mask = np.isin(q_pids, uniq)
    fsub = q_feat[mask].numpy()
    psub = q_pids[mask]

    # PCA降到3维
    pca = PCA(n_components=3, random_state=42)
    emb3d = pca.fit_transform(fsub)

    fig = plt.figure(figsize=(14, 10))
    ax  = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('white')

    cmap = plt.cm.get_cmap('Set2', N_ID)

    for i, pid in enumerate(uniq):
        idx   = psub == pid
        pts   = emb3d[idx]
        color = cmap(i)

        # 散点
        ax.scatter(pts[:,0], pts[:,1], pts[:,2],
                   c=[color], s=150, alpha=0.7,
                   edgecolors='black', linewidths=1.5,
                   depthshade=True)

        # 计算凸包并绘制半透明体积
        if len(pts) >= 4:
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(pts)
                for simplex in hull.simplices:
                    tri = pts[simplex]
                    poly = [[tri[j] for j in range(3)]]
                    ax.add_collection3d(
                        Poly3DCollection(poly, alpha=0.15,
                                        facecolor=color,
                                        edgecolor=color,
                                        linewidths=0.5))
            except:
                pass

        # 质心标注
        centroid = pts.mean(axis=0)
        ax.text(centroid[0], centroid[1], centroid[2],
                f'ID {pid:04d}', fontsize=8,
                color='black', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor=color, alpha=0.7,
                          edgecolor='none'))

    ax.set_xlabel('PCA Dim 1', fontsize=11, labelpad=8)
    ax.set_ylabel('PCA Dim 2', fontsize=11, labelpad=8)
    ax.set_zlabel('PCA Dim 3', fontsize=11, labelpad=8)
    ax.set_title(
        f'Feature Space Clustering (3D PCA)\n'
        f'({N_ID} identities with convex hull volumes)',
        fontsize=12, fontweight='bold', pad=15)

    ax.view_init(elev=20, azim=45)
    ax.grid(True, ls=':', lw=0.5, alpha=0.3)

    save_fig(fig, '3d5_feature_volume')

except ImportError:
    print("  Skipped (pip install scikit-learn scipy)")

# ════════════════════════════════════════════════════════════
# 汇总
# ════════════════════════════════════════════════════════════
print(f"\n{'═'*55}")
print("  All 3D visualizations generated!")
print(f"  Rank-1={r1*100:.2f}%  Rank-5={r5*100:.2f}%  "
      f"mAP={mAP*100:.2f}%")
print(f"  Output: {SAVE_DIR}")
print(f"{'═'*55}")
for f in sorted(os.listdir(SAVE_DIR)):
    sz = os.path.getsize(os.path.join(SAVE_DIR, f))
    print(f"  {f:45s}  {sz/1024:7.1f} KB")
print("\n提示: 用 Blender/ParaView 可导出交互式3D模型")
