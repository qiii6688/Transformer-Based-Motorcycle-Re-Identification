import os, re, torch, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family'       : 'DejaVu Serif',
    'font.size'         : 11,
    'axes.titlesize'    : 12,
    'axes.labelsize'    : 11,
    'axes.linewidth'    : 1.2,
    'xtick.direction'   : 'in',
    'ytick.direction'   : 'in',
    'xtick.major.size'  : 4,
    'ytick.major.size'  : 4,
    'figure.dpi'        : 150,
    'savefig.dpi'       : 200,
    'savefig.bbox'      : 'tight',
    'savefig.facecolor' : 'white',
})

CMAP_SIM  = LinearSegmentedColormap.from_list('sim', [
    '#ffffff','#d1e5f0','#92c5de','#4393c3','#2166ac','#053061'])
CMAP_DIST = LinearSegmentedColormap.from_list('dist', [
    '#053061','#2166ac','#4393c3','#92c5de',
    '#f7f7f7','#f4a582','#d6604d','#b2182b','#67001f'])
CMAP_CONF = LinearSegmentedColormap.from_list('conf', [
    '#f7fbff','#c6dbef','#6baed6','#2171b5','#08306b'])
CMAP_CORR = LinearSegmentedColormap.from_list('corr', [
    '#053061','#2166ac','#4393c3','#d1e5f0',
    '#f7f7f7','#fddbc7','#d6604d','#b2182b','#67001f'])
CMAP_ATTN = LinearSegmentedColormap.from_list('attn', [
    '#000004','#3b0f70','#8c2981','#de4968','#fe9f6d','#fcfdbf'])

# ── 自动探测文件路径 ──────────────────────────────────────────
DATA_ROOT = '/root/TransReID-main/data/MoRe-Final-Version'
CKPT_PATH = '/root/TransReID-main/output/more/best_model.pth'
SAVE_DIR  = '/root/TransReID-main/output/heatmaps'
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_H, IMG_W = 256, 128

os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Device: {DEVICE} | Save: {SAVE_DIR}")

# 自动探测 train/test 文件名
def find_split_file(root, name):
    for candidate in [name, name+'.txt', name+'.lst']:
        p = os.path.join(root, candidate)
        if os.path.exists(p):
            print(f"  Found: {p}")
            return p
    raise FileNotFoundError(
        f"Cannot find {name}[/.txt/.lst] in {root}\n"
        f"  Files: {os.listdir(root)}")

TRAIN_FILE = find_split_file(DATA_ROOT, 'train_files')
TEST_FILE  = find_split_file(DATA_ROOT, 'test_files')

# ════════════════════════════════════════════════════════════
# 数据解析
# ════════════════════════════════════════════════════════════
def parse_files(txt, root):
    data, pids = [], set()
    with open(txt, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line: continue
        rel  = line.split()[0]
        # 尝试多种路径拼接
        absp = os.path.join(root, rel)
        if not os.path.exists(absp):
            absp = rel
        if not os.path.exists(absp):
            absp = os.path.join(root, os.path.basename(rel))
        fname = os.path.basename(rel)
        mid  = re.search(r'id_(\d+)', fname)
        mcam = re.search(r'cam([AB])', fname, re.I)
        pid  = int(mid.group(1)) if mid else 0
        cam  = 0 if (mcam and mcam.group(1).upper()=='A') else 1
        data.append((absp, pid, cam))
        pids.add(pid)
    print(f"  Parsed {len(data)} lines → {len(pids)} IDs")
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

print("\nParsing dataset...")
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
sim_mat  = 1 - dist_mat
indices  = np.argsort(dist_mat, axis=1)
print(f"  dist_mat: {dist_mat.shape}")

# ════════════════════════════════════════════════════════════
# 评估
# ════════════════════════════════════════════════════════════
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
r1,r5,r10,r20 = cmc[0],cmc[4],cmc[9],cmc[19]
print(f"\nRank-1={r1*100:.2f}%  Rank-5={r5*100:.2f}%  "
      f"Rank-10={r10*100:.2f}%  mAP={mAP*100:.2f}%\n")

def save_fig(fig, name):
    pdf = os.path.join(SAVE_DIR, name+'.pdf')
    png = os.path.join(SAVE_DIR, name+'.png')
    fig.savefig(pdf, format='pdf')
    fig.savefig(png)
    plt.close(fig)
    kb = os.path.getsize(png)/1024
    print(f"  ✓ {name}.png  ({kb:.0f} KB)")

# ════════════════════════════════════════════════════════════
# HM-1  Query–Gallery 相似度矩阵
# ════════════════════════════════════════════════════════════
print("[HM-1] Query–Gallery Similarity Matrix...")
N_SHOW = 40
sel_qidx, seen = [], set()
for i, pid in enumerate(q_pids):
    if pid not in seen: seen.add(pid); sel_qidx.append(i)
    if len(sel_qidx) == N_SHOW: break
sel_gidx, seen_g = [], set()
for i, pid in enumerate(g_pids):
    if pid not in seen_g: seen_g.add(pid); sel_gidx.append(i)
    if len(sel_gidx) == N_SHOW: break

sub_sim = sim_mat[np.ix_(sel_qidx, sel_gidx)]
sub_qp  = q_pids[sel_qidx]
sub_gp  = g_pids[sel_gidx]
gt_mask = (sub_qp[:,None] == sub_gp[None,:])

fig = plt.figure(figsize=(14, 11))
gs  = gridspec.GridSpec(1, 2, width_ratios=[20,1], wspace=0.04)
ax  = fig.add_subplot(gs[0])
cax = fig.add_subplot(gs[1])
im  = ax.imshow(sub_sim, cmap=CMAP_SIM, aspect='auto',
                vmin=0.0, vmax=1.0, interpolation='nearest')
for qi in range(N_SHOW):
    for gi in range(N_SHOW):
        if gt_mask[qi, gi]:
            ax.add_patch(plt.Rectangle(
                (gi-.5,qi-.5),1,1, fill=False,
                edgecolor='#ff3333', lw=1.6, zorder=5))
step = 5
ax.set_xticks(range(0,N_SHOW,step))
ax.set_xticklabels([f'{sub_gp[i]:04d}' for i in range(0,N_SHOW,step)],
                   fontsize=8, rotation=45, ha='right')
ax.set_yticks(range(0,N_SHOW,step))
ax.set_yticklabels([f'{sub_qp[i]:04d}' for i in range(0,N_SHOW,step)],
                   fontsize=8)
ax.set_xlabel('Gallery Identity ID', fontsize=12, labelpad=8)
ax.set_ylabel('Query Identity ID',   fontsize=12, labelpad=8)
ax.set_title('Query–Gallery Cosine Similarity Matrix\n'
             '(Red boxes indicate ground-truth matches)',
             fontsize=13, fontweight='bold', pad=12)
ax.set_xticks(np.arange(-.5,N_SHOW,1), minor=True)
ax.set_yticks(np.arange(-.5,N_SHOW,1), minor=True)
ax.grid(which='minor', color='white', lw=0.3, alpha=0.4)
cb = fig.colorbar(im, cax=cax)
cb.set_label('Cosine Similarity', fontsize=11, labelpad=10)
handles = [
    mpatches.Patch(facecolor='#053061', label='Low similarity'),
    mpatches.Patch(facecolor='#4393c3', label='Medium'),
    mpatches.Patch(facecolor='#d1e5f0', label='High similarity'),
    plt.Line2D([0],[0], color='#ff3333', lw=2,
               label='Ground-truth match'),
]
ax.legend(handles=handles, loc='upper right', fontsize=9)
ax.text(0.02, 0.02,
        f'Rank-1 : {r1*100:.2f}%\n'
        f'Rank-5 : {r5*100:.2f}%\n'
        f'Rank-10: {r10*100:.2f}%\n'
        f'mAP    : {mAP*100:.2f}%',
        transform=ax.transAxes, fontsize=9, va='bottom',
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5',
                  fc='white', ec='#2166ac', alpha=0.93))
save_fig(fig, 'hm1_similarity_matrix')

# ════════════════════════════════════════════════════════════
# HM-2  类内 vs 类间距离块状热力图
# ════════════════════════════════════════════════════════════
print("[HM-2] Intra/Inter-class Distance Block Heatmap...")
N_ID, MAX_PER = 20, 3
all_feat = torch.cat([q_feat, g_feat], dim=0)
all_pids = np.concatenate([q_pids, g_pids])
pid_to_idx = defaultdict(list)
for i, pid in enumerate(all_pids):
    if pid != -1: pid_to_idx[pid].append(i)

sel_pids = sorted(pid_to_idx.keys())[:N_ID]
sel_idx, sel_plabel, bounds = [], [], []
for pid in sel_pids:
    idxs = pid_to_idx[pid][:MAX_PER]
    bounds.append(len(sel_idx))
    sel_idx.extend(idxs)
    sel_plabel.extend([pid]*len(idxs))
bounds.append(len(sel_idx))

sf   = all_feat[sel_idx]
sdst = (1 - torch.mm(sf, sf.t())).numpy()
sp   = np.array(sel_plabel)
NT   = len(sel_idx)

fig = plt.figure(figsize=(13, 11))
gs  = gridspec.GridSpec(1, 2, width_ratios=[20,1], wspace=0.04)
ax  = fig.add_subplot(gs[0])
cax = fig.add_subplot(gs[1])
im  = ax.imshow(sdst, cmap=CMAP_DIST, aspect='auto',
                vmin=0, vmax=1.2, interpolation='nearest')
for b in bounds[1:-1]:
    ax.axhline(b-.5, color='white', lw=2.0)
    ax.axvline(b-.5, color='white', lw=2.0)
for k in range(N_ID):
    s, e = bounds[k], bounds[k+1]
    if e > s:
        ax.add_patch(plt.Rectangle(
            (s-.5,s-.5), e-s, e-s,
            fill=False, edgecolor='#ffdd00',
            lw=2.2, ls='--', zorder=6))
mids = [(bounds[k]+bounds[k+1])/2 for k in range(N_ID)]
ax.set_xticks(mids)
ax.set_xticklabels([f'{p:04d}' for p in sel_pids],
                   fontsize=7.5, rotation=60, ha='right')
ax.set_yticks(mids)
ax.set_yticklabels([f'{p:04d}' for p in sel_pids], fontsize=7.5)
ax.set_xlabel('Identity ID', fontsize=12, labelpad=8)
ax.set_ylabel('Identity ID', fontsize=12, labelpad=8)
ax.set_title(
    'Intra- and Inter-class Cosine Distance Matrix\n'
    f'({N_ID} identities · {MAX_PER} samples/ID  '
    '— yellow dashes mark intra-class blocks)',
    fontsize=12, fontweight='bold', pad=12)
cb = fig.colorbar(im, cax=cax)
cb.set_label('Cosine Distance', fontsize=11, labelpad=10)
intra, inter = [], []
for i in range(NT):
    for j in range(i+1, NT):
        d = sdst[i,j]
        if sp[i]==sp[j]: intra.append(d)
        else:            inter.append(d)
im_ = np.mean(intra) if intra else 0
ie_ = np.mean(inter) if inter else 0
ax.text(0.02, 0.98,
        f'Intra-class μ : {im_:.4f}\n'
        f'Inter-class μ : {ie_:.4f}\n'
        f'Margin        : {ie_-im_:.4f}',
        transform=ax.transAxes, fontsize=9.5,
        va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5',
                  fc='white', ec='#b2182b', alpha=0.93))
save_fig(fig, 'hm2_intra_inter_class')

# ════════════════════════════════════════════════════════════
# HM-3  Per-Identity Rank-K 命中率
# ════════════════════════════════════════════════════════════
print("[HM-3] Per-Identity Rank-K Hit Rate...")
MAX_RANK = 20
pid_to_qi = defaultdict(list)
for i, pid in enumerate(q_pids): pid_to_qi[pid].append(i)
sel_qpids  = sorted(pid_to_qi.keys())[:40]
hit_mat    = np.zeros((len(sel_qpids), MAX_RANK))
for pi, pid in enumerate(sel_qpids):
    qidxs = pid_to_qi[pid]
    for qi in qidxs:
        order  = indices[qi]
        junk   = (g_pids[order]==pid) & (g_cams[order]==q_cams[qi])
        keep   = ~junk
        ranked = (g_pids[order[keep]]==pid).astype(int)
        cmc    = ranked[:MAX_RANK].cumsum()
        cmc[cmc>1] = 1
        hit_mat[pi] += cmc[:MAX_RANK]
    if len(qidxs): hit_mat[pi] /= len(qidxs)
hit_mat *= 100

fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(1, 2, width_ratios=[20,1], wspace=0.04)
ax  = fig.add_subplot(gs[0])
cax = fig.add_subplot(gs[1])
im  = ax.imshow(hit_mat, cmap=CMAP_CONF, aspect='auto',
                vmin=0, vmax=100, interpolation='nearest')
for pi in range(len(sel_qpids)):
    for ri in range(MAX_RANK):
        if ri % 4 == 0:
            v = hit_mat[pi,ri]
            c = 'white' if v>55 else '#222222'
            ax.text(ri, pi, f'{v:.0f}',
                    ha='center', va='center',
                    fontsize=6.5, color=c, fontweight='bold')
ax.set_xticks(range(MAX_RANK))
ax.set_xticklabels([f'R{i+1}' for i in range(MAX_RANK)], fontsize=9)
ax.set_yticks(range(len(sel_qpids)))
ax.set_yticklabels([f'ID {p:04d}' for p in sel_qpids], fontsize=7.5)
ax.set_xlabel('Rank', fontsize=12, labelpad=8)
ax.set_ylabel('Query Identity', fontsize=12, labelpad=8)
ax.set_title('Per-Identity Cumulative Hit Rate at Each Rank (%)\n'
             '(Darker blue = higher recognition rate)',
             fontsize=12, fontweight='bold', pad=12)
cb = fig.colorbar(im, cax=cax)
cb.set_label('Hit Rate (%)', fontsize=11, labelpad=10)
avg_row = hit_mat.mean(axis=0)
ax2 = ax.twinx()
ax2.plot(range(MAX_RANK), avg_row, 'o-',
         color='#ff6b35', lw=2.2, ms=5,
         label='Average CMC', zorder=10)
ax2.set_ylim(0, 110)
ax2.set_ylabel('Average CMC (%)', fontsize=11,
               color='#ff6b35', labelpad=8)
ax2.tick_params(axis='y', colors='#ff6b35')
ax2.spines['right'].set_color('#ff6b35')
ax2.legend(loc='lower right', fontsize=9)
save_fig(fig, 'hm3_per_identity_rank')

# ════════════════════════════════════════════════════════════
# HM-4  特征相关性（彻底修复对称问题）
# ════════════════════════════════════════════════════════════
print("[HM-4] Feature Correlation Heatmap...")
N_SAMP = min(256, len(q_feat))
N_DIM  = 64
feat_s = q_feat[:N_SAMP, :N_DIM].numpy().astype(np.float64)

# 手动计算相关矩阵（完全对称，无浮点误差）
feat_c = feat_s - feat_s.mean(axis=0, keepdims=True)
norms  = np.linalg.norm(feat_c, axis=0, keepdims=True) + 1e-8
feat_n = feat_c / norms
corr   = feat_n.T @ feat_n / N_SAMP          # [N_DIM, N_DIM]
corr   = np.clip(corr, -1.0, 1.0)
np.fill_diagonal(corr, 1.0)
# 强制对称（消除所有数值噪声）
corr   = (corr + corr.T) * 0.5
np.fill_diagonal(corr, 1.0)

try:
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    dist_c = 1.0 - np.abs(corr)
    np.fill_diagonal(dist_c, 0.0)
    dist_c = np.clip(dist_c, 0.0, 1.0)
    # 再次强制对称
    dist_c = (dist_c + dist_c.T) * 0.5
    np.fill_diagonal(dist_c, 0.0)

    # 检验对称性
    assert np.allclose(dist_c, dist_c.T, atol=1e-10), \
        "Still not symmetric!"

    condensed = squareform(dist_c)
    Z         = linkage(condensed, method='ward')
    order     = leaves_list(Z)
    corr_ord  = corr[np.ix_(order, order)]
    clustered = True
    print("  Hierarchical clustering ✓")
except Exception as e:
    corr_ord  = corr
    order     = list(range(N_DIM))
    clustered = False
    print(f"  Clustering skipped: {e}")

fig = plt.figure(figsize=(12, 10))
gs  = gridspec.GridSpec(1, 2, width_ratios=[20,1], wspace=0.04)
ax  = fig.add_subplot(gs[0])
cax = fig.add_subplot(gs[1])
im  = ax.imshow(corr_ord, cmap=CMAP_CORR, aspect='auto',
                vmin=-1, vmax=1, interpolation='nearest')
step = 8
ticks = list(range(0, N_DIM, step))
ax.set_xticks(ticks)
ax.set_xticklabels([f'd{order[t]}' for t in ticks],
                   fontsize=8, rotation=45, ha='right')
ax.set_yticks(ticks)
ax.set_yticklabels([f'd{order[t]}' for t in ticks], fontsize=8)
ax.set_xlabel('Feature Dimension (clustered)', fontsize=12, labelpad=8)
ax.set_ylabel('Feature Dimension (clustered)', fontsize=12, labelpad=8)
ax.set_title(
    f'Feature Dimension Pearson Correlation Matrix\n'
    f'(First {N_DIM} of {embed_dim}-d ViT-Base features'
    + (', hierarchically clustered)' if clustered else ')'),
    fontsize=12, fontweight='bold', pad=12)
cb = fig.colorbar(im, cax=cax)
cb.set_label('Pearson Correlation', fontsize=11, labelpad=10)
cb.set_ticks([-1,-0.5,0,0.5,1])
off = corr[~np.eye(N_DIM, dtype=bool)]
ax.text(0.02, 0.98,
        f'Mean |corr| (off-diag): {np.abs(off).mean():.4f}\n'
        f'Mean  corr (off-diag): {off.mean():.4f}\n'
        f'Sparsity (|r|<0.1):    {(np.abs(off)<0.1).mean()*100:.1f}%',
        transform=ax.transAxes, fontsize=9, va='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5',
                  fc='white', ec='#4393c3', alpha=0.93))
save_fig(fig, 'hm4_feature_correlation')

# ════════════════════════════════════════════════════════════
# HM-5  ViT Attention Rollout
# ════════════════════════════════════════════════════════════
print("[HM-5] ViT Attention Rollout...")
import torch.nn.functional as F

def get_attentions(mdl, img_t):
    attns, hooks = [], []
    def hook(m, inp, out): attns.append(out.detach().cpu())
    for blk in mdl.backbone.blocks[-3:]:
        hooks.append(blk.attn.register_forward_hook(hook))
    with torch.no_grad():
        mdl(img_t.unsqueeze(0).to(DEVICE))
    for h in hooks: h.remove()
    return attns

def rollout(attns, discard=0.9):
    result = torch.eye(attns[0].shape[-1])
    for a in attns:
        a_avg = a.mean(dim=1)[0]
        thr   = a_avg.flatten().quantile(discard)
        a_avg[a_avg < thr] = 0
        a_avg = a_avg + torch.eye(a_avg.shape[0])
        a_avg = a_avg / a_avg.sum(dim=-1, keepdim=True)
        result = torch.mm(a_avg, result)
    return result[0, 1:].numpy()

n_h = IMG_H // 16
n_w = IMG_W // 16

sel_q, seen_ids = [], set()
for i, (p, pid, cam) in enumerate(query):
    if pid not in seen_ids and os.path.exists(p):
        seen_ids.add(pid); sel_q.append(i)
    if len(sel_q) == 6: break

fig, axes = plt.subplots(3, 6, figsize=(15, 9))
fig.patch.set_facecolor('white')
fig.suptitle(
    'ViT-Base Attention Rollout Visualization on MoRe Dataset\n'
    'Row 1: Original Image  │  '
    'Row 2: Attention Heatmap  │  '
    'Row 3: Overlay',
    fontsize=13, fontweight='bold', y=1.01)

row_labels = ['Original', 'Attention\nMap', 'Overlay']
for ri, rl in enumerate(row_labels):
    axes[ri,0].set_ylabel(rl, fontsize=10,
                           rotation=0, labelpad=55, va='center')

for col, qi in enumerate(sel_q):
    p, pid, cam = query[qi]
    try:
        img_pil = Image.open(p).convert('RGB').resize((IMG_W,IMG_H))
    except:
        img_pil = Image.new('RGB',(IMG_W,IMG_H),(120,120,120))
    img_t   = tf_test(img_pil)
    img_arr = np.array(img_pil)

    try:
        attns   = get_attentions(model, img_t)
        mask    = rollout(attns).reshape(n_h, n_w)
        mask_t  = torch.tensor(mask).float().unsqueeze(0).unsqueeze(0)
        mask_up = F.interpolate(mask_t, size=(IMG_H,IMG_W),
                                mode='bilinear',
                                align_corners=False)[0,0].numpy()
        mn, mx  = mask_up.min(), mask_up.max()
        mask_up = (mask_up-mn)/(mx-mn+1e-8)
    except Exception as e:
        print(f"  col {col}: {e}")
        mask_up = np.zeros((IMG_H,IMG_W))

    axes[0,col].imshow(img_arr)
    axes[0,col].set_title(f'ID:{pid:04d} Cam:{cam}',
                           fontsize=9, fontweight='bold')
    axes[0,col].axis('off')

    axes[1,col].imshow(mask_up, cmap=CMAP_ATTN, vmin=0, vmax=1)
    axes[1,col].axis('off')

    hm      = plt.cm.inferno(mask_up)[:,:,:3]
    overlay = np.clip(0.45*img_arr/255.0 + 0.55*hm, 0, 1)
    axes[2,col].imshow(overlay)
    axes[2,col].axis('off')

fig.subplots_adjust(right=0.88, hspace=0.06, wspace=0.05)
cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
sm = plt.cm.ScalarMappable(cmap=CMAP_ATTN,
                            norm=plt.Normalize(0,1))
cb = fig.colorbar(sm, cax=cbar_ax)
cb.set_label('Attention Weight', fontsize=11, labelpad=10)
cb.set_ticks([0,0.25,0.5,0.75,1.0])
cb.set_ticklabels(['Low','','Mid','','High'])
save_fig(fig, 'hm5_attention_rollout')

# ════════════════════════════════════════════════════════════
# 汇总
# ════════════════════════════════════════════════════════════
print(f"\n{'═'*55}")
print("  All 5 heatmaps generated successfully!")
print(f"  Rank-1={r1*100:.2f}%  Rank-5={r5*100:.2f}%  "
      f"Rank-10={r10*100:.2f}%  mAP={mAP*100:.2f}%")
print(f"  Output: {SAVE_DIR}")
print(f"{'═'*55}")
for f in sorted(os.listdir(SAVE_DIR)):
    sz = os.path.getsize(os.path.join(SAVE_DIR,f))
    print(f"  {f:50s}  {sz/1024:7.1f} KB")
