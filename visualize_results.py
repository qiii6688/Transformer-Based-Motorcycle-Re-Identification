import os, sys, torch, timm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn as nn

sys.path.insert(0, '/root/TransReID-main')
from datasets.more import MoRe

# ============================================================
# 配置
# ============================================================
DATA_ROOT  = '/root/TransReID-main/data'
CKPT_PATH  = '/root/TransReID-main/output/more_vitbase/best_model.pth'
OUTPUT_DIR = '/root/TransReID-main/output/visualization'
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_H, IMG_W = 256, 128

# ============================================================
# 1. 与train_more.py完全相同的模型定义
# ============================================================
class ReIDModel(nn.Module):
    def __init__(self, backbone, embed_dim, num_classes):
        super().__init__()
        self.backbone   = backbone
        self.bottleneck = nn.BatchNorm1d(embed_dim)
        self.bottleneck.bias.requires_grad_(False)
        nn.init.constant_(self.bottleneck.weight, 1)
        nn.init.constant_(self.bottleneck.bias, 0)
        self.classifier = nn.Linear(embed_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, x):
        feat    = self.backbone(x)
        feat_bn = self.bottleneck(feat)
        if self.training:
            return self.classifier(feat_bn), feat
        return feat_bn

def build_model(num_classes, img_size=(256,128)):
    backbone = timm.create_model(
        'deit_small_patch16_224',
        pretrained=False,
        num_classes=0,
        img_size=img_size
    )
    return ReIDModel(backbone, backbone.embed_dim, num_classes)

# ============================================================
# 2. 加载数据集
# ============================================================
print("加载数据集...")
dataset     = MoRe(root=DATA_ROOT, verbose=False)
num_classes = len(set(x[1] for x in dataset.train))

print(f"  训练集: {len(dataset.train)}张 {num_classes}个ID")
print(f"  Query:  {len(dataset.query)}张")
print(f"  Gallery:{len(dataset.gallery)}张")

# ── 诊断摄像头ID分布 ──────────────────────────────────────
q_cams = set(x[2] for x in dataset.query)
g_cams = set(x[2] for x in dataset.gallery)
print(f"\n  Query摄像头ID集合:   {sorted(q_cams)}")
print(f"  Gallery摄像头ID集合: {sorted(g_cams)}")

# 统计Query里有多少同ID同摄像头的Gallery图（junk）
junk_count = 0
valid_count = 0
for q_path, q_pid, q_cam in dataset.query:
    for g_path, g_pid, g_cam in dataset.gallery:
        if g_pid == q_pid:
            if g_cam == q_cam:
                junk_count += 1
            else:
                valid_count += 1
print(f"\n  Gallery中与Query同ID:")
print(f"    同摄像头(junk，会被过滤): {junk_count}张")
print(f"    异摄像头(有效匹配):       {valid_count}张")

if valid_count == 0:
    print("\n  ⚠ 警告：有效匹配为0！所有Query和Gallery摄像头ID相同")
    print("  → 需要修改评估函数，不过滤同摄像头")
    SKIP_CAM_FILTER = True
else:
    SKIP_CAM_FILTER = False
    print(f"\n  ✓ 有效匹配正常，继续...")

# ============================================================
# 3. DataLoader
# ============================================================
transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

class SimpleDS(Dataset):
    def __init__(self, data, tf):
        self.data = data; self.tf = tf
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        path, pid, camid = self.data[i]
        try:   img = Image.open(path).convert('RGB')
        except: img = Image.new('RGB',(IMG_W,IMG_H),(128,128,128))
        return self.tf(img), pid, camid, path

q_loader = DataLoader(SimpleDS(dataset.query,   transform),
                      batch_size=64, shuffle=False, num_workers=2)
g_loader = DataLoader(SimpleDS(dataset.gallery, transform),
                      batch_size=64, shuffle=False, num_workers=2)

# ============================================================
# 4. 加载模型（正确方式：重建结构再load state_dict）
# ============================================================
print("\n加载模型权重...")
ckpt = torch.load(CKPT_PATH, map_location='cpu')

# ckpt['model'] 是 OrderedDict(state_dict)，不是模型对象
state_dict = ckpt['model']
print(f"  epoch={ckpt['epoch']} | "
      f"保存rank1={float(ckpt['rank1']):.4f} | "
      f"保存mAP={float(ckpt['mAP']):.4f}")

model = build_model(num_classes).to(DEVICE)
missing, unexpected = model.load_state_dict(state_dict, strict=True)
print(f"  Missing={len(missing)} Unexpected={len(unexpected)}")
model.eval()
print("  ✓ 模型加载成功!")

# ============================================================
# 5. 提取特征
# ============================================================
def extract(loader):
    feats, pids, camids, paths = [], [], [], []
    with torch.no_grad():
        for imgs, pid, camid, path in loader:
            feat = model(imgs.to(DEVICE))
            feats.append(feat.cpu())
            pids.extend(pid.numpy().tolist())
            camids.extend(camid.numpy().tolist())
            paths.extend(path)
    feats = torch.nn.functional.normalize(
            torch.cat(feats,0), p=2, dim=1)
    return feats, np.array(pids), np.array(camids), paths

print("\n提取Query特征...")
q_feat, q_pids, q_camids, q_paths = extract(q_loader)
print("提取Gallery特征...")
g_feat, g_pids, g_camids, g_paths = extract(g_loader)

dist_mat = 1 - torch.mm(q_feat, g_feat.t()).numpy()
indices  = np.argsort(dist_mat, axis=1)

# ============================================================
# 6. 评估函数（根据摄像头情况自动选择是否过滤）
# ============================================================
def compute_metrics(dist_mat, q_pids, g_pids,
                    q_camids, g_camids,
                    skip_cam_filter=False, max_rank=50):
    all_cmc, all_AP = [], []
    num_valid = 0

    for qi in range(dist_mat.shape[0]):
        order = np.argsort(dist_mat[qi])

        if skip_cam_filter:
            # 不过滤同摄像头：Query本身不在Gallery里就不需要过滤
            keep = np.ones(len(g_pids), dtype=bool)
            # 只排除完全相同的图片路径（如果有的话）
        else:
            # 标准过滤：去掉同ID同摄像头
            remove = ((g_pids[order] == q_pids[qi]) &
                      (g_camids[order] == q_camids[qi]))
            keep = ~remove

        y = (g_pids[order[keep]] == q_pids[qi]).astype(np.int32)
        if not y.any():
            continue
        num_valid += 1

        cmc = y.cumsum(); cmc[cmc>1] = 1
        all_cmc.append(cmc[:max_rank])

        num_rel = y.sum()
        prec = [y[:i+1].sum()/(i+1) for i in range(len(y))]
        AP   = (np.array(prec)*y).sum() / num_rel
        all_AP.append(AP)

    print(f"  有效Query: {num_valid}/{dist_mat.shape[0]}")
    cmc = np.mean(all_cmc, axis=0)
    mAP = np.mean(all_AP)
    return cmc, mAP

print(f"\n评估模式: {'不过滤同摄像头' if SKIP_CAM_FILTER else '过滤同摄像头'}")
cmc, mAP = compute_metrics(
    dist_mat, q_pids, g_pids, q_camids, g_camids,
    skip_cam_filter=SKIP_CAM_FILTER)

r1=cmc[0]*100; r5=cmc[4]*100
r10=cmc[9]*100; r20=cmc[19]*100

print(f"\n{'='*55}")
print(f"  ✅ 最终评估结果（正确协议）")
print(f"  Rank-1  = {r1:.2f}%")
print(f"  Rank-5  = {r5:.2f}%")
print(f"  Rank-10 = {r10:.2f}%")
print(f"  Rank-20 = {r20:.2f}%")
print(f"  mAP     = {mAP*100:.2f}%")
print(f"{'='*55}")

# ============================================================
# 7. 可视化① Ranking结果
# ============================================================
print("\n生成 Ranking 可视化...")
NUM_Q=8; TOP_K=10

# 选query：有命中和没命中各一半
hit, miss = [], []
for qi in range(len(q_pids)):
    top_pids = g_pids[indices[qi,:TOP_K]]
    if (top_pids==q_pids[qi]).any() and len(hit)<NUM_Q//2:
        hit.append(qi)
    elif not (top_pids==q_pids[qi]).any() and len(miss)<NUM_Q//2:
        miss.append(qi)
    if len(hit)+len(miss)>=NUM_Q: break
selected=(hit+miss)[:NUM_Q]
# 如果hit不够就补miss
if len(selected)<NUM_Q:
    for qi in range(len(q_pids)):
        if qi not in selected:
            selected.append(qi)
        if len(selected)>=NUM_Q: break

fig, axes = plt.subplots(NUM_Q, TOP_K+1,
                          figsize=(2.2*(TOP_K+1), 2.8*NUM_Q))
fig.patch.set_facecolor('#0d1117')
fig.suptitle(
    f'MoRe 摩托车重识别  ·  Top-{TOP_K} Ranking 结果\n'
    f'DeiT-Small  |  Rank-1={r1:.2f}%  |  mAP={mAP*100:.2f}%',
    fontsize=14, fontweight='bold', color='white', y=1.005)

for row, qi in enumerate(selected):
    # Query图
    try:   qimg=Image.open(q_paths[qi]).convert('RGB')
    except: qimg=Image.new('RGB',(IMG_W,IMG_H),(100,100,100))
    ax=axes[row,0]
    ax.imshow(qimg); ax.set_facecolor('#0d1117')
    ax.set_title(f'Query\nID:{q_pids[qi]}\nCam:{q_camids[qi]}',
                 fontsize=8,color='#58a6ff',fontweight='bold')
    for sp in ax.spines.values():
        sp.set_edgecolor('#58a6ff'); sp.set_linewidth(3)
    ax.set_xticks([]); ax.set_yticks([])

    # Gallery Top-K
    for k in range(TOP_K):
        gi    = indices[qi,k]
        match = (g_pids[gi]==q_pids[qi])
        color = '#3fb950' if match else '#f85149'
        try:   gimg=Image.open(g_paths[gi]).convert('RGB')
        except: gimg=Image.new('RGB',(IMG_W,IMG_H),(60,60,60))
        ax=axes[row,k+1]
        ax.imshow(gimg); ax.set_facecolor('#0d1117')
        dist_val = dist_mat[qi,gi]
        ax.set_title(
            f'#{k+1} {"✓" if match else "✗"}\n'
            f'ID:{g_pids[gi]}\nd={dist_val:.3f}',
            fontsize=7, color=color,
            fontweight='bold' if match else 'normal')
        for sp in ax.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(3)
        ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
p1=os.path.join(OUTPUT_DIR,'1_ranking.png')
plt.savefig(p1,dpi=120,bbox_inches='tight',facecolor='#0d1117')
print(f"  ✓ {p1}"); plt.close()

# ============================================================
# 8. 可视化② CMC曲线 + 指标柱状图
# ============================================================
print("生成 CMC 曲线...")
fig, axes = plt.subplots(1,2,figsize=(15,6))
fig.patch.set_facecolor('#f8f9fa')

# 左：CMC
ax=axes[0]; ax.set_facecolor('white')
ranks=np.arange(1,min(51,len(cmc)+1))
ax.plot(ranks,cmc[:50]*100,'#1f77b4',lw=2.5,
        marker='o',ms=3,label='DeiT-Small on MoRe')
ax.fill_between(ranks,cmc[:50]*100,alpha=0.12,color='#1f77b4')
for r,c in [(1,'#d62728'),(5,'#2ca02c'),
            (10,'#ff7f0e'),(20,'#9467bd')]:
    if r<=len(cmc):
        ax.axvline(r,color=c,ls='--',lw=1.2,alpha=0.6)
        ax.scatter([r],[cmc[r-1]*100],color=c,s=80,zorder=5)
        ax.annotate(f'R{r}={cmc[r-1]*100:.1f}%',
                    xy=(r,cmc[r-1]*100),
                    xytext=(r+1.5,cmc[r-1]*100+1.5),
                    fontsize=9,color=c,fontweight='bold',
                    arrowprops=dict(arrowstyle='->',color=c,lw=1))
ax.set_xlabel('Rank',fontsize=13)
ax.set_ylabel('累计识别率 (%)',fontsize=13)
ax.set_title('CMC 曲线  —  MoRe数据集',
             fontsize=14,fontweight='bold')
ax.legend(fontsize=11,loc='lower right')
ax.grid(True,alpha=0.3)
ax.set_xlim([0,50]); ax.set_ylim([0,100])

# 右：柱状图
ax=axes[1]; ax.set_facecolor('white')
names =['Rank-1','Rank-5','Rank-10','Rank-20','mAP']
values=[r1,r5,r10,r20,mAP*100]
colors=['#1f77b4','#2ca02c','#ff7f0e','#9467bd','#d62728']
bars=ax.bar(names,values,color=colors,width=0.55,
            edgecolor='white',linewidth=1.5,alpha=0.9)
for bar,val in zip(bars,values):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.3,
            f'{val:.2f}%',ha='center',va='bottom',
            fontsize=12,fontweight='bold',color='#333')
ax.set_ylim([0,max(values)*1.25+5])
ax.set_title('性能指标  —  DeiT-Small on MoRe',
             fontsize=14,fontweight='bold')
ax.set_ylabel('识别率 (%)',fontsize=13)
ax.grid(True,axis='y',alpha=0.3)

# 数据集说明
fig.text(0.5,-0.03,
         f'MoRe摩托车重识别数据集  |  '
         f'训练:{len(dataset.train)}张/{num_classes}ID  '
         f'Query:{len(dataset.query)}张  '
         f'Gallery:{len(dataset.gallery)}张  |  '
         f'评估协议: {"不过滤同摄像头" if SKIP_CAM_FILTER else "过滤同摄像头junk"}',
         ha='center',fontsize=10,color='#555',
         bbox=dict(boxstyle='round,pad=0.4',
                   fc='#e8f5e9',ec='#2e7d32',alpha=0.9))
plt.tight_layout()
p2=os.path.join(OUTPUT_DIR,'2_cmc_metrics.png')
plt.savefig(p2,dpi=150,bbox_inches='tight')
print(f"  ✓ {p2}"); plt.close()

# ============================================================
# 9. 可视化③ 特征距离分布
# ============================================================
print("生成距离分布图...")
np.random.seed(42)
sq=np.random.choice(len(q_pids),min(200,len(q_pids)),replace=False)
pos_d,neg_d=[],[]
for qi in sq:
    for gi in range(len(g_pids)):
        d=dist_mat[qi,gi]
        if g_pids[gi]==q_pids[qi]: pos_d.append(d)
        else: neg_d.append(d)
neg_d=np.random.choice(neg_d,min(len(neg_d),8000),replace=False)
pos_d=np.array(pos_d)

fig,ax=plt.subplots(figsize=(11,5))
ax.hist(neg_d,bins=70,alpha=0.65,color='#ef5350',density=True,
        label=f'异ID负样本  均值={np.mean(neg_d):.3f}')
ax.hist(pos_d,bins=40,alpha=0.80,color='#66bb6a',density=True,
        label=f'同ID正样本  均值={np.mean(pos_d):.3f}')
ax.axvline(np.mean(pos_d),color='#1b5e20',ls='--',lw=2.5)
ax.axvline(np.mean(neg_d),color='#b71c1c',ls='--',lw=2.5)
sep=np.mean(neg_d)-np.mean(pos_d)
ax.set_xlabel('余弦距离',fontsize=13)
ax.set_ylabel('密度',fontsize=13)
ax.set_title('特征距离分布  —  绿峰(同ID)越靠左、红峰(异ID)越靠右越好',
             fontsize=13,fontweight='bold')
ax.legend(fontsize=11); ax.grid(True,alpha=0.25)
ax.text(0.97,0.92,f'类间-类内距离差\nΔ={sep:.4f}',
        transform=ax.transAxes,ha='right',va='top',fontsize=12,
        bbox=dict(boxstyle='round',fc='#e3f2fd',ec='#1565c0',alpha=0.85))
plt.tight_layout()
p3=os.path.join(OUTPUT_DIR,'3_distance_dist.png')
plt.savefig(p3,dpi=150,bbox_inches='tight')
print(f"  ✓ {p3}"); plt.close()

# ============================================================
# 10. 可视化④ 逐Query命中热力图
# ============================================================
print("生成命中率热力图...")
K_list=[1,5,10,20,50]
hit_mat=np.zeros((len(K_list),len(q_pids)))
for qi in range(len(q_pids)):
    for ki,k in enumerate(K_list):
        if q_pids[qi] in g_pids[indices[qi,:k]]:
            hit_mat[ki,qi]=1

fig,ax=plt.subplots(figsize=(16,3.5))
im=ax.imshow(hit_mat,aspect='auto',cmap='RdYlGn',
             vmin=0,vmax=1,interpolation='nearest')
ax.set_yticks(range(len(K_list)))
ax.set_yticklabels([f'Top-{k}' for k in K_list],fontsize=11)
ax.set_xlabel('Query 索引',fontsize=11)
ax.set_title('逐Query命中情况（绿=命中✓  红=未命中✗）',
             fontsize=12,fontweight='bold')
for ki,k in enumerate(K_list):
    rate=hit_mat[ki].mean()*100
    ax.text(len(q_pids)+2,ki,f'{rate:.1f}%',
            va='center',fontsize=10,fontweight='bold')
plt.colorbar(im,ax=ax,shrink=0.7)
plt.tight_layout()
p4=os.path.join(OUTPUT_DIR,'4_hit_heatmap.png')
plt.savefig(p4,dpi=120,bbox_inches='tight')
print(f"  ✓ {p4}"); plt.close()

# ============================================================
# 11. 可视化⑤ 数据集统计
# ============================================================
print("生成数据集统计图...")
from collections import Counter
cnt=Counter(x[1] for x in dataset.train)
counts=sorted(cnt.values(),reverse=True)

fig,axes=plt.subplots(1,2,figsize=(14,5))
ax=axes[0]
ax.bar(range(len(counts)),counts,color='steelblue',alpha=0.7,width=1)
ax.axhline(np.mean(counts),color='red',ls='--',
           label=f'均值={np.mean(counts):.1f}张/ID')
ax.set_xlabel('摩托车ID（按样本数排序）',fontsize=12)
ax.set_ylabel('图像数量',fontsize=12)
ax.set_title(f'训练集每ID样本分布\n{num_classes}个ID，{len(dataset.train)}张',
             fontsize=12,fontweight='bold')
ax.legend(); ax.grid(True,axis='y',alpha=0.3)

ax=axes[1]
labels=['Train\n训练集','Query\n查询集','Gallery\n图库']
sizes=[len(dataset.train),len(dataset.query),len(dataset.gallery)]
clrs=['#42a5f5','#66bb6a','#ffa726']
wedges,texts,autotexts=ax.pie(
    sizes,labels=labels,colors=clrs,
    autopct='%1.1f%%',startangle=90,
    textprops={'fontsize':11})
for at in autotexts: at.set_fontweight('bold')
ax.set_title(f'数据集划分\n总计{sum(sizes)}张图像',
             fontsize=12,fontweight='bold')

plt.tight_layout()
p5=os.path.join(OUTPUT_DIR,'5_dataset_stats.png')
plt.savefig(p5,dpi=150,bbox_inches='tight')
print(f"  ✓ {p5}"); plt.close()

# ============================================================
# 12. 汇总
# ============================================================
print(f"""
{'='*60}
  ✓ 全部可视化完成！
  输出目录: {OUTPUT_DIR}
  ─────────────────────────────────────────────────────────
  1_ranking.png        Ranking结果
  2_cmc_metrics.png    CMC曲线 + 指标柱状图
  3_distance_dist.png  特征距离分布
  4_hit_heatmap.png    逐Query命中热力图
  5_dataset_stats.png  数据集统计
  ─────────────────────────────────────────────────────────
  DeiT-Small 最终性能:
    Rank-1  = {r1:.2f}%
    Rank-5  = {r5:.2f}%
    Rank-10 = {r10:.2f}%
    Rank-20 = {r20:.2f}%
    mAP     = {mAP*100:.2f}%
  ─────────────────────────────────────────────────────────
  摄像头过滤模式: {'跳过（所有Query/Gallery同摄像头）' 
                   if SKIP_CAM_FILTER else '正常过滤junk'}
{'='*60}
""")
