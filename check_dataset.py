import sys
sys.path.insert(0, '/root/TransReID-main')
from datasets.more import MoRe
from collections import defaultdict

# ★ 关键：数据集在MoRe-Final-Version子文件夹里
DATA_ROOT = '/root/TransReID-main/data/MoRe-Final-Version'

print("="*60)
print("数据集统计")
print("="*60)
dataset = MoRe(root=DATA_ROOT, verbose=True)

# 验证camid
q_cams = sorted(set(x[2] for x in dataset.query))
g_cams = sorted(set(x[2] for x in dataset.gallery))
print(f"\nQuery  摄像头集合: {q_cams}")
print(f"Gallery摄像头集合: {g_cams}")

# 前10个Query详情
print("\n前10个Query:")
for i, (path, pid, camid) in enumerate(dataset.query[:10]):
    import os
    # 找Gallery中同ID不同cam的数量
    matches_diff_cam = sum(1 for gp, gid, gcam in dataset.gallery
                          if gid == pid and gcam != camid)
    matches_same_cam = sum(1 for gp, gid, gcam in dataset.gallery
                          if gid == pid and gcam == camid)
    fname = os.path.basename(path)
    print(f"  [{i}] {fname}")
    print(f"       ID={pid} Cam={camid} | "
          f"Gallery同ID异cam={matches_diff_cam} "
          f"同ID同cam={matches_same_cam}(junk)")

# ID交集统计
q_ids = set(x[1] for x in dataset.query)
g_ids = set(x[1] for x in dataset.gallery)
print(f"\nQuery与Gallery ID交集: {len(q_ids & g_ids)}个")

# 有效匹配统计（异摄像头）
total_valid = 0
for path, pid, camid in dataset.query:
    valid = sum(1 for gp, gid, gcam in dataset.gallery
               if gid == pid and gcam != camid)
    total_valid += valid
print(f"有效匹配总数(异cam): {total_valid}")
print(f"平均每Query有效匹配: {total_valid/len(dataset.query):.2f}张")

if total_valid == 0:
    print("\n⚠ 警告：有效匹配=0！camid提取仍有问题")
    print("  显示前3个Query的路径:")
    for path, pid, camid in dataset.query[:3]:
        print(f"    {path}")
else:
    print("\n✓ camid修复成功！可以开始训练")
