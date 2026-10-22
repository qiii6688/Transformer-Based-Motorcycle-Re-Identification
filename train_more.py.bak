import os, sys, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from collections import defaultdict

sys.path.insert(0, '/root/TransReID-main')
from datasets.more import MoRe

# ================================================================ #
#  Dataset
# ================================================================ #
class ReIDDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, pid, camid = self.data[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            img = Image.new('RGB', (128, 256), 0)
        if self.transform:
            img = self.transform(img)
        return img, pid, camid

def get_train_transform(h=256, w=128):
    return transforms.Compose([
        transforms.Resize((h, w)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Pad(10),
        transforms.RandomCrop((h, w)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02,0.33)),
    ])

def get_test_transform(h=256, w=128):
    return transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

# ================================================================ #
#  Losses
# ================================================================ #
class LabelSmoothCE(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, pred, target):
        log_prob = F.log_softmax(pred, dim=-1)
        smooth   = self.smoothing / (self.num_classes - 1)
        with torch.no_grad():
            one_hot = torch.full_like(log_prob, smooth)
            one_hot.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return -(one_hot * log_prob).sum(dim=-1).mean()

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, feat, labels):
        n = feat.size(0)
        dist = torch.cdist(feat, feat, p=2)
        mask_pos = labels.unsqueeze(1).eq(labels.unsqueeze(0))
        mask_neg = ~mask_pos

        dist_ap = (dist * mask_pos.float()).max(dim=1)[0]
        dist_an = dist.clone()
        dist_an[mask_pos] = 1e9
        dist_an = dist_an.min(dim=1)[0]

        loss = F.relu(dist_ap - dist_an + self.margin).mean()
        return loss

# ================================================================ #
#  Model (DeiT-Small via timm)
# ================================================================ #
def build_model(num_classes, pretrained_path=None, img_size=(256,128)):
    import timm
    backbone = timm.create_model(
        'vit_base_patch16_224',
        pretrained=False,
        num_classes=0,
        img_size=img_size,
    )

    if pretrained_path and os.path.exists(pretrained_path):
        state = torch.load(pretrained_path, map_location='cpu')
        if 'model' in state:
            state = state['model']

        # 处理pos_embed尺寸不匹配
        if 'pos_embed' in state:
            pe = state['pos_embed']          # [1, N+1, D]
            cls_pe = pe[:, :1, :]
            patch_pe = pe[:, 1:, :]
            orig_size = int(patch_pe.shape[1] ** 0.5)  # 14 for 224
            new_h = img_size[0] // 16   # 16
            new_w = img_size[1] // 16   # 8
            patch_pe = patch_pe.reshape(1, orig_size, orig_size, -1
                       ).permute(0, 3, 1, 2)
            patch_pe = F.interpolate(patch_pe, size=(new_h, new_w),
                       mode='bicubic', align_corners=False)
            patch_pe = patch_pe.permute(0, 2, 3, 1
                       ).reshape(1, new_h*new_w, -1)
            state['pos_embed'] = torch.cat([cls_pe, patch_pe], dim=1)
            print(f"  pos_embed resized: {pe.shape} -> {state['pos_embed'].shape}")

        missing, unexpected = backbone.load_state_dict(state, strict=False)
        print(f"  Loaded pretrained | missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print("  No pretrained weights found, training from scratch")

    embed_dim = backbone.embed_dim  # 384

    class ReIDModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone   = backbone
            self.bottleneck = nn.BatchNorm1d(embed_dim)
            nn.init.constant_(self.bottleneck.weight, 1)
            nn.init.constant_(self.bottleneck.bias, 0)
            self.bottleneck.bias.requires_grad_(False)
            self.classifier = nn.Linear(embed_dim, num_classes, bias=False)
            nn.init.normal_(self.classifier.weight, std=0.001)

        def forward(self, x):
            feat    = self.backbone(x)          # [B, 384]
            feat_bn = self.bottleneck(feat)
            if self.training:
                logits = self.classifier(feat_bn)
                return logits, feat             # for loss
            return feat_bn                      # for retrieval

    return ReIDModel(), embed_dim

# ================================================================ #
#  Evaluation
# ================================================================ #
@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    feats, pids, camids = [], [], []
    for imgs, pid, camid in loader:
        f = model(imgs.to(device))
        feats.append(f.cpu())
        pids.extend(pid.tolist())
        camids.extend(camid.tolist())
    return torch.cat(feats), np.array(pids), np.array(camids)

def compute_metrics(dist_mat, q_pids, g_pids,
                    q_camids, g_camids, max_rank=50):
    nq = dist_mat.shape[0]
    indices = np.argsort(dist_mat, axis=1)

    all_cmc, all_AP = [], []
    for i in range(nq):
        qp, qc = q_pids[i], q_camids[i]
        order  = indices[i]

        # 去掉同cam同id (junk)
        keep = ~((g_pids[order] == qp) & (g_camids[order] == qc))
        gp_sorted = g_pids[order][keep]

        matches = (gp_sorted == qp).astype(np.int32)
        if matches.sum() == 0:
            continue

        # CMC
        cmc = matches.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])

        # AP
        num_rel = matches.sum()
        tmp_cmc = matches.cumsum() / (np.arange(len(matches)) + 1)
        AP = (tmp_cmc * matches).sum() / num_rel
        all_AP.append(AP)

    all_cmc = np.mean(all_cmc, axis=0) * 100
    mAP     = np.mean(all_AP)  * 100
    return all_cmc, mAP

def evaluate(model, query_loader, gallery_loader, device):
    print("  Extracting query features...")
    qf, q_pids, q_camids = extract_features(model, query_loader, device)
    print("  Extracting gallery features...")
    gf, g_pids, g_camids = extract_features(model, gallery_loader, device)

    # cosine distance
    qf = F.normalize(qf, p=2, dim=1)
    gf = F.normalize(gf, p=2, dim=1)
    dist = (1 - torch.mm(qf, gf.t())).numpy()

    cmc, mAP = compute_metrics(dist, q_pids, g_pids, q_camids, g_camids)
    return cmc, mAP

# ================================================================ #
#  Main
# ================================================================ #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',   default='/root/TransReID-main/data/MoRe-Final-Version')
    parser.add_argument('--pretrained',  default='/root/TransReID-main/pretrained/vit_base_patch16_224-cd65a155.pth')
    parser.add_argument('--output_dir',  default='/root/TransReID-main/output/more')
    parser.add_argument('--epochs',      type=int,   default=120)
    parser.add_argument('--batch_size',  type=int,   default=64)
    parser.add_argument('--lr',          type=float, default=5e-4)
    parser.add_argument('--warmup',      type=int,   default=10)
    parser.add_argument('--eval_freq',   type=int,   default=10)
    parser.add_argument('--img_h',       type=int,   default=256)
    parser.add_argument('--img_w',       type=int,   default=128)
    parser.add_argument('--num_workers', type=int,   default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # ---------- Data ----------
    dataset = MoRe(root=args.data_root)
    img_size = (args.img_h, args.img_w)

    train_loader = DataLoader(
        ReIDDataset(dataset.train, get_train_transform(*img_size)),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    query_loader = DataLoader(
        ReIDDataset(dataset.query, get_test_transform(*img_size)),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    gallery_loader = DataLoader(
        ReIDDataset(dataset.gallery, get_test_transform(*img_size)),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    num_classes = dataset.num_train_pids
    print(f"num_classes={num_classes}")

    # ---------- Model ----------
    model, embed_dim = build_model(num_classes, args.pretrained, img_size)
    model = model.to(device)
    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {nparams/1e6:.2f}M")

    # ---------- Loss ----------
    ce_loss  = LabelSmoothCE(num_classes, smoothing=0.1)
    tri_loss = TripletLoss(margin=0.3)

    # ---------- Optimizer (layerwise lr) ----------
    backbone_params = list(model.backbone.parameters())
    head_params     = (list(model.bottleneck.parameters()) +
                       list(model.classifier.parameters()))
    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': head_params,     'lr': args.lr},
    ], weight_decay=1e-4)

    # Warmup + Cosine scheduler
    def lr_lambda(epoch):
        if epoch < args.warmup:
            return (epoch + 1) / args.warmup
        progress = (epoch - args.warmup) / (args.epochs - args.warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---------- Train ----------
    best_rank1, best_mAP = 0.0, 0.0
    log_path = os.path.join(args.output_dir, 'log.txt')

    def log(msg):
        print(msg)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')

    log(f"{'='*55}")
    log(f"MoRe Training | DeiT-Small | {num_classes} classes")
    log(f"Train:{dataset.num_train_imgs} Query:{dataset.num_query_imgs} "
        f"Gallery:{dataset.num_gallery_imgs}")
    log(f"{'='*55}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = total_ce = total_tri = correct = total = 0

        for imgs, pids, camids in train_loader:
            imgs  = imgs.to(device)
            pids  = pids.to(device)

            logits, feat = model(imgs)

            loss_ce  = ce_loss(logits, pids)
            loss_tri = tri_loss(feat, pids)
            loss     = loss_ce + loss_tri

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_ce   += loss_ce.item()
            total_tri  += loss_tri.item()
            correct    += logits.argmax(1).eq(pids).sum().item()
            total      += pids.size(0)

        scheduler.step()
        lr  = optimizer.param_groups[1]['lr']
        acc = 100. * correct / total
        avg = total_loss / len(train_loader)
        elapsed = time.time() - t0

        msg = (f"Ep[{epoch:3d}/{args.epochs}] "
               f"loss={avg:.4f}(ce={total_ce/len(train_loader):.3f} "
               f"tri={total_tri/len(train_loader):.3f}) "
               f"acc={acc:.1f}% lr={lr:.2e} t={elapsed:.0f}s")
        log(msg)

        # ---------- Eval ----------
        if epoch % args.eval_freq == 0 or epoch == args.epochs:
            cmc, mAP = evaluate(model, query_loader, gallery_loader, device)
            r1, r5, r10 = cmc[0], cmc[4], cmc[9]

            eval_msg = (f"  >> Rank-1={r1:.2f}%  Rank-5={r5:.2f}%  "
                        f"Rank-10={r10:.2f}%  mAP={mAP:.2f}%")
            log(eval_msg)

            if r1 > best_rank1:
                best_rank1 = r1
                best_mAP   = mAP
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'rank1': r1,
                    'mAP':   mAP,
                }, os.path.join(args.output_dir, 'best_model.pth'))
                log(f"  ★ Best saved! Rank-1={best_rank1:.2f}% mAP={best_mAP:.2f}%")

    log(f"\n{'='*55}")
    log(f"Training Complete!")
    log(f"Best Rank-1 : {best_rank1:.2f}%")
    log(f"Best mAP    : {best_mAP:.2f}%")
    log(f"{'='*55}")

if __name__ == '__main__':
    main()
