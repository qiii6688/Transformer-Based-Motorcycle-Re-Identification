# datasets/more.py
import os
import glob
import re
from collections import defaultdict

class MoRe:
    dataset_dir = 'MoRe-Final-Version'
    
    def __init__(self, root='', **kwargs):
        self.root = os.path.join(root, self.dataset_dir)
        self.train_file = os.path.join(self.root, 'train_files')
        self.test_file = os.path.join(self.root, 'test_files')
        
        self._check()
        train = self._load(self.train_file)
        test = self._load(self.test_file)
        
        self.train = self._remap(train)
        self.query, self.gallery = self._split_qg(test)
        
        self.num_train_pids = len(set(x[1] for x in self.train))
        self.num_train_imgs = len(self.train)
        self.num_train_cams = len(set(x[2] for x in self.train))
        self.num_train_vids = self.num_train_cams
        
        print(f"=> MoRe loaded")
        print(f"  train: {len(self.train)} imgs, {self.num_train_pids} IDs, {self.num_train_cams} cams")
        print(f"  query: {len(self.query)} imgs")
        print(f"  gallery: {len(self.gallery)} imgs")
    
    def _check(self):
        if not os.path.exists(self.root):
            raise RuntimeError(f"'{self.root}' 不存在")
    
    def _load(self, fpath):
        """从文件加载数据，格式: pair01/camA/camA_id_03578_num_001.png"""
        data = []
        with open(fpath, 'r') as f:
            for line in f:
                rel_path = line.strip()
                if not rel_path:
                    continue
                
                img_path = os.path.join(self.root, rel_path)
                if not os.path.exists(img_path):
                    continue
                
                # 从文件名解析 pid 和 camid
                # 格式: camA_id_03578_num_001.png
                filename = os.path.basename(rel_path)
                
                # 提取 id
                pid_match = re.search(r'id[_\s](\d+)', filename)
                if pid_match:
                    pid = int(pid_match.group(1))
                else:
                    continue
                
                # 提取 camid (camA=0, camB=1, camC=2...)
                cam_match = re.search(r'cam([A-Z])', filename)
                if cam_match:
                    camid = ord(cam_match.group(1)) - ord('A')
                else:
                    camid = 0
                
                data.append((img_path, pid, camid))
        
        return data
    
    def _remap(self, data):
        """将PID重映射为从0开始的连续整数"""
        pids = sorted(set(x[1] for x in data))
        p2l = {p: i for i, p in enumerate(pids)}
        return [(img, p2l[pid], cam) for img, pid, cam in data]
    
    def _split_qg(self, data):
        """构建query和gallery"""
        pc = defaultdict(list)
        for img, pid, cam in data:
            pc[(pid, cam)].append(img)
        
        pid_cams = defaultdict(set)
        for pid, cam in pc:
            pid_cams[pid].add(cam)
        
        query, gallery = [], []
        for pid in sorted(pid_cams.keys()):
            cams = sorted(pid_cams[pid])
            if len(cams) > 1:
                q_cam = cams[0]
                imgs = pc[(pid, q_cam)]
                query.append((imgs[0], pid, q_cam))
                for img in imgs[1:]:
                    gallery.append((img, pid, q_cam))
                for cam in cams[1:]:
                    for img in pc[(pid, cam)]:
                        gallery.append((img, pid, cam))
            else:
                cam = cams[0]
                for img in pc[(pid, cam)]:
                    gallery.append((img, pid, cam))
        
        # 加载干扰图像
        dist_dir = os.path.join(self.root, 'Distractors')
        if os.path.exists(dist_dir):
            for ext in ['*.jpg', '*.png']:
                for img in glob.glob(os.path.join(dist_dir, '**', ext), recursive=True):
                    gallery.append((img, -1, -1))
        
        # 重映射
        all_pids = sorted(set(x[1] for x in query + gallery if x[1] != -1))
        p2l = {p: i for i, p in enumerate(all_pids)}
        query = [(img, p2l[pid], cam) for img, pid, cam in query]
        gallery = [(img, p2l.get(pid, -1), cam) for img, pid, cam in gallery]
        
        return query, gallery
