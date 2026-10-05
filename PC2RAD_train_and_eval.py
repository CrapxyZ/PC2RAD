from torchvision import transforms
import torch
import torch.nn as nn
from dataset import get_data_transforms, get_strong_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader, ConcatDataset
import math
from models.uad import PC2RAD
from models import vit_encoder
from models.utils import trunc_normal_

from dataset import MVTecDataset
from torchinfo import summary
import torch.backends.cudnn as cudnn
import argparse
from utils import evaluation_batch, global_cosine_hm_percent, WarmCosineScheduler
from torch.nn import functional as F
from functools import partial
from optimizers import StableAdamW
import warnings
import copy
import logging
from sklearn.metrics import roc_auc_score, average_precision_score
import itertools

warnings.filterwarnings("ignore")

def shuffle_patches(x: torch.Tensor, num_stay: int) -> torch.Tensor:
    """
    Randomly shuffle patches within one image, num_stay patches remain the original positions.
    input:
        x: [B, 3, 392, 392]
        num_stay: int
    output:
        out: [B, 3, 392, 392]
    """
    B, C, H, W = x.shape
    patch_size = 14
    num_patches = (H // patch_size) * (W // patch_size)  # 28*28=784
    Hn, Wn = H // patch_size, W // patch_size

    # patchify: [B, C, Hn, patch_size, Wn, patch_size] -> [B, Hn*Wn, C, patch_size, patch_size]
    patches = (
        x.unfold(2, patch_size, patch_size)
         .unfold(3, patch_size, patch_size)  # [B, C, Hn, Wn, 14, 14]
         .permute(0, 2, 3, 1, 4, 5)          # [B, Hn, Wn, C, 14, 14]
         .reshape(B, Hn*Wn, C, patch_size, patch_size)
    )

    out_patches = torch.empty_like(patches)
    
    for i in range(B):
        perm = torch.randperm(num_patches)
        fixed_idx = torch.randperm(num_patches)[:num_stay]
        perm[fixed_idx] = fixed_idx

        out_patches[i] = patches[i][perm]

    # Reshape
    out = (
        out_patches.reshape(B, Hn, Wn, C, patch_size, patch_size)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(B, C, H, W)
    )
    return out

def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(
    encoder_name,
    item_list, 
    used_layers, 
    fuse_layer_encoder, 
    image_size, 
    h, 
    w,
    K,
    prob_max,
    total_iters,
    batch_size,
    p_final,
    seed,
    device):
    setup_seed(seed)

    crop_size = image_size

    H = crop_size // 14
    W = crop_size // 14

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    train_data_list = []
    test_data_list = []
    for i, item in enumerate(item_list):
        train_path = os.path.join(args.data_path, item, 'train')
        test_path = os.path.join(args.data_path, item)

        train_data = ImageFolder(root=train_path, transform=data_transform)
        train_data.classes = item
        train_data.class_to_idx = {item: i}
        train_data.samples = [(sample[0], i) for sample in train_data.samples]

        test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
        train_data_list.append(train_data)
        test_data_list.append(test_data)

    train_data = ConcatDataset(train_data_list)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   drop_last=False)

    encoder = vit_encoder.load(encoder_name)
    # print(encoder)
    if 'small' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12

    model = PC2RAD(encoder=encoder, used_layers=used_layers, fuse_layer_encoder=fuse_layer_encoder, 
                    dim=embed_dim, H=H, W=W, h=h, w=w, K=K, p=prob_max)
    model = model.to(device)

    trainable = nn.ModuleList([model.DGA2, model.mlp2, model.sa2_1, model.sa2_2, 
                               model.DGA3, model.mlp3, model.sa3_1, model.sa3_2])


    for m in trainable.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    optimizer = StableAdamW([
    {'params': trainable.parameters()},
    {'params': [model.pos_embed]},
    ], 
     lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-8)

    lr_scheduler = WarmCosineScheduler(optimizer, base_value=1e-3, final_value=2e-5, total_iters=total_iters,
                                       warmup_iters=100)

    print_fn('train image number:{}'.format(len(train_data)))

    it = 0
    N = H*W
    lowest = int(N*prob_max)
    for epoch in range(int(np.ceil(total_iters / len(train_dataloader)))):
        model.train()

        loss_list = []
        for img, label in train_dataloader:
            img = img.to(device)
            label = label.to(device)
            num_stay = random.randint(lowest, N)
            img_train = shuffle_patches(img, num_stay)

            en, de = model(img, x_train=img_train)

            p = min(p_final * it / 1000, p_final)
            loss = global_cosine_hm_percent(en, de, p=p, factor=0.1)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.1)

            optimizer.step()
            loss_list.append(loss.item())
            lr_scheduler.step()

            if (it + 1) == (total_iters // 2) or it == 99:

                auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
                auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

                for item, test_data in zip(item_list, test_data_list):
                    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                                                  num_workers=4)
                    results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=224, add_mul=True)
                    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results

                    auroc_sp_list.append(auroc_sp)
                    ap_sp_list.append(ap_sp)
                    f1_sp_list.append(f1_sp)
                    auroc_px_list.append(auroc_px)
                    ap_px_list.append(ap_px)
                    f1_px_list.append(f1_px)
                    aupro_px_list.append(aupro_px)

                    print_fn(
                        '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                            item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))

                print_fn(
                    'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                        np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
                        np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))
                
                model.train()
            it += 1
            if it == total_iters:
                torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, f'last_iter.pth'))

                auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
                auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []
                for item, test_data in zip(item_list, test_data_list):
                    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                                                num_workers=4)
                    results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=224)
                    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results

                    auroc_sp_list.append(auroc_sp)
                    ap_sp_list.append(ap_sp)
                    f1_sp_list.append(f1_sp)
                    auroc_px_list.append(auroc_px)
                    ap_px_list.append(ap_px)
                    f1_px_list.append(f1_px)
                    aupro_px_list.append(aupro_px)

                    print_fn(
                        '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                            item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))

                print_fn(
                    'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                        np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
                        np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))

                break
            if it % 100 == 0:
                print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))

    return


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    import argparse

    parser = argparse.ArgumentParser(description='')
    ### MVTec AD
    parser.add_argument('--data_path', type=str, default='../mvtec_anomaly_detection')

    ### VisA
    # parser.add_argument('--data_path', type=str, default='../VisA_pytorch/1cls')

    ### BTAD
    # parser.add_argument('--data_path', type=str, default='../BTech_Dataset_transformed')

    parser.add_argument('--save_dir', type=str, default='./output_models')
    parser.add_argument('--save_name', type=str,
                        default='which_dataset_which_configuration_which_seed')
    args = parser.parse_args()
    ### MVTec AD
    item_list = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule',
                 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
    
    ### VisA
    # item_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                 # 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    
    ### BTAD
    # item_list = ['01', '02', '03']

    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info


    ### Configuration
    
    encoder_name = 'dinov2reg_vit_base_14'
    # encoder_name = 'dinov2reg_vit_small_14'

    used_layers=[0, 1, 2, 3, 4, 5, 6, 7, 8]
    fuse_layer_encoder=[[0, 1, 2], [3, 4, 5], [6, 7, 8]] # forward
    # fuse_layer_encoder=[[6, 7, 8], [3, 4, 5], [0, 1, 2]] # backward
    image_size = 392
    h = 2   # default: MVTec AD (2), VisA/BTAD (7)
    w = 2
    K = 4   # default: MVTec AD (4), VisA/BTAD (2). 2x2-top4 , 7x7-top2
    p = 0.2
    total_iters = 10000  # 10000 for MVTec AD/VisA, 5000 for BTAD
    batch_size = 16
    p_final = 0.9  # hard-mining ratio. note that it represents the "easy ratio", so 0.9 for MVTec AD, 0.8 for VisA/BTAD
    seed = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print_fn(device)

    train(encoder_name, item_list, used_layers, fuse_layer_encoder, image_size, 
            h, w, K, p, total_iters, batch_size, p_final, seed, device)

