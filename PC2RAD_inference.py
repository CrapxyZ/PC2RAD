from torchvision import transforms
import torch
import torch.nn as nn
from dataset import get_data_transforms, get_strong_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader, Dataset
import math
from models.uad import PC2RAD
from models import vit_encoder
from pathlib import Path
import torch.backends.cudnn as cudnn
import argparse
from torchvision.utils import save_image
from utils import get_gaussian_kernel, cal_anomaly_maps
from torch.nn import functional as F
import warnings
import copy
import logging
from PIL import Image
warnings.filterwarnings("ignore")

class DefectInferenceDataset(Dataset):
    def __init__(self, input_base_dir, output_base_dir, item_list, transform=None):
        self.input_base_dir = Path(input_base_dir)
        self.output_base_dir = Path(output_base_dir)
        self.item_list = item_list
        self.transform = transform
        
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # scan and build(input_path, output_path)
        self.samples = self._gather_files()

    def _gather_files(self):
        samples = []
        for item in self.item_list:
            test_dir = self.input_base_dir / item / 'test'

            if not test_dir.exists() or not test_dir.is_dir():
                continue

            for category_dir in test_dir.iterdir():
                # skip 'good'
                if not category_dir.is_dir() or category_dir.name == 'good':
                    continue

                for file_path in category_dir.iterdir():
                    if file_path.suffix.lower() in self.valid_extensions:
                        relative_path = file_path.relative_to(self.input_base_dir)

                        out_path = self.output_base_dir / relative_path
                        samples.append((str(file_path), str(out_path)))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, out_path = self.samples[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, out_path

def run_pipeline(
    input_folder,
    output_folder,
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
    device,
    num_workers=4):

    crop_size = image_size

    H = crop_size // 14
    W = crop_size // 14

    transform, _ = get_data_transforms(image_size, crop_size)

    encoder = vit_encoder.load(encoder_name)
    # print(encoder)
    if 'small' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12

    model = PC2RAD(encoder=encoder, used_layers=used_layers, fuse_layer_encoder=fuse_layer_encoder, 
                    dim=embed_dim, H=H, W=W, h=h, w=w, K=K, p=prob_max)
    model = model.to(device)

    os.makedirs(output_folder, exist_ok=True)

    state_dict = torch.load(os.path.join(args.save_dir, args.save_name, f'last_iter.pth'), map_location='cpu')
    if 'model' in state_dict:
        state_dict = state_dict['model']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    # filter total_ops / total_params non-parameter keys (if exist)
    clean_state_dict = {
        k: v for k, v in state_dict.items()
        if not (k.endswith('total_ops') or k.endswith('total_params'))
    }

    model.load_state_dict(clean_state_dict, strict=False)

    dataset = DefectInferenceDataset(input_folder, output_folder, item_list, transform=transform)
    
    if len(dataset) == 0:
        print("No legal images, check path and item_list")
        return

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    model = model.to(device)
    model.eval()

    print(f"Found {len(dataset)} images ...")

    with torch.no_grad():
        for batch_imgs, batch_out_paths in dataloader:
            batch_imgs = batch_imgs.to(device)
            en, de = model(batch_imgs)

            anomaly_map, amap_list = cal_anomaly_maps(en, de, batch_imgs.shape[-1])
            anomaly_map = F.interpolate(anomaly_map, size=224, mode='bilinear', align_corners=False)
            gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=3).to(device)
            anomaly_map = gaussian_kernel(anomaly_map)

            for i in range(len(batch_out_paths)):
                out_path = batch_out_paths[i]
                output_tensor = anomaly_map[i] # [1, H, W]

                os.makedirs(os.path.dirname(out_path), exist_ok=True)

                save_image(output_tensor, out_path, normalize=True)

    print("Results saved.")

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

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default='../mvtec_anomaly_detection')
    # parser.add_argument('--data_path', type=str, default='../VisA_pytorch/1cls')
    # parser.add_argument('--data_path', type=str, default='../BTech_Dataset_transformed')
    parser.add_argument('--save_dir', type=str, default='./output_models')
    parser.add_argument('--save_name', type=str,
                        default='which_dataset_which_configuration_which_seed')
    args = parser.parse_args()
    #
    item_list = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule',
                'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
    # item_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                # 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
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
    batch_size = 32
    p_final = 0.9  # hard-mining ratio. note that it represents the "easy ratio", so 0.9 for MVTec AD, 0.8 for VisA/BTAD
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print_fn(device)

    output_folder = './MVTec_AnoMaps'
    run_pipeline(args.data_path, output_folder, encoder_name, item_list, used_layers, fuse_layer_encoder, 
                    image_size, h, w, K, p, total_iters, batch_size, p_final, device)
