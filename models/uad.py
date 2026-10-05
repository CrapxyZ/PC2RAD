import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from sklearn.cluster import KMeans
import math
import random
import torchvision.transforms.functional as TF

class BatchTokenShuffle(nn.Module):
    
    def __init__(self, p=0.2, num_protected_tokens=0):
        super().__init__()
        self.p = p
        self.num_protected_tokens = num_protected_tokens
    def forward(self, x):
        # only enable during training
        p = random.uniform(0, self.p)
        if not self.training or p == 0:
            return x
            
        batch_size, num_tokens, feature_dim = x.shape
        if batch_size <= 1:
            return x
            
        num_shuffleable_tokens = num_tokens - self.num_protected_tokens
        if num_shuffleable_tokens <= 0:
            return x

        out = x.clone().detach()
        
        # generate mask
        shuffle_mask = torch.rand(num_shuffleable_tokens, device=x.device) < p
        
        # get shuffle indices
        shuffle_indices = torch.nonzero(shuffle_mask).squeeze(-1)

        if len(shuffle_indices) == 0:
            return out

        actual_indices = shuffle_indices + self.num_protected_tokens
        num_selected = len(actual_indices)
        
        perms = torch.rand(batch_size, num_selected, device=x.device).argsort(dim=0)
        
        perms_expanded = perms.unsqueeze(-1).expand(-1, -1, feature_dim)
        
        tokens_to_shuffle = x[:, actual_indices, :]
        
        shuffled_tokens = torch.gather(tokens_to_shuffle, dim=0, index=perms_expanded)
        
        out[:, actual_indices, :] = shuffled_tokens
        
        return out

class DGA(nn.Module):
    def __init__(self, dim, N_token, M_token, num_heads=12, attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0, "illegal num_heads"
        self.m = M_token

        self.proj_k1 = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.GELU(),
            nn.LayerNorm(dim//2),
            nn.Linear(dim//2, dim//2),
        )
        self.proj_k2 = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.GELU(),
            nn.LayerNorm(dim//2),
            nn.Linear(dim//2, dim//2),
        )
        self.proj_q1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim//2),
        )
        self.proj_q2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim//2),
        )
        self.head_dim = (dim // num_heads) // 2
        '''
        Project Q/K to half the dim (0.5d) to match the computational and params budget as standard Attn*.

        For keys, since they have extremely short sequence lengths, we can add an additional linear layer to 
        enhance the transformation capability without incurring excessive computational burden. We believe that 
        this trade-off is worthwhile. Removing this linear layer is also an option, and the performance metrics 
        will only experience a slight decline.
        '''
        self.drop = nn.Dropout(attn_drop)
        self.proj_v = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.relative_pos_ebd = nn.Parameter(torch.randn(1, 1, N_token, M_token) * 0.01)

    def _shape(self, x):
        B, N, C = x.shape  # B: batch size, N: sequence length, C: embedding size
        return x.view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)

    def forward(self, q1, q2, k1, k2):
        '''
        default: k1 also works as Value source.
        '''
        B, N, C = q1.shape
        q1 = self._shape(self.proj_q1(q1))  # [B, num_heads, N, head_dim]
        q2 = self._shape(self.proj_q2(q2))

        v = self._shape(self.proj_v(k1))

        k1 = self._shape(self.proj_k1(k1))
        k2 = self._shape(self.proj_k2(k2))

        attn_scores1 = torch.matmul(q1, k1.transpose(-2, -1) / torch.sqrt(torch.tensor(self.head_dim // 2, dtype=torch.float32)))  # [B, num_heads, N, N]
        attn_scores2 = torch.matmul(q2, k2.transpose(-2, -1) / torch.sqrt(torch.tensor(self.head_dim // 2, dtype=torch.float32)))

        attn_weights1 = F.softmax(attn_scores1 + self.relative_pos_ebd, dim=-1)
        attn_weights2 = F.softmax(attn_scores2 + self.relative_pos_ebd, dim=-1)
        attn_weights = 0.5 * (self.drop(attn_weights1) + self.drop(attn_weights2))

        attn_output = torch.matmul(attn_weights, v) # [B, num_heads, N, head_dim]

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, C)  # [B, N, embed_dim]

        output = self.out_proj(attn_output)  # [B, N, embed_dim]
        return output

def grid_topk_pooling(consensus_score, feat, H=28, W=28, h=2, w=2, K=4):

    B_size, N, _ = consensus_score.shape
    _, _, C = feat.shape
    G_H, G_W = H // h, W // w

    scores_2d = consensus_score.squeeze(-1).reshape(B_size, H, W)
    scores_flat = (scores_2d
              .reshape(B_size, h, G_H, w, G_W)
              .permute(0, 1, 3, 2, 4)
              .reshape(B_size, h, w, -1))
    topk_vals, topk_idx = torch.topk(scores_flat, k=K, dim=-1)
    weights = F.softmax(topk_vals, dim=-1)

    feat_used = feat.permute(0, 2, 1).reshape(B_size, C, H, W)
    feat_flat = (feat_used
              .reshape(B_size, C, h, G_H, w, G_W)
              .permute(0, 2, 4, 1, 3, 5)
              .reshape(B_size, h, w, C, -1))
    gather_idx = topk_idx.unsqueeze(3).expand(-1, -1, -1, C, -1)
    feat_selected = torch.gather(feat_flat, dim=-1, index=gather_idx)
    out = (feat_selected * weights.unsqueeze(3)).sum(dim=-1)
    out = out.reshape(B_size, -1, C)
    return out

class PC2RAD(nn.Module):
    def __init__(
            self,
            encoder,
            used_layers=[0, 1, 2, 3, 4, 5, 6, 7, 8],
            fuse_layer_encoder=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
            remove_class_token=False,
            encoder_require_grad_layer=[],
            dim=768, H=28, W=28, h=2, w=2, K=4, p=0.2
    ) -> None:
        super(PC2RAD, self).__init__()
        self.encoder = encoder
        max_idx = max(used_layers)
        self.encoder.blocks = self.encoder.blocks[:max_idx + 1]
        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0

        self.H = H
        self.W = W
        self.h = h
        self.w = w
        self.K = K
        self.used_layers = used_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer
        self.token_shuffle = BatchTokenShuffle(p=p)

        self.DGA2 = DGA(dim=dim, N_token= 1 + H * W + self.encoder.num_register_tokens, M_token= h * w)
        self.mlp2 = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(dim//2, dim),
        )

        self.sa2_1 = SimpleViTBlock(dim=dim, num_heads=dim//64)
        self.sa2_2 = SimpleViTBlock(dim=dim, num_heads=dim//64)

        self.DGA3 = DGA(dim=dim, N_token= 1 + H * W + self.encoder.num_register_tokens, M_token= h * w)
        self.mlp3 = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(dim//2, dim),
        )

        self.sa3_1 = SimpleViTBlock(dim=dim, num_heads=dim//64)
        self.sa3_2 = SimpleViTBlock(dim=dim, num_heads=dim//64)
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + H*W + self.encoder.num_register_tokens, dim) * 0.02)

    def forward(self, x, x_train=None):
        x = self.encoder.prepare_tokens(x)

        en_list = []
        en_list_train = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= max(self.used_layers):
                if i in self.encoder_require_grad_layer:
                    x = blk(x)
                else:
                    with torch.no_grad():
                        x = blk(x)
            else:
                continue
            if i in self.used_layers:
                en_list.append(x)
        
        ####
        if x_train is not None:
            x = self.encoder.prepare_tokens(x_train)
            ref_list = []
            for i, blk in enumerate(self.encoder.blocks):
                if i <= max(self.used_layers):
                    if i in self.encoder_require_grad_layer:
                        x = blk(x)
                    else:
                        with torch.no_grad():
                            x = blk(x)
                if i in self.used_layers:
                    en_list_train.append(x)
        else:
            en_list_train = en_list   
        en_train = [self.fuse_feature([en_list_train[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]

        pos_embed = self.pos_embed.expand(x.size(0), -1, -1)

        ref_1 = en_train[0]
        ref_2 = en_train[1]
        ref_3 = en_train[2]
        
        if self.training:
            ref_1 = self.token_shuffle(ref_1)
            ref_2 = self.token_shuffle(ref_2)
            ref_3 = self.token_shuffle(ref_3)
        
        # Centroid trick to compute consensus scores, reduce O(N^2 D) to O(ND)

        ref_1_norm  = F.normalize(ref_1[:, 1+self.encoder.num_register_tokens:, :], p=2, dim=-1) 
        centroid_sum = torch.sum(ref_1_norm, dim=1, keepdim=True)
        consensus1 = torch.matmul(ref_1_norm, centroid_sum.transpose(-2, -1))

        ref_2_norm  = F.normalize(ref_2[:, 1+self.encoder.num_register_tokens:, :], p=2, dim=-1) 
        centroid_sum = torch.sum(ref_2_norm, dim=1, keepdim=True)
        consensus2 = torch.matmul(ref_2_norm, centroid_sum.transpose(-2, -1))

        ref_3_norm  = F.normalize(ref_3[:, 1+self.encoder.num_register_tokens:, :], p=2, dim=-1) 
        centroid_sum = torch.sum(ref_3_norm, dim=1, keepdim=True)
        consensus3 = torch.matmul(ref_3_norm, centroid_sum.transpose(-2, -1))

        P1 = grid_topk_pooling(consensus1, ref_1[:, 1+self.encoder.num_register_tokens:, :])
        P2 = grid_topk_pooling(consensus2, ref_2[:, 1+self.encoder.num_register_tokens:, :])
        P3 = grid_topk_pooling(consensus3, ref_3[:, 1+self.encoder.num_register_tokens:, :])

        de2 = self.DGA2(ref_1, ref_2, P1, P2)
        de2 = self.mlp2(de2)
        de2 = self.sa2_1(de2 + pos_embed)
        de2 = self.sa2_2(de2)

        de3 = self.DGA3(ref_2, ref_3, P2, P3)
        de3 = self.mlp3(de3)

        de3 = self.sa3_1(de3 + pos_embed + de2)
        de3 = self.sa3_2(de3)

        de2 = de2[:, 1+self.encoder.num_register_tokens:, :]
        de3 = de3[:, 1+self.encoder.num_register_tokens:, :]
        de2 = de2.permute(0, 2, 1).reshape([x.shape[0], -1, self.H, self.W]).contiguous()
        de3 = de3.permute(0, 2, 1).reshape([x.shape[0], -1, self.H, self.W]).contiguous()

        if not self.training:
            en = []
            en.append(ref_2)
            en.append(ref_3)
        else:
            en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder[1:]]

        if not self.remove_class_token:  # class tokens have not been removed above
            en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, self.H, self.W]).contiguous() for e in en]
        de = [de2, de3]

        return en, de

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)

class SimpleViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.05):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ln1 = nn.LayerNorm(dim)

        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=0.1, bias=True)
        self.ln2 = nn.LayerNorm(dim)

        # MLP
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        lnx = self.ln1(x)

        attn_out,_ = self.mha(lnx, lnx, lnx)
        attn_out = self.dropout(attn_out)
        x_mid = x + attn_out

        # --- MLP Block ---
        x_mlp = self.ln2(x_mid)        # LN
        x_mlp = self.mlp(x_mlp)        # MLP
        x_mlp = self.dropout(x_mlp)

        out = x_mid + x_mlp
        return out