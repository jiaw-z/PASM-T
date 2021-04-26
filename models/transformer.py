#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

from typing import Optional

import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint
from utils.helper_data_processing import get_clones
from .linear_attention import LinearAttention, FullAttention

class Transformer(nn.Module):
    """
    Transformer computes self (intra image) and cross (inter image) attention
    """

    def __init__(self, hidden_dim: int = 128, nhead: int = 8, num_attn_layers: int = 6):
        super().__init__()

        self_attn_layer = TransformerSelfAttnLayer(hidden_dim, nhead)
        self.self_attn_layers = get_clones(self_attn_layer, num_attn_layers)

        cross_attn_layer = TransformerCrossAttnLayer(hidden_dim, nhead)
        self.cross_attn_layers = get_clones(cross_attn_layer, num_attn_layers)

        # self.norm = nn.LayerNorm(hidden_dim)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.num_attn_layers = num_attn_layers

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _alternating_attn(self, feat_left, feat_right, attn_left, attn_right):
        """
        Alternate self and cross attention with gradient checkpointing to save memory

        :param feat: image feature concatenated from left and right, [W,2HN,C]
        :param pos_enc: positional encoding, [W,HN,C]
        :param pos_indexes: indexes to slice positional encoding, [W,HN,C]
        :param hn: size of HN
        :return: attention weight [N,H,W,W]
        """
        # alternating
        if self.training:
            for idx, (self_attn, cross_attn) in enumerate(zip(self.self_attn_layers, self.cross_attn_layers)):
                layer_idx = idx
                # self-attention
                feat_left = self_attn(feat_left)
                feat_right = self_attn(feat_right)
                # cross-attention
                feat_left, feat_right, attn_weight, attn_weight_right = cross_attn(feat_left, feat_right)
                attn_left += attn_weight
                attn_right += attn_weight_right

            return feat_left, feat_right, attn_left, attn_right
        else:
            for idx, (self_attn, cross_attn) in enumerate(zip(self.self_attn_layers, self.cross_attn_layers)):
                #checkpoint self-attention
                def create_custom_self_attn(module):
                    def custom_self_attn(*inputs):
                        return module(*inputs)
                    return custom_self_attn
                feat_left = checkpoint(create_custom_self_attn(self_attn), feat_left)
                feat_right = checkpoint(create_custom_self_attn(self_attn), feat_right)
                # checkpoint cross-attention
                def create_custom_cross_attn(module):
                    def custom_cross_attn(*inputs):
                        return module(*inputs)
                    return custom_cross_attn
                feat_left, feat_right, attn_weight, attn_weight_right = checkpoint(create_custom_cross_attn(cross_attn), feat_left, feat_right)
                attn_left += attn_weight
                attn_right += attn_weight_right
            return feat_left, feat_right, attn_left, attn_right

    def forward(self, feat_left: torch.Tensor, feat_right: torch.Tensor, cost=None, pos_enc: Optional[Tensor] = None):
        """
        :param feat_left: feature descriptor of left image, [N,C,H,W]
        :param feat_right: feature descriptor of right image, [N,C,H,W]
        :param pos_enc: relative positional encoding, [N,C,H,2W-1]
        :return: cross attention values [N,H,W,W], dim=2 is left image, dim=3 is right image
        """
        # print('###################################################')
        # print(feat_left.shape)
        # print(pos_enc)

        # feat_left = feat_left.permute(0, 2, 3, 1)   # (n, h, w, c)
        # feat_right = feat_right.permute(0, 2, 3, 1)
        # print(feat_left.shape)
        # print(feat_right.shape)

        # compute attention
        feat_left, feat_right, attn_left, attn_right = self._alternating_attn(feat_left, feat_right, cost[0], cost[1])
        # feat_left = feat_left.permute(0, 3, 1, 2)   # (n, h, w, c)->(h, c, h, w)
        # feat_right = feat_right.permute(0, 3, 1, 2)

        return feat_left, feat_right, (attn_left, attn_right)


class TransformerSelfAttnLayer(nn.Module):
    """
    Self attention layer
    """

    def __init__(self, hidden_dim: int, nhead: int):
        super().__init__()
        self.nhead = nhead
        self.dim = hidden_dim // nhead

        # self-attention layer
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.self_attn = LinearAttention()
        self.merge = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # feed-forward layer
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )

        # normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor):
        """
        :param feat: image feature [bs, h, w, c]
        :return: updated image feature
        """
        bs, h, w, c = x.shape
        x = x.view(bs, -1, c)
        x_norm1 = self.norm1(x)
        query, key, value = x_norm1, x_norm1, x_norm1

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.self_attn(query, key, value)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))  # [N, L, C]
        # message = self.norm2(message)

        # feed-forward network
        # print(x.size())
        # print(message.size())
        # message = torch.cat([x, message], dim=-1)
        message = x + message
        message_norm = self.norm2(message)
        message_norm = self.mlp(message_norm)


        return (message + message_norm).view(bs, h, w, c)


class TransformerCrossAttnLayer(nn.Module):
    """
    Cross attention layer
    """

    def __init__(self, hidden_dim: int, nhead: int):
        super().__init__()
        self.nhead = nhead
        self.dim = hidden_dim // nhead

        # self-attention layer
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.cross_attn = FullAttention()
        self.merge = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # feed-forward layer
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )

        # normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)


    def forward(self, feat_left: Tensor, feat_right: Tensor):
        """
        :param feat_left: left image feature, [bs, h, w, c]
        :param feat_right: right image feature, [bs, h, w, c]
        :return: update image feature and attention weight [ bs, c, h, w]
        """
        bs, h, w, c = feat_left.shape
        feat_left_norm = self.norm1(feat_left)
        feat_right_norm = self.norm1(feat_right)

        query_l, key_l, value_l = feat_left_norm, feat_left_norm, feat_left_norm
        query_r, key_r, value_r = feat_right_norm, feat_right_norm, feat_right_norm

        query_l = self.q_proj(query_l).view(bs * h, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key_l = self.k_proj(key_l).view(bs * h, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value_l = self.v_proj(value_l).view(bs * h, -1, self.nhead, self.dim)
        query_r = self.q_proj(query_r).view(bs * h, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key_r = self.k_proj(key_r).view(bs * h, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value_r = self.v_proj(value_r).view(bs * h, -1, self.nhead, self.dim)

        # attention for left
        message_l, raw_attn_left = self.cross_attn(query_l, key_r, value_r)  # [N, L, (H, D)]
        raw_attn_left = raw_attn_left.sum(-1).view(bs, h, w, w)
        message_l = self.merge(message_l.view(bs, h, w, self.nhead * self.dim))  # [N, L, C]
        # message_l = self.norm2(message_l).view(bs, h, w, c)

        # feed-forward network for left
        # message_l = torch.cat([feat_left, message_l], dim=-1)
        message_l = feat_left + message_l
        message_l_norm = self.norm2(message_l)
        message_l_norm = self.mlp(message_l_norm).view(bs, h, w, c)
        message_l = message_l + message_l_norm

        # attention for right
        message_r, raw_attn_right = self.cross_attn(query_r, key_l, value_l)  # [N, L, (H, D)]
        raw_attn_right = raw_attn_right.sum(-1).view(bs, h, w, w)
        message_r = self.merge(message_r.view(bs, h, w, self.nhead * self.dim))  # [N, L, C]
        # message_r = self.norm2(message_r).view(bs, h, w, c)

        # feed-forward network for left
        # message_r = torch.cat([feat_right, message_r], dim=-1)
        message_r = feat_right + message_r
        message_r_norm = self.norm2(message_r)
        message_r_norm = self.mlp(message_r_norm).view(bs, h, w, c)
        message_r = message_r + message_r_norm

        return message_l, message_r, raw_attn_left, raw_attn_right



def build_transformer(args):
    return Transformer(
        hidden_dim=args.channel_dim,
        nhead=args.nheads,
        num_attn_layers=args.num_attn_layers
    )
