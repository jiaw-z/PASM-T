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

    def _alternating_attn(self, feat_left, feat_right, attn_left=None, attn_right=None):
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
                feat_left, feat_right, attn_left, attn_right = cross_attn(feat_left, feat_right, attn_left, attn_right)

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
                feat_left, feat_right, attn_left, attn_right = checkpoint(create_custom_cross_attn(cross_attn), feat_left, feat_right, attn_left, attn_right)

            return feat_left, feat_right, attn_left, attn_right

    def forward(self, feat_left: torch.Tensor, feat_right: torch.Tensor, cost=None, pos_enc: Optional[Tensor] = None):
        """
        :param feat_left: feature descriptor of left image, [N,C,H,W]
        :param feat_right: feature descriptor of right image, [N,C,H,W]
        :param pos_enc: relative positional encoding, [N,C,H,2W-1]
        :return: cross attention values [N,H,W,W], dim=2 is left image, dim=3 is right image
        """
        # compute attention
        if cost is not None:
            feat_left, feat_right, attn_left, attn_right = self._alternating_attn(feat_left, feat_right, cost[0], cost[1])
        else:
            feat_left, feat_right, attn_left, attn_right = self._alternating_attn(feat_left, feat_right)


        return feat_left, feat_right, (attn_left, attn_right)


class TransformerSelfAttnLayer(nn.Module):
    """
    Self attention layer
    """

    def __init__(self, hidden_dim: int, nhead: int, dp1 = 0.0, dp2 = 0.0):
        super().__init__()
        self.nhead = nhead
        self.dim = hidden_dim // nhead

        # self-attention layer
        self.kqv = nn.Linear(self.dim, 3 * self.dim, bias=False)
        self.self_attn = LinearAttention()
        self.dp = nn.Dropout(dp1)

        # feed-forward layer
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dp2),
        )

        # normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def mha(self, x, prev=None):
        bs, h, w, c = x.shape
        x = x.view(bs, -1, c)

        key, query, value = torch.split(self.kqv(x.view(bs, h * w, self.nhead, self.dim)), self.dim,
                                        dim=-1)  # B, T, h, emb_s
        return self.dp(self.self_attn(query, key, value)).view(bs, h, w, c)  # [N, L, (H, D)]

    def forward(self, x: Tensor, prev=None):
        """
        :param feat: image feature [bs, h, w, c]
        :return: updated image feature
        """
        bs, h, w, c = x.shape
        x = self.norm1(x + self.mha(x))
        x = self.norm2(x + self.ff(x))

        return x.view(bs, h, w, c)


class TransformerCrossAttnLayer(nn.Module):
    """
    Cross attention layer
    """

    def __init__(self, hidden_dim: int, nhead: int, dp1 = 0.0, dp2 = 0.0):
        super().__init__()
        self.nhead = nhead
        self.dim = hidden_dim // nhead

        # self-attention layer
        self.kqv = nn.Linear(self.dim, 3 * self.dim, bias=False)
        self.cross_attn = FullAttention()
        self.dp = nn.Dropout(dp1)

        # feed-forward layer
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dp2),
        )

        # normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def resmha(self, feat_left: Tensor, feat_right: Tensor, prev_l=None, prev_r=None):
        bs, h, w, c = feat_left.shape
        key_l, query_l, value_l = torch.split(self.kqv(feat_left.view(bs * h, w, self.nhead, self.dim)), self.dim,
                                        dim=-1)  # B, T, h, emb_s
        key_r, query_r, value_r = torch.split(self.kqv(feat_right.view(bs * h, w, self.nhead, self.dim)), self.dim,
                                              dim=-1)  # B, T, h, emb_s

        message_l, raw_attn_left = self.cross_attn(query_l, key_r, value_r, prev_l)  # [N, L, (H, D)]
        # raw_attn_left = raw_attn_left.sum(-1).view(bs, h, w, w)

        message_r, raw_attn_right = self.cross_attn(query_r, key_l, value_l, prev_r)  # [N, L, (H, D)]
        # raw_attn_right = raw_attn_right.sum(-1).view(bs, h, w, w)

        return [self.dp(message_l).view(bs, h, w, c), self.dp(message_r).view(bs, h, w, c)], [raw_attn_left, raw_attn_right]

    def forward(self, feat_left: Tensor, feat_right: Tensor, prev_l=None, prev_r=None):
        """
        :param feat_left: left image feature, [bs, h, w, c]
        :param feat_right: right image feature, [bs, h, w, c]
        :return: update image feature and attention weight [ bs, c, h, w]
        """
        bs, h, w, c = feat_left.shape
        rmha, prev = self.resmha(feat_left, feat_right, prev_l=prev_l, prev_r=prev_r)

        feat_left = self.norm1(feat_left + rmha[0])
        feat_left = self.norm2(feat_left + self.ff(feat_left)).view(bs, h, w, c)

        feat_right = self.norm1(feat_right + rmha[1])
        feat_right = self.norm2(feat_right + self.ff(feat_right)).view(bs, h, w, c)

        return feat_left, feat_right, prev[0], prev[1]



def build_transformer(args):
    return Transformer(
        hidden_dim=args.channel_dim,
        nhead=args.nheads,
        num_attn_layers=args.num_attn_layers
    )
