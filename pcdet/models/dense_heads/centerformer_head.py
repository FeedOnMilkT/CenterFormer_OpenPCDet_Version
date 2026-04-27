import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_

from ..model_utils import centernet_utils
from ..model_utils.centerformer_utils import CenterFormerPositionEmbedding, gather_bev_features
from ...utils import loss_utils


# ---------------------------------------------------------------------------
# 1-D Separate Head  (operates on N query tokens, not a 2-D feature map)
# ---------------------------------------------------------------------------

class SeparateHead1D(nn.Module):
    """Box regression sub-heads on query token sequence (B, D, N)."""

    def __init__(self, d_model, sep_head_dict, init_bias=-2.19):
        super().__init__()
        self.sep_head_dict = sep_head_dict
        for name, cfg in sep_head_dict.items():
            out_ch  = cfg['out_channels']
            n_conv  = cfg['num_conv']
            layers  = []
            for _ in range(n_conv - 1):
                layers += [
                    nn.Conv1d(d_model, d_model, 1, bias=False),
                    nn.BatchNorm1d(d_model),
                    nn.ReLU(inplace=True),
                ]
            layers.append(nn.Conv1d(d_model, out_ch, 1, bias=True))
            fc = nn.Sequential(*layers)
            if 'hm' in name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv1d):
                        kaiming_normal_(m.weight.data)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
            self.__setattr__(name, fc)

    def forward(self, x):
        # x: (B, D, N)
        return {name: self.__getattr__(name)(x) for name in self.sep_head_dict}


# ---------------------------------------------------------------------------
# Transformer Decoder Layer
# ---------------------------------------------------------------------------

class CenterFormerDecoderLayer(nn.Module):
    """Self-attn (global, among N proposals) + local cross-attn (each proposal
    attends to its own S×3×3 BEV neighborhood) + FFN.

    Per paper Fig. 3 left: cross-attn attending key = 3×3 window × S scales.
    """

    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.1):
        super().__init__()
        # Global self-attention among proposals
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                               batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)

        # Per-proposal local cross-attention
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                                batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop2 = nn.Dropout(dropout)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, query, local_kv, query_pos):
        """
        Args:
            query:     (B, N, D)
            local_kv:  (B, N, K, D)   K = S * 9
            query_pos: (B, N, D)
        Returns:
            query: (B, N, D)
        """
        B, N, D = query.shape
        K = local_kv.shape[2]

        # — Self-attention (pos embed added to Q & K; V = raw query) —
        q_p = query + query_pos
        sa_out, _ = self.self_attn(q_p, q_p, query)
        query = self.norm1(query + self.drop1(sa_out))

        # — Local cross-attention (reshape B*N into batch dim) —
        q_ca  = (query + query_pos).reshape(B * N, 1, D)   # (B*N, 1, D)
        kv_ca = local_kv.reshape(B * N, K, D)              # (B*N, K, D)
        ca_out, _ = self.cross_attn(q_ca, kv_ca, kv_ca)
        query = self.norm2(query + self.drop2(ca_out.reshape(B, N, D)))

        # — FFN —
        query = self.norm3(query + self.drop3(self.ffn(query)))
        return query


# ---------------------------------------------------------------------------
# Deformable cross-attention (CenterFormer deformable variant)
# ---------------------------------------------------------------------------

class DeformableCrossAttention(nn.Module):
    """Deformable cross-attention for CenterFormer.

    Replaces the fixed 3×3 window sampling with n_pts learned offset points
    per scale.  Attention weights are predicted from the query (Deformable DETR
    style) rather than computed by dot-product, avoiding O((S·K)²) cost.

    Reference: Zhu et al., "Deformable DETR", ICLR 2021 §3.2
    Paper deformable config: K=15 per scale, n_scales=3, 2 layers, 6 heads.
    """

    def __init__(self, d_model: int, n_heads: int,
                 n_scales: int = 3, n_pts: int = 15, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads  = n_heads
        self.n_scales = n_scales
        self.n_pts    = n_pts
        self.head_dim = d_model // n_heads

        # (dx, dy) offset per (scale, point) pair, predicted from each query
        self.offset_net = nn.Linear(d_model, n_scales * n_pts * 2)
        # Scalar attention weight per (head, scale, point)
        self.attn_net   = nn.Linear(d_model, n_heads * n_scales * n_pts)
        # Final output projection
        self.out_proj   = nn.Linear(d_model, d_model)
        self.drop       = nn.Dropout(dropout)

        # Zero-init offsets: at init all sampling points collapse to reference
        nn.init.zeros_(self.offset_net.weight)
        nn.init.zeros_(self.offset_net.bias)
        nn.init.zeros_(self.attn_net.bias)

    def forward(self, query: torch.Tensor, query_pos: torch.Tensor,
                reference_pts: torch.Tensor,
                scales: list, kv_proj_list: nn.ModuleList) -> torch.Tensor:
        """
        Args:
            query:         (B, N, D)
            query_pos:     (B, N, D)
            reference_pts: (B, N, 2)  grid_sample normalized coords [-1, 1]
            scales:        list of S (B, C, H_s, W_s) BEV feature maps
            kv_proj_list:  nn.ModuleList of S Linear(C → D), shared with head
        Returns:
            (B, N, D)
        """
        B, N, D = query.shape
        S, K    = self.n_scales, self.n_pts

        q = query + query_pos   # inject position for offset / weight prediction

        # Predict sampling offsets, bounded to ±0.5 in grid_sample space
        offsets = self.offset_net(q).reshape(B, N, S, K, 2).tanh() * 0.5

        # Predict normalised attention weights across all (scale, point) positions
        attn_w = self.attn_net(q).reshape(B, N, self.n_heads, S * K)
        attn_w = F.softmax(attn_w, dim=-1)                   # (B, N, n_heads, S*K)

        # Build per-scale sampling points and clamp to valid grid range
        ref        = reference_pts[:, :, None, None, :]      # (B, N, 1, 1, 2)
        sample_pts = (ref + offsets).clamp(-1.0, 1.0)        # (B, N, S, K, 2)

        # Sample and project features from each scale → cat to (B, N, S*K, D)
        values_list = []
        for s_idx, (scale, proj) in enumerate(zip(scales, kv_proj_list)):
            pts   = sample_pts[:, :, s_idx, :, :].reshape(B, N * K, 1, 2)
            feats = F.grid_sample(scale, pts, mode='bilinear',
                                  padding_mode='border', align_corners=True)
            # (B, C, N*K, 1) → (B, N, K, C)
            feats = feats.squeeze(-1).permute(0, 2, 1).reshape(B, N, K, scale.shape[1])
            values_list.append(proj(feats))                   # (B, N, K, D)
        values = torch.cat(values_list, dim=2)                # (B, N, S*K, D)

        # Multi-head weighted sum
        # (B, N, S*K, n_heads, head_dim) → (B, N, n_heads, S*K, head_dim)
        values = values.reshape(B, N, S * K, self.n_heads, self.head_dim)
        values = values.permute(0, 1, 3, 2, 4)
        # attn_w (B, N, n_heads, S*K) broadcast with values last dim
        out = (attn_w.unsqueeze(-1) * values).sum(dim=3)     # (B, N, n_heads, head_dim)
        out = self.drop(self.out_proj(out.reshape(B, N, D)))
        return out


class CenterFormerDeformableDecoderLayer(nn.Module):
    """CenterFormer decoder layer with deformable cross-attention.

    Self-attention (global among N proposals) and FFN are identical to
    CenterFormerDecoderLayer; only the cross-attention step uses learned
    sampling offsets instead of a fixed 3×3 window.

    Paper deformable config: n_layers=2, n_heads=6, n_pts=15.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int,
                 n_scales: int = 3, n_pts: int = 15, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                               batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)

        self.deform_cross = DeformableCrossAttention(
            d_model, n_heads, n_scales=n_scales, n_pts=n_pts, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop2 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, scales: list,
                reference_pts: torch.Tensor, query_pos: torch.Tensor,
                kv_proj_list: nn.ModuleList) -> torch.Tensor:
        """
        Args:
            query:         (B, N, D)
            scales:        list of S (B, C, H_s, W_s) BEV feature maps
            reference_pts: (B, N, 2)  grid_sample normalized coords [-1, 1]
            query_pos:     (B, N, D)
            kv_proj_list:  nn.ModuleList of S Linear(C → D)
        Returns:
            query: (B, N, D)
        """
        q_p = query + query_pos
        sa_out, _ = self.self_attn(q_p, q_p, query)
        query = self.norm1(query + self.drop1(sa_out))

        ca_out = self.deform_cross(query, query_pos, reference_pts, scales, kv_proj_list)
        query = self.norm2(query + self.drop2(ca_out))

        query = self.norm3(query + self.drop3(self.ffn(query)))
        return query


# ---------------------------------------------------------------------------
# CenterFormerHead
# ---------------------------------------------------------------------------

class CenterFormerHead(nn.Module):
    """CenterFormer detection head (ECCV 2022).

    Architecture:
        1. Heatmap branch   – 2-D conv on stride-4 BEV → focal loss
        2. Transformer decoder – GT-forced proposals → multi-scale cross-attn
        3. Box head         – 1-D conv on N query tokens → L1 loss

    Training:  GT centers forced as proposals; L1 box loss on GT proposals only.
    Inference: top-N heatmap proposals; box decoded from decoder output.
    """

    def __init__(self, model_cfg, input_channels, num_class, class_names,
                 grid_size, point_cloud_range, voxel_size,
                 predict_boxes_when_training=True):
        super().__init__()

        self.model_cfg   = model_cfg
        self.num_class   = num_class
        self.grid_size   = grid_size
        self.point_cloud_range = point_cloud_range  # [x0,y0,z0,x1,y1,z1]
        self.voxel_size  = voxel_size
        self.feature_map_stride = model_cfg.TARGET_ASSIGNER_CONFIG.FEATURE_MAP_STRIDE
        self.predict_boxes_when_training = predict_boxes_when_training

        D        = model_cfg.HIDDEN_CHANNEL
        n_heads  = model_cfg.NUM_HEADS
        n_layers = model_cfg.NUM_DECODER_LAYERS
        dropout  = model_cfg.get('DROPOUT', 0.1)
        self.num_proposals = model_cfg.NUM_PROPOSALS

        # ── class name book-keeping (same pattern as CenterHead) ──────────────
        self.class_names = class_names
        self.class_names_each_head   = []
        self.class_id_mapping_each_head = []
        for cur_class_names in model_cfg.CLASS_NAMES_EACH_HEAD:
            valid = [x for x in cur_class_names if x in class_names]
            self.class_names_each_head.append(valid)
            self.class_id_mapping_each_head.append(
                torch.from_numpy(np.array(
                    [class_names.index(x) for x in valid]
                )).cuda()
            )

        # ── heatmap branch ────────────────────────────────────────────────────
        shared_ch = model_cfg.SHARED_CONV_CHANNEL
        self.shared_conv = nn.Sequential(
            nn.Conv2d(input_channels, shared_ch, 3, 1, 1,
                      bias=model_cfg.get('USE_BIAS_BEFORE_NORM', False)),
            nn.BatchNorm2d(shared_ch, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.heatmap_heads = nn.ModuleList()
        for cur_names in self.class_names_each_head:
            self.heatmap_heads.append(nn.Sequential(
                nn.Conv2d(shared_ch, shared_ch, 3, 1, 1, bias=False),
                nn.BatchNorm2d(shared_ch, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_ch, len(cur_names), 3, 1, 1, bias=True),
            ))
        for hm in self.heatmap_heads:
            hm[-1].bias.data.fill_(-2.19)

        # ── query initialisation ──────────────────────────────────────────────
        self.query_proj = nn.Linear(shared_ch, D)
        self.pos_embed  = CenterFormerPositionEmbedding(D)   # learnable, 2→D

        # ── K/V projection per scale (input_channels = backbone_2d output channels) ──
        self.kv_proj = nn.ModuleList([nn.Linear(input_channels, D) for _ in range(3)])

        # ── transformer decoder ───────────────────────────────────────────────
        # DECODER_TYPE: 'standard' (fixed 3×3, paper base config)
        #             or 'deformable' (K=15 learned offsets, paper Appendix B)
        self.decoder_type = model_cfg.get('DECODER_TYPE', 'standard')
        n_deform_pts      = model_cfg.get('NUM_DEFORMABLE_POINTS', 15)
        if self.decoder_type == 'deformable':
            self.decoder_layers = nn.ModuleList([
                CenterFormerDeformableDecoderLayer(
                    D, n_heads, ffn_dim=D * 4, n_scales=3,
                    n_pts=n_deform_pts, dropout=dropout)
                for _ in range(n_layers)
            ])
        else:
            self.decoder_layers = nn.ModuleList([
                CenterFormerDecoderLayer(D, n_heads, ffn_dim=D * 4, dropout=dropout)
                for _ in range(n_layers)
            ])

        # ── box regression head ───────────────────────────────────────────────
        sep_head_dict = copy.deepcopy(model_cfg.SEPARATE_HEAD_CFG.HEAD_DICT)
        self.box_head = SeparateHead1D(D, sep_head_dict)
        self.head_order = model_cfg.SEPARATE_HEAD_CFG.HEAD_ORDER

        self.forward_ret_dict = {}
        self._build_losses()

    # ── losses ────────────────────────────────────────────────────────────────

    def _build_losses(self):
        self.hm_loss_func  = loss_utils.FocalLossCenterNet()
        self.reg_loss_func = loss_utils.RegLossCenterNet()

    # ── coordinate helpers ────────────────────────────────────────────────────

    def _pixel_to_gridsample_norm(self, centers_xy, H, W):
        """Heatmap pixel (cx,cy) → grid_sample [-1,1].

        Args:
            centers_xy: (..., 2)  x=col, y=row  (float, in [0, W/H-1])
        Returns:
            (..., 2) in [-1, 1]
        """
        xn = 2.0 * centers_xy[..., 0] / max(W - 1, 1) - 1.0
        yn = 2.0 * centers_xy[..., 1] / max(H - 1, 1) - 1.0
        return torch.stack([xn, yn], dim=-1)

    def _pixel_to_pos_embed_norm(self, centers_xy):
        """Heatmap pixel (cx,cy) → world [0,1] for position embedding.

        Uses FEATURE_MAP_STRIDE to convert pixel → real-world coords.
        """
        pcr    = self.point_cloud_range
        vs     = self.voxel_size
        stride = self.feature_map_stride
        x_world = pcr[0] + centers_xy[..., 0] * vs[0] * stride
        y_world = pcr[1] + centers_xy[..., 1] * vs[1] * stride
        xn = ((x_world - pcr[0]) / (pcr[3] - pcr[0])).clamp(0, 1)
        yn = ((y_world - pcr[1]) / (pcr[4] - pcr[1])).clamp(0, 1)
        return torch.stack([xn, yn], dim=-1)

    # ── heatmap target assignment (reused from CenterHead) ───────────────────

    def assign_target_of_single_head(
        self, num_classes, gt_boxes, feature_map_size, feature_map_stride,
        num_max_objs=500, gaussian_overlap=0.1, min_radius=2,
    ):
        heatmap   = gt_boxes.new_zeros(num_classes, feature_map_size[1], feature_map_size[0])
        ret_boxes = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        inds      = gt_boxes.new_zeros(num_max_objs).long()
        mask      = gt_boxes.new_zeros(num_max_objs).long()
        ret_boxes_src = gt_boxes.new_zeros(num_max_objs, gt_boxes.shape[-1])
        ret_boxes_src[:gt_boxes.shape[0]] = gt_boxes

        x, y, z  = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x  = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y  = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x  = torch.clamp(coord_x, 0, feature_map_size[0] - 0.5)
        coord_y  = torch.clamp(coord_y, 0, feature_map_size[1] - 0.5)
        center   = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int       = center.int()
        center_int_float = center_int.float()

        dx = gt_boxes[:, 3] / self.voxel_size[0] / feature_map_stride
        dy = gt_boxes[:, 4] / self.voxel_size[1] / feature_map_stride
        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue
            if not (0 <= center_int[k][0] <= feature_map_size[0] and
                    0 <= center_int[k][1] <= feature_map_size[1]):
                continue
            cur_class_id = (gt_boxes[k, -1] - 1).long()
            centernet_utils.draw_gaussian_to_heatmap(heatmap[cur_class_id], center[k], radius[k].item())
            inds[k] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0]
            mask[k] = 1
            ret_boxes[k, 0:2] = center[k] - center_int_float[k].float()
            ret_boxes[k, 2]   = z[k]
            ret_boxes[k, 3:6] = gt_boxes[k, 3:6].log()
            ret_boxes[k, 6]   = torch.cos(gt_boxes[k, 6])
            ret_boxes[k, 7]   = torch.sin(gt_boxes[k, 6])
            if gt_boxes.shape[1] > 8:
                ret_boxes[k, 8:] = gt_boxes[k, 7:-1]
        return heatmap, ret_boxes, inds, mask, ret_boxes_src

    def assign_targets(self, gt_boxes, feature_map_size):
        feature_map_size  = feature_map_size[::-1]   # [H,W] → [x,y]
        cfg     = self.model_cfg.TARGET_ASSIGNER_CONFIG
        B       = gt_boxes.shape[0]
        all_names = np.array(['bg', *self.class_names])
        ret = {'heatmaps': [], 'target_boxes': [], 'inds': [], 'masks': [],
               'target_boxes_src': []}

        for head_idx, cur_class_names in enumerate(self.class_names_each_head):
            hm_list, tb_list, ind_list, msk_list, src_list = [], [], [], [], []
            for b in range(B):
                cur_gt = gt_boxes[b]
                gt_names = all_names[cur_gt[:, -1].cpu().long().numpy()]
                gt_single = []
                for i, name in enumerate(gt_names):
                    if name not in cur_class_names:
                        continue
                    tmp      = cur_gt[i].clone()
                    tmp[-1]  = cur_class_names.index(name) + 1
                    gt_single.append(tmp[None])
                gt_single = (torch.cat(gt_single, 0) if gt_single else cur_gt[:0])
                hm, rb, ind, msk, src = self.assign_target_of_single_head(
                    len(cur_class_names), gt_single.cpu(),
                    feature_map_size, cfg.FEATURE_MAP_STRIDE,
                    cfg.NUM_MAX_OBJS, cfg.GAUSSIAN_OVERLAP, cfg.MIN_RADIUS,
                )
                hm_list.append(hm.to(cur_gt.device))
                tb_list.append(rb.to(cur_gt.device))
                ind_list.append(ind.to(cur_gt.device))
                msk_list.append(msk.to(cur_gt.device))
                src_list.append(src.to(cur_gt.device))
            ret['heatmaps'].append(torch.stack(hm_list))
            ret['target_boxes'].append(torch.stack(tb_list))
            ret['inds'].append(torch.stack(ind_list))
            ret['masks'].append(torch.stack(msk_list))
            ret['target_boxes_src'].append(torch.stack(src_list))
        return ret

    # ── proposal helpers ──────────────────────────────────────────────────────

    @torch.no_grad()
    def _gt_proposals(self, gt_boxes, H, W):
        """Compute GT center positions in heatmap pixel space.

        Returns:
            centers: (B, M_pad, 2)  float pixel coords
            valid:   (B, M_pad)     bool
        """
        pcr    = self.point_cloud_range
        vs     = self.voxel_size
        stride = self.feature_map_stride
        B, M   = gt_boxes.shape[:2]
        x      = gt_boxes[..., 0]
        y      = gt_boxes[..., 1]
        cx     = (x - pcr[0]) / vs[0] / stride
        cy     = (y - pcr[1]) / vs[1] / stride
        valid  = gt_boxes[..., -1] > 0   # class_id > 0 → valid
        cx     = cx.clamp(0, W - 1)
        cy     = cy.clamp(0, H - 1)
        return torch.stack([cx, cy], dim=-1), valid

    @torch.no_grad()
    def _build_train_proposals(self, hm_all_sigmoid, gt_boxes, H, W):
        """GT-forced proposals for training (paper Appendix A).

        First M entries = GT centers; remaining N-M = top heatmap peaks
        excluding GT positions.

        Returns:
            proposals:   (B, N, 2)  float pixel coords
            gt_mask:     (B, N)     bool  (True = GT-forced proposal)
            gt_box_idx:  (B, N)     long  (-1 for non-GT proposals)
        """
        B, N = gt_boxes.shape[0], self.num_proposals
        device = gt_boxes.device
        gt_centers, gt_valid = self._gt_proposals(gt_boxes, H, W)  # (B, M, 2), (B, M)

        # Global heatmap score: max over classes
        hm_score = hm_all_sigmoid.max(dim=1)[0]   # (B, H, W)

        all_proposals, all_gt_mask, all_gt_idx = [], [], []
        for b in range(B):
            valid_mask = gt_valid[b]              # (M,)
            valid_ctrs = gt_centers[b][valid_mask]# (m, 2)
            m          = valid_ctrs.shape[0]

            # Build pixel-index for GT positions to mask from heatmap
            gt_xi = valid_ctrs[:, 0].long().clamp(0, W - 1)
            gt_yi = valid_ctrs[:, 1].long().clamp(0, H - 1)
            gt_flat_idx = gt_yi * W + gt_xi         # (m,)

            hm_flat = hm_score[b].flatten().clone() # (H*W,)
            if m > 0:
                hm_flat[gt_flat_idx] = -1e6         # mask GT positions

            n_extra   = min(N - m, H * W)
            extra_xy  = gt_boxes.new_zeros(0, 2)
            if n_extra > 0:
                topk_idx     = torch.topk(hm_flat, n_extra)[1]  # (n_extra,)
                extra_y      = (topk_idx // W).float()
                extra_x      = (topk_idx  % W).float()
                extra_xy     = torch.stack([extra_x, extra_y], dim=-1)  # (n_extra, 2)

            # Concatenate GT + heatmap proposals, then pad to N
            props    = torch.cat([valid_ctrs, extra_xy], dim=0)   # (m + n_extra, 2)
            gt_idx_v = torch.full((props.shape[0],), -1, dtype=torch.long, device=device)
            gt_mask_v= torch.zeros(props.shape[0], dtype=torch.bool, device=device)
            gt_idx_v[:m]  = torch.arange(m, device=device)
            gt_mask_v[:m] = True

            pad = N - props.shape[0]
            if pad > 0:
                props    = torch.cat([props, props.new_zeros(pad, 2)], dim=0)
                gt_idx_v = torch.cat([gt_idx_v, gt_idx_v.new_full((pad,), -1)], dim=0)
                gt_mask_v= torch.cat([gt_mask_v, gt_mask_v.new_zeros(pad)], dim=0)
            else:
                props, gt_idx_v, gt_mask_v = props[:N], gt_idx_v[:N], gt_mask_v[:N]

            all_proposals.append(props)
            all_gt_mask.append(gt_mask_v)
            all_gt_idx.append(gt_idx_v)

        return (torch.stack(all_proposals),    # (B, N, 2)
                torch.stack(all_gt_mask),      # (B, N)
                torch.stack(all_gt_idx))       # (B, N)

    @torch.no_grad()
    def _build_infer_proposals(self, hm_all_sigmoid):
        """Top-N proposals for inference."""
        B   = hm_all_sigmoid.shape[0]
        N   = self.num_proposals
        _, _, H, W = hm_all_sigmoid.shape
        hm_score = hm_all_sigmoid.max(dim=1)[0]  # (B, H, W)
        hm_flat  = hm_score.reshape(B, -1)
        topk_idx = torch.topk(hm_flat, min(N, hm_flat.shape[1]))[1]  # (B, N)
        cy = (topk_idx // W).float()
        cx = (topk_idx  % W).float()
        return torch.stack([cx, cy], dim=-1)     # (B, N, 2)

    # ── box target encoding ────────────────────────────────────────────────────

    def _encode_gt_boxes(self, gt_boxes, proposals_pixel, gt_box_idx):
        """Encode GT boxes as regression targets relative to proposal centers.

        Returns:
            targets: (B, N, 10)  [dx, dy, z, log_w, log_l, log_h, cos, sin, vx, vy]
            reg_mask:(B, N)      bool  True = valid regression target
        """
        B, N = proposals_pixel.shape[:2]
        pcr, vs = self.point_cloud_range, self.voxel_size
        stride  = self.feature_map_stride

        targets  = proposals_pixel.new_zeros(B, N, 10)
        reg_mask = torch.zeros(B, N, dtype=torch.bool, device=proposals_pixel.device)

        for b in range(B):
            valid_prop = gt_box_idx[b] >= 0          # (N,)
            if not valid_prop.any():
                continue
            prop_xy  = proposals_pixel[b][valid_prop]  # (k, 2) in pixel space
            box_idx  = gt_box_idx[b][valid_prop]        # (k,)
            gt_b     = gt_boxes[b]                      # (M, 8+)

            # GT centers in pixel space (float)
            gt_cx = (gt_b[box_idx, 0] - pcr[0]) / vs[0] / stride
            gt_cy = (gt_b[box_idx, 1] - pcr[1]) / vs[1] / stride

            dx = gt_cx - prop_xy[:, 0]   # sub-pixel offset
            dy = gt_cy - prop_xy[:, 1]
            z  = gt_b[box_idx, 2]

            log_w = gt_b[box_idx, 3].log()
            log_l = gt_b[box_idx, 4].log()
            log_h = gt_b[box_idx, 5].log()

            cos_a = gt_b[box_idx, 6].cos()
            sin_a = gt_b[box_idx, 6].sin()

            has_vel = gt_b.shape[1] > 8
            vx = gt_b[box_idx, 7] if has_vel else torch.zeros_like(dx)
            vy = gt_b[box_idx, 8] if has_vel else torch.zeros_like(dx)

            targets[b, valid_prop] = torch.stack(
                [dx, dy, z, log_w, log_l, log_h, cos_a, sin_a, vx, vy], dim=-1
            )
            reg_mask[b, valid_prop] = True

        return targets, reg_mask

    # ── query / kv construction ────────────────────────────────────────────────

    def _build_query(self, x_shared, proposals_pixel):
        """Build query tokens and position embeddings (common to both decoder types).

        Args:
            x_shared:        (B, shared_ch, H, W)
            proposals_pixel: (B, N, 2)  float pixel coords at heatmap resolution
        Returns:
            query:    (B, N, D)
            query_pos:(B, N, D)
            ctrs_norm:(B, N, 2)  grid_sample coords [-1, 1] (valid for all scales)
        """
        B, N = proposals_pixel.shape[:2]
        H_hm, W_hm = x_shared.shape[2], x_shared.shape[3]

        ctrs_norm  = self._pixel_to_gridsample_norm(proposals_pixel, H_hm, W_hm)
        q_raw      = gather_bev_features(x_shared, ctrs_norm, radius=0)  # (B, N, shared_ch)
        query      = self.query_proj(q_raw)                               # (B, N, D)

        ctrs_world = self._pixel_to_pos_embed_norm(proposals_pixel)       # (B, N, 2)
        query_pos  = self.pos_embed(ctrs_world.reshape(B * N, 2))
        query_pos  = query_pos.reshape(B, N, -1)                          # (B, N, D)

        return query, query_pos, ctrs_norm

    def _build_local_kv(self, ctrs_norm, scales):
        """Standard decoder: pre-sample fixed 3×3 patches from each BEV scale.

        Args:
            ctrs_norm: (B, N, 2)  grid_sample coords (shared across all scales)
            scales:    list of 3 (B, C, H_s, W_s)
        Returns:
            local_kv:  (B, N, 27, D)   (3 scales × 9 points each)
        """
        B, N    = ctrs_norm.shape[:2]
        kv_list = []
        for s_idx, scale in enumerate(scales):
            raw = gather_bev_features(scale, ctrs_norm, radius=1)     # (B, N, C*9)
            raw = raw.reshape(B, N, 9, scale.shape[1])                # (B, N, 9, C)
            kv_list.append(self.kv_proj[s_idx](raw))                  # (B, N, 9, D)
        return torch.cat(kv_list, dim=2)                              # (B, N, 27, D)

    # ── box prediction decode (inference) ─────────────────────────────────────

    def _decode_boxes(self, box_preds, proposals_pixel, hm_all_sigmoid):
        """Decode 1-D box head output to world-coordinate boxes.

        Args:
            box_preds:       dict of (B, out_ch, N)
            proposals_pixel: (B, N, 2)
            hm_all_sigmoid:  (B, total_cls, H, W)
        Returns:
            list of dicts (one per batch item) with pred_boxes/scores/labels
        """
        B, N = proposals_pixel.shape[:2]
        pcr, vs = self.point_cloud_range, self.voxel_size
        stride  = self.feature_map_stride
        post_cfg = self.model_cfg.POST_PROCESSING

        # Scores: sample heatmap at proposal positions
        H_hm, W_hm = hm_all_sigmoid.shape[2], hm_all_sigmoid.shape[3]
        ctrs_norm   = self._pixel_to_gridsample_norm(proposals_pixel, H_hm, W_hm)
        # (B, total_cls, N, 1) → (B, N, total_cls)
        hm_at_ctrs  = gather_bev_features(hm_all_sigmoid, ctrs_norm, radius=0)

        pred_scores, pred_labels = hm_at_ctrs.max(dim=-1)   # (B, N)

        # Decode predicted box offset
        center_off = box_preds['center'].permute(0, 2, 1)    # (B, N, 2)
        cx_pred = proposals_pixel[..., 0] + center_off[..., 0]
        cy_pred = proposals_pixel[..., 1] + center_off[..., 1]
        x_world = pcr[0] + cx_pred * vs[0] * stride
        y_world = pcr[1] + cy_pred * vs[1] * stride
        z_world = box_preds['center_z'].permute(0, 2, 1)[..., 0]  # (B, N)
        dim     = box_preds['dim'].permute(0, 2, 1).exp()          # (B, N, 3)
        rot     = box_preds['rot'].permute(0, 2, 1)                # (B, N, 2)
        angle   = torch.atan2(rot[..., 1], rot[..., 0])            # (B, N)

        box_parts = [x_world.unsqueeze(-1), y_world.unsqueeze(-1),
                     z_world.unsqueeze(-1), dim, angle.unsqueeze(-1)]
        if 'vel' in box_preds:
            box_parts.append(box_preds['vel'].permute(0, 2, 1))    # (B, N, 2)
        final_boxes = torch.cat(box_parts, dim=-1)                  # (B, N, 7 or 9)

        limit_range = torch.tensor(post_cfg.POST_CENTER_LIMIT_RANGE,
                                   device=final_boxes.device).float()
        score_thresh = post_cfg.SCORE_THRESH

        ret = []
        for b in range(B):
            mask  = (final_boxes[b, :, :3] >= limit_range[:3]).all(-1)
            mask &= (final_boxes[b, :, :3] <= limit_range[3:]).all(-1)
            mask &= pred_scores[b] > score_thresh
            ret.append({
                'pred_boxes':  final_boxes[b, mask],
                'pred_scores': pred_scores[b, mask],
                'pred_labels': pred_labels[b, mask].long(),
            })
        return ret

    # ── loss ──────────────────────────────────────────────────────────────────

    def get_loss(self):
        pred_hm_list   = self.forward_ret_dict['pred_hm_list']
        target_dicts   = self.forward_ret_dict['target_dicts']
        box_preds      = self.forward_ret_dict['box_preds']
        reg_targets    = self.forward_ret_dict['reg_targets']   # (B, N, 10)
        reg_mask       = self.forward_ret_dict['reg_mask']      # (B, N)

        loss_weights = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        tb = {}
        total_loss = 0.0

        # — Heatmap focal loss (per head) —
        for idx, (pred_hm, tgt_hm) in enumerate(zip(pred_hm_list,
                                                      target_dicts['heatmaps'])):
            hm_sig = torch.clamp(pred_hm.sigmoid(), 1e-4, 1 - 1e-4)
            hm_loss = self.hm_loss_func(hm_sig, tgt_hm) * loss_weights['cls_weight']
            total_loss += hm_loss
            tb[f'hm_loss_head_{idx}'] = hm_loss.item()

            # 2-D box regression (over heatmap, same as CenterHead)
            pred_boxes_2d = torch.cat(
                [box_preds.get(k, box_preds.get(k)) for k in self.head_order
                 if k in box_preds], dim=1
            )
            # NB: box_preds here comes from SeparateHead1D (1-D).
            # We use only the heatmap-based reg loss for the 2-D branch.
            # (The 1-D transformer box loss is computed separately below.)

        # — Transformer 1-D box regression loss —
        if reg_mask.any():
            code_w = reg_targets.new_tensor(loss_weights['code_weights'])  # (10,)
            # Concatenate box_preds in order: dx,dy, z, w,l,h, cos,sin, vx,vy
            pred_1d_list = []
            for k in self.head_order:
                pred_1d_list.append(box_preds[k].permute(0, 2, 1))  # (B,N,out_ch)
            pred_encoded = torch.cat(pred_1d_list, dim=-1)           # (B, N, 10)

            # L1 loss only on GT-assigned proposals
            diff = (pred_encoded - reg_targets).abs()                # (B, N, 10)
            diff = (diff * code_w).sum(-1)                           # (B, N)
            n_pos = reg_mask.float().sum().clamp(min=1)
            reg_loss = (diff * reg_mask.float()).sum() / n_pos
            reg_loss = reg_loss * loss_weights['loc_weight']
            total_loss += reg_loss
            tb['reg_loss_1d'] = reg_loss.item()
        else:
            tb['reg_loss_1d'] = 0.0

        tb['rpn_loss'] = total_loss.item()
        return total_loss, tb

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, batch_dict):
        x        = batch_dict['spatial_features_2d']           # (B, cpn_ch, H, W)
        scales   = batch_dict['multi_scale_bev_features']      # list [s0, s1, s2]
        B, _, H, W = x.shape

        # ── heatmap ──────────────────────────────────────────────────────────
        x_shared   = self.shared_conv(x)                       # (B, shared_ch, H, W)
        pred_hm_list = [head(x_shared) for head in self.heatmap_heads]
        hm_all_sig   = torch.cat([hm.sigmoid() for hm in pred_hm_list], dim=1)  # (B,C,H,W)

        # ── proposals ────────────────────────────────────────────────────────
        if self.training:
            gt_boxes = batch_dict['gt_boxes']
            proposals, gt_mask, gt_box_idx = self._build_train_proposals(
                hm_all_sig.detach(), gt_boxes, H, W)
        else:
            proposals = self._build_infer_proposals(hm_all_sig.detach())

        # ── query construction (common to both decoder types) ────────────────
        query, query_pos, ctrs_norm = self._build_query(x_shared, proposals)

        # ── decoder ──────────────────────────────────────────────────────────
        if self.decoder_type == 'deformable':
            for layer in self.decoder_layers:
                query = layer(query, scales, ctrs_norm, query_pos, self.kv_proj)
        else:
            local_kv = self._build_local_kv(ctrs_norm, scales)
            for layer in self.decoder_layers:
                query = layer(query, local_kv, query_pos)      # (B, N, D)

        # ── box head ─────────────────────────────────────────────────────────
        box_preds = self.box_head(query.transpose(1, 2))       # dict of (B, out_ch, N)

        # ── training targets / loss storage ──────────────────────────────────
        if self.training:
            target_dicts = self.assign_targets(gt_boxes, feature_map_size=[H, W])
            reg_targets, reg_mask = self._encode_gt_boxes(gt_boxes, proposals, gt_box_idx)
            self.forward_ret_dict = {
                'pred_hm_list':  pred_hm_list,
                'target_dicts':  target_dicts,
                'box_preds':     box_preds,
                'reg_targets':   reg_targets,
                'reg_mask':      reg_mask,
            }

        # ── inference ────────────────────────────────────────────────────────
        if not self.training:
            pred_dicts = self._decode_boxes(box_preds, proposals, hm_all_sig)
            batch_dict['final_box_dicts'] = pred_dicts

        return batch_dict
