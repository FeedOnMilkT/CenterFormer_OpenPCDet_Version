import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBnReLU(nn.Sequential):
    """Single conv block: Conv2d(3×3) + BN + ReLU, as described in paper Fig. 6."""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )


def _make_conv_block(in_ch, out_ch, num_convs):
    layers = [ConvBnReLU(in_ch, out_ch)]
    for _ in range(num_convs - 1):
        layers.append(ConvBnReLU(out_ch, out_ch))
    return nn.Sequential(*layers)


class ChannelAttention(nn.Module):
    """CBAM channel attention: avg + max pool → shared MLP → sigmoid."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x):
        avg = x.mean(dim=[2, 3])        # (B, C)
        mx  = x.amax(dim=[2, 3])        # (B, C)
        att = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * att[:, :, None, None]


class SpatialAttention(nn.Module):
    """CBAM spatial attention: channel avg + max → Conv(7×7) → sigmoid."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)   # (B, 1, H, W)
        mx  = x.amax(dim=1, keepdim=True)   # (B, 1, H, W)
        att = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * att


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al., ECCV 2018).

    Applied at the end of each CPN scale, as described in CenterFormer §3.2.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ch  = ChannelAttention(channels, reduction)
        self.spa = SpatialAttention()

    def forward(self, x):
        return self.spa(self.ch(x))


# ---------------------------------------------------------------------------
# CenterFormerCPN
# ---------------------------------------------------------------------------

class CenterFormerCPN(nn.Module):
    """Multi-scale Center Proposal Network (CenterFormer, ECCV 2022, §3.2 + Fig. 6).

    Takes the stride-8 BEV feature from HeightCompression and builds three FPN
    scales via one downsample and one upsample:

        scale[0]  stride-4   (2H × 2W)   highest res — used for heatmap
        scale[1]  stride-8   (H  × W)    base BEV
        scale[2]  stride-16  (H/2× W/2)  lowest res

    Each scale has a CBAM block at its output (paper: "At the end of each scale,
    we add a CBAM to enhance the feature via channel-wise and spatial attention").

    Config key: NUM_FILTERS (int) — hidden channel count for all three scales.

    batch_dict keys written:
        'spatial_features_2d'      → scale[0], for the heatmap head
        'multi_scale_bev_features' → [scale[0], scale[1], scale[2]], for cross-attention
    """

    def __init__(self, model_cfg, input_channels):
        super().__init__()
        C = model_cfg.NUM_FILTERS

        # ── mid-res branch: Conv×6 on the backbone BEV feature ──────────────
        self.conv_base = _make_conv_block(input_channels, C, num_convs=6)
        self.cbam_mid  = CBAM(C)

        # ── low-res branch: stride-2 conv + Conv×5 ───────────────────────────
        self.downsample = nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv_low   = _make_conv_block(C, C, num_convs=5)
        self.cbam_low   = CBAM(C)

        # ── high-res branch: restore from low-res + skip from input ──────────
        # upsample low-res (stride-16) back to stride-8, then Conv×1
        self.upsample_low = nn.ConvTranspose2d(C, C, kernel_size=2, stride=2)
        self.conv_fuse    = _make_conv_block(C, C, num_convs=1)

        # direct stride-4 skip from the original backbone BEV feature
        self.upsample_bev = nn.ConvTranspose2d(input_channels, C, kernel_size=2, stride=2)

        # 1×1 projection after concat (2C → C) + CBAM
        self.conv_proj = nn.Sequential(
            nn.Conv2d(C * 2, C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.cbam_high = CBAM(C)

        self.num_bev_features = C  # channels of spatial_features_2d, used by head

    def forward(self, batch_dict):
        x = batch_dict['spatial_features']   # (B, C_in, H, W)

        # ── mid-res ──────────────────────────────────────────────────────────
        x_mid = self.cbam_mid(self.conv_base(x))          # (B, C, H, W)

        # ── low-res ──────────────────────────────────────────────────────────
        x_low = self.cbam_low(self.conv_low(self.downsample(x_mid)))  # (B, C, H/2, W/2)

        # ── high-res ─────────────────────────────────────────────────────────
        # restore stride-16 → stride-8 then Conv×1
        x_fused = self.conv_fuse(self.upsample_low(x_low))   # (B, C, H, W)
        # upsample stride-8 → stride-4 via bilinear (avoids size-mismatch on
        # odd H/W that ConvTranspose2d can produce)
        x_fused_2x = F.interpolate(x_fused, scale_factor=2,
                                   mode='bilinear', align_corners=False)  # (B, C, 2H, 2W)
        # direct skip: backbone BEV → stride-4
        x_bev_up = self.upsample_bev(x)                               # (B, C, 2H, 2W)

        x_high = self.cbam_high(
            self.conv_proj(torch.cat([x_fused_2x, x_bev_up], dim=1))  # (B, C, 2H, 2W)
        )

        batch_dict['multi_scale_bev_features'] = [x_high, x_mid, x_low]
        batch_dict['spatial_features_2d'] = x_high   # heatmap uses highest res
        return batch_dict
