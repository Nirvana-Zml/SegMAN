"""Reflection suppression (RSM) before VSSM in LASS encoder."""

import torch
import torch.nn as nn


def masked_gap(feat, mask, eps=1e-6):
    """在 mask 指定区域内做全局平均池化（GAP）。

    Args:
        feat: B x C x H x W，输入特征图
        mask: B x 1 x H x W，1 表示参与统计的区域（此处为背景）
        eps: 防止除零的小常数

    Returns:
        B x C x 1 x 1，mask 区域内的通道均值
    """
    # 将单通道 mask 广播到与 feat 相同的 C 维，便于逐元素相乘
    w = mask.expand_as(feat)
    # 对空间维 (H,W) 求 mask 权重之和，作为分母；clamp 避免空 mask 除零
    denom = w.sum(dim=(2, 3), keepdim=True).clamp_min(eps)
    # 加权求和后除以有效像素数，得到 mask 区域内的通道均值
    return (feat * w).sum(dim=(2, 3), keepdim=True) / denom


class ReflectionSuppression(nn.Module):
    """在 SS2D/VSSM 之前抑制背景反射分量。

    核心公式：
        F_clean = v - gamma * M_bg * mu_bg
        F_in    = F_clean + delta * v

    其中 mu_bg 为背景区域内的特征均值，M_bg 为背景空间 mask。
  """

    def __init__(self, channels, gamma_init=0.5, delta_init=0.5):
        super().__init__()
        # gamma：从特征中减去多少“背景反射”分量（可学习）
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        # delta：向输出中保留多少原始特征 v 的残差（可学习）
        self.delta = nn.Parameter(torch.tensor(float(delta_init)))

    def forward(self, v, m_bg):
        """
        Args:
            v: B x C x H x W，进入 VSSM 之前的特征
            m_bg: B x 1 x H x W，背景区域为 1，前景为 0
        """
        # 若 mask 与特征空间尺寸不一致，最近邻插值对齐到 v 的 H、W
        if m_bg.shape[-2:] != v.shape[-2:]:
            m_bg = torch.nn.functional.interpolate(
                m_bg, size=v.shape[-2:], mode='nearest')

        # 步骤 1：在背景 mask 内统计 v 的通道均值 mu_bg (B x C x 1 x 1)
        mu_bg = masked_gap(v, m_bg)

        # 步骤 2：将背景均值广播到全图，得到“背景反射”估计 f_refl
        #         仅在 m_bg=1 的位置保留，前景位置为 0
        f_refl = m_bg * mu_bg

        # 步骤 3：从原特征中减去 gamma 倍的背景反射，得到去反射特征
        f_clean = v - self.gamma * f_refl

        # 步骤 4：叠加 delta 倍原始特征，平衡抑制强度与信息保留
        return f_clean + self.delta * v
