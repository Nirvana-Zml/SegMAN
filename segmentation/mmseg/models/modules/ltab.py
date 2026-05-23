"""Low-texture aware weighting (LTAB) for LASS encoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LTAB(nn.Module):
    """对 SSM 输出做低纹理感知的空间加权。

    设计文档 §4.2 对应公式：
        T       = AvgPool(|dF/dx| + |dF/dy|)
        T_norm  = (T - mean) / std
        W_lt    = sigmoid(alpha * (tau - T_norm))
        F_out   = F_ssm * (1 + beta * W_lt)

    低纹理区域（梯度小）T_norm 较低，W_lt 较高，特征被适度放大；
    高纹理区域则 W_lt 较低，改动较小。
    """

    def __init__(self, channels, beta_init=0.1, alpha_init=1.0, tau_init=0.0):
        super().__init__()
        # beta：低纹理加权强度（可学习）
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        # alpha：sigmoid 斜率，控制纹理阈值附近的过渡陡峭程度
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        # tau：归一化纹理图上的阈值，决定“低/高纹理”分界
        self.tau = nn.Parameter(torch.tensor(float(tau_init)))

    def _texture_map(self, x):
        """从特征 x 估计空间纹理强度图 T (B x 1 x H x W)。

        对通道均值后的灰度图求 x/y 方向一阶差分，取梯度幅值并平滑。
        """
        # 步骤 1：沿通道维求均值，得到单通道“灰度”特征，降低通道间差异干扰
        xm = x.mean(dim=1, keepdim=True)

        # 步骤 2：水平 / 垂直方向相邻像素差分，近似 |dF/dx|、|dF/dy|
        gx = xm[:, :, :, 1:] - xm[:, :, :, :-1]
        gy = xm[:, :, 1:, :] - xm[:, :, :-1, :]

        # 步骤 3：差分后尺寸各少 1，用零填充恢复到 H x W，便于与 x 对齐
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = F.pad(gy, (0, 0, 0, 1))

        # 步骤 4：梯度幅值 = |gx| + |gy|，纹理越强该值越大
        t = (gx.abs() + gy.abs())

        # 步骤 5：3x3 平均池化平滑，抑制噪声、稳定纹理估计
        t = F.avg_pool2d(t, kernel_size=3, stride=1, padding=1)
        return t

    def forward(self, x):
        """
        Args:
            x: B x C x H x W，SSM 输出的特征

        Returns:
            经低纹理感知加权后的特征，形状与 x 相同
        """
        # 步骤 1：计算空间纹理强度图 T
        t = self._texture_map(x)

        # 步骤 2：对 T 做逐样本的空间维标准化，得到 T_norm
        mu = t.mean(dim=(2, 3), keepdim=True)
        std = t.std(dim=(2, 3), keepdim=True) + 1e-6
        t_norm = (t - mu) / std

        # 步骤 3：低纹理权重 W_lt
        #       T_norm 小（低纹理）→ (tau - T_norm) 大 → sigmoid 接近 1
        #       T_norm 大（高纹理）→ (tau - T_norm) 小 → sigmoid 接近 0
        w_lt = torch.sigmoid(self.alpha * (self.tau - t_norm))

        # 步骤 4：在低纹理区域按 (1 + beta * W_lt) 放大特征，高纹理区域接近恒等
        return x * (1.0 + self.beta * w_lt)
