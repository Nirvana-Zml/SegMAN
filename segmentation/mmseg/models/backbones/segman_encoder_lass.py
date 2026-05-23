# LASS encoder: SegMAN backbone with LTAB + RSM in stage 1-3.
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import einsum, rearrange
from natten.functional import na2d, na2d_av, na2d_qk
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from mmseg.models.builder import BACKBONES
from mmseg.models.modules.ltab import LTAB
from mmseg.models.modules.lass_utils import semantic_seg_to_mbg
from mmseg.models.modules.rsm import ReflectionSuppression

from .segman_encoder import (
    Attention,
    BasicLayer,
    FFN,
    LayerNorm2d,
    LayerScale,
    RoPE,
    SegMANEncoder,
    VSSM,
    _cfg,
    stem,
    theta_shift,
    toodd,
)


def _parse_lass_cfg(lass_cfg):
    lass_cfg = lass_cfg or {}
    return dict(
        enable_stages=lass_cfg.get('enable_stages', [0, 1, 2]),
        enable_ltab=lass_cfg.get('enable_ltab', True),
        enable_rsm=lass_cfg.get('enable_rsm', True),
        ltab_cfg=lass_cfg.get('ltab', {}),
        rsm_cfg=lass_cfg.get('rsm', {}),
        dilate_kernel=lass_cfg.get('dilate_kernel', 5),
        ignore_index=lass_cfg.get('ignore_index', 255),
    )


class AttentionLASS(Attention):
    """Attention with RSM (before VSSM) and LTAB (after VSSM) on local branches."""

    def __init__(self,
                 embed_dim,
                 num_heads,
                 window_size,
                 window_dilation,
                 global_mode=False,
                 image_size=None,
                 use_rpb=False,
                 sr_ratio=1,
                 fused_na=True,
                 ssm_ratio=1,
                 ssm_split=False,
                 enable_ltab=True,
                 enable_rsm=True,
                 ltab_cfg=None,
                 rsm_cfg=None):
        super().__init__(
            embed_dim,
            num_heads,
            window_size,
            window_dilation,
            global_mode,
            image_size,
            use_rpb,
            sr_ratio,
            fused_na=fused_na,
            ssm_ratio=ssm_ratio,
            ssm_split=ssm_split,
        )
        self.enable_ltab = enable_ltab and (not global_mode)
        self.enable_rsm = enable_rsm and (not global_mode)
        ltab_cfg = ltab_cfg or {}
        rsm_cfg = rsm_cfg or {}
        if self.enable_ltab:
            self.ltab = LTAB(
                embed_dim,
                beta_init=ltab_cfg.get('beta_init', 0.1),
                alpha_init=ltab_cfg.get('alpha_init', 1.0),
                tau_init=ltab_cfg.get('tau_init', 0.0),
            )
        if self.enable_rsm:
            self.rsm = ReflectionSuppression(
                embed_dim,
                gamma_init=rsm_cfg.get('gamma_init', 0.5),
                delta_init=rsm_cfg.get('delta_init', 0.5),
            )

    def forward(self, x, pos_enc, m_bg=None):
        B, C, H, W = x.shape

        qkv = self.qkv(x)
        lepe = self.lepe(qkv[:, -C:, ...])
        q, k, v = rearrange(qkv, 'b (m n c) h w -> m b n h w c', m=3, n=self.num_heads)

        sin, cos = pos_enc
        q = theta_shift(q, sin, cos) * self.scale
        k = theta_shift(k, sin, cos)

        rpb = self.rpb[0] if hasattr(self, 'rpb') else None

        if self.fused_na:
            q = rearrange(q, 'b n h w c -> b h w n c')
            k = rearrange(k, 'b n h w c -> b h w n c')
            v = rearrange(v, 'b n h w c -> b h w n c')
            x = na2d(
                q, k, v,
                kernel_size=toodd(self.window_size),
                dilation=self.window_dilation,
                scale=float(q.size(-1)**0.5))
            q = rearrange(q, 'b h w n c -> b n h w c')
            k = rearrange(k, 'b h w n c -> b n h w c')
            x = rearrange(x, 'b h w n c -> b n h w c')
        else:
            attn = na2d_qk(
                q, k,
                kernel_size=toodd(self.window_size),
                dilation=self.window_dilation,
                rpb=rpb)
            attn = torch.softmax(attn, dim=-1)
            x = na2d_av(
                attn, v,
                kernel_size=toodd(self.window_size),
                dilation=self.window_dilation)

        if not self.global_mode:
            q = rearrange(q, 'b n h w c -> b n c h w').contiguous()
            k = rearrange(k, 'b n h w c -> b n c h w').contiguous()
            v = rearrange(x, 'b n h w c -> b n c h w').contiguous()

            v_r = v.flatten(1, 2)
            v = self.dwconv(v_r)
            v = F.silu(v)
            if self.enable_rsm and m_bg is not None:
                v = self.rsm(v, m_bg)
            v = self.ssm(v)
            v = v.reshape(B, -1, H, W).contiguous()
            if self.enable_ltab:
                v = self.ltab(v)
            v = self.norm(v)
            x = v + v_r
        else:
            q = rearrange(q, 'b n h w c -> b n (h w) c')
            k = rearrange(k, 'b n h w c -> b n (h w) c')
            v = rearrange(x, 'b n h w c -> b n (h w) c')
            attn = einsum(q, k, 'b n l c, b n m c -> b n l m')
            if hasattr(self, 'rpb'):
                if attn.size(-1) != self.rpb[-1].size(1) or x.size(-2) != self.rpb[-1].size(2):
                    attn = attn + F.interpolate(
                        self.rpb[-1].unsqueeze(0),
                        size=attn.shape[2:],
                        mode='bicubic',
                        align_corners=False)
                else:
                    attn = attn + self.rpb[-1]
            attn = torch.softmax(attn, dim=-1)
            x = einsum(attn, v, 'b n l m, b n m c -> b n c l').reshape(
                B, -1, H, W).contiguous()

        x = x + lepe
        x = self.proj(x)
        return x


class BlockLASS(nn.Module):

    def __init__(self,
                 image_size=None,
                 embed_dim=64,
                 num_heads=2,
                 window_size=7,
                 window_dilation=1,
                 global_mode=False,
                 use_rpb=False,
                 sr_ratio=1,
                 ffn_dim=256,
                 drop_path=0,
                 layerscale=False,
                 layer_init_values=1e-6,
                 norm_layer=LayerNorm2d,
                 fused_na=False,
                 ssm_ratio=1.0,
                 ssm_split=False,
                 enable_ltab=True,
                 enable_rsm=True,
                 ltab_cfg=None,
                 rsm_cfg=None):
        super().__init__()
        self.layerscale = layerscale
        self.embed_dim = embed_dim
        self.cpe1 = nn.Conv2d(
            embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.norm1 = norm_layer(embed_dim)
        self.token_mixer = AttentionLASS(
            embed_dim,
            num_heads,
            window_size,
            window_dilation,
            global_mode,
            image_size,
            use_rpb,
            sr_ratio,
            fused_na=fused_na,
            ssm_ratio=ssm_ratio,
            ssm_split=ssm_split,
            enable_ltab=enable_ltab,
            enable_rsm=enable_rsm,
            ltab_cfg=ltab_cfg,
            rsm_cfg=rsm_cfg,
        )
        self.cpe2 = nn.Conv2d(
            embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.norm2 = norm_layer(embed_dim)
        self.mlp = FFN(embed_dim, ffn_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        if layerscale:
            self.layer_scale1 = LayerScale(
                embed_dim, init_value=layer_init_values)
            self.layer_scale2 = LayerScale(
                embed_dim, init_value=layer_init_values)
        else:
            self.layer_scale1 = nn.Identity()
            self.layer_scale2 = nn.Identity()

    def forward(self, x, pos_enc, m_bg=None):
        x = x + self.cpe1(x)
        x = x + self.drop_path(
            self.layer_scale1(
                self.token_mixer(self.norm1(x), pos_enc, m_bg=m_bg)))
        x = x + self.cpe2(x)
        x = x + self.drop_path(self.layer_scale2(self.mlp(self.norm2(x))))
        return x


class BasicLayerLASS(nn.Module):

    def __init__(self,
                 image_size=None,
                 embed_dim=64,
                 depth=4,
                 num_heads=4,
                 window_size=7,
                 window_dilation=1,
                 global_mode=False,
                 use_rpb=False,
                 sr_ratio=1,
                 ffn_dim=96,
                 drop_path=0,
                 layerscale=False,
                 layer_init_values=1e-6,
                 norm_layer=LayerNorm2d,
                 use_checkpoint=0,
                 ssm_ratio=1.0,
                 ssm_split=False,
                 fused_na=False,
                 enable_ltab=True,
                 enable_rsm=True,
                 ltab_cfg=None,
                 rsm_cfg=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.rope = RoPE(embed_dim, num_heads)
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = BlockLASS(
                embed_dim=embed_dim,
                num_heads=num_heads,
                window_size=window_size,
                window_dilation=window_dilation,
                global_mode=global_mode,
                ffn_dim=ffn_dim,
                drop_path=drop_path[i]
                if isinstance(drop_path, list) else drop_path,
                layerscale=layerscale,
                layer_init_values=layer_init_values,
                norm_layer=norm_layer,
                image_size=image_size,
                use_rpb=use_rpb,
                sr_ratio=sr_ratio,
                ssm_ratio=ssm_ratio,
                ssm_split=ssm_split,
                fused_na=fused_na,
                enable_ltab=enable_ltab,
                enable_rsm=enable_rsm,
                ltab_cfg=ltab_cfg,
                rsm_cfg=rsm_cfg,
            )
            self.blocks.append(block)

    def forward(self, x, m_bg=None):
        pos_enc = self.rope((x.shape[2:]))
        for i, blk in enumerate(self.blocks):
            if i < self.use_checkpoint and x.requires_grad:
                x = checkpoint.checkpoint(
                    blk, x, pos_enc, m_bg, use_reentrant=False)
            else:
                x = blk(x, pos_enc, m_bg=m_bg)
        return x


class SegMANEncoderLASS(SegMANEncoder):
    """SegMAN encoder with LASS (LTAB + RSM) on selected stages."""

    def __init__(self,
                 image_size=224,
                 in_chans=3,
                 num_classes=1000,
                 embed_dims=None,
                 depths=None,
                 num_heads=None,
                 window_size=None,
                 window_dilation=None,
                 use_rpb=False,
                 sr_ratio=None,
                 mlp_ratios=None,
                 drop_path_rate=0,
                 projection=1024,
                 layerscales=None,
                 layer_init_values=1e-6,
                 norm_layer=LayerNorm2d,
                 drop_rate=0,
                 use_checkpoint=None,
                 ssm_split=False,
                 fused_na=False,
                 ssm_ratio=1.0,
                 pretrained=None,
                 lass_cfg=None,
                 **kwargs):
        self.lass_cfg_parsed = _parse_lass_cfg(lass_cfg)
        nn.Module.__init__(self)

        embed_dims = embed_dims or [64, 128, 256, 512]
        depths = depths or [2, 2, 6, 2]
        num_heads = num_heads or [2, 4, 8, 16]
        window_size = window_size or [7, 7, 7, 7]
        window_dilation = window_dilation or [1, 1, 1, 1]
        sr_ratio = sr_ratio or [8, 4, 2, 1]
        mlp_ratios = mlp_ratios or [4, 4, 4, 4]
        layerscales = layerscales or [False] * 4
        use_checkpoint = use_checkpoint or [0, 0, 0, 0]

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dims[0]
        self.num_features = embed_dims[-1]
        self.mlp_ratios = mlp_ratios
        self.pretrained = pretrained
        self.dilate_kernel = self.lass_cfg_parsed['dilate_kernel']
        self.seg_ignore_index = self.lass_cfg_parsed['ignore_index']

        self.patch_embed = stem(in_chans=in_chans, embed_dim=embed_dims[0])

        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        image_size = to_2tuple(image_size)
        image_size = [
            (image_size[0] // 2**(i + 2), image_size[1] // 2**(i + 2))
            for i in range(4)
        ]

        cfg = self.lass_cfg_parsed
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            use_lass = (i_layer in cfg['enable_stages']) and (i_layer < 3)
            if use_lass:
                layer = BasicLayerLASS(
                    embed_dim=embed_dims[i_layer],
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size[i_layer],
                    window_dilation=window_dilation[i_layer],
                    global_mode=False,
                    use_rpb=use_rpb,
                    sr_ratio=sr_ratio[i_layer],
                    ffn_dim=int(mlp_ratios[i_layer] * embed_dims[i_layer]),
                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    layerscale=layerscales[i_layer],
                    layer_init_values=layer_init_values,
                    norm_layer=norm_layer,
                    image_size=image_size[i_layer],
                    use_checkpoint=use_checkpoint[i_layer],
                    ssm_split=ssm_split,
                    ssm_ratio=ssm_ratio,
                    fused_na=fused_na,
                    enable_ltab=cfg['enable_ltab'],
                    enable_rsm=cfg['enable_rsm'],
                    ltab_cfg=cfg['ltab_cfg'],
                    rsm_cfg=cfg['rsm_cfg'],
                )
            else:
                layer = BasicLayer(
                    embed_dim=embed_dims[i_layer],
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size[i_layer],
                    window_dilation=window_dilation[i_layer],
                    global_mode=(i_layer == 3),
                    use_rpb=use_rpb,
                    sr_ratio=sr_ratio[i_layer],
                    ffn_dim=int(mlp_ratios[i_layer] * embed_dims[i_layer]),
                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    layerscale=layerscales[i_layer],
                    layer_init_values=layer_init_values,
                    norm_layer=norm_layer,
                    image_size=image_size[i_layer],
                    use_checkpoint=use_checkpoint[i_layer],
                    ssm_split=ssm_split,
                    ssm_ratio=ssm_ratio,
                    fused_na=fused_na,
                )

            downsample = nn.Sequential(
                nn.Conv2d(
                    embed_dims[i_layer],
                    embed_dims[i_layer + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                nn.BatchNorm2d(embed_dims[i_layer + 1]),
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers.append(layer)
            self.layers.append(downsample)

        self.classifier = nn.Sequential(
            nn.Conv2d(self.num_features, projection, kernel_size=1),
            nn.BatchNorm2d(projection),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(projection, num_classes, kernel_size=1)
            if num_classes > 0 else nn.Identity(),
        )
        del self.classifier

        self.apply(self._init_weights)
        if torch.distributed.is_initialized():
            self = nn.SyncBatchNorm.convert_sync_batchnorm(self)

    def forward_features(self, x, seg_map=None):
        x = self.patch_embed(x)
        m_bg = None
        if seg_map is not None:
            m_bg = semantic_seg_to_mbg(
                seg_map,
                target_size=x.shape[-2:],
                dilate_kernel=self.dilate_kernel,
                ignore_index=self.seg_ignore_index,
            )

        outs = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, BasicLayerLASS):
                if m_bg is not None:
                    m_bg_i = F.interpolate(
                        m_bg,
                        size=x.shape[-2:],
                        mode='nearest',
                        align_corners=None)
                    x = layer(x, m_bg=m_bg_i)
                else:
                    x = layer(x, m_bg=None)
            else:
                x = layer(x)
            if i % 2 == 0:
                outs.append(x)
        return outs

    def forward(self, x, seg_map=None):
        return self.forward_features(x, seg_map=seg_map)


@BACKBONES.register_module()
def SegMANEncoderLASS_b(pretrained=None, pretrained_cfg=None, lass_cfg=None,
                        **args):
    model = SegMANEncoderLASS(
        embed_dims=[96, 160, 364, 560],
        depths=[4, 4, 18, 4],
        num_heads=[4, 8, 13, 20],
        window_size=[11, 9, 7, 7],
        window_dilation=[1, 1, 1, 1],
        mlp_ratios=[4, 4, 3, 3],
        layerscales=[True, True, True, True],
        layer_init_values=1e-6,
        use_rpb=True,
        pretrained=pretrained,
        norm_layer=LayerNorm2d,
        lass_cfg=lass_cfg,
        **args,
    )
    model.default_cfg = _cfg(crop_pct=0.95)
    return model
