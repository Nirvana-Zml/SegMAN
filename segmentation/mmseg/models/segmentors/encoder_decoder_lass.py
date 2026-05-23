# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.ops import resize

from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder


@SEGMENTORS.register_module()
class EncoderDecoderLASS(EncoderDecoder):
    """EncoderDecoder that passes GT semantic mask to LASS backbone for RSM."""

    def extract_feat(self, img, seg_map=None):
        if seg_map is not None and seg_map.dim() == 4 and seg_map.size(1) == 1:
            seg_map = seg_map.squeeze(1)
        if seg_map is not None:
            seg_map = seg_map.long()
        x = self.backbone(img, seg_map=seg_map)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img, img_metas):
        x = self.extract_feat(img, seg_map=None)
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def forward_train(self, img, img_metas, gt_semantic_seg):
        seg_map = gt_semantic_seg.squeeze(1).long()
        x = self.extract_feat(img, seg_map=seg_map)
        losses = dict()
        loss_decode = self._decode_head_forward_train(
            x, img_metas, gt_semantic_seg)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)
        return losses
