# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn
# import torch.nn.functional as F
from fast_reid.fastreid.config import configurable
from fast_reid.fastreid.modeling.backbones import build_backbone
from fast_reid.fastreid.modeling.heads import build_heads
from fast_reid.fastreid.modeling.losses import *
# from torch.nn import Parameter
import math
from .build import META_ARCH_REGISTRY


class CombinedMarginLoss(nn.Module):
    def __init__(self, s, m1, m2, m3, interclass_filtering_threshold=0):
        super().__init__()
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.interclass_filtering_threshold = interclass_filtering_threshold

        self.cos_m = math.cos(self.m2)
        self.sin_m = math.sin(self.m2)
        self.theta = math.cos(math.pi - self.m2)
        self.sinmm = math.sin(math.pi - self.m2) * self.m2
        self.easy_margin = False

    def forward(self, logits, labels):
        logits = logits.float()
        index_positive = torch.where(labels != -1)[0]

        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                dirty = logits > self.interclass_filtering_threshold
                dirty = dirty.float()
                mask = torch.ones([index_positive.size(0), logits.size(1)], device=logits.device)
                mask.scatter_(1, labels[index_positive].view(-1, 1), 0)
                dirty[index_positive] *= mask
                tensor_mul = 1 - dirty    
            logits = tensor_mul * logits

        target_logit = logits[index_positive, labels[index_positive].view(-1)]

        if self.m1 == 1.0 and self.m3 == 0.0:
            target_logit_arccos = torch.clamp(target_logit, -1 + 1e-7, 1 - 1e-7).arccos()
            logits_arccos = torch.clamp(logits, -1 + 1e-7, 1 - 1e-7).arccos()
            final_target_logit = target_logit_arccos + self.m2
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            logits = logits_arccos.cos()
            logits = logits * self.s
        elif self.m3 > 0:
            final_target_logit = target_logit - self.m3
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            logits = logits * self.s
        else:
            raise ValueError("Invalid margin parameters.")

        # if torch.isnan(logits).any() or torch.isinf(logits).any():
        #     print("Combined Margin Loss computation resulted in NaN or Inf values.")
        #     print(f"Logits: {logits}")
        #     print(f"Target Logit: {target_logit}")

        logits = torch.relu(logits)  # 确保返回非负值

        return logits.half()

class ArcFace(nn.Module):
    def __init__(self, s=64.0, margin=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        logits = logits.float()
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        target_logit_arccos = torch.clamp(target_logit, -1 + 1e-7, 1 - 1e-7).arccos()
        logits_arccos = torch.clamp(logits, -1 + 1e-7, 1 - 1e-7).arccos()
        final_target_logit = target_logit_arccos + self.margin
        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits_arccos.cos()
        logits = logits * self.s
        
        # if torch.isnan(logits).any() or torch.isinf(logits).any():
        #     print("ArcFace Loss computation resulted in NaN or Inf values.")
        #     print(f"Logits: {logits}")
        #     print(f"Target Logit: {target_logit}")

        logits = torch.relu(logits)  # 确保返回非负值

        return logits.half() 
    
    
@META_ARCH_REGISTRY.register()
class Baseline(nn.Module):
    """
    Baseline architecture. Any models that contains the following two components:
    1. Per-image feature extraction (aka backbone)
    2. Per-image feature aggregation and loss computation
    """

    @configurable
    def __init__(
            self,
            *,
            backbone,
            heads,
            pixel_mean,
            pixel_std,
            loss_kwargs=None,
            arcface=None,
            combined_margin_loss=None
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone:
            heads:
            pixel_mean:
            pixel_std:
            arcface: ArcFace loss function
            combined_margin_loss: CombinedMarginLoss function
        """
        super().__init__()
        # backbone
        self.backbone = backbone

        # head
        self.heads = heads

        self.loss_kwargs = loss_kwargs
        self.arcface = arcface
        self.combined_margin_loss = combined_margin_loss

        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(1, -1, 1, 1), False)


    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        heads = build_heads(cfg)
        arcface = ArcFace(cfg.MODEL.LOSSES.ARCFACE.SCALE, cfg.MODEL.LOSSES.ARCFACE.MARGIN)
        combined_margin_loss = CombinedMarginLoss(
            cfg.MODEL.LOSSES.COMBINED_MARGIN_LOSS.SCALE,
            cfg.MODEL.LOSSES.COMBINED_MARGIN_LOSS.M1,
            cfg.MODEL.LOSSES.COMBINED_MARGIN_LOSS.M2,
            cfg.MODEL.LOSSES.COMBINED_MARGIN_LOSS.M3,
        )
        return {
            'backbone': backbone,
            'heads': heads,
            'pixel_mean': cfg.MODEL.PIXEL_MEAN,
            'pixel_std': cfg.MODEL.PIXEL_STD,
            'arcface': arcface,
            'combined_margin_loss': combined_margin_loss,
            'loss_kwargs': {
                # loss name
                'loss_names': cfg.MODEL.LOSSES.NAME,

                # loss hyperparameters
                'ce': {
                    'eps': cfg.MODEL.LOSSES.CE.EPSILON,
                    'alpha': cfg.MODEL.LOSSES.CE.ALPHA,
                    'scale': cfg.MODEL.LOSSES.CE.SCALE
                },
                'tri': {
                    'margin': cfg.MODEL.LOSSES.TRI.MARGIN,
                    'norm_feat': cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                    'hard_mining': cfg.MODEL.LOSSES.TRI.HARD_MINING,
                    'scale': cfg.MODEL.LOSSES.TRI.SCALE
                },
                'circle': {
                    'margin': cfg.MODEL.LOSSES.CIRCLE.MARGIN,
                    'gamma': cfg.MODEL.LOSSES.CIRCLE.GAMMA,
                    'scale': cfg.MODEL.LOSSES.CIRCLE.SCALE
                },
                'cosface': {
                    'margin': cfg.MODEL.LOSSES.COSFACE.MARGIN,
                    'gamma': cfg.MODEL.LOSSES.COSFACE.GAMMA,
                    'scale': cfg.MODEL.LOSSES.COSFACE.SCALE
                },
            }
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images)

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"]
            outputs = self.heads(features, targets)
            losses = self.losses(outputs, targets)
            return losses
        else:
            outputs = self.heads(features)
            return outputs

    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs['images']
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))

        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def losses(self, outputs, gt_labels):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # model predictions
        # fmt: off
        pred_class_logits = outputs['pred_class_logits'].detach()
        cls_outputs       = outputs['cls_outputs']
        pred_features     = outputs['features']
        # fmt: on
        # print(f"Pred Class Logits: {pred_class_logits}")
        # print(f"GT Labels: {gt_labels}")
        # Log prediction accuracy
        log_accuracy(pred_class_logits, gt_labels)

        loss_dict = {}
        loss_names = self.loss_kwargs['loss_names']

        if 'CrossEntropyLoss' in loss_names:
            ce_kwargs = self.loss_kwargs.get('ce')
            loss_dict['loss_cls'] = cross_entropy_loss(
                cls_outputs,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale')

        if 'TripletLoss' in loss_names:
            tri_kwargs = self.loss_kwargs.get('tri')
            loss_dict['loss_triplet'] = triplet_loss(
                pred_features,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale')

        if 'CircleLoss' in loss_names:
            circle_kwargs = self.loss_kwargs.get('circle')
            loss_dict['loss_circle'] = pairwise_circleloss(
                pred_features,
                gt_labels,
                circle_kwargs.get('margin'),
                circle_kwargs.get('gamma')
            ) * circle_kwargs.get('scale')

        if 'Cosface' in loss_names:
            cosface_kwargs = self.loss_kwargs.get('cosface')
            loss_dict['loss_cosface'] = pairwise_cosface(
                pred_features,
                gt_labels,
                cosface_kwargs.get('margin'),
                cosface_kwargs.get('gamma'),
            ) * cosface_kwargs.get('scale')

           # Adding custom losses
        if 'CombinedMarginLoss' in loss_names:
            combined_margin_loss = self.combined_margin_loss(
                pred_class_logits, gt_labels
            )
            # if torch.isnan(combined_margin_loss).any():
            #     print("Combined Margin Loss contains NaN values!")
            loss_dict['loss_combined_margin'] = combined_margin_loss.mean()  # 確保需要梯度

        if 'ArcFace' in loss_names:
            arcface_loss = self.arcface(
                pred_class_logits, gt_labels
            )
            # if torch.isnan(arcface_loss).any():
            #     print("ArcFace Loss contains NaN values!")
            loss_dict['loss_arcface'] = arcface_loss.mean()  # 確保需要梯度

        return loss_dict

