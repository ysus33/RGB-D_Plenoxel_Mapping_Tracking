import torch
import torch.nn as nn


class Criterion(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.rgb_weight = args.criteria["rgb_weight"]
        self.depth_weight = args.criteria["depth_weight"]
        self.max_dpeth = args.data_specs["max_depth"]

    def forward(self, pred_color, pred_depth, 
                img, depth, 
                use_color_loss=True,
                use_depth_loss=True):
        loss = 0
        loss_dict = {}

        # ray_mask = outputs["ray_mask"] # for transmittance masking
        # gt_depth = depth[ray_mask]
        # gt_color = img[ray_mask]
        gt_depth = depth
        gt_color = img

        if use_color_loss:
            color_loss = (gt_color - pred_color).abs().mean()
            loss += self.rgb_weight * color_loss
            loss_dict["color_loss"] = color_loss.item()

        if use_depth_loss:
            valid_depth = (gt_depth > 0.01) & (gt_depth < self.max_dpeth)
            depth_loss = (gt_depth - pred_depth).abs()
            depth_loss = depth_loss[valid_depth].mean()
            loss += self.depth_weight * depth_loss
            loss_dict["depth_loss"] = depth_loss.item()

        loss_dict["loss"] = loss.item()
        return loss, loss_dict