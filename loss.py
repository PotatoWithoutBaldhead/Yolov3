import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import build_targets
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class YOLOv3Loss(nn.Module):
    def __init__(self, num_classes, anchors, image_size, ignore_thresh=0.5):
        super(YOLOv3Loss, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.anchors_num = len(anchors)
        self.image_size = image_size
        self.ignore_thresh = ignore_thresh
        self.obj_scale = 1
        self.noobj_scale = 0.5
        
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss(reduction='sum')
        
    # def compute_iou(self, box1, box2):
    #     b1_x1, b1_y1 = box1[..., 0] - box1[..., 2] / 2, box1[..., 1] - box1[..., 3] / 2
    #     b1_x2, b1_y2 = box1[..., 0] + box1[..., 2] / 2, box1[..., 1] + box1[..., 3] / 2
    #     b2_x1, b2_y1 = box2[..., 0] - box2[..., 2] / 2, box2[..., 1] - box2[..., 3] / 2
    #     b2_x2, b2_y2 = box2[..., 0] + box2[..., 2] / 2, box2[..., 1] + box2[..., 3] / 2

    #     inter_rect_x1 = torch.max(b1_x1, b2_x1)
    #     inter_rect_y1 = torch.max(b1_y1, b2_y1)
    #     inter_rect_x2 = torch.min(b1_x2, b2_x2)
    #     inter_rect_y2 = torch.min(b1_y2, b2_y2)
    #     inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)

    #     b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    #     b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    #     union_area = b1_area + b2_area - inter_area

    #     return inter_area / union_area
    
    def forward(self, pred, targets):
        batch_size = pred.size(0)
        grid_size = pred.size(2)
        
        stride = self.image_size / grid_size
        
        x = torch.sigmoid(pred[..., 0])
        y = torch.sigmoid(pred[..., 1])
        w = pred[..., 2]
        h = pred[..., 3]
        conf = torch.sigmoid(pred[..., 4])
        pred_cls = torch.sigmoid(pred[..., 5:])
        
        grid_x = torch.arange(grid_size, dtype=torch.float32).repeat(grid_size, 1).view([1, 1, grid_size, grid_size]).type_as(pred)
        grid_y = torch.arange(grid_size, dtype=torch.float32).repeat(grid_size, 1).t().view([1, 1, grid_size, grid_size]).type_as(pred)
        scaled_anchors = [(a_w / stride, a_h / stride) for a_w, a_h in self.anchors]
        
        anchor_w = torch.tensor([anchor[0] for anchor in scaled_anchors], dtype=torch.float32, device='cuda').view(1, self.anchors_num, 1, 1)
        anchor_h = torch.tensor([anchor[1] for anchor in scaled_anchors], dtype=torch.float32, device='cuda').view(1, self.anchors_num, 1, 1)
        
        pred_boxes = torch.zeros_like(pred[..., :4])
        pred_boxes[..., 0] = x + grid_x
        pred_boxes[..., 1] = y + grid_y
        pred_boxes[..., 2] = torch.exp(w) * anchor_w
        pred_boxes[..., 3] = torch.exp(h) * anchor_h
        
        obj_mask = torch.zeros_like(conf)
        noobj_mask = torch.ones_like(conf)
        tx = torch.zeros_like(x)
        ty = torch.zeros_like(y)
        tw = torch.zeros_like(w)
        th = torch.zeros_like(h)
        tconf = torch.zeros_like(conf)
        tcls = torch.zeros_like(pred_cls)
        
        # target_boxes = targets[:, 2:6] * grid_size
        obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(pred_boxes, pred_cls, targets, scaled_anchors, 0.5, self.image_size)
        
        obj_mask = obj_mask.bool()
        ## 计算loss
        # 计算目标框的loss
        loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
        loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
        loss_w = self.mse_loss((torch.exp(w) * anchor_w)[obj_mask], tw[obj_mask])
        loss_h = self.mse_loss((torch.exp(h) * anchor_h)[obj_mask], th[obj_mask])
        
        loss_conf_obj = self.bce_loss(conf[obj_mask], tconf[obj_mask])
        loss_conf_noobj = self.bce_loss(conf[noobj_mask], tconf[noobj_mask])
        loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
        loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
        loss = 5 * loss_x + 5 * loss_y + 5 * loss_w + 5 * loss_h + loss_conf + loss_cls
        
        
        return loss
    