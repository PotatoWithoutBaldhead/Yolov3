import torch
from PIL import Image
import torchvision.transforms.functional as F
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# def bbox_wh_iou(wh1, wh2):
#     wh2 = wh2.t()
#     w1, h1 = wh1[0], wh1[1]
#     w2, h2 = wh2[0], wh2[1]
#     inter_area = torch.min(w1, w2) * torch.min(h1, h2)
#     union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
#     return inter_area / union_area

def bbox_wh_iou(anchor, gwh):
    # anchor = torch.tensor(anchors).to(gwh.device)
    if anchor.dim() == 1:
        anchor = anchor.unsqueeze(0)  # 将 (2,) 转换为 (1, 2)
    # 计算所有 anchor 和 gwh 的交集区域
    inter_w = torch.min(anchor[:, 0].unsqueeze(0), gwh[:, 0])
    inter_h = torch.min(anchor[:, 1].unsqueeze(0), gwh[:, 1])
    inter_area = inter_w * inter_h

    # 计算所有 anchor 和 gwh 的联合区域
    anchor_area = anchor[:, 0].unsqueeze(1) * anchor[:, 1].unsqueeze(1)
    gwh_area = gwh[:, 0] * gwh[:, 1]
    union_area = anchor_area + gwh_area - inter_area

    # 计算 IOU
    iou = inter_area / (union_area + 1e-16)
    return iou


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two boundigrid_size boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordianchors_numtes
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordianchors_numtes of boundigrid_size boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdianchors_numtes of the intersection rectagrid_sizele
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres, img_size):
    batch_size = pred_boxes.size(0)  # batch size
    anchors_num = pred_boxes.size(1)  # number of anchors
    classes_num = pred_cls.size(-1)  # number of classes
    grid_size = pred_boxes.size(2)  # grid size

    # Output tensors
    obj_mask = torch.zeros((batch_size, anchors_num, grid_size, grid_size), dtype=torch.int, requires_grad=False, device='cuda')  # anchor包含物体则为1，默认为0，考虑前景
    noobj_mask = torch.ones((batch_size, anchors_num, grid_size, grid_size), dtype=torch.int, requires_grad=False, device='cuda')  # anchor不包含物体则为1，默认为1，考虑背景
    # class_mask = torch.zeros(batch_size, anchors_num, grid_size, grid_size, requires_grad=False)  # anchor包含物体的类别, 类别正确则为1，默认为0
    # iou_scores = torch.zeros(batch_size, anchors_num, grid_size, grid_size, requires_grad=False)  # 预测框与真实框的iou得分
    
    tx = torch.zeros((batch_size, anchors_num, grid_size, grid_size), requires_grad=False, device='cuda')  # 真实框相对于网格的位置
    ty = torch.zeros((batch_size, anchors_num, grid_size, grid_size), requires_grad=False, device='cuda')  
    tw = torch.zeros((batch_size, anchors_num, grid_size, grid_size), requires_grad=False, device='cuda')  
    th = torch.zeros((batch_size, anchors_num, grid_size, grid_size), requires_grad=False, device='cuda') 
    tcls = torch.zeros((batch_size, anchors_num, grid_size, grid_size, classes_num), requires_grad=False, device='cuda')  # 真实框的类别
    
    target_boxes = target[:, 2:6]  # 真实框相对于网格的位置
    gxy = target_boxes[:, :2]  # 真实框的中心点坐标
    gwh = target_boxes[:, 2:]  # 真实框的宽高
    
    anchors = torch.tensor(anchors, device='cuda')
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors]).squeeze(1)  # 计算每个anchor与真实框的iou
    _, best_n = ious.max(0)
    
    stride = img_size / grid_size
    
    b, target_labels = target[:, :2].long().t()  # 真实框所对应的batch，以及每个框所代表的实际类别
    gx, gy = (gxy / stride).t()  # 真实框在特征图上的中心点坐标
    gw, gh = (gwh / stride).t()  # 真实框在特征图上的宽高
    gi, gj = (gxy / stride).long().t() #位置信息, 向下取整
    
    obj_mask[b, best_n, gj, gi] = 1  # 包含物体的mask
    noobj_mask[b, best_n, gj, gi] = 0  # 不包含物体的mask
    
    for i, anchor_ious in enumerate(ious.t()): # IOU超过了指定的阈值就相当于有物体了
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0
    
    tx[b, best_n, gj, gi] = gx - gi  # 真实框的中心点x坐标相对于网格的偏移量
    ty[b, best_n, gj, gi] = gy - gj  # 真实框的中心点y坐标相对于网格的偏移量
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)

    tcls[b, best_n, gj, gi, target_labels] = 1  # 真实框的类别（使用独热编码）
    # class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    # iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False) #与真实框想匹配的预测框之间的iou值
    
    tconf = obj_mask.float()
    
    return obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf
    
def resize_and_pad(image, boxes, target_size):
    origianchors_numl_size = image.size
    ratio = min(target_size[0] / origianchors_numl_size[0], target_size[1] / origianchors_numl_size[1])
    new_size = (int(origianchors_numl_size[0] * ratio), int(origianchors_numl_size[1] * ratio))
    
    # Resize the image
    image = F.resize(image, new_size)
    
    # Create a new image with the target size and paste the resized image onto it
    new_image = Image.new("RGB", target_size)
    new_image.paste(image, ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2))
    
    # Adjust boundigrid_size boxes
    new_boxes = []
    dx = (target_size[0] - new_size[0]) // 2
    dy = (target_size[1] - new_size[1]) // 2
    for box in boxes:
        x_center, y_center, width, height = box
        x_center = x_center * ratio + dx
        y_center = y_center * ratio + dy
        width = width * ratio
        height = height * ratio
        new_boxes.append([x_center, y_center, width, height])
    
    
    return new_image, new_boxes
    
    