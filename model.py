import torch
import torch.nn as nn
   
class DBL(nn.Module):
    def __init__(self, input_ch, output_ch, kernel=3, stride=1, padding=0):
        super(DBL, self).__init__()
        self.con = nn.Sequential(
            nn.Conv2d(input_ch, output_ch, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(output_ch),
            nn.LeakyReLU()
        )
    def forward(self, x):
        return self.con(x)
       
class res_block(nn.Module): #  res block
    def __init__(self, input_ch, output_ch):
        super(res_block, self).__init__()
        self.block = nn.Sequential(
            DBL(input_ch, output_ch, kernel=1),
            DBL(output_ch, input_ch, kernel=3, padding=1)
        )
    def forward(self, x):
        out = self.block(x)
        return x + out

class DarkNet53(nn.Module):
    def __init__(self):
        super(DarkNet53, self).__init__()
        self.conv1 = DBL(3, 32, kernel=3, stride=1, padding=1)
        self.con2 = DBL(32, 64, kernel=3, stride=2, padding=1)
        self.res_block1 = self._make_res_block(64, 32)
        self.conv3 = DBL(64, 128, kernel=3, stride=2, padding=1)
        self.res_block2 = self._make_res_block(128, 64, 2)
        self.conv4 = DBL(128, 256, kernel=3, stride=2, padding=1)
        self.res_block3 = self._make_res_block(256, 128, 8)
        self.conv5 = DBL(256, 512, kernel=3, stride=2, padding=1)
        self.res_block4 = self._make_res_block(512, 256, 8)
        self.conv6 = DBL(512, 1024, kernel=3, stride=2, padding=1)
        self.res_bloc5 = self._make_res_block(1024, 512, 4)
        
    def _make_res_block(self, input_ch, output_ch, nums=1):
        layers = []
        for _ in range(nums):
            layers.append(res_block(input_ch, output_ch))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.con2(x)
        x = self.res_block1(x)
        x = self.conv3(x)
        x = self.res_block2(x)
        x = self.conv4(x)
        x = self.res_block3(x)
        output1 = x  # 第一幅特征图 n*52*52*256
        x = self.conv5(x)
        x = self.res_block4(x)
        output2 = x  # 第二幅特征图 n*26*26*512
        x = self.conv6(x)
        x = self.res_bloc5(x)
        output3 = x  # 第三幅特征图 n*13*13*1024
               
        return output1, output2, output3     
       
class YOLOv3_Layer(nn.Module):
    def __init__(self, in_channel, anchors, num_classes, img_dim=416):
        super(YOLOv3_Layer, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = len(anchors)
        self.anchors = anchors
        self.ignore_threshold = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0
        self.conv = nn.Conv2d(in_channel, self.num_anchors * (self.num_classes + 5), 1)
        
    def compute_grid_offsets(self, grid_size):
        self.grid_size = grid_size
        g = self.grid_size
        self.stride = self.img_dim / self.grid_size
        #将网格做成坐标盘
        self.grid_x = torch.arange(g).repeat(g).view([1, 1, g, g]).type(torch.Tensor).to('cuda')
        self.grid_y = torch.arange(g).repeat(g).t().view([1, 1, g, g]).type(torch.Tensor).to('cuda')
        #实际anchors框的大小经过比例缩放后，与grid的比例尺一致
        self.scaled_anchors = torch.Tensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors]).to('cuda')
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1)).to('cuda')
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1)).to('cuda')
    
    def forward(self, x, img_dim = 416):
        x = self.conv(x)
        batch_size = x.size(0)
        grid_size = x.size(2)
        
        self.img_dim = img_dim
        batch_size = x.size(0)
        grid_size = x.size(2)
        
        pred = (
            x.view(batch_size, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
            )
        x = torch.sigmoid(pred[..., 0])
        y = torch.sigmoid(pred[..., 1])
        w = pred[..., 2]
        h = pred[..., 3]
        pred_conf = torch.sigmoid(pred[..., 4])
        pred_cls = torch.sigmoid(pred[..., 5:])
        
        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size)
            
        # pred_boxes = torch.zeros_like(pred[..., :4])
        # pred_boxes[..., 0] = x.data + self.grid_x
        # pred_boxes[..., 1] = y.data + self.grid_y
        # pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        # pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        # output = torch.cat( 
        #     (
        #         pred_boxes[..., :4] * self.stride, #还原到原始图中
        #         pred_conf.view(batch_size, 3, grid_size, grid_size, 1),
        #         pred_cls.view(batch_size, 3, grid_size, grid_size, self.num_classes),
        #     ),
        #     dim = -1,
        # )
        
        output = torch.cat((x.unsqueeze(-1),
                             y.unsqueeze(-1),
                             w.unsqueeze(-1),
                             h.unsqueeze(-1),
                             pred_conf.unsqueeze(-1),
                             pred_cls),
                           dim=-1)

        return output
    
class YOLOv3(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv3, self).__init__()
        self.backbone = DarkNet53()
        self.num_classes = num_classes
        
        self.yolo_layer1 = YOLOv3_Layer(256, anchors=[(10, 13), (16, 30), (33, 23)], num_classes=num_classes)
        self.yolo_layer2 = YOLOv3_Layer(128, anchors=[(30, 61), (62, 45), (59, 119)], num_classes=num_classes)
        self.yolo_layer3 = YOLOv3_Layer(512, anchors=[(116, 90), (156, 198), (373, 326)], num_classes=num_classes)
        
        # output3 layer DBL
        self.out3_layer5 = nn.Sequential(
            DBL(1024, 512, kernel=3, stride=1, padding=1),
            DBL(512, 1024, kernel=1),
            DBL(1024, 512, kernel=3, stride=1, padding=1),
            DBL(512, 1024, kernel=1),
            DBL(1024, 512, kernel=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=1)
        )
        self.out3_conv = DBL(1024, 512, kernel=3, stride=1, padding=1)
        # for cat
        self.conv3 = DBL(1024, 256, kernel=1, stride=1, padding=0)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')
        
        # output2 layer DBL
        self.out2_layer5 = nn.Sequential(
            DBL(768, 512, kernel=3, stride=1, padding=1),
            DBL(512, 256, kernel=1),
            DBL(256, 512, kernel=3, stride=1, padding=1),
            DBL(512, 256, kernel=1),
            DBL(256, 512, kernel=3, stride=1, padding=1),
            nn.Conv2d(512, 256, kernel_size=1)
        )
        self.out2_conv = DBL(256, 128, kernel=3, stride=1, padding=1)
        # for cat
        self.conv2 = DBL(256, 128, kernel=1, stride=1, padding=0)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')

        # output1 layer DBL
        self.out1_layer5 = nn.Sequential(
            DBL(384, 256, kernel=3, stride=1, padding=1),
            DBL(256, 384, kernel=1),
            DBL(384, 256, kernel=3, stride=1, padding=1),
            DBL(256, 384, kernel=1),
            DBL(384, 256, kernel=3, stride=1, padding=1),
            nn.Conv2d(256, 128, kernel_size=1)
        )
        self.out1_conv = DBL(128, 256, kernel=3, stride=1, padding=1)
        
    def forward(self, x):
        out1, out2, out3 = self.backbone(x)  # out1: 52*52*256 out2: 26*26*512 out3: 13*13*1024
        
        # output3
        out3 = self.out3_layer5(out3)
        detections3 = self.out3_conv(out3)
        detections3 = self.yolo_layer3(detections3)
        
        # unsample3 
        out3_for_cat = self.conv3(out3)
        out3_for_cat = self.upsample3(out3_for_cat)
        
        out2 = torch.cat([out3_for_cat, out2], dim=1)
        # output2
        out2 = self.out2_layer5(out2)
        detections2 = self.out2_conv(out2)
        detections2 = self.yolo_layer2(detections2)
        
        # unsample2
        out2_for_cat = self.conv2(out2)
        out2_for_cat = self.upsample2(out2_for_cat)
        out1 = torch.cat([out2_for_cat, out1], dim=1)
        
        out1 = self.out1_layer5(out1)
        detections1 = self.out1_conv(out1)
        detections1 = self.yolo_layer1(detections1)
        
        detections = [detections1, detections2, detections3]
        return detections

        
def print_shape(module, input, output):
    print(f"{module.__class__.__name__}: {input[0].shape}")
    print(f"{module.__class__.__name__}: {output.shape}")
       
if __name__ == "__main__":
    input = torch.rand((3, 3, 128, 128))
    temp = DBL(input_ch=3, output_ch=32, kernel=3,padding=1)
    for layer in temp.children():
        layer.register_forward_hook(print_shape)
    output = temp(input)
    

    