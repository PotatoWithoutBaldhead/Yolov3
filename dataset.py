import os
from PIL import Image
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
from utils import resize_and_pad
import os
import xml.etree.cElementTree as ET
 
class_code = {"person": 0, "bird": 1, "cat": 2, "cow": 3, "dog": 4, "horse": 5, "sheep": 6, "aeroplane": 7,
              "bicycle": 8, "boat": 9, "bus": 10, "car": 11, "motorbike": 12, "train": 13, "bottle": 14,
              "chair": 15, "diningtable": 16, "pottedplant": 17, "sofa": 18, "tvmonitor": 19}
 
class GetobjectdetectiondatafromVoc2012():
    def __init__(self, base_root):
        self.baseroot = base_root
 
    def findobjectdetectiondatafromvoc2012(self):                    # 在VOC2012找到目标检测相关dataset划分并分类返回
        allobjtxtlsit = []
        obj_traintxtfile = os.path.join(self.baseroot, "ImageSets", "Main", "train.txt")
        obj_valtxtfile = os.path.join(self.baseroot, "ImageSets", "Main", "val.txt")
        obj_trainvaltxtfile = os.path.join(self.baseroot, "ImageSets", "Main", "trainval.txt")
        allobjtxtlsit.append(obj_traintxtfile)
        allobjtxtlsit.append(obj_valtxtfile)
        allobjtxtlsit.append(obj_trainvaltxtfile)
        # print(allobjtxtlsit)
 
        datanames = {}
        for txtfilename in allobjtxtlsit:
            with open(txtfilename, "r") as f:
                txtfiledirname = os.path.basename(txtfilename)
                datanames[txtfiledirname] = f.readlines()
 
        return datanames["train.txt"], datanames["val.txt"], datanames["trainval.txt"]
 
    def __call__(self, target_dir, mode):
        if os.path.exists(target_dir):                                     # 存放新txt目标文件夹检测，若没有则创建
            print("lables数据文件夹已存在：{}".format(target_dir))
        else:
            os.mkdir(target_dir)
            print("lables数据文件夹已创立：{}".format(target_dir))
        traindatanameslist, valdatanameslist, trainvaldatanameslist = self.findobjectdetectiondatafromvoc2012()
 
        target_dataset = []
        if mode == "train":
            target_dataset.append(traindatanameslist)
        elif mode == "val":
            target_dataset.append(valdatanameslist)
        elif mode == "trainval":
            target_dataset.append(trainvaldatanameslist)
        elif mode == "all":
            target_dataset.append(traindatanameslist)
            target_dataset.append(valdatanameslist)
            target_dataset.append(trainvaldatanameslist)
 
        order = ["train", "val", "trainval"]
        txtfilename_index = 0
        for mission in target_dataset:
            picnum = 0
            boxnum = 0
 
            if mode == "all":
                txtfilename = order[txtfilename_index]
            else:
                txtfilename = mode
 
            for datanames in mission:
                lable = []
 
                lable.append(datanames[:-1] + ".jpg")
                xml_file = os.path.join(self.baseroot, "Annotations", datanames[:-1] + ".xml")
                tree = ET.parse(xml_file)
                try:
                    for obj in tree.iter("object"):
                        lable.append(class_code[obj.findtext("name")])
 
                        xmin = int(obj.findtext("bndbox/xmin"))
                        ymin = int(obj.findtext("bndbox/ymin"))
                        xmax = int(obj.findtext("bndbox/xmax"))
                        ymax = int(obj.findtext("bndbox/ymax"))
 
                        cx = (xmin + xmax) / 2
                        cy = (ymin + ymax) / 2
                        w = xmax - xmin
                        h = ymax - ymin
                        lable.append(str(cx))
                        lable.append(str(cy))
                        lable.append(str(w))
                        lable.append(str(h))
 
                        boxnum += 1
 
                except:
                    print(lable[0])
 
                with open(os.path.join(target_dir, txtfilename + "_annotations.txt"), "a+") as f:
                    lable = [str(e) + " " for e in lable]
                    lable.append("\n")
                    f.writelines(lable)
 
                picnum += 1
            print("{}_labes.txt成功共转化{}张图片的标签，共计{}个物体框".format(txtfilename, picnum, boxnum))
            txtfilename_index += 1
 
 
def txtGet():  # 从xml文件中读取box和label，并写入txt文件
    database_root = "D:\yyt_code\DeepLearning\VOC2012\VOC2012"          # VOC2012数据文件夹地址
    target_root = "D:\yyt_code\DeepLearning\Yolo\label_txt"         # 存放txt文件地址
    labletxtgentor = GetobjectdetectiondatafromVoc2012(database_root)
    labletxtgentor(target_root, "all")
    
class YOLODataset(Dataset):
    def __init__(self, annotations_file, images_dir, transform=None):
        with open(annotations_file, 'r') as f:
            self.lines = f.readlines()
        
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip().split()
        image_file = line[0]
        image_path = os.path.join(self.images_dir, image_file)
        
        image = Image.open(image_path).convert("RGB")
        boxes = []
        labels = []
        
        annotations = line[1:]
        for i in range(0, len(annotations), 5):
            class_id = int(annotations[i])
            x_center = float(annotations[i+1])
            y_center = float(annotations[i+2])
            width = float(annotations[i+3])
            height = float(annotations[i+4])
            
            # Convert YOLO format (x_center, y_center, width, height) to (x_min, y_min, x_max, y_max)
            # x_min = x_center - width / 2
            # y_min = y_center - height / 2
            # x_max = x_center + width / 2
            # y_max = y_center + height / 2
            
            boxes.append([x_center, y_center, width, height])
            labels.append(class_id)
        new_image, new_boxes = resize_and_pad(image, boxes, (416,416))
        # image.show()
        # new_image.show()
        new_image = F.to_tensor(new_image) 
        new_boxes = torch.as_tensor(new_boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {}
        target["boxes"] = new_boxes
        target["labels"] = labels
        
        return new_image, target
    
def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    
    batch_targets = []
    for i, target in enumerate(targets):
        boxes = target['boxes']
        labels = target['labels']
        batch_index = torch.full((labels.size(0), 1), i, dtype=torch.int64)
        batch_targets.append(torch.cat((batch_index, labels.unsqueeze(1), boxes), dim=1))
    
    batch_targets = torch.cat(batch_targets, dim=0)
    
    return images, batch_targets

def data_loader(annotations_file, images_dir, batch_size, transform=None, collate_fn=collate_fn):
    
    dataset = YOLODataset(annotations_file, images_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    return dataloader

if __name__ == "__main__":
    # transform = transforms.Compose([
    #     transforms.Resize((416, 416)),
    #     transforms.ToTensor(),
    #     ])
    annotations_file = "D:\yyt_code\DeepLearning\Yolo\label_txt\\train_annotations.txt"
    images_dir = 'D:\yyt_code\DeepLearning\VOC2012\VOC2012\JPEGImages'
    train_data =  data_loader(annotations_file, images_dir, 64)
