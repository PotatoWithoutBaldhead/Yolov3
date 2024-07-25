from dataset import data_loader
from model import YOLOv3
import torch
from loss import YOLOv3Loss
import torch.optim as optim
from tqdm import tqdm
import time

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_annotations_file = "D:\yyt_code\DeepLearning\Yolo\label_txt\\train_annotations.txt"
    val_annotations_file = "D:\yyt_code\DeepLearning\Yolo\label_txt\\val_annotations.txt"
    images_dir = 'D:\yyt_code\DeepLearning\VOC2012\VOC2012\JPEGImages'
    
    anchors = [[(10, 13), (16, 30), (33, 23)], 
               [(30, 61), (62, 45), (59, 119)], 
               [(116, 90), (156, 198), (373, 326)]]
    num_classes = 20
    image_size = 416
    batch_size = 16
    epochs = 100
    learning_rate = 1e-3
    gradient_accumulations = 20
    train_loader =  data_loader(train_annotations_file, images_dir, batch_size)
    val_loader =  data_loader(train_annotations_file, images_dir, batch_size)
    model = YOLOv3(num_classes).to(device)
    criterion = []
    for i in range(3):
        criterion_loss  = YOLOv3Loss( num_classes, anchors[i], image_size).to(device)
        criterion.append(criterion_loss)
        
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    min_loss = float('inf')

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for batch_i, (images, targets) in enumerate(train_loader):
                # batches_done = len(train_loader) * epoch + batch_i
                optimizer.zero_grad()
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss = 0.0
                for i in range(3):
                    loss += criterion[i](outputs[i], targets)
                loss.backward()
                optimizer.step()
                # if batches_done % gradient_accumulations:
                # # Accumulates gradient before each step
                #     optimizer.step()
                #     optimizer.zero_grad()

                running_loss += loss.item()
                pbar.set_postfix({'loss': loss.item() / batch_size})
                pbar.update(1)
        
        end_time = time.time()
        print(f"Time taken for epoch {epoch+1}: {end_time - start_time:.2f} seconds, Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss = 0.0
                for i in range(3):
                    loss += criterion[i](outputs[i], targets)
                val_loss += loss.item()
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}")
        if val_loss/len(val_loader) < min_loss:
            min_loss = val_loss/len(val_loader)
            torch.save(model.state_dict(), 'best_model.pth')