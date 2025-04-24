from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import os
from PIL import Image
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np

class ConvertToFloatTensor(object):
    def __call__(self, tensor):
        return tensor.float()
    
class TemporalImageDataset(Dataset):
    def __init__(self, root_dir,is_train):  #sequence_length: num of week
        self.samples_path = [] 
        
        # # Define  transform
        # self.transform = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        # ])

        ids_folder = os.path.join(root_dir,"train" if is_train else "val")
        for id in os.listdir(ids_folder):
            times = sorted(os.listdir(os.path.join(ids_folder, id)))
            no_tumor_sequence = [] #temporal sequence 
            tumor_sequence = []
            no_tumor_label_sequence = None
            tumor_label_sequence = None
            
            for time in times:
                class_dir_paths = sorted(os.listdir(os.path.join(ids_folder, id, time)))
                for class_dir in class_dir_paths:
                        images_dir_path = os.path.join(ids_folder, id, time, class_dir)  #") # for i in range(self.sequence_length)]                    
                        images_files = sorted(os.listdir(images_dir_path))
                        if images_files:
                            img_path=os.path.join(images_dir_path,images_files[0]) #choose first image path of each folder 
                            if class_dir == "no tumor":
                                no_tumor_sequence.append(img_path)
                                no_tumor_label_sequence=class_dir  
                            if class_dir == "tumor":
                                tumor_sequence.append(img_path)
                                tumor_label_sequence=class_dir             
            
            if no_tumor_sequence:             
                self.samples_path.append((no_tumor_sequence,no_tumor_label_sequence))
            if tumor_sequence:             
                self.samples_path.append((tumor_sequence,tumor_label_sequence))

    def __len__(self):
        return len(self.samples_path) #the total number of images in the dataset

    def __getitem__(self, idx):  #idx : index of animals ID from 0 to ... 
        image_paths, label= self.samples_path[idx] #image_paths : image paths in each sequence of a animal
        images = [Image.open(img_path).convert("RGB")  for img_path in image_paths]  #Image.open(img_path).convert("RGB")

        # # Convert images to RGB 
        # images = [np.stack((image,)*3,axis=-1).astype(np.uint8) for image in images]
        # print(images.shape)
        # images = [Image.fromarray(image) for image in images]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images
            transforms.ToTensor(),          # Convert images to Tensor and scale pixels between 0 and 1
            ConvertToFloatTensor(),  # Ensure tensor is float32
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
        ])
    
        if self.transform:
            images = [self.transform(image) for image in images]
         
        # for i, img_tensor in enumerate(images):
        #     if i<1:
        #         print(f"Transformed image {i} shape: {img_tensor.shape}")
        # Stack images along the new temporal dimension
        image = torch.stack(images, dim=0) #the temporal sequence of images for that animals
        #convert label to tensor (0 for "no tumor", 1 for "tumor")
        label = torch.tensor(0 if label == "no tumor" else 1, dtype=torch.long)
        # One-hot encode the label
        label=F.one_hot(label,num_classes=2).float()  # Convert the scalar label to one-hot encoded tensor of shape [1,2] and ensure it's float
        #remove any unnecessary dimensions
        label= label.squeeze(0)
         
        return image,label



if __name__ == '__main__':
    
    # dataset
    root_dir='E:\Tiankuo\ViT\data'
    is_train=True
    dataset_train = TemporalImageDataset(root_dir,is_train) #sequence_length=5
    # DataLoader
    data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=10)

    # Now iterating over data_loader will return batches with shape [batch_size, time, channels, height, width]
    for image,label in data_loader:
        print("x.shape: {},label:{}".format(image.shape,label) ) 
        # output :x.shape: torch.Size([1, 5, 3, 224, 224]),label:tensor([[1., 0.]])
        #         x.shape: torch.Size([1, 5, 3, 224, 224]),label:tensor([[0., 1.]])
