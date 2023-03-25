from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os

IMAGE_SIZE = 128
CHANNELS_IMG = 1
BATCH_SIZE = 32


transforms = transforms.Compose(
    [
     transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
     transforms.ToTensor(),
     transforms.Normalize(
         [0.5 for _ in range(CHANNELS_IMG)],[0.5 for _ in range(CHANNELS_IMG)]
     )
    ]
)

class ImageDataSet(Dataset):
  # img_dir
  def __init__(self,img_dir,transform=None,target_transform=None):
    self.img_dir = img_dir
    self.file_list = os.listdir(self.img_dir)
    self.transform = transform
    self.target_transform = target_transform
  
  def __len__(self):
    return len(self.file_list)
  
  def __getitem__(self,idx):
    img_path = os.path.join(self.img_dir, self.file_list[idx])
    if self.transform:
      image = self.transform(Image.open(img_path))
    return image
  
