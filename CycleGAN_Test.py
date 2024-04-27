import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from tqdm import tqdm
import itertools
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Downsample
        self.downsample = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Upsample
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
        # Attention Gates
        self.att1 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.att2 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.att3 = AttentionGate(F_g=32, F_l=32, F_int=16)

    def forward(self, x):
        # Encoder
        d1 = self.downsample[0:2](x) # 32
        d2 = self.downsample[2:4](d1) # 64
        d3 = self.downsample[4:6](d2) # 128
        d4 = self.downsample[6:8](d3) # 256
        
        # Decoder with attention
        u3 = self.upsample[0:2](d4) 
        u3 = self.att1(g=u3, x=d3)
        u2 = self.upsample[2:4](u3) 
        u2 = self.att2(g=u2, x=d2)
        u1 = self.upsample[4:6](u2)
        out = self.upsample[6:8](u1)
        
        return out



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1),  # Output a single scalar
            nn.Sigmoid()  # Output in [0, 1] range (for binary classification)
        )

    def forward(self, x):
        return self.model(x)


# Define dataset paths
data_root = r"E:\Work\Classes\Sem2\ml\CycleGAN\kaggletest\images"
# Define transformations for data preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to a consistent size
    transforms.Grayscale(),          # Convert images to grayscale
    transforms.ToTensor(),           # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize images to the range [-1, 1]
])

# Initialize hyperparameters
batch_size = 4
learning_rate = 0.0002
num_epochs = 40
lambda_cycle = 10  # Weight for cycle consistency loss

import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = self._load_image_paths()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = self._read_image(img_path)

        if self.transform:
            img = self.transform(img)

        return img

    def _load_image_paths(self):
        image_paths = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def _read_image(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        img = img.astype(np.float32) / 255.0  # Convert to float32 and normalize to range [0, 1]
        return img


data_dir = "E:\Work\Classes\Sem2\ml\CycleGAN\kaggletest\images"
transform = ToTensor()  # Convert images to PyTorch tensors

# Create custom datasets
train_dataset_A = CustomImageDataset(root_dir=os.path.join(data_dir, 'trainA'), transform=transform)
train_dataset_B = CustomImageDataset(root_dir=os.path.join(data_dir, 'trainB'), transform=transform)
test_dataset_A = CustomImageDataset(root_dir=os.path.join(data_dir, 'testA'), transform=transform)
test_dataset_B = CustomImageDataset(root_dir=os.path.join(data_dir, 'testB'), transform=transform)

print("Length of train_dataset_A:", len(train_dataset_A))
print("Length of train_dataset_B:", len(train_dataset_B))
print("Length of test_dataset_A:", len(test_dataset_A))
print("Length of test_dataset_B:", len(test_dataset_B))
# Define data loaders
train_loader_A = DataLoader(dataset=train_dataset_A, batch_size=batch_size, shuffle=True)
train_loader_B = DataLoader(dataset=train_dataset_B, batch_size=batch_size, shuffle=True)
test_loader_A = DataLoader(dataset=test_dataset_A, batch_size=batch_size, shuffle=False)
test_loader_B = DataLoader(dataset=test_dataset_B, batch_size=batch_size, shuffle=False)

# Initialize generator and discriminator
G_AB = Generator()
G_BA = Generator()
D_A = Discriminator()
D_B = Discriminator()

# Print Generator model structure
print("Generator Model:")
print(G_AB)

# Print Discriminator model structure
print("\nDiscriminator Model:")
print(D_A)

# Define loss functions
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()

# Initialize optimizers
optimizer_G = optim.AdamW(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=learning_rate)
optimizer_D_A = optim.AdamW(D_A.parameters(), lr=learning_rate)
optimizer_D_B = optim.AdamW(D_B.parameters(), lr=learning_rate)


# Define the directory to save images
output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the generator and discriminator models to the selected device
G_AB.to(device)
G_BA.to(device)
D_A.to(device)
D_B.to(device)

# Initialize models with suffixes
G_AB_new = Generator()
G_BA_new = Generator()
D_A_new = Discriminator()
D_B_new = Discriminator()

# Path to your saved model weights
model_dir = r"E:\Work\Classes\Sem2\ml\CycleGAN\kaggletest\cgan66epoch\saved_models"

# Load the weights into the new model instances
G_AB_new.load_state_dict(torch.load(os.path.join(model_dir, 'best_generator_AB_weights.pth')))
G_BA_new.load_state_dict(torch.load(os.path.join(model_dir, 'best_generator_BA_weights.pth')))
D_A_new.load_state_dict(torch.load(os.path.join(model_dir, 'best_discriminator_A_weights.pth')))
D_B_new.load_state_dict(torch.load(os.path.join(model_dir, 'best_discriminator_B_weights.pth')))

# Move models to GPU if CUDA is available
if torch.cuda.is_available():
    G_AB_new.cuda()
    G_BA_new.cuda()
    D_A_new.cuda()
    D_B_new.cuda()

print("Models with new suffixes loaded successfully and ready for use!")



def perform_inference(G_AB, G_BA, test_loader_A, test_loader_B, device, output_dir):
    """
    Perform inference using provided generator models on test data.
    Args:
    - G_AB: Generator model from domain A to B.
    - G_BA: Generator model from domain B to A.
    - test_loader_A: DataLoader for domain A test images.
    - test_loader_B: DataLoader for domain B test images.
    - device: Device to run the inference on (e.g., 'cuda' or 'cpu').
    - output_dir: Directory to save generated images.
    """
    # Set generator models to evaluation mode
    G_AB.eval()
    G_BA.eval()

    # Test loop with no gradients needed for inference
    with torch.no_grad():
        for i, (real_A, real_B) in enumerate(zip(test_loader_A, test_loader_B)):
            # Move real_A and real_B to the device
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            # Generate fake images and reconstruct the original images
            fake_B = G_AB(real_A[0].unsqueeze(0))  # Generate B from A
            rec_A = G_BA(fake_B)  # Reconstruct A from fake B
            
            fake_A = G_BA(real_B[0].unsqueeze(0))  # Generate A from B
            rec_B = G_AB(fake_A)  # Reconstruct B from fake A

            # Save generated images
            save_image(fake_B, os.path.join(output_dir, f'test_fake_B_{i}.png'))
            save_image(fake_A, os.path.join(output_dir, f'test_fake_A_{i}.png'))
            save_image(rec_A, os.path.join(output_dir, f'test_rec_A_{i}.png'))
            save_image(rec_B, os.path.join(output_dir, f'test_rec_B_{i}.png'))
            
            # Print and display images
            plt.figure(figsize=(10, 10))  # Set figure size
            plt.subplot(3, 2, 1)
            plt.imshow(real_A[0].permute(1, 2, 0).cpu().numpy())
            plt.title("Real A")
            
            plt.subplot(3, 2, 2)
            plt.imshow(fake_B[0].permute(1, 2, 0).cpu().numpy())
            plt.title("Generated B from A")

            plt.subplot(3, 2, 3)
            plt.imshow(rec_A[0].permute(1, 2, 0).cpu().numpy())
            plt.title("Reconstructed A from B")
            
            plt.subplot(3, 2, 4)
            plt.imshow(real_B[0].permute(1, 2, 0).cpu().numpy())
            plt.title("Real B")

            plt.subplot(3, 2, 5)
            plt.imshow(fake_A[0].permute(1, 2, 0).cpu().numpy())
            plt.title("Generated A from B")

            plt.subplot(3, 2, 6)
            plt.imshow(rec_B[0].permute(1, 2, 0).cpu().numpy())
            plt.title("Reconstructed B from A")
            
            plt.show()

            if i == 24:  # Stop after processing 25 images
                break
# Example of how to call the function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_dir = 'best_output_images'
os.makedirs(image_dir, exist_ok=True)
perform_inference(G_AB_new, G_BA_new, test_loader_A, test_loader_B, device, image_dir)


