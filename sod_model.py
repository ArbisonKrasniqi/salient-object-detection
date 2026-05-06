import torch
import torch.nn as nn

class SODModel(nn.Module):
    def __init__(self):
        super().__init__()

        #Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(3,32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2))
        self.enc2 = nn.Sequential(nn.Conv2d(32,64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2))
        self.enc3 = nn.Sequential(nn.Conv2d(64,128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2))
        self.enc4 = nn.Sequential(nn.Conv2d(128,256, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2))
        
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=2, stride = 2), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=2, stride = 2), nn.ReLU())
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=2, stride = 2), nn.ReLU())
        self.dec4 = nn.Sequential(nn.ConvTranspose2d(32, 1, kernel_size=2, stride = 2), nn.Sigmoid())

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        return x

       
if __name__ == "__main__":
    model = SODModel()
    dummy = torch.randn(1, 3, 128, 128)  # fake batch of 1 image
    output = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {output.shape}")