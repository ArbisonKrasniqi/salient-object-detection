import torch
import torch.nn as nn

class UNetSODModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)

        # encoding, zooming out
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.enc4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU())

        # decoding, zooming in
        # everystep upsamples and then concats with its corresponding encoding layer
        self.dec1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec1_conv = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU())

        self.dec2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2_conv = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU())

        self.dec3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3_conv = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU())

        self.output = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x) # 128x128, 64ch
        e2 = self.enc2(self.pool(e1)) # 64x64,  128ch
        e3 = self.enc3(self.pool(e2)) # 32x32,  256ch
        e4 = self.enc4(self.pool(e3)) # 16x16,  512ch

        # Decoder path with skip connections
        x = self.dec1(e4) # 32x32, 256ch
        x = self.dec1_conv(torch.cat([x, e3], dim=1)) # concat with e3 (256ch) → 512ch in → 256ch out

        x = self.dec2(x) # 64x64, 128ch
        x = self.dec2_conv(torch.cat([x, e2], dim=1)) # concat with e2 (128ch) → 256ch in → 128ch out

        x = self.dec3(x) # 128x128, 64ch
        x = self.dec3_conv(torch.cat([x, e1], dim=1)) # concat with e1 (64ch) → 128ch in → 64ch out

        return self.sigmoid(self.output(x))

if __name__ == "__main__":
    model = UNetSODModel()
    dummy = torch.randn(1, 3, 128, 128)
    output = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {output.shape}")