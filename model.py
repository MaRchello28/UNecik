import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as transforms

# DoubleConv to podstawowy blok UNet z dwoma warstwami konwolucyjnymi
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

# Główny model UNet
class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Downward path (encoding)
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Reverse path (decoding)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

# Funkcja do ładowania pretrenowanego modelu z ImageNet
def load_pretrained_weights(model):
    pretrained_model = models.resnet18(pretrained=True)  # Możesz zmienić na inny model (np. ResNet50)
    
    # Przekopiowanie wag z pretrenowanego modelu do UNet
    model_dict = model.state_dict()
    pretrained_dict = pretrained_model.state_dict()
    
    # Filtrujemy tylko te wagi, które odpowiadają warstwom UNet
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    
    # Załaduj wagę
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print("Wagi pretrenowane zostały załadowane do modelu UNet.")

# Funkcja do konwersji obrazu czarno-białego na RGB (powielanie kanałów)
def convert_to_rgb(image):
    return image.repeat(1, 1, 3)

# Inicjalizacja modelu UNet
model = UNET(in_channels=3, out_channels=1)

load_pretrained_weights(model)

def test():
    x = torch.randn((1, 3, 512, 512))
    preds = model(x)
    print(f"Predykcja ma kształt: {preds.shape}")

if __name__ == "__main__":
    test()