import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp

def my_unet():
    model = smp.Unet(
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,                      # model output channels (number of classes in your dataset)
    )
    return model

# - baseline 中没有实现的
# - The dropout rate is set to values between 0.3 and 0.5 with the lowest dropout rate in the first downsampling/last upsampling block  ???
# - our architecture applies zero-padding in the con-volutional layers
# - The loss function is of the form MSE − SDC

# - The two-channel output masks contain one channel for the suture landmarks and one for the background
# - The predicted output heatmap is converted into a binary mask by thresholding. 
# - Then, the centre of mass for each region is determined as point of interest.

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

# baseline Unet
class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[16, 32, 64, 128]):

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
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return torch.sigmoid(self.final_conv(x))

def test():
    x = torch.randn((3, 3, 288, 512))
    model = my_unet()
    preds = model(x)
    print(x.shape)
    print(preds.shape)
    # assert preds.shape == x.shape

if __name__ == "__main__":
    test()

