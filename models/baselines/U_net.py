import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.bottleneck = self.conv_block(512, 1024)

        self.decoder4 = self.upconv_block(1024, 512)
        self.decoder3 = self.upconv_block(512, 256)
        self.decoder2 = self.upconv_block(256, 128)
        self.decoder1 = self.upconv_block(128, 64)

        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool3d(enc1, kernel_size=2))
        enc3 = self.encoder3(F.max_pool3d(enc2, kernel_size=2))
        enc4 = self.encoder4(F.max_pool3d(enc3, kernel_size=2))
        bottleneck = self.bottleneck(F.max_pool3d(enc4, kernel_size=2))

        dec4 = self.decoder4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec3 = self.decoder3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec2 = self.decoder2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec1 = self.decoder1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)

        return self.final_conv(dec1)

def conv(in_planes, output_channels, kernel_size, stride, dropout_rate):
    return nn.Sequential(
        nn.Conv3d(in_planes, output_channels, kernel_size=kernel_size,
                  stride=stride, padding=(kernel_size - 1) // 2, bias = False),
        nn.BatchNorm3d(output_channels),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Dropout(dropout_rate)
    )

def deconv(input_channels, output_channels):
    return nn.Sequential(
        nn.ConvTranspose3d(input_channels, output_channels, kernel_size=4,
                           stride=2, padding=1),
        nn.LeakyReLU(0.1, inplace=True)
    )

def output_layer(input_channels, output_channels, kernel_size, stride, dropout_rate):
    return nn.Conv3d(input_channels, output_channels, kernel_size=kernel_size,
                     stride=stride, padding=(kernel_size - 1) // 2)

class U_net(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dropout_rate):
        super(U_net, self).__init__()
        self.input_channels = input_channels
        self.conv1 = conv(input_channels, 64, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv2 = conv(64, 128, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv3 = conv(128, 256, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv3_1 = conv(256, 256, kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)
        self.conv4 = conv(256, 512, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv4_1 = conv(512, 512, kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)
        self.conv5 = conv(512, 1024, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv5_1 = conv(1024, 1024, kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)

        self.deconv4 = deconv(1024, 256)
        self.deconv3 = deconv(768, 128)
        self.deconv2 = deconv(384, 64)
        self.deconv1 = deconv(192, 32)
        self.deconv0 = deconv(96, 16)
    
        self.output_layer = output_layer(16 + input_channels, output_channels, 
                                         kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)


    def forward(self, x):

        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))

        out_deconv4 = self.deconv4(out_conv5)
        concat4 = torch.cat((out_conv4, out_deconv4), 1)
        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3, out_deconv3), 1)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2, out_deconv2), 1)
        out_deconv1 = self.deconv1(concat2)
        concat1 = torch.cat((out_conv1, out_deconv1), 1)
        out_deconv0 = self.deconv0(concat1)
        concat0 = torch.cat((x, out_deconv0), 1)
        out = self.output_layer(concat0)

        return out
    

    
    
