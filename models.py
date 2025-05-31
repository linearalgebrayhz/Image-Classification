import torch
import torch.nn as nn
import torch.nn.functional as F

#TODO: ResNet
class ResNet(nn.Module):
    def __init__(self, num_blocks, num_classes):
        super().__init__()
        
        self.in_dim = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding = 1, bias = False)
        self.bn = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(64, num_blocks, stride=1)
        self.layer2 = self.make_layer(128, num_blocks, stride=1)
        self.layer3 = self.make_layer(256, num_blocks, stride=1)
        self.layer4 = self.make_layer(512, num_blocks, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512,num_classes)
        
    def forward(self,x):
        out = self.relu(self.bn(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = out.view(out.shape[0],-1)
        out = self.fc(out) 
        return out
    
    def make_layer(self,out_dim, num_blocks, stride):
        layer = []
        layer.append(ResNetBlock(self.in_dim, out_dim, stride=2))
        self.in_dim = out_dim
        for _ in range(1, num_blocks):
            layer.append(ResNetBlock(out_dim, out_dim))        
        return nn.Sequential(*layer)
    

class ResNetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size = 3, stride = 1, non_linear = "relu"):
        super().__init__()
        self.non_linear = non_linear
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size = kernel_size, stride = stride, padding = 1, bias = True)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size = kernel_size, stride = 1, padding = 1, bias = True)
        self.relu = nn.ReLU()
        if non_linear == "sigmoid":
            self.sigmoid = nn.Sigmoid()
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.skip = nn.Sequential()
        
        # reshape required if dimensions do not match
        if stride != 1 or in_dim != out_dim:
            self.skip = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, stride = stride, bias = False), 
                nn.BatchNorm2d(out_dim)
            )
        
    def forward(self, x): 
        out = self.bn1(self.conv1(x))
        if self.non_linear == "relu":
            out = self.relu(out)
        elif self.non_linear == "sigmoid":
            out = self.sigmoid(out) 
        out = self.bn2(self.conv2(out))
        out += self.skip(x)
        out = self.relu(out)
        return out

#TODO: ViT

#TODO: Mamba

# unit test
if __name__ == "__main__":
    # print("---Res Block Testing---")
    # r1 = ResNetBlock(3, 64)
    # r2 = ResNetBlock(3, 64, stride = 2)
    # print(r1)
    # x = torch.randn(16, 3, 100, 100)
    # print(r1(x).shape) # torch.Size([16, 64, 100, 100])
    # print(r2(x).shape) # torch.Size([16, 64, 50, 50])
    # print()
    print("---ResNet Testing---")
    net = ResNet(3, 100)
    print(net)
    x = torch.randn(16, 3, 100, 100)
    print(net(x).shape)
