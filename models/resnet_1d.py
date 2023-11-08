import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=True)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=True)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x;

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class model_ResNet(nn.Module):
    def __init__(self, layers, inchannel, block=BasicBlock, num_classes=2, dropout_rate=0.5, is_training=True):
        super(model_ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv1d(inchannel, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)


        self.avgpool = nn.AdaptiveAvgPool1d(1)  # TODO
        self.fc = nn.Linear(512, num_classes)   # the value is undecided yet.
        self.dropout = nn.Dropout(dropout_rate)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, kernel_size=3, stride=1):
        downsample = None;
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                    nn.Conv1d(self.inplanes, planes*block.expansion,
                    kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm1d(planes*block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, kernel_size=kernel_size, stride=stride,  downsample=downsample))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=kernel_size))
        
        return  nn.Sequential(*layers)
    
    def forward(self, x):
        # x = torch.transpose(x, 1, 2) # dimensions of dim-1 and dim 2 are swappted. [B, Length, C] => [B, C, Length]
        x = self.conv1(x);
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        output = x.view(x.size(0), -1)
        # output = self.fc(output)
        return output
    
class model_ResNet_dualmodal(nn.Module):
    def __init__(self, layers, inchannel1, inchannel2, block=BasicBlock, num_classes=2, dropout_rate=0.5, is_training=True):
        super(model_ResNet_dualmodal, self).__init__()
        self.inplanes = 128
        self.conv1_1 = nn.Conv1d(inchannel1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm1d(64)
        self.conv1_2 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm1d(64)
        self.relu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(inchannel2, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 1024, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, kernel_size=3, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, kernel_size=kernel_size, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        x1 = self.conv1_1(x1)
        x1 = self.bn1_1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)

        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x2 = self.maxpool(x2)

        x = torch.cat((x1, x2), dim=1)  # Concatenate the two modalities

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.dropout(x)
        # output = self.fc(x)
        output = x
        return output