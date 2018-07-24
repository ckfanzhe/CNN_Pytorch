# 人脸识别神经网络模型
# Pytorch 0.4.0  author: fanzhe

# 导入所需的库

import torch.nn as nn




class RecognizeNet(nn.Module):
    def __init__(self, num_classes=3):
        super(RecognizeNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(kernel_size=2) # (32,32,32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(kernel_size=2) # (64,16,16)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()

        self.avgpool = nn.AvgPool2d(kernel_size=4) # ()

        self.cnn_net = nn.Sequential(self.conv1, self.relu1, self.conv2, self.relu2, self.conv3, self.relu3, self.avgpool)

        self.fc1 = nn.Linear(in_features=64*16*16, out_features=128)
        self.drop1 = nn.Dropout(0.5)
        self.relu4 = nn.ReLU()

        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.drop2 = nn.Dropout(0.75)
        self.relu5 = nn.ReLU()

        self.fc3 = nn.Linear(in_features=128, out_features=num_classes)

        self.fc_net = nn.Sequential(self.fc1,self.drop1, self.relu4, self.fc2,self.drop2, self.relu5, self.fc3)



    def forward(self, input):
        output = self.cnn_net(input)  # 卷积层

        output = output.view(-1, 64*16*16)

        output = self.fc_net(output)  # 全连接层


        return output



if __name__ == '__main__':
    model = RecognizeNet()
    print(model)
