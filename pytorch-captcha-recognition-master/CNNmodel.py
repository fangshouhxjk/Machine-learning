# 神经网络模型
import torch.nn as nn
import captcha_setting

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),  # 以50%的比例丢弃神经元节点
            nn.ReLU(),
            nn.MaxPool2d(2))#逐渐降低数据体的空间尺寸，减少网络中参数 ，
                            # 减少计算资源的消耗，同时也能够有效地控制过拟合。
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Sequential(
            nn.Linear(
                (captcha_setting.IMAGE_WIDTH//8)*
                (captcha_setting.IMAGE_HEIGHT//8)*64, 1024),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU())
        self.rfc = nn.Sequential(
            nn.Linear(1024, captcha_setting.MAX_CAPTCHA*
                      captcha_setting.ALL_CHAR_SET_LEN),)



    def forward(self, x):
        #卷积--激活--池化
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)#-1表示自适应
        out = self.fc(out)
        out = self.rfc(out)
        return out

