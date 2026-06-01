import os
import importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "common", os.path.join(os.path.dirname(__file__), "16_classify-leaves-common.py"))
_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_common)

device = _common.device
train_and_evaluate = _common.train_and_evaluate
build_loaders = _common.build_loaders


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


class ResNet(nn.Module):
    def __init__(self, arch=[2, 2, 2, 2], in_channels=3, num_classes=176):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b2 = nn.Sequential(*resnet_block(64, 64, arch[0], first_block=True))
        self.b3 = nn.Sequential(*resnet_block(64, 128, arch[1]))
        self.b4 = nn.Sequential(*resnet_block(128, 256, arch[2]))
        self.b5 = nn.Sequential(*resnet_block(256, 512, arch[3]))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    train_loader, val_loader, test_loader, label2idx, idx2label, num_classes = build_loaders()

    NUM_EPOCHS = 20
    LR = 1e-3

    model = ResNet(arch=[2, 2, 2, 2], in_channels=3, num_classes=num_classes)
    result = train_and_evaluate(
        'resnet18', model,
        train_loader, val_loader, test_loader,
        NUM_EPOCHS, LR, device, idx2label)

    test_df = pd.read_csv(_common.TEST_CSV)
    submission = pd.DataFrame({
        'filename': test_df['filename'],
        'label': [result['test_results'].get(fn, '') for fn in test_df['filename']]
    })
    submission.to_csv('16_submission_resnet18.csv', index=False)
    print(f"提交文件已保存: 16_submission_resnet18.csv ({len(submission)} 条)")
