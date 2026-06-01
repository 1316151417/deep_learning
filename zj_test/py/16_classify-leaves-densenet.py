import os
import importlib.util
import torch
import torch.nn as nn
import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "common", os.path.join(os.path.dirname(__file__), "16_classify-leaves-common.py"))
_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_common)

device = _common.device
train_and_evaluate = _common.train_and_evaluate
build_loaders = _common.build_loaders


def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1)
    )


class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super().__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)
        return X


def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, num_convs_in_dense_blocks=[6, 12, 24, 16],
                 in_channels=3, num_classes=176):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        num_channels = 64
        blks = []
        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            blks.append(DenseBlock(num_convs, num_channels, growth_rate))
            num_channels += num_convs * growth_rate
            if i != len(num_convs_in_dense_blocks) - 1:
                blks.append(transition_block(num_channels, num_channels // 2))
                num_channels = num_channels // 2
        self.blks = nn.Sequential(*blks)

        self.head = nn.Sequential(
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_channels, num_classes)
        )

    def forward(self, x):
        x = self.b1(x)
        x = self.blks(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    train_loader, val_loader, test_loader, label2idx, idx2label, num_classes = build_loaders()

    NUM_EPOCHS = 20
    LR = 1e-3

    model = DenseNet(growth_rate=32,
                     num_convs_in_dense_blocks=[6, 12, 24, 16],
                     in_channels=3, num_classes=num_classes)
    result = train_and_evaluate(
        'densenet121', model,
        train_loader, val_loader, test_loader,
        NUM_EPOCHS, LR, device, idx2label)

    test_df = pd.read_csv(_common.TEST_CSV)
    submission = pd.DataFrame({
        'filename': test_df['filename'],
        'label': [result['test_results'].get(fn, '') for fn in test_df['filename']]
    })
    submission.to_csv('16_submission_densenet121.csv', index=False)
    print(f"提交文件已保存: 16_submission_densenet121.csv ({len(submission)} 条)")
