import os
import importlib.util
import pandas as pd
from torchvision.models import resnet18

_spec = importlib.util.spec_from_file_location(
    "common", os.path.join(os.path.dirname(__file__), "16_classify-leaves-common.py"))
_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_common)

device = _common.device
train_and_evaluate = _common.train_and_evaluate
build_loaders = _common.build_loaders


if __name__ == "__main__":
    train_loader, val_loader, test_loader, label2idx, idx2label, num_classes = build_loaders()

    NUM_EPOCHS = 20
    LR = 1e-3

    model = resnet18(num_classes=num_classes)
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
