import os
import time
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'classify-leaves')
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
TEST_CSV = os.path.join(DATA_DIR, 'test.csv')
IMG_DIR = DATA_DIR

IMG_SIZE = 224
BATCH_SIZE = 128

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class LeafDataset(Dataset):
    def __init__(self, csv_file, img_dir, label2idx=None, transform=None):
        df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.has_label = 'label' in df.columns

        if self.has_label and label2idx is None:
            labels = sorted(df['label'].unique())
            self.label2idx = {label: idx for idx, label in enumerate(labels)}
        else:
            self.label2idx = label2idx

        self.images = df['image'].tolist()
        if self.has_label:
            self.labels = [self.label2idx[l] for l in df['label']]
        else:
            self.labels = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.has_label:
            return image, self.labels[idx]
        else:
            return image, self.images[idx]


class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def _gpu_prefetch(loader, device):
    use_cuda = device.type == 'cuda'
    non_blk = use_cuda and getattr(loader, 'pin_memory', False)
    stream = torch.cuda.Stream() if use_cuda else None

    loader_iter = iter(loader)

    try:
        next_data, next_target = next(loader_iter)
    except StopIteration:
        return
    next_data = next_data.to(device, non_blocking=non_blk)
    if isinstance(next_target, torch.Tensor):
        next_target = next_target.to(device, non_blocking=non_blk)

    while True:
        data, target = next_data, next_target

        has_next = False
        try:
            next_data, next_target = next(loader_iter)
            has_next = True
            if stream is not None:
                with torch.cuda.stream(stream):
                    next_data = next_data.to(device, non_blocking=non_blk)
                    if isinstance(next_target, torch.Tensor):
                        next_target = next_target.to(device, non_blocking=non_blk)
            else:
                next_data = next_data.to(device)
                if isinstance(next_target, torch.Tensor):
                    next_target = next_target.to(device)
        except StopIteration:
            pass

        if stream is not None and has_next:
            torch.cuda.current_stream().wait_stream(stream)

        yield data, target

        if not has_next:
            break


def train_epoch(model, loader, criterion, optimizer, scheduler, device, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    use_amp = scaler is not None

    for batch_idx, (data, target) in enumerate(_gpu_prefetch(loader, device), 1):
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            output = model(data)
            loss = criterion(output, target)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * data.size(0)
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % 50 == 0:
            print(f"  batch {batch_idx}/{len(loader)}: "
                  f"loss={loss.item():.4f}, acc={100.*correct/total:.2f}%")

    if scheduler is not None:
        scheduler.step()

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    use_amp = device.type == 'cuda'

    with torch.inference_mode():
        for data, target in _gpu_prefetch(loader, device):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                output = model(data)
                loss = criterion(output, target)

            running_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    return running_loss / total, correct / total


def predict_test(model, loader, device, idx2label):
    model.eval()
    results = {}

    with torch.inference_mode():
        for data, filenames in _gpu_prefetch(loader, device):
            output = model(data)
            _, predicted = output.max(1)
            for fn, pred in zip(filenames, predicted.cpu().numpy()):
                results[fn] = idx2label[pred]

    return results


def train_and_evaluate(model_name, model, train_loader, val_loader, test_loader,
                       num_epochs, lr, device, idx2label):
    print("\n" + "=" * 60)
    print(f"  训练模型: {model_name}")
    print("=" * 60)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler(device.type) if use_amp else None
    if use_amp:
        print("  ✓ 启用 AMP 混合精度训练 (FP16)")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数量: {total_params:,} (可训练: {trainable_params:,})")
    print(f"Epochs: {num_epochs}, Batch size: {BATCH_SIZE}, LR: {lr}")
    print("-" * 60)

    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, scaler)
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device)
        elapsed = time.time() - start_time

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"16_{model_name}_best.pth")

        print(f"Epoch {epoch}/{num_epochs} [{elapsed:.1f}s] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    print("-" * 60)
    print(f"最佳验证准确率: {best_val_acc * 100:.2f}%")

    model.load_state_dict(torch.load(f"16_{model_name}_best.pth",
                                     map_location=device, weights_only=True))
    test_results = predict_test(model, test_loader, device, idx2label)

    torch.save(model.state_dict(), f"16_{model_name}_final.pth")
    print(f"模型已保存: 16_{model_name}_best.pth / 16_{model_name}_final.pth")

    return {
        'name': model_name,
        'best_val_acc': best_val_acc,
        'test_results': test_results,
        'total_params': total_params,
    }


def build_loaders():
    print(f"使用设备: {device}")
    print(f"数据路径: {DATA_DIR}")

    full_dataset = LeafDataset(TRAIN_CSV, IMG_DIR)
    label2idx = full_dataset.label2idx
    idx2label = {v: k for k, v in label2idx.items()}
    num_classes = len(label2idx)
    print(f"类别数: {num_classes}")

    total_len = len(full_dataset)
    train_len = int(total_len * 0.8)
    val_len = total_len - train_len
    train_subset, val_subset = torch.utils.data.random_split(
        full_dataset, [train_len, val_len])

    train_dataset = TransformSubset(train_subset, train_transform)
    val_dataset = TransformSubset(val_subset, val_transform)

    PIN = device.type == 'cuda'
    NUM_WORKERS = 0 if os.name == 'nt' else min(os.cpu_count() or 4, 12)
    if device.type != 'cuda':
        NUM_WORKERS = 0

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=PIN, persistent_workers=NUM_WORKERS > 0,
                              prefetch_factor=4 if NUM_WORKERS > 0 else None)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS,
                            pin_memory=PIN, persistent_workers=NUM_WORKERS > 0,
                            prefetch_factor=4 if NUM_WORKERS > 0 else None)

    test_dataset = LeafDataset(TEST_CSV, IMG_DIR, label2idx=label2idx,
                               transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=PIN, persistent_workers=NUM_WORKERS > 0,
                             prefetch_factor=4 if NUM_WORKERS > 0 else None)

    print(f"训练集: {train_len}, 验证集: {val_len}, 测试集: {len(test_dataset)}")

    return train_loader, val_loader, test_loader, label2idx, idx2label, num_classes
