"""
手写数字识别 - 训练 + HTTP 推理服务

用法:
    python 08_lenet_server.py

然后浏览器打开 08_digit_recognizer.html 即可手写识别。
首次运行会自动训练并保存模型，之后直接加载。
"""

import os
import io
import json
import base64
from http.server import HTTPServer, BaseHTTPRequestHandler

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageOps

# ============================
# 配置
# ============================
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "08_lenet_model.pth")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
HOST, PORT = "localhost", 8000

# ============================
# 模型定义（和 08_lenet_modern.py 一致）
# ============================

class ModernCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ============================
# 训练（仅在没有模型文件时执行）
# ============================

def train_and_save():
    print("未找到模型，开始训练...")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"  设备: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    model = ModernCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch [{epoch+1}/10] Loss: {total_loss / len(loader):.4f}")

    # 保存模型参数
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"模型已保存: {MODEL_PATH}\n")
    return model


# ============================
# 推理预处理（必须和训练一致）
# ============================

# 训练时: ToTensor() 把 [0,255] → [0.0,1.0]，然后 Normalize
infer_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def preprocess(image_bytes):
    """
    canvas 图片 → 模型输入 tensor(1,1,28,28)

    canvas 是白底黑字，MNIST 是黑底白字，所以需要反转。
    步骤: 灰度 → 反转 → 裁剪 → 等比缩放 → 居中 28x28 → normalize
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("L")

    # 白底黑字 → 黑底白字（匹配 MNIST）
    img = ImageOps.invert(img)

    # 裁剪到数字区域
    bbox = img.getbbox()
    if not bbox:
        return None
    img = img.crop(bbox)

    # 等比缩放: 让最长边 = 20px（MNIST 数字约占 20x20 区域）
    w, h = img.size
    scale = 20.0 / max(w, h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    img = img.resize((new_w, new_h), Image.LANCZOS)

    # 居中放到 28x28 黑色画布
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(img, ((28 - new_w) // 2, (28 - new_h) // 2))

    # 和训练完全一致的 transform
    return infer_transform(canvas).unsqueeze(0)


# ============================
# HTTP 服务
# ============================

class Handler(BaseHTTPRequestHandler):

    def do_POST(self):
        if self.path != "/predict":
            self.send_error(404)
            return

        # 读取请求体
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))

        # 解码 base64 图片
        image_b64 = body["image"].split(",")[1]
        image_bytes = base64.b64decode(image_b64)

        # 预处理
        tensor = preprocess(image_bytes)
        if tensor is None:
            result = {"digit": -1, "confidence": 0, "probs": [0] * 10}
        else:
            with torch.no_grad():
                output = model(tensor)
                probs = torch.softmax(output, dim=1)[0].tolist()
                digit = int(max(range(10), key=lambda i: probs[i]))
            result = {
                "digit": digit,
                "confidence": round(probs[digit], 4),
                "probs": [round(p, 4) for p in probs]
            }

        # 返回 JSON
        body = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, fmt, *args):
        # 简化日志
        print(f"  {args[0]}")


# ============================
# 启动
# ============================

if __name__ == "__main__":
    # 加载或训练模型
    model = ModernCNN()
    if os.path.exists(MODEL_PATH):
        print(f"加载模型: {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    else:
        model = train_and_save()
    model.eval().to("cpu")

    print(f"\n服务已启动: http://{HOST}:{PORT}")
    print("请用浏览器打开 08_digit_recognizer.html 开始手写识别\n")

    server = HTTPServer((HOST, PORT), Handler)
    server.serve_forever()
