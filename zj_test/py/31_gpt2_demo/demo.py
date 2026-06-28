"""GPT-2 中文预训练模型 (uer/gpt2-chinese-cluecorpussmall) 本地推理 demo。

用 HuggingFace 标准的 model.generate 自回归生成 —— 它内部就是「不断预测下一个 token」，
无需手写循环。采样参数 (do_sample / top_k / temperature / repetition_penalty) 直接传给 generate。

两个中文模型要知道的坑 (代码里已处理):
  * 分词器是 BertTokenizer（字级 WordPiece，词表 21128），不是英文 GPT-2 的字节级 BPE。
  * 解码出的文本字与字之间会带空格（很 久 之 前），故最后 .replace(" ", "") 去掉。
    （纯贪心会严重重复，所以要 do_sample=True + repetition_penalty。）

运行:
    python demo.py
    PROMPT="人工智能的未来" MAX_NEW=120 python demo.py
    HF_ENDPOINT=https://hf-mirror.com python demo.py    # 国内镜像加速下载
首次运行会从 HuggingFace 下载模型 (~400MB) 到 ~/.cache/huggingface。
"""
import logging
import os

os.environ.setdefault("TQDM_DISABLE", "1")   # 关掉权重加载进度条 (须在 import transformers 前设)
logging.disable(logging.WARNING)             # 压制 transformers/HF 的告警噪声 (bos/eos、LOAD REPORT、未鉴权提示)；ERROR 仍可见

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "uer/gpt2-chinese-cluecorpussmall"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    prompt = os.getenv("PROMPT", "这是很久之前的事情了")
    max_new = int(os.getenv("MAX_NEW", 120))
    device = pick_device()

    print(f"loading {MODEL_ID} on {device} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device).eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new,
        do_sample=True,
        top_k=40,
        temperature=0.9,
        repetition_penalty=1.3,
        pad_token_id=tokenizer.pad_token_id,   # 显式给 pad，避免用越界的 eos(50256)
    )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True).replace(" ", "")
    print(text)


if __name__ == "__main__":
    main()
