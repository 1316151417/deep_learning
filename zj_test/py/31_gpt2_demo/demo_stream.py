"""GPT-2 中文预训练模型 (uer/gpt2-chinese-cluecorpussmall) —— 逐 token 流式生成。

模型每预测出一个 token 就立刻打印，而不是一次性输出整段 (与 demo.py 的区别)。

实现：用 HuggingFace 标准的 model.generate 做生成 (采样/重复惩罚等都交给它)，
只挂一个自定义 CharStreamer 把每个 token 单独 decode 后即时打印。
  * 为什么不用现成的 TextStreamer？它对「整段」做 decode，中文会变成「很 久 之 前」
    (字间插空格)。而「单 token」decode 没有空格 (每个汉字本身就是一个 token)，
    所以 CharStreamer 一个个 decode、一个个打印，输出干净。
  * 分词器是 BertTokenizer (字级 WordPiece，词表 21128)，不是英文 GPT-2 的字节级 BPE。
  * 纯贪心会严重重复，所以 do_sample=True + repetition_penalty。

运行:
    python demo2.py
    PROMPT="人工智能的未来" MAX_NEW=120 python demo2.py
    HF_ENDPOINT=https://hf-mirror.com python demo2.py    # 国内镜像加速下载
首次运行会从 HuggingFace 下载模型 (~400MB) 到 ~/.cache/huggingface。
"""
import logging
import os

os.environ.setdefault("TQDM_DISABLE", "1")   # 关掉权重加载进度条 (须在 import transformers 前设)
logging.disable(logging.WARNING)             # 压制 transformers/HF 的告警噪声 (bos/eos、LOAD REPORT、未鉴权提示)；ERROR 仍可见

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.streamers import BaseStreamer

MODEL_ID = "uer/gpt2-chinese-cluecorpussmall"


class CharStreamer(BaseStreamer):
    """逐 token 流式打印：每个 token 单独 decode (中文单字无空格)，算出一个就打印一个。"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def put(self, value):
        # value 是新生成 (或首次传入 prompt) 的 token id 张量；逐个 decode 并立即打印
        for tid in value.flatten().tolist():
            piece = self.tokenizer.decode([int(tid)], skip_special_tokens=True)
            if piece:                                  # 跳过 [CLS]/[SEP] 等空串
                print(piece, end="", flush=True)

    def end(self):
        print(flush=True)


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
    model.generate(
        input_ids,
        max_new_tokens=max_new,
        do_sample=True,
        top_k=40,
        temperature=0.9,
        repetition_penalty=1.3,
        pad_token_id=tokenizer.pad_token_id,   # 显式给 pad，避免用越界的 eos(50256)
        streamer=CharStreamer(tokenizer),       # 边生成边逐字打印
    )


if __name__ == "__main__":
    main()
