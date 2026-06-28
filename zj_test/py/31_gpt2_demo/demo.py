"""GPT-2 中文预训练模型 (uer/gpt2-chinese-cluecorpussmall) 本地推理 demo。

逐 token 自回归地「不断预测下一个 token」，流式打印续写结果。

模型与分词器 (与官方 from-scratch GPT-2 的关键区别)：
  * 模型仍是 GPT2LMHeadModel (12 层 / 768 维 / 约 102M 参数)，架构没变。
  * 但分词器是 **BertTokenizer** (字级中文 WordPiece)：每个汉字 ≈ 1 个 token，
    词表只有 21128，而非英文 GPT-2 的 50257 字节级 BPE。这是 UER 在中文语料
    (CLUECorpusSmall) 上训练时采用的分词方式。
  * 模型 config 里的 bos/eos=50256 是「英文 GPT-2 遗留值」，超出本词表范围 (0..21127)，
    所以不能依赖它做停止符 —— 本 demo 自己控制起止。

采样策略：top-k + 温度 + 重复惩罚。纯贪心 (argmax) 的中文 GPT-2 会陷入严重重复。

运行:
    python demo.py
    PROMPT="人工智能的未来" MAX_NEW=120 python demo.py
    # 国内下载慢可走镜像:
    HF_ENDPOINT=https://hf-mirror.com python demo.py
首次运行会从 HuggingFace 下载模型 (~400MB) 到 ~/.cache/huggingface。
"""
import os
import sys
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "uer/gpt2-chinese-cluecorpussmall"
CLS_ID = 101        # [CLS]：作为序列起始，匹配模型训练时的 [CLS] 文本 分布
SEP_ID = 102        # [SEP]：若模型自己生成出来，视作「自然结束」


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load(device: torch.device):
    """加载字级中文分词器与 GPT-2 模型。"""
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device).eval()
    return tok, model


@torch.no_grad()
def next_token_logits(model, ids, device) -> torch.Tensor:
    """前向传播，返回序列最后一个位置在词表上的 logits。"""
    x = torch.tensor([ids], dtype=torch.long, device=device)
    return model(x).logits[0, -1]


def sample_next(logits, ids_so_far, *, temperature, top_k, repetition_penalty) -> int:
    """从 logits 采样下一个 token id (重复惩罚 → 温度 → top-k 截断 → 多项采样)。"""
    # 重复惩罚：已出现过的 token 的 logit 除以 penalty (>1 则压低其概率)
    if repetition_penalty != 1.0:
        for t in set(ids_so_far):
            logits[t] /= repetition_penalty
    # 温度
    logits = logits / max(temperature, 1e-6)
    # top-k：只保留概率最高的 k 个，其余置 -inf
    if top_k and top_k < logits.size(-1):
        kth = torch.topk(logits, top_k).values[-1]
        logits[logits < kth] = float("-inf")
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


def generate_stream(tok, model, prompt, *, n_new, temperature, top_k,
                    repetition_penalty, device, n_ctx, stop_on_sep=True):
    """自回归地不断预测下一个 token，边预测边把对应汉字打印出来。

    返回完整 token id 列表。上下文超过 n_ctx 时自动滑动窗口 (只保留最近 n_ctx 个)。
    """
    # 起始：[CLS] + 提示词的字级 token (不加 [SEP]，让模型自由续写)
    ids = [CLS_ID] + tok.encode(prompt, add_special_tokens=False)
    sys.stdout.write(prompt)
    sys.stdout.flush()

    for _ in range(n_new):
        ctx = ids[-n_ctx:]                                   # 滑动窗口，防超长
        logits = next_token_logits(model, ctx, device)
        nxt = sample_next(logits, ids, temperature=temperature,
                          top_k=top_k, repetition_penalty=repetition_penalty)
        ids.append(nxt)
        if stop_on_sep and nxt == SEP_ID:                    # 模型主动结束
            break
        # 单 token → 汉字 (跳过特殊 token，如 [UNK] 之类)
        piece = tok.convert_ids_to_tokens([nxt])[0]
        if piece.startswith("[") and piece.endswith("]"):
            continue
        sys.stdout.write(piece)
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()
    return ids


def main():
    prompt = os.getenv("PROMPT", "人工智能的未来")
    n_new = int(os.getenv("MAX_NEW", 120))
    temperature = float(os.getenv("TEMP", 0.9))
    top_k = int(os.getenv("TOP_K", 40))
    rep = float(os.getenv("REP_PENALTY", 1.3))
    seed = os.getenv("SEED")
    if seed:
        torch.manual_seed(int(seed))
    device = pick_device()

    print(f"device = {device}", file=sys.stderr)
    print(f"loading {MODEL_ID} ...", file=sys.stderr)
    tok, model = load(device)
    n_ctx = model.config.n_positions
    n_params = sum(p.numel() for p in model.parameters())
    print(f"vocab = {tok.vocab_size}  |  n_ctx = {n_ctx}  |  params = {n_params/1e6:.0f}M",
          file=sys.stderr)
    print(f"prompt = {prompt!r}  |  max_new = {n_new}  temp = {temperature}  "
          f"top_k = {top_k}  rep = {rep}", file=sys.stderr)
    print("-" * 50, file=sys.stderr)

    t0 = time.time()
    out_ids = generate_stream(
        tok, model, prompt, n_new=n_new, temperature=temperature, top_k=top_k,
        repetition_penalty=rep, device=device, n_ctx=n_ctx)
    print("-" * 50, file=sys.stderr)
    print(f"生成 {len(out_ids)} tokens，用时 {time.time() - t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
