"""GPT-2 复现入口：训练 byte-level 分词器 → 无监督语言模型预训练 → 零样本任务演示。

GPT-2 与 GPT-1 最大的不同：**不做下游微调**。所有任务 (翻译/问答/摘要) 都通过把任务
写成「文本续写」的提示词，让纯语言模型零样本完成 —— 即「无监督多任务学习」。

运行:
    python main.py
可用环境变量调节规模 (保持小规模以便快速运行):
    PRETRAIN_EPOCHS=20  BPE_VOCAB=500  python main.py
"""
import os
from pathlib import Path

import torch
from torch.nn import functional as F

from model import GPT, GPTConfig
from tokenizer import ByteBPETokenizer
import data
import train


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def generate(model, tok, prompt, n_new, device, temperature=0.8, top_k=40):
    """从提示词续写文本 (温度采样 + top-k 截断)，GPT-2 的标准生成方式。

    遇到 <|endoftext|> 即停止。top-k 只在最高 k 个 logits 中采样，抑制低概率尾部噪声。
    """
    model.eval()
    ids = tok.encode(prompt)
    n_ctx = model.cfg.n_ctx
    for _ in range(n_new):
        x = torch.tensor([ids[-n_ctx:]], dtype=torch.long, device=device)
        logits = model(x)[0, -1] / max(temperature, 1e-6)
        if top_k and top_k < logits.size(-1):
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[-1]] = float("-inf")
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
        if nxt == tok.endoftext_id:                          # 文档结束符
            break
        ids.append(nxt)
    return tok.decode(ids)


@torch.no_grad()
def zero_shot_accuracy(model, tok, device):
    """零样本「预测下一个 token」命中率 (近似 LAMBADA / Children's Book Test)。

    比较模型 argmax 的下一个 token id 与「正确续写」的第一个 token id 是否一致。
    教学小语料下表现有限，仅作机制演示。
    """
    model.eval()
    hits, total = 0, 0
    for prompt, expected in data.ZERO_SHOT_EVAL:
        prompt_ids = tok.encode(prompt)
        full_ids = tok.encode(prompt + expected)
        if len(full_ids) <= len(prompt_ids):        # 续写未产生新 token
            continue
        target_id = full_ids[len(prompt_ids)]        # 正确的下一个 token
        x = torch.tensor([prompt_ids[-model.cfg.n_ctx:]], dtype=torch.long, device=device)
        pred_id = int(model(x)[0, -1].argmax())
        hits += int(pred_id == target_id)
        total += 1
    return hits / max(1, total), total


def save_checkpoint(model, path):
    torch.save({"cfg": model.cfg, "state_dict": model.state_dict()}, path)


def load_gpt(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = GPT(cfg)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device)


def plot_losses(history, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("  (未安装 matplotlib，跳过绘图)")
        return
    steps, losses = zip(*history)
    plt.figure(figsize=(7, 4))
    plt.plot(steps, losses, "-o", ms=3)
    plt.xlabel("Optimization step")
    plt.ylabel("Language modeling loss")
    plt.title("GPT-2 unsupervised pretraining loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  预训练损失曲线已保存: {path}")


def main():
    data.seed_everything(42)
    here = Path(__file__).resolve().parent
    device = pick_device()
    pre_epochs = int(os.getenv("PRETRAIN_EPOCHS", 10))
    bpe_vocab = int(os.getenv("BPE_VOCAB", 500))

    print("=" * 64)
    print("GPT-2 复现：无监督语言模型 → 零样本多任务")
    print("=" * 64)
    print(f"device = {device}")

    # ---- 1. 训练 byte-level BPE 分词器 ----
    print("\n[1] 训练 byte-level BPE 分词器 ...")
    tok = ByteBPETokenizer()
    if not tok.has_regex:
        print("  (未安装 regex 模块，使用标准库 re 近似切分；安装 `pip install regex` 可完全对齐 GPT-2)")
    corpus = data.full_corpus()
    tok.train(corpus, vocab_size=bpe_vocab)
    print(f"  词表大小 = {tok.vocab_size} (256 字节基底 + {tok.n_merges} 合并 + <|endoftext|>)")
    print(f"  空格在词表中以 'Ġ' 表示 (bytes_to_unicode 的产物)")
    demo = "the cat"
    print(f"  encode(\"{demo}\") -> {tok.encode(demo)} -> {tok.encode(demo) and [tok.decoder[i] for i in tok.encode(demo)]}")

    train_ids, val_ids = data.split_corpus(tok.encode(corpus), frac=0.9)
    print(f"  语料 token 数 = {len(train_ids) + len(val_ids)} (训练 {len(train_ids)} / 验证 {len(val_ids)})")

    # ---- 2. 构建 GPT-2 模型 ----
    cfg = GPTConfig(vocab_size=tok.vocab_size, n_ctx=128, n_embd=128, n_layer=4, n_head=4)
    model = GPT(cfg)
    print(f"\n[2] GPT-2 (教学规模) 参数量 = {model.num_parameters():,}")
    print("     论文 4 种规模 (n_layer / n_embd / n_head，词表用论文的 50257):")
    GPT2_VOCAB = 50257                                  # GPT-2 真实字节级词表: 256 字节 + 50000 合并 + 1 特殊
    for name, preset in [("Small", GPTConfig.gpt2_small), ("Medium", GPTConfig.gpt2_medium),
                         ("Large", GPTConfig.gpt2_large), ("XL", GPTConfig.gpt2_xl)]:
        pcfg = preset(vocab_size=GPT2_VOCAB)
        pcount = sum(p.numel() for p in GPT(pcfg).parameters())
        print(f"       GPT-2 {name:6s}: {pcfg.n_layer}/{pcfg.n_embd}/{pcfg.n_head} -> {pcount/1e6:,.0f}M")

    # ---- 3. 无监督语言模型预训练 ----
    print(f"\n[3] 无监督语言模型预训练 ({pre_epochs} epochs, Adam β2=0.999, wd=0.01) ...")
    history = train.pretrain(
        model, train_ids, block_size=cfg.n_ctx, batch_size=32,
        epochs=pre_epochs, lr=3e-3, weight_decay=0.01, warmup_ratio=0.1,
        device=device, log_every=50)
    if history:
        print(f"  最终 LM 损失 ≈ {history[-1][1]:.3f}")
    ppl = train.perplexity(model, val_ids, block_size=cfg.n_ctx, device=device)
    print(f"  验证集困惑度 PPL ≈ {ppl:.2f}  (越低越好；1.0 = 完美预测)")
    ckpt_path = here / "gpt2_pretrained.pth"
    save_checkpoint(model, ckpt_path)
    print(f"  预训练权重已保存: {ckpt_path.name}")
    plot_losses(history, here / "pretrain_loss.png")

    # ---- 4. 续写示例 (top-k 采样) ----
    print("\n[4] 语言模型续写示例 (temperature=0.8, top_k=40):")
    for prompt in ["the cat", "the capital of france is :", "the weather was"]:
        text = generate(model, tok, prompt, n_new=16, device=device)
        print(f"  «{prompt}» -> {text[len(prompt):].strip() or '(空)'}")

    # ---- 5. 零样本任务：翻译 / 问答 / 摘要 (无任何微调) ----
    print("\n[5] 零样本任务演示 (任务 = 文本续写，无参数微调):")
    demos = [
        ("翻译", data.translate_prompt("the cat", "french")),
        ("翻译", data.translate_prompt("the house", "french")),
        ("问答", data.qa_prompt("what is the capital of france ?")),
        ("问答", data.qa_prompt("what is the capital of japan ?")),
        ("摘要", data.summarize_prompt("a fast red car drove down the empty street at midnight .")),
    ]
    for kind, prompt in demos:
        text = generate(model, tok, prompt, n_new=6, device=device, temperature=0.0, top_k=0)
        cont = text[len(prompt):].strip() or "(空)"
        print(f"  [{kind}] {prompt!r} -> {cont!r}")

    # ---- 6. 零样本「下一词」命中率 ----
    print("\n[6] 零样本「预测下一个 token」评估 (argmax，近似 LAMBADA/CBT):")
    acc, n = zero_shot_accuracy(model, tok, device)
    print(f"  命中 {acc:.0%} ({n} 条提示)。")
    print("  说明：这些提示多属训练分布，高命中说明模型已学会「提示→补全」的关联 ——")
    print("        这正是 GPT-2 无监督多任务的机制。对训练时未见过的全新任务/语言，")
    print("        真正的零样本泛化需 WebText 量级 (~40GB) 数据才能涌现。")

    print("\n完成。")
    print("注：GPT-2 与 GPT-1 的根本区别 —— 没有「任务头 / 微调」阶段，"
          "所有任务能力均来自纯语言建模 + 提示词 (unsupervised multitask learner)。")


if __name__ == "__main__":
    main()
