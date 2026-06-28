"""GPT-1 复现入口：预训练 -> 保存 -> 微调 (对比 预训练初始化 vs 从零训练)。

运行:
    python main.py
可用环境变量调节规模 (保持小规模以便快速运行):
    PRETRAIN_EPOCHS=8  FINETUNE_EPOCHS=30  BPE_VOCAB=300  python main.py
"""
import os
from pathlib import Path

import torch
from torch.nn import functional as F

from model import GPT, GPTConfig, ClassificationHead
from tokenizer import BPETokenizer
import data
import train


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def generate(model, tok, prompt, n_new, device, temperature=0.8, top_k=20):
    """用预训练好的语言模型从提示词续写文本 (温度采样 + top-k 截断)。"""
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
        ids.append(nxt)
        if nxt in tok.special_to_id.values():
            break
    return tok.decode(ids)


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
    plt.ylabel("Language modeling loss (L1)")
    plt.title("GPT-1 unsupervised pretraining loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  预训练损失曲线已保存: {path}")


def plot_comparison(acc_pre, acc_scratch, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    plt.figure(figsize=(5, 4))
    bars = plt.bar(["Pretrained init\n(pretrain+finetune)", "From scratch\n(finetune only)"],
                   [acc_pre, acc_scratch], color=["#4C9F70", "#C0C0C0"])
    plt.ylabel("Sentiment accuracy")
    plt.title("Benefit of pretraining")
    plt.ylim(0, 1.0)
    for b, v in zip(bars, [acc_pre, acc_scratch]):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  对比图已保存: {path}")


def main():
    torch.manual_seed(42)
    here = Path(__file__).resolve().parent
    device = pick_device()
    pre_epochs = int(os.getenv("PRETRAIN_EPOCHS", 8))
    ft_epochs = int(os.getenv("FINETUNE_EPOCHS", 60))
    bpe_vocab = int(os.getenv("BPE_VOCAB", 300))

    print("=" * 64)
    print("GPT-1 复现：生成式预训练 + 任务微调")
    print("=" * 64)
    print(f"device = {device}")

    # ---- 1. 训练 BPE 分词器 ----
    print("\n[1] 训练 BPE 分词器 ...")
    tok = BPETokenizer()
    tok.train(data.PRETRAIN_CORPUS, target_size=bpe_vocab)
    print(f"  词表大小 = {tok.vocab_size} (含特殊 token {list(tok.special_to_id)})")

    token_ids = tok.encode(data.PRETRAIN_CORPUS)
    print(f"  语料 token 数 = {len(token_ids)}")

    # ---- 2. 构建 GPT 模型 ----
    cfg = GPTConfig(vocab_size=tok.vocab_size, n_ctx=64, n_embd=128, n_layer=4, n_head=4)
    model = GPT(cfg)
    print(f"\n[2] GPT 参数量 = {model.num_parameters():,}")
    print(f"     (论文 GPT-1 small: 12 层 / 768 维 ≈ 117M)")

    # ---- 3. 无监督预训练 (L1) ----
    print(f"\n[3] 无监督语言模型预训练 ({pre_epochs} epochs) ...")
    history = train.pretrain(
        model, token_ids, block_size=cfg.n_ctx, batch_size=32,
        epochs=pre_epochs, lr=3e-3, warmup_ratio=0.1, device=device, log_every=50)
    if history:
        print(f"  最终 LM 损失 ≈ {history[-1][1]:.3f}")
    ckpt_path = here / "gpt1_pretrained.pth"
    save_checkpoint(model, ckpt_path)
    print(f"  预训练权重已保存: {ckpt_path.name}")
    plot_losses(history, here / "pretrain_loss.png")

    # ---- 4. 预训练模型续写示例 ----
    print("\n[4] 预训练模型续写示例:")
    for prompt in ["the cat", "the food was", "the movie"]:
        text = generate(model, tok, prompt, n_new=12, device=device)
        print(f"  «{prompt}» -> {text}")

    # ---- 5. 下游任务：情感分类微调 ----
    ft_lr = 1e-3
    print(f"\n[5] 下游任务微调 (情感分类, {ft_epochs} epochs, lr={ft_lr}, λ=0.5 辅助 LM 目标)")
    train_set, val_set = data.split_data(data.SENTIMENT_DATA, frac=0.75)
    n_classes = len({lbl for _, lbl in data.SENTIMENT_DATA})
    print(f"  训练 {len(train_set)} 条 / 验证 {len(val_set)} 条 / 类别数 {n_classes}")

    # 5a. 基于预训练初始化微调
    print("  [5a] 预训练初始化 + 微调:")
    pre_model = load_gpt(ckpt_path, device)
    clf_pre = ClassificationHead(cfg.n_embd, n_classes)
    hist_pre = train.finetune(pre_model, clf_pre, train_set, tok, n_ctx=cfg.n_ctx, batch_size=8,
                              epochs=ft_epochs, lr=ft_lr, lm_weight=0.5, warmup_ratio=0.1, device=device)
    train_acc_pre = train.evaluate(pre_model, clf_pre, train_set, tok, n_ctx=cfg.n_ctx,
                                   batch_size=8, device=device)
    acc_pre = train.evaluate(pre_model, clf_pre, val_set, tok, n_ctx=cfg.n_ctx,
                             batch_size=8, device=device)

    # 5b. 从零训练 (相同结构, 随机初始化, 仅微调) 作为对照
    print("  [5b] 从零训练 (对照, 仅微调):")
    scratch_model = GPT(cfg).to(device)
    clf_scratch = ClassificationHead(cfg.n_embd, n_classes)
    hist_scratch = train.finetune(scratch_model, clf_scratch, train_set, tok, n_ctx=cfg.n_ctx, batch_size=8,
                                  epochs=ft_epochs, lr=ft_lr, lm_weight=0.5, warmup_ratio=0.1, device=device)
    train_acc_scratch = train.evaluate(scratch_model, clf_scratch, train_set, tok, n_ctx=cfg.n_ctx,
                                       batch_size=8, device=device)
    acc_scratch = train.evaluate(scratch_model, clf_scratch, val_set, tok, n_ctx=cfg.n_ctx,
                                 batch_size=8, device=device)

    print(f"\n  末轮损失:        预训练+微调 = {hist_pre[-1][1]:.3f}   从零训练 = {hist_scratch[-1][1]:.3f}")
    print(f"  训练集准确率:    预训练+微调 = {train_acc_pre:.2%}   从零训练 = {train_acc_scratch:.2%}")
    print(f"  验证集准确率:    预训练+微调 = {acc_pre:.2%}   从零训练 = {acc_scratch:.2%}")
    plot_comparison(acc_pre, acc_scratch, here / "finetune_compare.png")

    # ---- 6. 论文 Figure 2 的四种任务输入变换示例 ----
    print("\n[6] 论文 Fig.2 四种任务输入变换示例 (展示序列拼接与 [Extract] 位置):")
    print("  分类:   ", data.classification_input(tok, "the food was delicious"))
    print("  蕴含:   ", data.entailment_input(tok, "a dog runs", "an animal moves"))
    print("  相似度:  ", data.similarity_inputs(tok, "it is sunny", "the sun is out"))
    print("  多选:   ", data.multiple_choice_input(tok, "paris is the capital",
                                                    "the capital of france is",
                                                    ["london", "paris", "berlin"]))

    print("\n完成。")


if __name__ == "__main__":
    main()
