from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRANSCRIPT = ROOT / "transcript-cn.txt"
OUTPUT = ROOT / "transcript-cn-learning.html"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def load_paragraphs() -> list[dict[str, object]]:
    lines = TRANSCRIPT.read_text(encoding="utf-8").splitlines()
    paragraphs: list[dict[str, object]] = []
    for number, text in enumerate(lines, start=1):
        text = text.strip()
        if text:
            paragraphs.append({"line": number, "text": text})
    return paragraphs


CHAPTERS = [
    {
        "id": "opening",
        "title": "开场：问题不是魔法，而是心智模型",
        "range": (1, 1),
        "thesis": "这支视频要回答一个朴素但关键的问题：你在 ChatGPT 文本框里按下回车后，背后到底发生了什么。",
        "viewpoints": [
            "理解 LLM 最重要的不是背术语，而是建立能解释优点、局限和风险的心智模型。",
            "模型既能令人惊叹，也有很多“锋利边缘”，学习时要同时看能力和失效模式。",
            "整条主线会沿着“如何构建一个类似 ChatGPT 的系统”展开。",
        ],
        "definitions": [
            ("LLM", "Large Language Model，大语言模型；以词元序列为输入和输出的神经网络模型。"),
            ("心智模型", "帮助你预测系统行为的一组简化理解，比如“它在预测下一个词元”。"),
            ("锋利边缘", "看起来很强但容易误用的边界条件，如幻觉、心算错误、过度自信。"),
        ],
        "examples": [
            "同一个文本框可以输入问题、文章、代码或指令，但模型看到的最终都是一串词元。",
            "“返回给你的文字是什么？”这个问题会贯穿预训练、SFT、RL 和工具调用。",
        ],
        "questions": [
            "当你把 LLM 当成“会补全文字的系统”时，哪些使用习惯会自然改变？",
            "你目前最想解释的现象是：幻觉、推理、翻译，还是写代码？",
        ],
    },
    {
        "id": "data",
        "title": "预训练第一步：把互联网变成高质量文本",
        "range": (3, 17),
        "thesis": "预训练不是把整个互联网原封不动塞进模型，而是先抓取、清洗、过滤、去重，得到大规模高质量文本。",
        "viewpoints": [
            "数据质量和数据多样性共同决定基座模型能学到什么。",
            "语言过滤是一个产品取舍：保留什么语言，未来模型就更擅长什么语言。",
            "“44 TB 文本”听起来大，但相对整个互联网并不夸张，因为文本很轻且经过大量过滤。",
        ],
        "definitions": [
            ("FineWeb", "Hugging Face 构建的高质量网页文本数据集，视频中作为生产级数据处理的代表例子。"),
            ("Common Crawl", "从 2007 年起抓取互联网网页的组织，很多训练语料的原始来源之一。"),
            ("URL 过滤", "用黑名单等规则剔除恶意、垃圾、营销、成人等不想纳入的数据来源。"),
            ("文本抽取", "从网页 HTML 中提取正文，丢掉导航、样式、脚本等非正文内容。"),
            ("PII 移除", "移除个人身份信息，如地址、社会安全号码等。"),
        ],
        "examples": [
            "FineWeb 只保留英语占比超过 65% 的页面，因此偏英语能力。",
            "最终数据里可能同时有龙卷风报道、医学介绍等非常不同的网页。",
        ],
        "questions": [
            "如果你要训练一个中文很强的模型，数据过滤策略应该和 FineWeb 有什么不同？",
            "为什么“更多数据”不一定等于“更好数据”？",
        ],
    },
    {
        "id": "tokenization",
        "title": "分词：把文本压成有限符号序列",
        "range": (19, 29),
        "thesis": "神经网络不能直接吃“文字意义”，它吃的是一维 token 序列；分词是在词表大小和序列长度之间做权衡。",
        "viewpoints": [
            "比特序列太长，字节序列短一些，BPE 继续把常见片段合并，让序列更短。",
            "token ID 只是符号编号，不要把它当成有大小意义的数字。",
            "分词会影响模型的拼写、计数、多语言表现，以及上下文窗口的使用效率。",
        ],
        "definitions": [
            ("UTF-8", "把文本编码成计算机可存储的比特序列的一种标准。"),
            ("字节", "8 个比特组成的单位，共有 256 种可能取值。"),
            ("BPE", "Byte Pair Encoding，字节对编码；不断合并常见相邻符号，得到更大的词表和更短的序列。"),
            ("token", "词元；模型实际处理的最小符号单元，可以是字、词、空格加词、标点或更奇怪的片段。"),
            ("词表", "模型允许出现的所有 token 集合。GPT-4 相关分词器约有 10 万个符号。"),
        ],
        "examples": [
            "“hello world”在 GPT-4 的 CL100K base 分词器中可被切成两个 token。",
            "大小写、空格数量变化会改变分词结果，所以“Hello world”和“hello world”不一定一样。",
        ],
        "questions": [
            "为什么上下文窗口宝贵时，我们愿意用更大的词表换更短的 token 序列？",
            "中文、英文、代码在分词上可能有什么不同体验？",
        ],
    },
    {
        "id": "training",
        "title": "训练与推理：预测下一个词元",
        "range": (31, 47),
        "thesis": "LLM 训练的核心任务非常朴素：给一段上下文，预测真实数据里接下来的 token，并不断调参数降低错误。",
        "viewpoints": [
            "训练不是写规则，而是通过海量例子调整参数，让输出概率符合数据统计。",
            "Transformer 是一个巨大数学函数，输入 token 和固定参数混合后输出下一个 token 的概率分布。",
            "推理时模型不会学习新参数，它只是反复采样 token，把结果接回上下文继续预测。",
        ],
        "definitions": [
            ("上下文", "当前喂给模型的一段 token 序列，长度有最大限制。"),
            ("标签", "训练数据中真实出现的下一个 token。"),
            ("参数/权重", "神经网络里可被训练调整的数字，可理解成大量旋钮。"),
            ("损失", "衡量预测与真实 token 不一致程度的指标，训练目标是让它下降。"),
            ("推理", "模型训练完后，用固定参数生成输出的过程。"),
            ("采样", "根据概率分布随机选出下一个 token，因此同一输入可得到不同答案。"),
        ],
        "examples": [
            "给 4 个 token 的窗口，模型输出 100,277 个概率，表示每个 token 成为下一个 token 的可能性。",
            "如果真实下一个 token 是“post”，训练会让“post”的概率稍微上升，让其他 token 概率稍微下降。",
        ],
        "questions": [
            "为什么同一个问题问两次，模型可能给出不同答案？",
            "如果模型只是在预测下一个 token，它为什么还能表现得像“懂知识”？",
        ],
    },
    {
        "id": "compute",
        "title": "规模：GPT-2、GPU 与训练成本",
        "range": (49, 59),
        "thesis": "简单目标乘上海量数据、参数和 GPU 并行计算，就变成了现代 AI 的基础设施竞赛。",
        "viewpoints": [
            "GPT-2 是现代 LLM 技术栈的早期代表；今天同类训练成本已大幅下降。",
            "GPU 适合矩阵乘法并行计算，所以成为训练神经网络的核心硬件。",
            "数据中心里的大量 GPU，本质上都在协同优化“预测下一个 token”。",
        ],
        "definitions": [
            ("GPT-2", "OpenAI 2019 年发布的语言模型，视频中用来展示训练与推理的具体过程。"),
            ("GPU", "图形处理器，擅长大规模并行矩阵运算。"),
            ("H100", "NVIDIA 的高端数据中心 GPU，视频中用作训练算力例子。"),
            ("节点", "一台含多块 GPU 的训练机器；多个节点可组成更大的训练集群。"),
        ],
        "examples": [
            "视频里复现 GPT-2 的 lm.c 项目，约一天训练，成本约 600 美元。",
            "8 块 H100 组成一个节点，多个节点通过网络协同训练更大的模型。",
        ],
        "questions": [
            "为什么模型训练的突破常常依赖硬件、数据和软件效率同时进步？",
            "如果训练更大模型，瓶颈可能出现在算力、数据、通信还是成本？",
        ],
    },
    {
        "id": "base-model",
        "title": "基座模型：互联网文档模拟器",
        "range": (61, 73),
        "thesis": "预训练产物是基座模型，它像互联网文本的有损压缩包；它会续写文档，但还不是可靠助手。",
        "viewpoints": [
            "基座模型拥有大量知识，但这些知识是模糊、概率性、统计性的回忆。",
            "它会记忆高频或重复出现的文本，也会在未知事实上生成“平行宇宙”。",
            "只靠提示词也能诱导基座模型做任务，但本质仍是续写符合上下文模式的 token。",
        ],
        "definitions": [
            ("基座模型", "只经过预训练、尚未被系统训练成对话助手的模型。"),
            ("有损压缩", "参数保留了互联网文本的整体统计印象，但不是逐字可查的数据库。"),
            ("幻觉", "模型在缺乏可靠知识时生成看似合理但不真实的内容。"),
            ("少样本提示", "在提示词中给几个输入输出例子，让模型在上下文里学会模式。"),
            ("上下文学习", "模型在当前上下文中识别模式并临时按模式续写的能力。"),
        ],
        "examples": [
            "问基座模型“2+2 等于几”，它可能只是补全网页式文本，不稳定地回答。",
            "给几个英文到韩语的词对，再写 teacher:，模型可以续写出韩语翻译。",
            "把提示词写成人类和 AI 助手的对话，基座模型会继续扮演助手。",
        ],
        "questions": [
            "为什么“会回答问题”和“被训练成助手”不是一回事？",
            "你能想到哪些任务适合少样本提示？哪些不适合？",
        ],
    },
    {
        "id": "sft",
        "title": "监督微调：用对话样本塑造助手",
        "range": (75, 93),
        "thesis": "SFT 把训练数据从互联网文档换成对话数据，让模型模仿人类标注员写出的理想助手回复。",
        "viewpoints": [
            "后训练的算法和预训练类似，关键变化是数据集变成了人类与助手的对话。",
            "助手不是用传统代码写出来的，而是通过大量示例被“编程”。",
            "你得到的普通聊天回答，可以理解成模型对标注员写作过程的统计模拟。",
        ],
        "definitions": [
            ("后训练", "基座模型之后的训练阶段，用来塑造助手行为、推理习惯和安全边界。"),
            ("SFT", "Supervised Fine-Tuning，监督微调；用理想对话回复继续训练模型。"),
            ("标注指南", "告诉人类标注员什么回答更 helpful、truthful、harmless 的规则文档。"),
            ("特殊词元", "用于标记对话结构的 token，如 im_start、im_end、user、assistant。"),
            ("对话协议", "把多轮对话编码成一维 token 序列的一套格式规则。"),
        ],
        "examples": [
            "“2+2 等于几？”之后，助手应答“4”；遇到不该帮助的请求，则学习拒绝。",
            "InstructGPT 论文展示了早期用人工提示词和理想回复进行微调的方法。",
            "UltraChat 等现代数据集大量使用合成数据，再由人工编辑和筛选。",
        ],
        "questions": [
            "如果标注指南改变，助手的“性格”会如何改变？",
            "为什么 SFT 通常比预训练便宜得多，却能显著改变用户体验？",
        ],
    },
    {
        "id": "psychology",
        "title": "LLM 心理学：幻觉、工具与认知缺口",
        "range": (95, 161),
        "thesis": "LLM 的许多奇怪行为不是偶然，而是由训练数据、token 表示、有限计算和上下文机制共同塑造。",
        "viewpoints": [
            "幻觉来自模型学会了“自信回答”的格式，却未必知道自己不知道。",
            "参数中的知识像模糊回忆；上下文窗口里的内容像工作记忆，可靠性更高。",
            "模型每生成一个 token 的计算量有限，所以复杂推理需要展开成多个 token 或交给工具。",
            "“瑞士奶酪”能力模型提醒我们：模型很多地方很强，但某些简单问题会突然失手。",
        ],
        "definitions": [
            ("工具调用", "模型输出特殊 token 或请求，让外部搜索、代码解释器等工具补充信息或执行计算。"),
            ("上下文窗口", "模型当前可直接访问的 token 范围，可视为工作记忆。"),
            ("系统消息", "隐藏在对话最前面的指令，用来设定身份、规则和行为边界。"),
            ("思维链", "把推理步骤展开成一串中间 token，让计算分布在多个生成步骤上。"),
            ("瑞士奶酪模型", "模型能力有洞：整体强大，但存在随机、局部、反直觉的失败点。"),
        ],
        "examples": [
            "面对杜撰人物 Orson Kovats，旧模型可能编造履历，而不是说不知道。",
            "让模型总结《傲慢与偏见》第一章时，把原文贴进上下文通常比让它凭记忆更可靠。",
            "数 strawberry 里的字母、比较 9.11 和 9.9，都是视频中用来说明认知缺口的例子。",
        ],
        "questions": [
            "什么时候应该让模型搜索？什么时候应该让它使用代码？",
            "你在工作中最需要防范的是幻觉、算错、误解上下文，还是过度自信？",
        ],
    },
    {
        "id": "rl",
        "title": "强化学习：从模仿答案到练出思考",
        "range": (162, 259),
        "thesis": "RL 让模型在可验证任务上自己试错，发现对它有效的解题策略，因此能涌现更长、更像思考的推理过程。",
        "viewpoints": [
            "预训练像读教材，SFT 像看已解例题，RL 像自己做练习题。",
            "在数学和代码这类可验证领域，答案能自动判定，模型可以大量采样并强化有效解法。",
            "思维链不是人工逐句写进去的，而是在优化过程中自然涌现。",
            "AlphaGo 的第 37 手说明：RL 能脱离人类模仿，发现人类罕见甚至未知的策略。",
        ],
        "definitions": [
            ("强化学习", "通过试错和奖励信号改进行为的训练方式。"),
            ("可验证领域", "答案能被明确检查的任务，如数学题、代码运行结果。"),
            ("奖励", "告诉模型某个候选解法好或坏的信号。"),
            ("思考模型", "经过更强推理训练、会在回答前生成较长推理过程的模型。"),
            ("第 37 手", "AlphaGo 对李世石的著名一步，代表 RL 发现非人类常规策略的能力。"),
        ],
        "examples": [
            "同一道苹果橙子题，模型可并行采样多个解法，只强化最终答案正确的路径。",
            "DeepSeek R1 在数学题上展示了回溯、重新评估、修正路径的长推理过程。",
            "ChatGPT 中标注“高级推理”的模型属于这一类思考模型的体验入口。",
        ],
        "questions": [
            "为什么“能自动验答案”的任务特别适合 RL？",
            "思考模型的能力能否迁移到写作、创意等不可验证领域？视频把它视为开放问题。",
        ],
    },
    {
        "id": "rlhf",
        "title": "RLHF：把人类偏好变成奖励，也会被钻空子",
        "range": (261, 281),
        "thesis": "RLHF 用奖励模型模拟人类偏好，让主观任务也能训练；但奖励模型不是现实本身，容易被对抗样本利用。",
        "viewpoints": [
            "写诗、讲笑话、摘要这类任务没有唯一正确答案，不能像数学题那样直接验算。",
            "人类更容易判断两个回答哪个好，而不一定容易亲手写出最优回答。",
            "奖励模型能扩展人类反馈，但它会被 RL 找到漏洞，所以 RLHF 通常不能无限跑。",
        ],
        "definitions": [
            ("不可验证领域", "没有明确标准答案、需要人类主观判断质量的任务。"),
            ("RLHF", "Reinforcement Learning from Human Feedback，基于人类反馈的强化学习。"),
            ("奖励模型", "学习模仿人类偏好排序的独立模型，用来给候选回答打分。"),
            ("判别器-生成器差距", "判断好坏通常比亲自生成高质量作品更容易。"),
            ("对抗样本", "专门钻模型漏洞、让模型给出异常高分的输入或输出。"),
        ],
        "examples": [
            "人类给多首诗或多个笑话排序，奖励模型学习这种排序偏好。",
            "奖励模型可能把无意义重复文本打高分，于是 RL 会把笑话训练到崩坏。",
        ],
        "questions": [
            "为什么 Karpathy 会强调“RLHF 不是 RL”？",
            "你能想到哪些工作任务属于不可验证领域？该如何检查结果？",
        ],
    },
    {
        "id": "future",
        "title": "未来与使用入口：多模态、智能体、模型生态",
        "range": (283, 307),
        "thesis": "视频最后把技术图景拉远：模型会走向多模态、智能体和更深的工具整合，使用者也需要学会跟进与选择。",
        "viewpoints": [
            "多模态并不是完全换一套方法，而是把音频、图像也转成 token 或类似 token 的表示。",
            "智能体会把单次回答扩展成长时间、多步骤、可监督的任务执行。",
            "排行榜和资讯能帮助跟进领域，但最终仍要用自己的任务测试模型。",
            "本地小模型、开放权重和云端专有模型会共同组成使用生态。",
        ],
        "definitions": [
            ("多模态", "模型同时处理文本、音频、图像等不同信息形式。"),
            ("智能体", "能持续执行多步任务、汇报进展并与工具互动的模型系统。"),
            ("测试时训练", "让模型在部署后的任务过程中更新自身能力的研究方向，区别于只扩展上下文。"),
            ("开放权重", "模型参数可被下载、托管和再使用的发布方式。"),
            ("蒸馏模型", "把大模型能力压缩到更小模型中的模型，便于本地运行。"),
        ],
        "examples": [
            "音频可切成频谱图片段，图像可切成 patch，再进入类似 token 的处理流程。",
            "LMArena、AI News、X/Twitter 是视频中提到的跟进资源。",
            "OpenAI、Gemini、together.ai、Hyperbolic、LM Studio 是视频中提到的使用入口。",
        ],
        "questions": [
            "你自己的学习或工作流程里，哪些步骤最可能被智能体化？",
            "选择模型时，你会更看重开放权重、推理能力、成本，还是隐私？",
        ],
    },
    {
        "id": "wrap",
        "title": "回到起点：按下回车后到底发生了什么",
        "range": (309, 325),
        "thesis": "一次 ChatGPT 对话可以被还原为：分词、套入对话协议、固定数学函数逐 token 采样；但不同训练阶段决定了它像谁、会什么、哪里会错。",
        "viewpoints": [
            "GPT-4o 类普通助手主要像遵循标注指南的人类标注员的有损模拟。",
            "思考模型在 RL 后不只是模仿标注员，还可能形成新的推理策略。",
            "最可靠的使用姿势是：把它当强力工具，用于灵感和初稿，但检查、验证并对结果负责。",
        ],
        "definitions": [
            ("有损模拟", "模型近似模仿人类写回复的过程，但受 token、参数和有限计算限制。"),
            ("固定数学函数", "部署后的模型参数固定；每个 token 的输出由同一个网络前向计算得到。"),
            ("工具心态", "把模型当作协作者和加速器，而不是无条件可信的权威。"),
        ],
        "examples": [
            "你的查询先变成一维 token 序列，再由模型不断追加 token 形成回答。",
            "模型可能帮你很快起草、解释、改写，也可能随机幻觉、算错或漏掉简单细节。",
        ],
        "questions": [
            "下次使用 LLM 时，你会如何给它更多工作记忆、更多思考空间或更好的工具？",
            "你应该在哪些场景里强制自己做事实核查？",
        ],
    },
]


STAGE_CARDS = [
    ("预训练", "互联网文档", "预测下一个 token", "基座模型", "#data"),
    ("SFT", "对话样本", "模仿理想助手回复", "普通助手", "#sft"),
    ("RL", "练习题与奖励", "试错并强化有效路径", "思考模型", "#rl"),
    ("RLHF", "人类偏好排序", "训练奖励模型再优化", "偏好对齐", "#rlhf"),
]


def paragraph_slice(paragraphs: list[dict[str, object]], start: int, end: int) -> list[dict[str, object]]:
    return [p for p in paragraphs if start <= int(p["line"]) <= end]


def render_list(items: list[str]) -> str:
    return "\n".join(f"<li>{esc(item)}</li>" for item in items)


def render_definitions(items: list[tuple[str, str]]) -> str:
    return "\n".join(
        f"<li><strong>{esc(term)}</strong><span>{esc(definition)}</span></li>"
        for term, definition in items
    )


def render_original(items: list[dict[str, object]]) -> str:
    parts = []
    for item in items:
        line = int(item["line"])
        text = str(item["text"])
        parts.append(
            f'<p class="original-paragraph"><span class="line-no">L{line}</span>{esc(text)}</p>'
        )
    return "\n".join(parts)


def render_chapter(index: int, chapter: dict[str, object], paragraphs: list[dict[str, object]]) -> str:
    start, end = chapter["range"]
    original = paragraph_slice(paragraphs, start, end)
    original_count = len(original)
    return f"""
    <section class="chapter" id="{esc(chapter['id'])}">
      <aside class="chapter-meta">
        <div class="chapter-index">第 {index:02d} 章</div>
        <h2>{esc(chapter["title"])}</h2>
        <p>{esc(chapter["thesis"])}</p>
        <a class="mini-link" href="#top">回到顶部</a>
      </aside>
      <div class="chapter-body">
        <div class="thesis">
          <span>核心观点</span>
          <p>{esc(chapter["thesis"])}</p>
        </div>
        <div class="learning-grid">
          <section class="learning-block accent-a">
            <h3>观点</h3>
            <ul>{render_list(chapter["viewpoints"])}</ul>
          </section>
          <section class="learning-block accent-b">
            <h3>定义</h3>
            <ul class="definition-list">{render_definitions(chapter["definitions"])}</ul>
          </section>
          <section class="learning-block accent-c">
            <h3>示例</h3>
            <ul>{render_list(chapter["examples"])}</ul>
          </section>
          <section class="learning-block accent-d">
            <h3>学习问题</h3>
            <ul>{render_list(chapter["questions"])}</ul>
          </section>
        </div>
        <details class="original">
          <summary>
            <span>原文</span>
            <small>字幕行 {start}-{end}，{original_count} 段</small>
          </summary>
          <div class="original-text">
            {render_original(original)}
          </div>
        </details>
      </div>
    </section>
    """


def render_nav() -> str:
    links = []
    for index, chapter in enumerate(CHAPTERS, start=1):
        links.append(
            f'<a href="#{esc(chapter["id"])}"><span>{index:02d}</span>{esc(chapter["title"].split("：")[0])}</a>'
        )
    return "\n".join(links)


def render_stage_cards() -> str:
    parts = []
    for stage, data, method, output, href in STAGE_CARDS:
        parts.append(
            f"""
            <a class="stage-card" href="{href}">
              <strong>{esc(stage)}</strong>
              <span>{esc(data)}</span>
              <em>{esc(method)}</em>
              <b>{esc(output)}</b>
            </a>
            """
        )
    return "\n".join(parts)


def render_glossary() -> str:
    seen: set[str] = set()
    items = []
    for chapter in CHAPTERS:
        for term, definition in chapter["definitions"]:
            if term in seen:
                continue
            seen.add(term)
            items.append(f"<dt>{esc(term)}</dt><dd>{esc(definition)}</dd>")
    return "\n".join(items)


def build_html() -> str:
    paragraphs = load_paragraphs()
    total_chars = sum(len(str(p["text"])) for p in paragraphs)
    all_chapters = "\n".join(
        render_chapter(index, chapter, paragraphs)
        for index, chapter in enumerate(CHAPTERS, start=1)
    )
    nav = render_nav()
    stage_cards = render_stage_cards()
    glossary = render_glossary()
    chapter_titles = [
        {"id": chapter["id"], "title": chapter["title"]} for chapter in CHAPTERS
    ]
    chapter_json = json.dumps(chapter_titles, ensure_ascii=False)

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LLM 学习可视化 | transcript-cn</title>
  <style>
    :root {{
      --bg: #f7f8f7;
      --paper: #ffffff;
      --ink: #18211f;
      --muted: #60706b;
      --line: #d9e0dc;
      --green: #2d6a56;
      --blue: #2d5f87;
      --coral: #b85b4c;
      --gold: #a67a28;
      --violet: #66558a;
      --mint: #e8f3ee;
      --sky: #e9f1f8;
      --rose: #f8ece9;
      --wheat: #f6efd9;
      --lav: #efedf7;
      --shadow: 0 18px 55px rgba(24, 33, 31, 0.08);
    }}

    * {{ box-sizing: border-box; }}

    html {{ scroll-behavior: smooth; }}

    body {{
      margin: 0;
      color: var(--ink);
      background:
        linear-gradient(90deg, rgba(45, 106, 86, 0.04), transparent 28%, rgba(184, 91, 76, 0.04) 72%, transparent),
        var(--bg);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      line-height: 1.68;
      letter-spacing: 0;
    }}

    a {{ color: inherit; text-decoration: none; }}

    .wrap {{
      width: min(1180px, calc(100% - 40px));
      margin: 0 auto;
    }}

    .hero {{
      position: relative;
      padding: 54px 0 32px;
      border-bottom: 1px solid var(--line);
      background:
        linear-gradient(135deg, rgba(45, 95, 135, 0.10), rgba(45, 106, 86, 0.05) 40%, rgba(184, 91, 76, 0.08)),
        #fbfcfb;
      overflow: hidden;
    }}

    .hero::after {{
      content: "";
      position: absolute;
      inset: auto 0 0;
      height: 8px;
      background: linear-gradient(90deg, var(--green), var(--blue), var(--coral), var(--gold), var(--violet));
    }}

    .kicker {{
      margin: 0 0 12px;
      color: var(--green);
      font-size: 13px;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}

    h1 {{
      max-width: 880px;
      margin: 0;
      font-size: clamp(34px, 6vw, 70px);
      line-height: 1.03;
      letter-spacing: 0;
    }}

    .lead {{
      max-width: 820px;
      margin: 20px 0 0;
      color: #33403d;
      font-size: 18px;
    }}

    .hero-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.1fr) minmax(280px, 0.9fr);
      gap: 28px;
      align-items: end;
    }}

    .hero-panel {{
      background: rgba(255, 255, 255, 0.72);
      border: 1px solid rgba(217, 224, 220, 0.9);
      border-radius: 8px;
      padding: 18px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }}

    .hero-panel h2 {{
      margin: 0 0 12px;
      font-size: 18px;
    }}

    .stats {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
    }}

    .stat {{
      min-height: 84px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--paper);
    }}

    .stat strong {{
      display: block;
      font-size: 24px;
      line-height: 1.1;
    }}

    .stat span {{
      color: var(--muted);
      font-size: 13px;
    }}

    .toolbar {{
      position: sticky;
      top: 0;
      z-index: 30;
      border-bottom: 1px solid var(--line);
      background: rgba(247, 248, 247, 0.92);
      backdrop-filter: blur(14px);
    }}

    .toolbar-inner {{
      display: grid;
      grid-template-columns: minmax(180px, 340px) 1fr auto;
      gap: 14px;
      align-items: center;
      min-height: 66px;
    }}

    .search {{
      width: 100%;
      height: 40px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0 12px;
      color: var(--ink);
      background: var(--paper);
      font-size: 14px;
      outline: none;
    }}

    .search:focus {{
      border-color: var(--blue);
      box-shadow: 0 0 0 3px rgba(45, 95, 135, 0.12);
    }}

    .nav-strip {{
      display: flex;
      gap: 8px;
      overflow-x: auto;
      padding: 8px 0;
      scrollbar-width: thin;
    }}

    .nav-strip a {{
      display: inline-flex;
      align-items: center;
      gap: 7px;
      white-space: nowrap;
      min-height: 34px;
      padding: 6px 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.75);
      color: #33403d;
      font-size: 13px;
    }}

    .nav-strip a.active {{
      border-color: rgba(45, 106, 86, 0.35);
      background: var(--mint);
      color: var(--green);
    }}

    .nav-strip span {{
      font-variant-numeric: tabular-nums;
      color: var(--muted);
    }}

    .toolbar-actions {{
      display: flex;
      gap: 8px;
    }}

    button {{
      min-height: 40px;
      padding: 0 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--paper);
      color: var(--ink);
      font: inherit;
      font-size: 14px;
      cursor: pointer;
    }}

    button:hover {{
      border-color: rgba(45, 95, 135, 0.35);
      background: var(--sky);
    }}

    .progress {{
      height: 3px;
      background: transparent;
    }}

    .progress span {{
      display: block;
      width: 0;
      height: 100%;
      background: linear-gradient(90deg, var(--green), var(--blue), var(--coral));
    }}

    main {{
      padding-bottom: 72px;
    }}

    .overview {{
      padding: 34px 0 22px;
      border-bottom: 1px solid var(--line);
    }}

    .section-title {{
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 20px;
      margin-bottom: 18px;
    }}

    .section-title h2 {{
      margin: 0;
      font-size: 26px;
      letter-spacing: 0;
    }}

    .section-title p {{
      max-width: 620px;
      margin: 0;
      color: var(--muted);
    }}

    .pipeline {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }}

    .stage-card {{
      position: relative;
      min-height: 188px;
      padding: 18px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--paper);
      box-shadow: 0 10px 30px rgba(24, 33, 31, 0.05);
    }}

    .stage-card::before {{
      content: "";
      position: absolute;
      left: 18px;
      right: 18px;
      top: 0;
      height: 5px;
      border-radius: 0 0 8px 8px;
      background: var(--green);
    }}

    .stage-card:nth-child(2)::before {{ background: var(--blue); }}
    .stage-card:nth-child(3)::before {{ background: var(--coral); }}
    .stage-card:nth-child(4)::before {{ background: var(--violet); }}

    .stage-card strong {{
      display: block;
      margin-bottom: 10px;
      font-size: 22px;
    }}

    .stage-card span,
    .stage-card em,
    .stage-card b {{
      display: block;
      margin-top: 8px;
      color: var(--muted);
      font-style: normal;
      font-weight: 500;
    }}

    .stage-card b {{
      color: var(--ink);
      font-weight: 800;
    }}

    .model-map {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-top: 18px;
    }}

    .model-map section {{
      min-height: 210px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--paper);
      padding: 18px;
    }}

    .model-map h3 {{
      margin: 0 0 14px;
      font-size: 18px;
    }}

    .flow {{
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 8px;
      align-items: stretch;
    }}

    .flow div {{
      min-height: 74px;
      padding: 11px;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: #fbfcfb;
    }}

    .flow strong {{
      display: block;
      margin-bottom: 4px;
      color: var(--green);
    }}

    .memory {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }}

    .memory div {{
      padding: 14px;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: #fbfcfb;
    }}

    .memory strong {{
      display: block;
      margin-bottom: 6px;
    }}

    .chapter {{
      display: grid;
      grid-template-columns: minmax(220px, 300px) minmax(0, 1fr);
      gap: 34px;
      padding: 42px 0;
      border-bottom: 1px solid var(--line);
    }}

    .chapter.hidden {{ display: none; }}

    .chapter-meta {{
      position: sticky;
      top: 88px;
      align-self: start;
      padding-left: max(20px, calc((100vw - 1180px) / 2));
    }}

    .chapter-meta h2 {{
      margin: 8px 0 12px;
      font-size: 24px;
      line-height: 1.22;
      letter-spacing: 0;
    }}

    .chapter-meta p {{
      margin: 0;
      color: var(--muted);
      font-size: 14px;
    }}

    .chapter-index {{
      color: var(--coral);
      font-size: 13px;
      font-weight: 800;
    }}

    .mini-link {{
      display: inline-block;
      margin-top: 16px;
      color: var(--blue);
      font-size: 13px;
      font-weight: 700;
    }}

    .chapter-body {{
      width: min(860px, calc(100vw - 360px));
      padding-right: max(20px, calc((100vw - 1180px) / 2));
    }}

    .thesis {{
      margin-bottom: 16px;
      padding: 18px;
      border: 1px solid rgba(45, 106, 86, 0.22);
      border-radius: 8px;
      background: var(--mint);
    }}

    .thesis span {{
      display: block;
      margin-bottom: 8px;
      color: var(--green);
      font-size: 13px;
      font-weight: 800;
    }}

    .thesis p {{
      margin: 0;
      font-size: 18px;
      font-weight: 700;
      line-height: 1.48;
    }}

    .learning-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}

    .learning-block {{
      min-height: 238px;
      padding: 17px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--paper);
    }}

    .learning-block h3 {{
      margin: 0 0 10px;
      font-size: 16px;
    }}

    .learning-block ul {{
      margin: 0;
      padding-left: 18px;
    }}

    .learning-block li {{
      margin: 7px 0;
      color: #30403c;
    }}

    .definition-list {{
      padding-left: 0 !important;
      list-style: none;
    }}

    .definition-list li {{
      margin: 9px 0;
    }}

    .definition-list strong {{
      display: block;
      color: var(--ink);
      font-weight: 800;
    }}

    .definition-list span {{
      display: block;
      color: var(--muted);
    }}

    .accent-a {{ background: linear-gradient(180deg, var(--paper), var(--sky)); }}
    .accent-b {{ background: linear-gradient(180deg, var(--paper), var(--lav)); }}
    .accent-c {{ background: linear-gradient(180deg, var(--paper), var(--wheat)); }}
    .accent-d {{ background: linear-gradient(180deg, var(--paper), var(--rose)); }}

    .original {{
      margin-top: 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--paper);
      overflow: hidden;
    }}

    .original summary {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      min-height: 54px;
      padding: 0 16px;
      cursor: pointer;
      font-weight: 800;
      list-style: none;
    }}

    .original summary::-webkit-details-marker {{ display: none; }}

    .original small {{
      color: var(--muted);
      font-weight: 500;
    }}

    .original-text {{
      max-height: 520px;
      overflow: auto;
      padding: 0 16px 16px;
      border-top: 1px solid var(--line);
      background: #fcfdfc;
    }}

    .original-paragraph {{
      margin: 16px 0;
      color: #27332f;
      font-size: 15px;
    }}

    .line-no {{
      display: inline-block;
      min-width: 44px;
      margin-right: 10px;
      color: var(--coral);
      font-size: 12px;
      font-weight: 800;
      font-variant-numeric: tabular-nums;
    }}

    .glossary-section,
    .review-section {{
      padding: 40px 0;
      border-bottom: 1px solid var(--line);
    }}

    .glossary {{
      columns: 3 260px;
      column-gap: 24px;
      margin: 0;
    }}

    .glossary dt {{
      break-after: avoid;
      margin: 0 0 4px;
      font-weight: 850;
      color: var(--green);
    }}

    .glossary dd {{
      break-inside: avoid;
      margin: 0 0 16px;
      color: var(--muted);
    }}

    .review-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }}

    .review-card {{
      min-height: 170px;
      padding: 18px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--paper);
    }}

    .review-card strong {{
      display: block;
      margin-bottom: 8px;
      color: var(--blue);
      font-size: 18px;
    }}

    .review-card p {{
      margin: 0;
      color: var(--muted);
    }}

    .empty-state {{
      display: none;
      padding: 28px;
      border: 1px dashed var(--line);
      border-radius: 8px;
      color: var(--muted);
      background: var(--paper);
      text-align: center;
    }}

    .empty-state.show {{ display: block; }}

    @media (max-width: 980px) {{
      .hero-grid,
      .model-map,
      .chapter,
      .learning-grid,
      .review-grid {{
        grid-template-columns: 1fr;
      }}

      .toolbar-inner {{
        grid-template-columns: 1fr;
        padding: 10px 0;
      }}

      .toolbar-actions {{
        justify-content: stretch;
      }}

      .toolbar-actions button {{
        flex: 1;
      }}

      .pipeline {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}

      .chapter-meta {{
        position: static;
        padding: 0 20px;
      }}

      .chapter-body {{
        width: auto;
        padding: 0 20px;
      }}
    }}

    @media (max-width: 620px) {{
      .wrap {{
        width: min(100% - 28px, 1180px);
      }}

      .hero {{
        padding-top: 36px;
      }}

      .lead {{
        font-size: 16px;
      }}

      .stats,
      .pipeline,
      .flow,
      .memory {{
        grid-template-columns: 1fr;
      }}

      .chapter {{
        gap: 14px;
        padding: 30px 0;
      }}

      .chapter-meta h2 {{
        font-size: 22px;
      }}

      .thesis p {{
        font-size: 16px;
      }}

      .learning-block {{
        min-height: auto;
      }}

      .original summary {{
        align-items: start;
        flex-direction: column;
        justify-content: center;
        padding: 10px 14px;
      }}
    }}
  </style>
</head>
<body id="top">
  <header class="hero">
    <div class="wrap hero-grid">
      <div>
        <p class="kicker">字幕可视化学习页</p>
        <h1>大语言模型从预训练到思考模型</h1>
        <p class="lead">基于 <strong>transcript-cn.txt</strong> 生成的学习版 HTML：把长字幕拆成章节，把每章整理成原文、观点、定义、示例和复习问题。</p>
      </div>
      <div class="hero-panel">
        <h2>学习材料概览</h2>
        <div class="stats">
          <div class="stat"><strong>{len(CHAPTERS)}</strong><span>个章节</span></div>
          <div class="stat"><strong>{len(paragraphs)}</strong><span>段原文</span></div>
          <div class="stat"><strong>{total_chars:,}</strong><span>个中文字符左右</span></div>
        </div>
      </div>
    </div>
  </header>

  <div class="toolbar">
    <div class="wrap toolbar-inner">
      <input class="search" id="search" type="search" placeholder="搜索章节、概念或原文" />
      <nav class="nav-strip" id="navStrip" aria-label="章节导航">
        {nav}
      </nav>
      <div class="toolbar-actions">
        <button id="expandAll" type="button">展开原文</button>
        <button id="collapseAll" type="button">收起原文</button>
      </div>
    </div>
    <div class="progress"><span id="progressBar"></span></div>
  </div>

  <main>
    <section class="overview">
      <div class="wrap">
        <div class="section-title">
          <h2>一张学习地图</h2>
          <p>把整支视频压缩成四个动作：先学互联网文本，再学对话，接着在可验证任务上练习，最后用偏好反馈补齐主观任务。</p>
        </div>
        <div class="pipeline">
          {stage_cards}
        </div>
        <div class="model-map">
          <section>
            <h3>从文本框到回答</h3>
            <div class="flow">
              <div><strong>1. 分词</strong><span>用户输入变成一维 token 序列。</span></div>
              <div><strong>2. 协议</strong><span>user / assistant 等结构被特殊 token 标记。</span></div>
              <div><strong>3. 采样</strong><span>固定网络反复预测并追加下一个 token。</span></div>
            </div>
          </section>
          <section>
            <h3>两种“知识”</h3>
            <div class="memory">
              <div><strong>参数里的知识</strong><span>像模糊回忆，来自预训练，有损、概率性、可能幻觉。</span></div>
              <div><strong>上下文里的知识</strong><span>像工作记忆，模型当前可直接读取，适合摘要、核查和工具结果。</span></div>
            </div>
          </section>
        </div>
      </div>
    </section>

    <div class="wrap">
      <div class="empty-state" id="emptyState">没有找到匹配内容，换个关键词试试。</div>
    </div>

    {all_chapters}

    <section class="glossary-section">
      <div class="wrap">
        <div class="section-title">
          <h2>概念索引</h2>
          <p>复习时可以先扫这一页，把术语和直觉对上，再回到对应章节读原文。</p>
        </div>
        <dl class="glossary">
          {glossary}
        </dl>
      </div>
    </section>

    <section class="review-section">
      <div class="wrap">
        <div class="section-title">
          <h2>复习路径</h2>
          <p>第一次学习建议按顺序读；第二次可以按下面三个问题倒推整套机制。</p>
        </div>
        <div class="review-grid">
          <div class="review-card"><strong>它从哪学来知识？</strong><p>回看预训练、数据清洗、分词、预测下一个 token 和基座模型。</p></div>
          <div class="review-card"><strong>它为什么像助手？</strong><p>回看 SFT、对话协议、标注员示例和系统消息。</p></div>
          <div class="review-card"><strong>它为什么会错？</strong><p>回看幻觉、有限 token 计算、工具调用、RLHF 和瑞士奶酪模型。</p></div>
        </div>
      </div>
    </section>
  </main>

  <script>
    const chapters = {chapter_json};
    const searchInput = document.getElementById('search');
    const emptyState = document.getElementById('emptyState');
    const chapterEls = Array.from(document.querySelectorAll('.chapter'));
    const detailsEls = Array.from(document.querySelectorAll('details.original'));
    const navLinks = Array.from(document.querySelectorAll('.nav-strip a'));
    const progressBar = document.getElementById('progressBar');

    function filterChapters() {{
      const query = searchInput.value.trim().toLowerCase();
      let visible = 0;
      chapterEls.forEach((chapter) => {{
        const hit = !query || chapter.textContent.toLowerCase().includes(query);
        chapter.classList.toggle('hidden', !hit);
        if (hit) visible += 1;
      }});
      emptyState.classList.toggle('show', visible === 0);
    }}

    function updateProgress() {{
      const doc = document.documentElement;
      const max = doc.scrollHeight - window.innerHeight;
      const pct = max > 0 ? (window.scrollY / max) * 100 : 0;
      progressBar.style.width = pct + '%';
    }}

    function updateActiveNav() {{
      let current = chapterEls[0]?.id;
      for (const chapter of chapterEls) {{
        if (chapter.classList.contains('hidden')) continue;
        const top = chapter.getBoundingClientRect().top;
        if (top < 140) current = chapter.id;
      }}
      navLinks.forEach((link) => {{
        link.classList.toggle('active', link.getAttribute('href') === '#' + current);
      }});
    }}

    searchInput.addEventListener('input', () => {{
      filterChapters();
      updateActiveNav();
    }});

    document.getElementById('expandAll').addEventListener('click', () => {{
      detailsEls.forEach((details) => details.open = true);
    }});

    document.getElementById('collapseAll').addEventListener('click', () => {{
      detailsEls.forEach((details) => details.open = false);
    }});

    window.addEventListener('scroll', () => {{
      updateProgress();
      updateActiveNav();
    }}, {{ passive: true }});

    filterChapters();
    updateProgress();
    updateActiveNav();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    OUTPUT.write_text(build_html(), encoding="utf-8")
    print(f"Wrote {OUTPUT}")
