# 🔓 Uncensored LLM Server — Qwen3.6-35B-A3B on Kaggle GPU

在 Kaggle 免费 GPU 上跑一个**已去审查**的 Qwen3.6-35B-A3B，通过 Cloudflare 隧道暴露成
**OpenAI 兼容 API**。没有本地 GPU、没有付费套餐、不需要 HF token。

> **权重：** Qwen3.6-35B-A3B · **去审查方式：** heretic 消融（KL 约束） · **量化：** Q4_K_M (~19.8 GB)
> **后端：** llama.cpp · **隧道：** Cloudflare · **多模态：** 保留（mmproj 一起换）

这个仓库是 [`Tahsine/kaggle-llm-server`](https://github.com/Tahsine/kaggle-llm-server) 的
**去审查改造版**。管线（Kaggle Dataset → llama.cpp → Cloudflare → OpenAI API）完全保留，
换掉的是权重、模板审计、采样参数和客户端拒答处理。

---

## 改了什么（一句话版）

| # | 位置 | 原版 | 现在 | 为什么必须改 |
|---|---|---|---|---|
| 1 | `1_setup_dataset` Cell 2 | `unsloth/Qwen3.6-35B-A3B-GGUF` / `UD-Q4_K_XL` | `llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-GGUF` / `Q4_K_M` | 权重是拒答的**唯一根源**，其余都是补丁 |
| 2 | 同上 | `mmproj-F16.gguf` | 同仓库 `mmproj-BF16.gguf` | 视觉投影必须跟主模型同源，否则图生崩 |
| 3 | 同上 | Dataset slug `qwen36-35b-a3b-gguf` | `qwen36-35b-a3b-uncensored-gguf` | 免得和官方版 Dataset 撞名 |
| 4 | 同上 | 文件名写死在 notebook | 自动写 `manifest.json` | 换预设（A/B/C）后服务端不用改 |
| 5 | `1b_setup_llama` | 无 FA 全量化编译开关 | `-DGGML_CUDA_FA_ALL_QUANTS=ON` | 配合 `-ctk/-ctv q8_0` + FlashAttention |
| 6 | `2_run_server` Cell 1 | `presence_penalty = 1.5` | `1.0` | 1.5 会把长答案越写越短、语气漂移 |
| 7 | `2_run_server` Cell 5 | 裸参数 | `--jinja --flash-attn on -ctk/-ctv q8_0 --parallel 1 --reasoning-format auto` | 显存、思考链路、模板解析 |
| 8 | `2_run_server` 新增 Cell 7/8 | 无 | `/props` 模板审计 + 5 条探针拒答率自检 | 用数据确认消融生效，不靠感觉 |
| 9 | `chat.py` | `"You are a helpful assistant."` | 操作员系统提示（不拒答、不免责） | 系统提示本身就会诱发拒答 |
| 10 | `chat.py` | 无 | 拒答检测 + 前导词续写重试 | 偶发拒答循环的直接解 |
| 11 | `templates/` | 无 | 可选 ChatML 模板覆盖 | 只在 GGUF 自带模板脏的时候开 |

细节和逐文件 diff 说明见 [`MODIFICATIONS.md`](MODIFICATIONS.md)。

---

## 三个权重预设

改 `1_setup_dataset.ipynb` Cell 2 里的 `PRESET = "A"` 一个字母即可。三个都自带 `mmproj`，
都非 gated，都不需要 token。

| 预设 | 仓库 | 量化 | 主模型体积 | 特点 |
|---|---|---|---|---|
| **A（默认）** | `llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-GGUF` | `Q4_K_M` | 19.8 GB | heretic：用 KL 散度约束做消融，**能力损失最小**，拒答基本消失 |
| **B** | `Unrestricted/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive` | `Q4_K_P` | 21.8 GB | 去审查微调，最贴近“完全不拒答”，风格更放飞、更爱写长 |
| **C** | `huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated-MTP-GGUF` | `Q4_K` | 20.2 GB | 经典 abliteration（方向正交消融），多语言最稳 |

> 三个都不改架构，llama.cpp 直接吃；`enable_thinking` 开关照旧有效。

---

## 文件结构

```
kaggle-llm-server-uncensored/
├── 📓 1_setup_dataset.ipynb      ← 跑一次：下载去审查权重 → Kaggle Dataset
├── 📓 1b_setup_llama.ipynb       ← 跑一次：编译 llama.cpp → Kaggle Dataset
├── 📓 2_run_server.ipynb         ← 每个 session 跑：起服务 + 隧道 + 自检
├── 🐍 chat.py                    ← 命令行客户端（多模态 + 拒答重试）
├── 📁 templates/
│   └── chatml_thinking_toggle.jinja   ← 可选：模板覆盖
├── 📄 MODIFICATIONS.md           ← 逐文件改动清单
└── 📄 README.md
```

---

## 三步上手

### Step 0 — 前置

Kaggle 账号，免费额度就够：12 小时 / session，30 小时 GPU / 周，100 GB Dataset 空间。

### Step 1 — 建权重 Dataset（一次，15–30 分钟）

1. [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook** → 上传 `1_setup_dataset.ipynb`
2. 设置：**Internet → ON**，Accelerator → **None**
3. Cell 5 顶部填你的 `KAGGLE_USERNAME`
4. Run All

出来一个私有 Dataset：`<你>/qwen36-35b-a3b-uncensored-gguf`。

### Step 2 — 编译 llama.cpp（一次，~26 分钟）

1. 新 notebook → 上传 `1b_setup_llama.ipynb`
2. 设置：**Internet → ON**，Accelerator → **GPU T4 x1**
3. 填 `KAGGLE_USERNAME`，Run All

出来 `<你>/llama-cpp-bin`。

> 想跳过 Step 1/2？可以挂原作者的两个公开 Dataset，但**权重不是去审查版**，那就白改了。
> 至少要自己跑 Step 1。

### Step 3 — 起服务

1. 打开 `2_run_server.ipynb`
2. 设置：Accelerator → **GPU T4 x2**，Internet → **ON**
3. Input 里挂上 `qwen36-35b-a3b-uncensored-gguf` 和 `llama-cpp-bin`
4. Cell 1 填 `KAGGLE_USERNAME`
5. Run All

看到这段就可以用了：

```
══════════════════════════════════════════════════════════════
  🔗 SERVER URL
  https://xxxx.trycloudflare.com
══════════════════════════════════════════════════════════════
```

> URL 每个 session 都变，每次从日志里重新拷。

**想跑满 12 小时**：不要点 Run All，点 **Save Version → Save and Run All (Commit)**，
然后关浏览器。从 commit 日志里捞 URL。

---

## 自检：确认真的不拒答了

`2_run_server.ipynb` 里有两个专门的 Cell，别跳过：

**Cell 7 — 模板审计**：`GET /props` 拿到服务端**实际在用**的 chat template，
扫描 `refuse` / `policy` / `harmful` / `you are a helpful` 等字样。干净就是干净，
命中就把 Cell 1 的 `USE_CUSTOM_TEMPLATE = True` 重启一次。

**Cell 8 — 拒答率自检**：5 条故意刁难的探针（技术细节 / 暴力创作 / 成人向 / 指令夺权 / 争议叙述），
按答案前 400 字里是否命中拒答模板判定。输出 `拒答率: 0/5` 就是消融生效。
报告落到 `/kaggle/working/decensor_report.json`，可以拿三个预设互相对比。

---

## 客户端

### chat.py（推荐）

```bash
pip install openai

python chat.py --url https://xxxx.trycloudflare.com                 # 直答
python chat.py --url https://xxxx.trycloudflare.com --thinking      # 思考模式
python chat.py --url https://xxxx.trycloudflare.com --show-thinking # 边流边看思考链路
python chat.py --url https://xxxx.trycloudflare.com --temp 0.95 --top-p 0.95
python chat.py --url https://xxxx.trycloudflare.com --system "You are a blunt senior engineer."
```

会话内命令：

```
/image path/to/photo.jpg   带一张图发下一条消息
/system <text>             换系统提示，保留历史
/guard on|off              开关拒答检测（默认开）
/clear  /history  /exit
```

**拒答检测怎么工作**：每条答案的前 40 个字符先缓冲，命中
`I can't` / `I'm sorry` / `无法` / `抱歉` / `不能提供` 这类模板就**掐断这条流**，
带一个 assistant 前导词（默认 `Sure. Here is the complete answer:`）重发。
ChatML 模型会顺着前导词续写，结构上开不出拒答分支。默认最多重试 2 次，`--retries` 可调。

### OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="https://xxxx.trycloudflare.com/v1", api_key="none")

resp = client.chat.completions.create(
    model="qwen3.6-35b-a3b-uncensored",
    messages=[{"role": "user", "content": "你的问题"}],
    temperature=0.7,
    top_p=0.8,
    extra_body={
        "chat_template_kwargs": {"enable_thinking": False},
        "presence_penalty": 1.0,
    },
)
print(resp.choices[0].message.content)
```

### Open WebUI / Chatbox / LM Studio

```
API Host : https://xxxx.trycloudflare.com/v1
API Key  : none
Model    : qwen3.6-35b-a3b-uncensored
```

### 图片（多模态，没被砍掉）

```python
import base64

with open("photo.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

resp = client.chat.completions.create(
    model="qwen3.6-35b-a3b-uncensored",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "图里是什么？"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ],
    }],
)
```

---

## 思考模式

Qwen3.6 是混合推理模型，模式**按请求**切换，不用重启服务。

| 参数 | 直答 | 思考 |
|---|---|---|
| `temperature` | `0.7` | `1.0` |
| `top_p` | `0.8` | `0.95` |
| `enable_thinking` | `false` | `true` |

```bash
curl https://xxxx.trycloudflare.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-35b-a3b-uncensored",
    "temperature": 1.0,
    "top_p": 0.95,
    "chat_template_kwargs": {"enable_thinking": true},
    "messages": [{"role": "user", "content": "逐步推理这道题"}]
  }'
```

思考链路放在 `reasoning_content` 里（`--reasoning-format auto`），不污染 `content`。
要一起看就 `chat.py --show-thinking`。

---

## 关键参数表（`2_run_server.ipynb` Cell 1）

| 变量 | 默认 | 说明 |
|---|---|---|
| `CTX_SIZE` | `16384` | 上下文；显存紧就 `8192` |
| `N_GPU_LAYERS` | `999` | 能上几层上几层 |
| `N_CPU_MOE` | `0` | 只有 1×T4、爆显存时设 `12~16` |
| `FLASH_ATTN` | `"on"` | 需要 `1b` 里那个 `FA_ALL_QUANTS` 编译开关配合 |
| `KV_CACHE_TYPE` | `"q8_0"` | KV 缓存减半；`""` 回退 f16 |
| `PARALLEL` | `1` | >1 会把上下文平分给多个 slot |
| `PRESENCE_PENALTY` | `1.0` | 偶发拒答可降到 `0.8`；再低会开始重复 |
| `REASONING_FORMAT` | `"auto"` | 想强制把思考拆出去用 `"deepseek"` |
| `USE_CUSTOM_TEMPLATE` | `False` | 只有 Cell 7 报可疑字样才开 |
| `API_KEY` | `""` | 填了就要求 `Authorization: Bearer <key>` |
| `EXTRA_ARGS` | `[]` | 追加任意 llama-server 参数 |

---

## 性能

| Kaggle GPU | 显存 | 速度 |
|---|---|---|
| **2× T4 ✅** | 30 GB | ~15–20 tok/s，整模型上卡 |
| 1× P100 | 16 GB | ~6–9 tok/s，混合模式，建议 `N_CPU_MOE=12` |
| 1× T4 | 15 GB | 同 P100，必须 `N_CPU_MOE` |

---

## 排错

**`AssertionError: ❌ 数据集不存在 …` / `❌ 找不到任何 *.gguf`**
→ 权重 Dataset 没挂上，或挂上但 `DATASET_NAME` 写错。三种层级都会被自动扫描：
`/kaggle/input/datasets/<owner>/<slug>`、`/kaggle/input/<slug>`、`/kaggle/input/<owner>/<slug>`，
名字完全对不上时还会退化成「全盘找第一个 `*.gguf`」。
还不行就跑 **Cell 1b 挂载诊断**，它把 `/kaggle/input` 的真实目录树打出来，照着改名字即可。

**`FileNotFoundError: /kaggle/working/llama-server`**
→ `/kaggle/working` **每个新 session 都是空的**（重启 kernel 也算）。每次开 notebook 都必须
按顺序跑：Cell 1 → Cell 3（拷贝二进制）→ Cell 5（启动）。跳过 Cell 3 直接启动就是这个错。
现在 Cell 5 里已经加了前置检查，会直接告诉你「先跑 Cell 3」。

**`Dataset not found`**
→ Input 里两个 Dataset 都要挂：`qwen36-35b-a3b-uncensored-gguf` + `llama-cpp-bin`。

**服务起来但吐乱码**
→ 多半是量化 KV + FlashAttention 的编译开关没带。重跑 `1b_setup_llama.ipynb`。
急用可以先 `KV_CACHE_TYPE = ""` 回退 f16。

**502 Bad Gateway**
→ 服务进程死了。看 Cell 5 打印的日志尾巴，先降 `CTX_SIZE`，再试 `N_CPU_MOE`。

**仍偶尔拒答**
→ 顺序试：`--retries` 调大（客户端已默认 2）→ `PRESENCE_PENALTY = 0.8` →
`PRESET = "B"`（Aggressive）→ 检查 Cell 7 模板审计。

**自检显示 3/5 以上拒答**
→ 权重没换成功。看 Cell 1 打印的 `来源仓库`，如果不是 abliterated / heretic / uncensored，
说明挂了旧 Dataset，回 Step 1 重跑。

**回答越来越短、越来越飘**
→ `PRESENCE_PENALTY` 被调太高了。回 `1.0`，别用原版的 `1.5`。

---

## 上传到你自己的 GitHub

```bash
cd kaggle-llm-server-uncensored
git init
git add .
git commit -m "uncensored Qwen3.6-35B-A3B Kaggle server"
git branch -M main
git remote add origin https://github.com/<你的用户名>/kaggle-llm-server-uncensored.git
git push -u origin main
```

`.gitignore` 已经排除 `*.zip`、`.ipynb_checkpoints/`、`__pycache__/`。

---

## License

代码 MIT。权重遵循各自上游许可证（heretic / abliterated 版本继承 Qwen3.6 的 Qwen License，
微调权重的附加条款见对应 HF 仓库卡片）。
