# LLM pairing

Scan your computer and get an honest answer to: **which local LLMs can
this machine actually run, roughly how fast, and exactly which file
should I download?**

[繁體中文版在下方](#llm-pairing繁體中文) / Traditional Chinese version below.

Core belief: **if we don't know, we say we don't know — we never
guess.** Every number carries an evidence tier (T0 = spec-sheet
estimate, T1 = calibrated on your machine, T2 = exact), every
approximation carries an honesty flag, and the system prefers a
conservative refusal (false reject) over an inflated promise (false
accept — design target ≤ 2%, validated against a growing
ground-truth confusion matrix; 10 cells scored so far: 8 true accepts,
0 false accepts).

## What it does

1. **Probe** (`llmpairing probe`) — read-only, zero upload, no admin
   rights. Reads basic CPU/RAM/GPU facts; anything it cannot obtain is
   honestly reported as UNKNOWN.
2. **Pair** (`llmpairing recommend`) — matches the scan against a
   versioned model-catalog snapshot using a pure-function memory model
   (weights + KV cache + activations + logits) and classifies every
   combination: fits / tight / runs-but-slower (partial offload) /
   long-context OOM / won't load / unsupported.
3. **Speed prediction** — single-stream decode is memory-bandwidth
   bound (roofline model): tok/s ≈ effective bandwidth × efficiency ÷
   bytes read per token. After a one-time local calibration (T-003)
   predictions upgrade from literature priors (T0) to your machine's
   measured product (T1).
4. **Interactive demo** — `demo/llm-pairing-demo.html`:
   recommendation-first cards, with the full matrix (memory breakdown,
   trade-off curves, decay curves) as the advanced view.
   **[Try it live](https://waynechou-bot.github.io/llm-pairing/)**
   (reference machines only — build it locally to see *your* machine).

## Install

Requires Python 3.11+. Conda recommended:

```bash
conda create -n llmpairing python=3.11
conda activate llmpairing
pip install -e ".[dev]"
```

## Quick start

```bash
llmpairing probe                 # see what the read-only scan finds
llmpairing recommend             # scan + pair + Top-3 picks with download commands
llmpairing recommend --ctx 32768 # re-rank for long-context use
```

Going further (run from the repo root):

```bash
# rebuild the model-catalog snapshot (needs network; output is an
# immutable content-hashed file)
python tools/catalog/build_catalog.py --top 25

# one-time local speed calibration (needs local Ollama and one
# whitelisted-architecture model)
python tools/calibrate/run_calibration.py \
    --ollama-model qwen3.5:4b --catalog-id Qwen/Qwen3.5-4B --quant Q4_K_M

# build the interactive demo (auto-uses the newest snapshot + your scan)
python tools/demo/build_demo.py && open demo/llm-pairing-demo.html

# collect ground truth and score predictions (TA/FA/FR/TR confusion matrix)
python tools/groundtruth/run_groundtruth.py
python tools/groundtruth/analyze.py data/groundtruth/<machine>/<run>.jsonl
```

## Current support status (honest disclosure)

| Area | Status |
|---|---|
| Windows: CPU / Intel iGPU | ✅ verified on real hardware (with ground truth) |
| Windows: NVIDIA discrete GPU | ⚠ nvidia-smi parsing implemented but **not yet verified on a real NVIDIA card** — results always carry the `NVIDIA_SMI_PARSER_UNVERIFIED_ON_REAL_HW` flag |
| Dual GPU (iGPU + data-less dGPU) | ✅ degrades to the usable pool with a flag |
| True multi-GPU (all with data) | ❌ honestly refused in v1 (UNSUPPORTED_TOPOLOGY) |
| macOS / Linux GPU | ❌ not implemented (T-001 S6; the probe says so instead of fabricating) |
| Inference engines | llama.cpp family (incl. Ollama). Others are honestly refused |
| Architecture whitelist | llama / qwen2 / qwen3(_moe) / mistral / gemma3(_text) / gemma4(_text) / qwen3.5(_moe). Anything else reports "unsupported" (e.g. deepseek_v4: triple-K-cache semantics only exist in an experimental port — refused rather than approximated) |

## Methodology

Developed spec-first in the author's working repository: every memory
formula is pinned by hand-computed golden vectors before
implementation, every architecture addition is an amendment with cited
sources, and predictions are scored against real-machine ground truth
(TA/FA/FR/TR). The public tree carries the code and the versioned
catalog snapshots; the working ledger and spec pack are not published.

## Privacy

The scan runs entirely on your machine, read-only, and uploads
nothing. You build the catalog snapshot yourself; pairing never
touches the network (reproducibility over freshness).

## License

MIT — see [LICENSE](LICENSE). Each model in a catalog snapshot carries
its own `license` field; check it before downloading.

---

# LLM pairing（繁體中文）

掃描你的電腦，誠實告訴你「哪些本地 LLM 跑得動、大概多快、該下載哪個檔」。

核心信念：**不知道就說不知道，絕不用猜的**。每個數字都標注證據等級
（T0 = 規格表估算、T1 = 本機實測校準、T2 = 精確值），每個近似都掛
誠實旗標，寧可保守拒絕（false-reject）也不誇口承諾（false-accept，
設計目標 ≤ 2%——以持續累積的 ground-truth 混淆矩陣驗證中，
目前 10 格計分：8 個 true accept、0 個 false accept）。

## 它做什麼

1. **掃描**（`llmpairing probe`）——唯讀、零上傳、不需系統管理員。
   讀 CPU/RAM/GPU 基本資訊；拿不到的欄位誠實標 UNKNOWN。
2. **配對**（`llmpairing recommend`）——把掃描結果對上版本化的模型
   目錄快照，用純函式記憶體模型（權重 + KV cache + activation +
   logits）逐一判定：可以跑 / 勉強可跑 / 可跑・會變慢（部分卸載）/
   長文跑不動 / 跑不動 / 尚未支援。
3. **速度預測**——單流解碼是記憶體頻寬瓶頸（roofline 模型）：
   tok/s ≈ 有效頻寬 × 效率 ÷ 每 token 讀取量。跑一次本機校準
   （T-003）後，預測從文獻先驗（T0）升級成你機器的實測乘積（T1）。
4. **互動 Demo**——`demo/llm-pairing-demo.html`，推薦卡優先、完整
   矩陣為進階視圖，含記憶體分解、trade-off 曲線與衰減曲線。
   **[線上試玩](https://waynechou-bot.github.io/llm-pairing/)**
   （僅參考機——想看「你的機器」請在本機建置）。

## 安裝

需求：Python 3.11+。建議 conda：

```bash
conda create -n llmpairing python=3.11
conda activate llmpairing
pip install -e ".[dev]"
```

## 快速開始

```bash
llmpairing probe                 # 看看掃描到什麼（唯讀）
llmpairing recommend             # 掃描 + 配對 + Top-3 推薦（附下載指令）
llmpairing recommend --ctx 32768 # 長文條件下重新推薦
```

進一步（都在 repo 根目錄執行）：

```bash
# 重建模型目錄快照（需要網路；輸出含內容雜湊的不可覆寫檔名）
python tools/catalog/build_catalog.py --top 25

# 本機速度校準（需要本機 Ollama 與一個白名單架構的模型）
python tools/calibrate/run_calibration.py \
    --ollama-model qwen3.5:4b --catalog-id Qwen/Qwen3.5-4B --quant Q4_K_M

# 建互動 Demo（會自動用最新快照與你的掃描檔）
python tools/demo/build_demo.py && start demo/llm-pairing-demo.html

# 收集 ground truth 並對預測評分（TA/FA/FR/TR 混淆矩陣）
python tools/groundtruth/run_groundtruth.py
python tools/groundtruth/analyze.py data/groundtruth/<machine>/<run>.jsonl
```

## 目前支援範圍（誠實聲明）

| 面向 | 狀態 |
|---|---|
| Windows：CPU / Intel iGPU | ✅ 已在真實機器驗證（含 ground truth） |
| Windows：NVIDIA 獨顯 | ⚠ nvidia-smi 解析已實作，但**尚未在真 N 卡驗證**——結果一律掛 `NVIDIA_SMI_PARSER_UNVERIFIED_ON_REAL_HW` 旗標 |
| 雙顯（iGPU＋無資料獨顯） | ✅ 自動降級到可用池並掛旗標 |
| 真多卡（皆有資料） | ❌ v1 誠實拒絕（UNSUPPORTED_TOPOLOGY） |
| macOS / Linux GPU | ❌ 未實作（T-001 S6；probe 會明說而不是編造） |
| 推論引擎 | llama.cpp 系（含 Ollama）。其他引擎誠實拒絕 |
| 模型架構白名單 | llama / qwen2 / qwen3(_moe) / mistral / gemma3(_text) / gemma4(_text) / qwen3.5(_moe)。白名單外一律回報「尚未支援」（如 deepseek_v4：其三重 K 快取語意目前只有實驗性移植——寧可拒絕也不近似） |

## 方法論

本專案在作者的工作庫中以「規格先行」開發：每條記憶體公式在實作前
先以手算 golden 釘住、每次新增架構都有引用來源的修正案、預測持續
與真實機器的 ground truth 對分（TA/FA/FR/TR）。公開樹包含程式碼與
版本化目錄快照；工作帳本與規格文件不隨公開版發佈。

## 隱私

掃描完全在本機執行、唯讀、不上傳任何資料。目錄快照由你自己建立，
配對時不連網（可重現性優先於即時性）。

## License

MIT — 見 [LICENSE](LICENSE)。目錄快照中各模型的授權各自標示於
快照的 `license` 欄位，下載前請自行確認。
