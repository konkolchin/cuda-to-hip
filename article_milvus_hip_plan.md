# Plan: one article, three placements (Milvus CUDA → HIP)

Status: first draft in `article_milvus_hip_v1.md`. Maxim section is still a placeholder.  
Audience of this file: Konstantin + manager (can be forwarded).  
Canonical text language: **English** (AMD blog / Medium / LinkedIn). A Russian LinkedIn cut can be made later from the same outline — not a second paper.

---

## 0. Reply to the manager (short)

Story frame is now the **compromise** (agreed): not a fake startup, not a dry “customer PoC” either.

1. I started on the machine I actually had: **Ryzen 7 7700X + Radeon RX 7900 XTX** (`Deck/Slide12.jpg`) — a consumer, low-budget workstation. No price ratio in the article; readers will get it. Milvus GPU search did not run on consumer Radeon (CUDA/cuVS; hipVS 0.1.0 is Instinct-only — `Deck/Slide2.jpg`).
2. The port ran on that AMD box first. Results looked real. Colleagues pushed. I **found funding for the NVIDIA box specifically to benchmark, compare, and publish** (`Deck/Slide4–6.jpg` — dashboard plots).
3. **Why Milvus** — yes, we can say it plainly: it is the most popular open-source RAG / ANN vector database. That answers «потому что других вариантов не было?» better than a tour of alternatives. See §3.1.
4. **One paper.** Medium and the AMD blog get the same body. LinkedIn is a cut. **Public numbers and plots come from the manager deck / nightly dashboard**, not from older one-shot harness tables when they disagree.

Open item: **Maxim** must supply the agent-system description and architecture diagram. Until then that section is a placeholder.

Confirm before publish: exact **Intel CPU** on the 4080 box (Slide 12 says “16 logical CPUs, model not recorded”). The NVIDIA GPU is **RTX 4080 16 GB**, not “4080 XTX.”

---

## 1. Encoding / `.cursorrules` (done)

What was wrong: `C:\CUDA-to-Hip\.cursorrules` was saved as **Windows-1251**. Cursor/UTF-8 readers showed `�` / replacement characters on Russian examples. Hex of “инновационный” was `E8 ED ED EE …` (CP1251), not UTF-8 `D0 B8 D0 BD …`.

What we did:

- Rewrote `.cursorrules` as UTF-8, scoped to **article drafts only** (so CMake/HIP chats are not forced into Telegram-voice).
- Replaced the generic “crappy laptop / secret niche” CONTENT CONTEXT with the real Milvus PoC.
- Added `.cursor/rules/milvus-hip-article.mdc` (globs on article markdown) so the voice does not leak into the rest of the repo.

---

## 2. Why one paper (and how the three sites differ)

| Placement | What it is | Length | What changes |
|---|---|---|---|
| **Canonical paper** | The only full text | ~1.8–2.5k words | Full stack, ugly bugs, four plots, leftover work |
| **Medium** | Same canonical text | same | Title + dek, maybe a 2-line intro |
| **AMD blog** | Same canonical text | same | AMD-required byline, legal, “not an official ROCm product claim” disclaimer |
| **LinkedIn** | **Cut**, not a rewrite | ~800–1,000 words or a 400-word hook + link | Hook + one plot + “full write-up here” |

Facts, plot axes, and caveats must match everywhere. If LinkedIn says “parity” and the paper says “0.60× on GIST PQ”, we lose the AMD blog.

---

## 3. Story frame (compromise — agreed)

Not a fictional startup. Not “we always had a matched NVIDIA pair.” Sequence:

1. **Only the AMD box.** Ryzen 7 7700X, 64 GB, RX 7900 XTX 24 GB, Ubuntu 22.04.5, ROCm 7.0.2 (`Deck/Slide12.jpg`). Consumer workstation, low budget — describe the SKUs, do not quote a price multiple. I needed GPU vector search. The GPU on the desk was a Radeon. Stock Milvus GPU search is CUDA. hipVS 0.1.0 is Instinct-only. Consumer Radeon: no GPU search (`Deck/Slide2.jpg`).
2. **Port on that box.** Public tech-preview fork, Milvus v2.5.4, two GPU indexes end-to-end: `GPU_IVF_FLAT` and `GPU_IVF_PQ` (`Deck/Slide3.jpg`).
3. **Then the 4080.** Results were good enough that colleagues pushed and I got funding for the NVIDIA machine — **to compare and to publish**, not because the port required NVIDIA to exist. Nightly CI, same queries, both cards (`Deck/Slide4–6.jpg`).

One-sentence “why vector search” (honest): embeddings in, nearest neighbors out — RAG / semantic search. We are not claiming we shipped a named RAG product.

### 3.1 Why Milvus — reply to «потому что других вариантов не было?»

**Yes. Lead with popularity.** That is the answer the manager was fishing for. Do not open with a list of rivals.

Paste-ready (RU):

> Выбор пал на Milvus потому, что это самая популярная open-source векторная БД для RAG / ANN (в районе 46 тыс. звёзд на GitHub — слайд 2). Не потому, что «других вариантов не было»: FAISS — библиотека, pgvector — расширение Postgres, облачные сервисы нам не подходили. Нужен был именно тот стек, который ставят в RAG, с GPU-индексами. У Milvus GPU-путь на NVIDIA из коробки, на consumer Radeon его не было. Вот эту дырку мы и закрывали.

**Article (EN), two sentences after the hook:**

I picked Milvus because it is the most widely used open-source vector database for RAG / ANN — on the order of 46k GitHub stars (`Deck/Slide2.jpg`). Its GPU search sits on NVIDIA cuVS; on a consumer Radeon that path did not exist. That is the hole we ported.

Wording that survives a fact-check:

- Prefer **“the most popular open-source vector database for RAG”** (Deck already says “about 46,000 GitHub stars”).
- Avoid **“the most popular database on earth”** and avoid **“there were no other options.”**
- FAISS is not a database; Pinecone is not open-source. We do not need to lecture about them unless a reviewer asks.

---

## 4. Article structure (canonical)

Working title (not final):  
**Milvus GPU search on a Radeon: a CUDA-to-HIP port that returns the same neighbors**

Dek (deck-aligned):  
Ryzen + 7900 XTX on the desk (consumer box); two GPU indexes ported; then a funded RTX 4080 so we could publish a fair nightly compare. Uncompressed search: Radeon ahead. Compressed: it depends on nprobe.

### 1. Hook (screen, not slogan)

Open on the desk, not on a strategy slide: Ryzen 7 7700X + RX 7900 XTX — the low-budget workstation I already had. I wanted GPU vector search in the RAG stack people actually use. That is Milvus. Its GPU path is CUDA. hipVS 0.1.0 does not cover consumer Radeon. Empty/wrong neighbors before the IVF packer fix (`kIndexGroupSize` / warp 32 vs 64) is the ugly beat if we need another paragraph.

### 2. The job

- GPU ANN in Milvus on the **Ryzen + RX 7900 XTX** already in the room (`Deck/Slide12.jpg`).
- Why not CPU-only: the gap is GPU search on a card people already bought (`Deck/Slide2.jpg`).
- NVIDIA **RTX 4080 16 GB** came **later**, funded for benchmarking and this publication — same dataset, same queries, nightly CI (`Deck/Slide4.jpg`).
- Config table in the article = Slide 12 (fill Intel CPU SKU).

### 3. Why this stack (short, honest — after §3.1)

```
pymilvus / VectorDBBench
        → Milvus server
            → Knowhere (CPU + GPU indexes)
                → GPU_IVF_FLAT / GPU_IVF_PQ / GPU_CAGRA
                    → NVIDIA: RAFT / cuVS
                    → AMD:    hipRAFT / hipVS   (ROCm-DS + our gfx1100 patches)
```

Milvus does not call cuVS from Python. The port is Knowhere + the GPU library underneath, then a **source build** (stock AMD Docker does not give you this path).

### 4. How we worked — layers (the actual method)

Do not present it as one heroic rewrite.

| Layer | Focus | Pass gate |
|---|---|---|
| 0 | ROCm + CPU Milvus | `gfx1100` visible; Standalone healthy |
| 1 | hipRAFT / hipVS | Build; neighbors tests |
| 1.5 | IVF packer on RDNA3 | Float IVF gtests 98/98 after `kIndexGroupSize=32` (warp32) |
| 2 | Knowhere + HIP | Catch2 GPU L2 search green |
| 3 | Milvus source | Sealed `GPU_IVF_*` build/load on HIP |
| 4 | Fair benches | SIFT/GIST recall match vs CUDA; QPS plots |

### 5. Agent-assisted porting *(placeholder — Maxim)*

What we can say **now** without Maxim:

- Work was done in **Cursor** with **Grok** (and other agents on the team).
- Slide claim already used internally: ~**2 weeks** calendar vs **6–8 weeks** estimated without this assistance — **reconfirm before publishing**.
- Humans owned design, measurement, and “do we ship this number.”
- At least one optimization was **reverted** (PQ score-loop micro-opt: no QPS win). That sentence is gold; it sounds like a person.

What we **must not** invent: a multi-agent architecture diagram, tool names, or “the agent system designed the port.”

**Ask Maxim for:**

- 1 diagram (boxes: who proposes patches / who runs builds / who gates numbers)
- What is human-only (claims, plot captions, “parity” language)
- What is OK to name publicly (Cursor, Grok, internal wrapper?)
- Whether AMD blog can mention third-party agents at all

Until that lands, section 5 in the draft is 4–6 sentences + `[DIAGRAM FROM MAXIM]`.

### 6. What actually got ported (algorithms)

Public claim matches `Deck/Slide3.jpg`: **two** GPU index types, one Knowhere fork, one test suite on both backends.

| Index | Public article | Lab note |
|---|---|---|
| `GPU_IVF_FLAT` | Done, dashboard GIST-1M | Uncompressed; Radeon ahead every nprobe |
| `GPU_IVF_PQ` | Done, dashboard GIST-1M | Compressed; AMD then NVIDIA as nprobe rises |
| `GPU_CAGRA` | **Next step** (Slide 10) | Unfiltered lab path exists; do not headline as shipped |
| CPU indexes | Already worked | Control / not the story |

Ugly-but-true engineering (one subsection, not a dump):

- RDNA3 IVF packer / `USE_WARPSIZE_32`
- hipVS launch heuristic: AMD-default `blockDim=512` (~+21–26% library PQ QPS on the same GPU)
- CMake / host-cxx offload flags / source build vs NVIDIA Docker

### 7. How we measured (dashboard — use the deck)

The dashboard **exists**. Slide 7: live view, nightly CI on two dedicated machines, checksummed artifacts. Slides 5–6 **are** dashboard plots. Use those as the public figures.

Publication protocol (from `Deck/Slide4.jpg`):

- Same dataset, queries, and Milvus build lineage on both cards
- Nightly CI: the same query sweep **1,000 times** per card
- Report **median** QPS; count only sweeps with **correct** results (exact-kNN gate)
- Whiskers = **±1σ** across those 1,000 sweeps
- Dataset: **GIST-1M** (960-d, fits in VRAM on both: 24 GB vs 16 GB)
- Pair: RX 7900 XTX vs RTX 4080 (same-generation consumer cards, Steam-popular; released four weeks apart in late 2022)

**When older harness one-shot plots disagree with the deck, the deck wins** for this article. The `.tex` SIFT runs and library-only microbenches can be a footnote, not the headline charts.

Fair-compare box (Slide 4 + 12 + 13):

- Not identical silicon; 24 GB vs 16 GB VRAM; GIST-1M (~3.8 GB) fits both
- This bench does **not** test collections larger than VRAM
- Recall/gate = port correctness; QPS = hardware+stack speed
- Kernels were written for NVIDIA and run through HIP; Radeon-specific tuning of compressed search is still ahead

### 8. Outcome — did the original job get done?

**Yes.** GPU search runs on the consumer Radeon. Two index types, end to end. Neighbors pass the same correctness gate as NVIDIA. We then bought/funded the 4080 so those numbers could be published.

Public QPS (dashboard / `Deck/Slide5–6`, GIST-1M, median of 1,000 sweeps):

| Index | Pattern | Headline number |
|---|---|---|
| `GPU_IVF_FLAT` (uncompressed) | Radeon ahead at **every** nprobe | **+26%** AMD at nprobe 8 (largest lead) |
| `GPU_IVF_PQ` (compressed) | AMD leads to nprobe 32, NVIDIA after | **+36%** at nprobe 8; **−22%** at nprobe 128 (`1.36× → 0.78×`) |

Tell the mixed story (`Deck/Slide13.jpg`): uncompressed, Radeon is already faster with **no Radeon-specific tuning**. Compressed high-nprobe is the tuning target (close the remaining ~22%).

Do **not** lead with older SIFT “stack-limited ~1×” tables if the deck’s GIST dashboard is what the manager will stand behind.

### 9. What is left / next steps

Align with `Deck/Slide10.jpg`:

- More GPU index types, **starting with CAGRA** (lab: unfiltered CAGRA exists; do not headline it as done)
- Tune compressed search at high nprobe on Radeon (close the 22%)
- Larger-than-VRAM collections (explicitly not in this bench)
- Instinct later — same software line, different SKU
- Agent-loop write-up only after Maxim

Slides 8–9 (205-library gap matrix) are **sales-deck material**. Out of scope for this paper unless the manager insists on one “what’s next” sentence.

### 10. Close

Repos (reproducible, not a dump of slogans):

- https://github.com/konkolchin/milvus-hip (v2.5.4)
- https://github.com/konkolchin/knowhere-hip (2.5)
- https://github.com/konkolchin/ann-harness-amd
- https://github.com/konkolchin/hipVS (+ ROCm-DS)

One dry sentence: we needed GPU vector search on a Radeon; the indexes build, the neighbors match, the QPS depends on the recipe.

---

## 5. Data we already have vs holes

### In hand (use these, do not regenerate from memory)

| Item | Where |
|---|---|
| **Public plots (dashboard)** | `Deck/Slide5.jpg` FLAT, `Deck/Slide6.jpg` PQ |
| Headline % | `Deck/Slide4.jpg` (+26% FLAT, +36% PQ at nprobe 8, −22% PQ at 128) |
| How to read / mixed verdict | `Deck/Slide13.jpg` |
| Hardware table | `Deck/Slide12.jpg` |
| Gap (consumer Radeon vs Instinct hipVS) | `Deck/Slide2.jpg` |
| What shipped | `Deck/Slide3.jpg` (2 indexes, 1 engine fork, 1 test suite × 2 backends) |
| Dashboard / CI / evidence pack | `Deck/Slide7.jpg` |
| Layer gates, packer, library microbench | `ann_harness_repo/docs/porting_milvus_gpu_to_amd.tex` (supporting, not headline) |
| CAGRA restriction | `_tmp_AMD_HIP_PORT.md` — next-step, not Slide 3 “done” |
| Agent calendar claim (2 vs 6–8 weeks) | `docs/porting_milvus_amd_slides.tex` — **reconfirm** |

Deck map for the writer (13 slides):

| Slide | Use in article? |
|---|---|
| 1 Title | Working title source |
| 2 Gap | Yes — hypothesis + Instinct vs Radeon split |
| 3 What we did | Yes — two indexes |
| 4–6 Bench + plots | **Yes — canonical figures** |
| 7 Harness / dashboard | Yes — measurement section |
| 8–9 Library gap matrix | No (other paper / sales) |
| 10 Close / next | Yes — leftover + CAGRA |
| 11 Glossary | Sidebar / footnotes |
| 12 Config | Yes — hardware table |
| 13 How to read results | Yes — mixed QPS story |

### Must collect before writing

| Item | Owner | Blocker? |
|---|---|---|
| Agent architecture + what we may name | **Maxim** | Yes for that section |
| AMD blog: can we name Cursor/Grok/Luxoft? | Manager + AMD comms | Yes for AMD placement |
| Byline (Konstantin only vs team) | Manager | |
| **Intel CPU SKU** on the 4080 box | Konstantin | Slide 12 blank |
| Export of Slide 5–6 plots as clean PNG (not whole DXC slide) | Whoever owns the dashboard | Nice for Medium/AMD |
| Legal: “not official AMD product / not full parity” | AMD blog editor | For AMD only |
| Reconfirm 2-week vs 6–8-week line | Konstantin + Maxim | Do not publish if shaky |

---

## 6. Voice (so it does not read as generated)

Aligned with the fixed `.cursorrules`, but **English** for the canonical paper:

- Specific nouns: sealed index, nprobe, warp32, empty neighbors, `blockDim=512`.
- One reverted optimization.
- Short sentences next to long ones.
- No “journey”, “unlock”, “seamless”, “game-changer”.
- First person: “I” for sitting at the 7900 box; “we” for the PoC.

LinkedIn cut: same voice, fewer layers. If only one plot: **Slide 6 (PQ)** — the lines cross; it does not look like a press release. If two: Slide 5 + 6.

---

## 7. Proposed writing sequence

1. Manager agrees the **compromise frame** (AMD box first, 4080 funded later for the paper) and **one-paper** rule.  
2. Maxim sends agent paragraph + diagram (or “omit architecture, mention Cursor/Grok only”).  
3. Freeze **Slide 5 + Slide 6** (+ Slide 4 headlines) as the only public QPS figures.  
4. Fill Intel CPU SKU (optional). No price multiple.  
5. Draft canonical English article from this outline (human pass to kill AI cadence).  
6. AMD disclaimer + byline pass.  
7. LinkedIn cut from the frozen text.  
8. Optional Russian LinkedIn abridgement — same numbers.

---

## 8. Open questions for Konstantin / manager / Maxim

1. Byline / employer: DXC on every deck slide — name Luxoft/DXC in the article, or engineer-only first person?
2. Maxim: one diagram or a short paragraph only?
3. May we publish the 2 vs 6–8 weeks estimate?
4. LinkedIn: full cut on the platform, or hook + Medium/AMD URL?
5. Library-only PQ plot (`.tex` bt512): footnote or omit? Easy to mix with dashboard QPS.
6. Russian version: LinkedIn only, or not at all for v1?
7. Confirm Intel CPU model if we name it; otherwise “the NVIDIA workstation” is enough.
8. Manager OK with the Why-Milvus paragraph (§3.1)?
