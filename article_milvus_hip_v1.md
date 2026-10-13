# Getting Milvus nearest-neighbour lookup to run on the 7900 XTX

*First draft. One piece of text, three places: Medium, LinkedIn, AMD blog if they take it.*

<!-- Before publish: crop Deck/Slide5.jpg and Slide6.jpg to the plots (full DXC slides today). Machine list is Deck/Slide12.jpg. Dashboard plots win over old one-shot harness tables. -->

---

I needed a RAG database on the box I already had. I wanted Milvus: it is the most popular open-source one for that (~46k GitHub stars). The GPU in that box was a consumer Radeon. Milvus GPU indexes run on NVIDIA cards. Not on this one.

What RAG retrieval actually does: embed the documents, embed the question, then find which stored vectors sit nearest to the question. ANN. Approximate nearest neighbours. I wanted that lookup on the GPU, inside Milvus, not as a side library.

CPU Milvus was fine, by the way. Same box, same collection, nearest neighbours came back, they were just slower, and I already knew the CPU path worked. The NVIDIA GPU docker images are NVIDIA, obviously. Under the hood the search engine is Knowhere, and Knowhere talks to cuVS / RAFT. AMD's counterpart is hipVS; version 0.1.0 is for Instinct in the datacentre. The Radeon was in the case. Milvus had no GPU indexes I could run on it. I started with the CUDA search code and tried to get it through HIP on this machine.

We ended up with two index types working all the way through on the 7900 XTX. `GPU_IVF_FLAT` keeps the vectors uncompressed inside IVF clusters. `GPU_IVF_PQ` compresses them with product quantization, which saves VRAM and costs a bit of recall. Line is Milvus v2.5.4, Knowhere rebuilt for HIP, hipVS underneath. The same tests run on the NVIDIA build and on the AMD one. If the neighbour IDs diverge, we don't get to talk about speed.

I didn't have an RTX 4080 at the beginning. Didn't need one to make search work. After it did work, people here said I should measure it against NVIDIA if we were going to write anything down, and I got funding for a 4080 (16 GB) for that — same queries, both cards, numbers we could put in public. The port did not depend on that second box. A blog post without it would have been hand-waving.

## What actually moved

Python never talks to cuVS. pymilvus hits the Milvus server, the server hits Knowhere, Knowhere owns the GPU indexes:

```
pymilvus
    -> Milvus server
        -> Knowhere
            -> GPU_IVF_FLAT / GPU_IVF_PQ
                -> NVIDIA: RAFT / cuVS
                -> AMD:    hipRAFT / hipVS   (ROCm-DS + gfx1100 patches)
```

Pulling `milvusdb/milvus:*-gpu` on a Radeon is a waste of an afternoon. Milvus Lite / CPU also doesn't go near this stack. GPU on AMD meant compiling from source. I still trip over that when I switch machines.

We didn't do a weekend find-and-replace of cuVS for hipVS. It went in layers. I didn't start the next one until the previous gate was green.

0. ROCm plus CPU Milvus: `gfx1100` shows up, Standalone answers.
1. hipRAFT / hipVS builds, neighbour tests pass.
1.5. IVF packer on RDNA3. This is the warp-32 mess below. After `kIndexGroupSize = 32`, float IVF gtests were 98/98.
2. Knowhere with HIP: GPU L2 search tests green.
3. Milvus from source, sealed `GPU_IVF_*` build / load / search.
4. Fair benches against CUDA, QPS vs nprobe, same correctness gate.

Layer 1.5 still annoys me. NVIDIA warps are 32 threads; a pile of RDNA3 code still assumes 64. The IVF packer grouped vectors the wrong way, and search came back with empty neighbour lists, or with IDs that belonged to someone else. The run itself looked healthy. Recall didn't. Fix was `kIndexGroupSize = 32` plus `USE_WARPSIZE_32`. After that the float IVF gtests were 98/98.

We also poked at a “smarter” PQ score loop. QPS didn't move. I reverted it. What did stick, on the library side only, is launching hipVS PQ search with `blockDim = 512` by default on this GPU — stock hipVS picks block sizes with an NVIDIA occupancy heuristic, and gfx1100 doesn't love it. That's a library microbench. The Milvus plots below are a different measurement. If you overlay them you will confuse yourself.

`GPU_CAGRA` can search the whole index on this GPU in the lab. If you need to skip rows — a filter expression, deletes, TTL — don't use it here yet. That is a later piece of work, not this article.

## Two different ports, not one story

There were two pieces of work. They look similar from far away — HIP, Knowhere, benches on a Radeon — and they are not the same job.

The first is mine: I sit in the code with an assistant. The second is Maksim's: a closed agent pipeline that was supposed to drive the port and the benches with almost no one typing C++. His tree is not on public GitHub. What we have is the workflow picture and what he told me. The next subsection is a draft for him to correct.

### 1. AI assistance (this article's main line)

I work in Cursor, mostly with Grok. Other people on the team use other agents. They are useful for grepping, for first-cut patches, and for the kind of CMake error that makes you want to go home. I still run the gate, and I still decide if a number is allowed to leave the lab. The layers, the packer fix, the plots below — that line is this.

### 2. Closed agent pipeline (Maksim) — draft, please correct

*Maksim: written from our chat plus `milvus_pipeline_architecture_v3.png`. Cross out anything that is wrong. We will not invent a public repo.*

Maksim did not patch Milvus by hand. He built a loop that can be pointed at the next CUDA-only library later. For this project the loop was told to adapt hipVS and hipRAFT, get Knowhere to build, hang tests and checks on it, and review whether a gfx1100 run was allowed to start. It was not told to rewrite NVIDIA kernels into RDNA3-native ones. HIP on consumer AMD, yes. Architecture-specific kernel work, no. That is a separate job. If someone later asks the same pipeline for a full kernel rewrite, he thinks it should cope. We have not shown that here.

Proof, in his words: benches on his dashboard, which has to be re-run because the board died, and a private branch that holds the pipeline. Not a GitHub URL we can paste.

The picture:

![Milvus to AMD RDNA agentic porting pipeline, v3](Deck/milvus_pipeline_architecture_v3.png)

Grey boxes are ordinary build/data steps. Orange is the agent. Green is a gate. Purple is a human.

It starts from upstream hipVS / hipRAFT (ROCm-DS 25.10, Instinct-tested). A hermetic preflight pins ROCm and CMake, hides the iGPU, and asserts `gfx1100`, warp size, `kIndexGroupSize`. Then the agent loop: wave64 toward wave32, masks and reductions, runtime warpSize, indexes in order BRUTE_FORCE → IVF_FLAT → IVF_PQ → CAGRA. A CUDA golden (per-kernel diff) sits on the side for debug. hipVS gtests are bucketed: real port fail, harness, env, or a known gap. New fails outside a frozen manifest stop the run. A human is allowed to add a gap to that list. Knowhere is a build-system port (Conan, shims, dual CUDA/HIP flags), then the GPU library swap, cuVS for hipVS, then Milvus itself on gfx1100.

Recall is gated against exact-kNN, including a peer-GPU check. After that a *read-only* agent looks at the diff: instrumentation is fine, weakening the test is not. A person looks again. CI on an internal GHES fork, self-hosted gfx1100 runner, clean rebuild. Benchmarks are not run by the agent: VectorDBBench, GPU isolated, GIST-1M, nprobe sweep, after the harness has checked itself. A provenance gate wants a sealed `GPU_CUVS_*` (or the HIP equivalent), flush, load, and rocm-smi VRAM/util in a sane window. Metrics: recall, QPS, p99, VRAM. Triage is labelled, a human looks, and the verdict is correctness. Slow QPS is not a fail by itself.

So: I used agents as a pair of extra hands. Maksim tried to close the loop and keep people at review and verdict. Same stack, two methods. His numbers are not the plots in the next section; those are the nightly pair of boxes after the 4080 existed. If his dashboard comes back, we can put his run next to that — after he says this description is not nonsense.

## Two cards, same night job

Once the 4080 was on the floor, both machines got the same dataset, the same queries, the same index settings, every night.

AMD box:

- GPU: Radeon RX 7900 XTX, 24 GB
- CPU: Ryzen 7 7700X, 8 cores / 16 threads
- RAM: 64 GB DDR5
- OS: Ubuntu 22.04.5 LTS
- GPU stack: ROCm 7.0.2

NVIDIA box:

- GPU: GeForce RTX 4080, 16 GB
- CPU: Intel, 16 logical CPUs (exact SKU was not in the lab notes)
- RAM: about 61 GiB as the OS sees it
- OS: Ubuntu 22.04.5 LTS (same as the AMD box)
- GPU stack: CUDA 11.8 for the Milvus build; CUDA 12.8 in a separate cuVS container

They shipped four weeks apart in late 2022, so this is not a 2020 card vs a 2024 card. It is also not the same chip. 24 GB on one side, 16 GB on the other. GIST-1M — a million vectors, 960 dimensions, roughly 3.8 GB — fits in VRAM on both, which is why we used it. If your collection is bigger than the card, these plots are not your plots.

The nightly job repeats the same query sweep a thousand times on each GPU. We throw away a run if it fails the exact-kNN gate (wrong neighbours). What's left: median QPS vs nprobe, whiskers ±1σ over those thousand sweeps. nprobe is how many IVF clusters you scan for a query; crank it up and recall goes up and the GPU does more work. That's why we plot a sweep instead of one heroic bar.

If the IDs are wrong, QPS is a video-game score. We look at correctness first.

## Uncompressed: the 7900 is faster here

`GPU_IVF_FLAT`, GIST-1M, median of those 1,000 sweeps, one night.

![GPU_IVF_FLAT on GIST-1M: median QPS vs nprobe, RX 7900 XTX vs RTX 4080](Deck/Slide5.jpg)

Red is the 7900 XTX, green is the 4080. Up is more queries per second. As you go right, nprobe grows and each query looks at more clusters.

The 7900 is ahead at every nprobe we ran. Biggest gap in this nightly run was **+26%** at nprobe 8.

I should say this clearly: those kernels were written for NVIDIA and we are running them through HIP. Nobody sat down and rewrote FLAT for RDNA3. In this recipe it is already quicker on the AMD card.

## Compressed: they trade places

`GPU_IVF_PQ`, GIST-1M, same night, same two boxes.

![GPU_IVF_PQ on GIST-1M: median QPS vs nprobe, RX 7900 XTX vs RTX 4080](Deck/Slide6.jpg)

Same axes as the previous figure. Somewhere around nprobe 32 the lines cross. Left of that, AMD. Right of that, NVIDIA. PQ has packed each vector into a short code, so you are in a different kernel than FLAT, which is the whole point of the second plot.

At nprobe 8 the 7900 is about **+36%**. At nprobe 128 it is **−22%**. If you like ratios, that's roughly 1.36× down to 0.78×.

I would tune that high-nprobe PQ gap next, on the Radeon, instead of pretending HIP already ate the 22%. Profiling first, then we see. “Which GPU is faster?” is a bad question anyway. Uncompressed, here, the 7900. Compressed and you asked for a lot of clusters: the 4080, today.

## So did it work?

Yes. I can build two IVF indexes on the 7900 XTX I already owned, load them, and get neighbours that pass the same gate as the 4080. The 4080 is in the room so I can show those two figures instead of asking you to trust me.

We have not matched every NVIDIA feature. We have not run this on Instinct. We have not timed collections that don't fit in VRAM. CAGRA is not a product index in this write-up. This is also not an AMD-blessed Milvus package. It is a tech-preview fork, two consumer cards, a dashboard that still ticks over at night.

What I want next on Milvus is CAGRA (done properly) and that PQ hole at high nprobe. The rest of the machinery — nightly CI, two boxes, hashes on the artifacts, the live QPS view — I would point at the next CUDA-only library rather than rebuild it. Hide the iGPU (`ROCR_VISIBLE_DEVICES=0`). And if you leave Docker `milvus-standalone` running next to the HIP binary, you will spend an hour debugging the wrong `libcuvs`. I did that once.

## Code

Public trees for the AI-assisted line (section 1), not Maksim's closed pipeline:

- Knowhere HIP: [https://github.com/konkolchin/knowhere-hip](https://github.com/konkolchin/knowhere-hip)
- hipVS (gfx1100 patches, packer / launch defaults): [https://github.com/konkolchin/hipVS](https://github.com/konkolchin/hipVS)
- Harness, layer scripts, recipes: [https://github.com/konkolchin/ann-harness-amd](https://github.com/konkolchin/ann-harness-amd)

Maksim's agent pipeline lives in a private branch. We are not linking a GitHub URL for it until he publishes one.

README in the Knowhere/hipVS forks, then `run_milvus_layer4*` in the harness, is the least painful way in.
