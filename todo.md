# μ²Tokenizer 复现 & Qwen 移植 TODO

> **状态（2026-05-27）**：smoke 端到端跑通（SFT / DPO 数据构造 / DPO 训练 / 评估）。
> 所有 21 个起步坑已固化在 `script/slurm/HOTFIXES.md`。
> 接下来要做的是从 smoke 放大到全量训练 + 真实评估。

## 🎯 总体目标

1. **理解** μ²Tokenizer 的模型设计与训练思路（先读懂再动手）。
2. **复现** μ²Tokenizer 项目的完整流程：训练 (SFT + DPO) → 推理 → 评估。
3. **移植**：将本项目适配到 **Qwen3.5-9B**（<10B 约束下的最终选择）。
   - 用户原意：Qwen3.6 优先，<10B；但 Qwen3.6 开源最小是 27B dense（2026-04 发布），无 <10B 版本。
   - 回退方案：**Qwen3.5-9B (dense, Reasoning)**，2026-03 发布；备选 Qwen3.5-4B。
   - 复现范围：**SFT + DPO 两阶段全做**。

## 🖥️ 运行环境

- **集群**：Cray HPC，SLURM 调度（`sbatch / srun / squeue / sinfo` 都在 `/usr/bin/`）。
- **节点架构**：ARM (NVIDIA Grace) + 4× NVIDIA GH200 GPU / 节点，288 CPU、~460GB RAM。
- **分区**：`workq`（默认）。QOS：`workq_qos` 最长 1 天；`interactive` 最长 8h / ≤4 节点。
- **关键模块**：`cuda/12.6`、`gcc-native/14.2`、`cray-python/3.11.7`、`PrgEnv-gnu`。
- **现有环境**：
  - `~/miniconda3/envs/{torchcodec,vllm}` —— vllm 可复用做 GREEN/数据合成的 server
  - `~/conda_vllm.tar.gz` —— 打包的 vllm 环境
- ⚠️ **ARM 平台注意**：PyTorch / monai / flash-attn 等需 aarch64 + CUDA 12 wheel；
  M3D-CLIP 预训练 ViT 与现成镜像可能需手动编译。建议建一个干净 conda env（如 `u2t`）。

> 所有训练/评估任务都必须经 `sbatch` 提交，**不要**直接在登录节点跑 GPU。

---

## 📋 项目结构速览

- `src/train/train_stage1.py` — Stage 1: SFT（已支持 llama / phi3 / qwen3）
- `src/train/train_stage2.py` — Stage 2: DPO
- `src/train/sft_u2Trainer.py`、`src/train/dpo_u2trainer.py` — 自定义 Trainer
- `src/model/language_model/` — `u2llama.py` / `u2phi3.py` / `u2qwen3.py`
- `src/model/multimodal_encoder/` — 3D ViT（M3D-CLIP）
- `src/model/multimodal_projector/` — SPP 投影层
- `src/model/u2tokenizer/` — μ²Tokenizer 核心模块
- `src/dataset/` — CT-RATE / AMOS-MM / Fused 数据集加载器
- `src/preprocess/` — 数据合成、JSONL 处理、vLLM 启动脚本
- `script/qwen3-1.7B-ft.sh` — Stage1 SFT 示例脚本
- `script/qwen3-1.7B-dpo.sh` — （当前为空，需要补全）
- `script/evaluation/ct_rate.py` — CT-RATE 多 GPU 评估（BLEU/ROUGE/BERT/METEOR/GREEN）
- `config/accelerate_config*.yaml` `config/ds_config*.json` — 分布式训练配置
- `config/project.json.template` — OpenAI / vLLM 服务器配置模板
- `demo.py` — 推理入口（README Quickstart 即此脚本）

---

## 🗺️ 阶段 -1：读懂模型与训练设计（先看代码 + 论文）

> 目标：能口头讲清「视觉特征怎么进 LLM、μ²Tokenizer 做了什么、两阶段训练分别优化谁」。

- [ ] -1.1 通读论文 `assets/` 配图 + arXiv (2507.00316)，记录：
  - μ² Tokenizer 的 **multi-scale / multi-modal** 是怎么定义的
  - 与 LLaVA 风格 projector 的差别（差分时序 / RPE / DMTP）
  - GREEN 指标如何驱动 DPO
- [ ] -1.2 自顶向下读代码：
  1. `src/model/multimodal_encoder/` — 3D ViT 是怎么对 CT 体素切 patch 的（参考 `image_size=(256,256,32)` `patch_size=(4,16,16)`）
  2. `src/model/multimodal_projector/` — SPP 投影 `proj_out_num` 的语义
  3. `src/model/u2tokenizer/` — 核心：multi-scale attention、`enable_rpe / enable_diffts / enable_dmtp` 三个开关分别加什么模块
  4. `src/model/u2_arch.py` — 看 `u2MetaModel` / `u2MetaForCausalLM` 如何把图像特征替换 `<im_patch>` 占位 token
  5. `src/model/language_model/u2qwen3.py` — Qwen3 集成方式（对照 llama / phi3 找差异）
- [ ] -1.3 数据流：从 `src/dataset/fused_dataset.py` 看 (image_path, question, answer, thinking) → tensor
- [ ] -1.4 训练阶段拆解：
  - Stage1 (`train_stage1.py` + `sft_u2Trainer.py`)：监督 SFT，冻结策略由 `--freeze_*` 决定
  - Stage2 (`train_stage2.py` + `dpo_u2trainer.py`)：DPO，需要 chosen/rejected 报告对
- [ ] -1.5 写一份 `docs/architecture.md`（≤2 页），用框图 + 公式总结自己的理解，作为复现前的"答辩"

## 🗺️ 阶段 0：环境准备 & SLURM 脚手架

- [x] 0.6 `script/slurm/_env.sh`（共用 loader：剥离 uv .venv → 加载 cuda/12.6 → activate conda u2t → 设缓存到 /lus）
- [x] 0.7 SLURM 模板：
  - `setup_env.sbatch` —— 建 `u2t` conda env + 装 aarch64 cu126 torch + requirements-aarch64.txt
  - `download_assets.sbatch` —— Qwen3-1.7B-Instruct + M3D-CLIP ViT
  - `download_data.sbatch` —— CT-RATE-Thinking + CT-RATE 验证集（`FULL=1` 拉完整训练集）
  - `train_stage1.sbatch` —— Stage1 SFT，参数走环境变量
  - `train_stage2.sbatch` —— DPO
  - `eval_ct_rate.sbatch` —— 多 GPU 评估
  - `infer_demo.sbatch` —— demo.py 单图跑通
  - `vllm_green_server.sbatch` —— 本地 GREEN 评分服务
- [ ] 0.1 ⏳ `setup_env.sbatch` 当前在跑（jobid 4812649），等 torch + deps 装完
- [ ] 0.3 拷贝 `config/project.json.template` → `config/project.json` 并填 API key（用于 GREEN/数据合成）
- [ ] 0.8 验证：跑 setup 输出的 `torch.cuda.is_available()` = True、`device_count` = 1（job 申请 1 卡）

## 🗺️ 阶段 1：数据准备

- [ ] 1.1 下载 CT-RATE 原始 CT 体素 + 报告 (HuggingFace `ibrahimhamamci/CT-RATE`)
- [ ] 1.2 下载 CT-RATE-Thinking VQA + 思维链 (HuggingFace `AlpachinoNLP/CT-RATE-Thinking`)
- [ ] 1.3 （可选）下载 AMOS-MM 数据作为验证集
- [ ] 1.4 下载预训练 ViT 权重：M3D-CLIP `pretrained_ViT.bin`
- [ ] 1.5 下载基座 LLM 权重：`Qwen/Qwen3-1.7B-Instruct`（或 4B / Llama-3.2-1B）
- [ ] 1.6 整理目录结构：
  ```
  $PROJECT_PATH/
    pretrained_models/{Qwen3-1.7B-Instruct, M3D-CLIP/pretrained_ViT.bin}
    datasets/Fused_Dataset/{train,val}/*.jsonl
    datasets/CT-RATE/{train,valid}/*.nii.gz
  ```
- [ ] 1.7 用 `src/preprocess/json2jsonl.py` / `merge_jsonl.py` 整合成训练 JSONL
- [ ] 1.8 抽样校验数据集字段（image path、question、answer、thinking）

## 🗺️ 阶段 2：Stage 1 SFT 训练

- [ ] 2.1 修改 `script/qwen3-1.7B-ft.sh`：
  - `PROJECT_PATH` 改成本机绝对路径
  - 选定 `--model_name_or_path` 与 `--model_type qwen3`
  - 调整 batch / grad-acc / epoch / lr 以匹配硬件
- [ ] 2.2 试跑 smoke test：`max_steps=10`、单 GPU、小数据集
- [ ] 2.3 正式启动：`bash script/qwen3-1.7B-ft.sh`
- [ ] 2.4 监控 loss / eval 指标（W&B / tensorboard）
- [ ] 2.5 产物：`checkpoint/ct_rate_mu2@...` 用于 Stage 2

## 🗺️ 阶段 3：Stage 2 DPO 训练

- [ ] 3.1 **构造偏好数据**（自建）：
  - 3.1.a 用 Stage1 ckpt 对训练子集每条样本采样 N 次（temperature>0），落盘候选
  - 3.1.b 用 GREEN（OpenAI 或本地 vLLM）对每个候选打分
  - 3.1.c 按分数选 top/bottom 组成 (chosen, rejected) 对，导出 JSONL
  - 3.1.d 写 SLURM 脚本 `script/slurm/build_dpo_pairs.sbatch`
- [ ] 3.2 补全 `script/qwen3-1.7B-dpo.sh`（当前为空）+ `script/slurm/train_stage2.sbatch`，参考 `train_stage2.py` 的 ModelArguments
- [ ] 3.3 启动 DPO 训练，监控 reward / KL / chosen-rejected margin
- [ ] 3.4 产物：DPO 后的 checkpoint，进入阶段 4/5

## 🗺️ 阶段 4：推理验证

- [ ] 4.1 用 `demo.py` 加载训练好的 ckpt + 一张 `.nii.gz`，跑通端到端
- [ ] 4.2 验证 `<im_patch>` token 数量 / `proj_out_num` 配置一致
- [ ] 4.3 对比 HuggingFace 公开 ckpt 输出，排查偏差

## 🗺️ 阶段 5：评估

- [ ] 5.1 准备 `valid_labels.csv`（已自带），确认与 valid 数据集对齐
- [ ] 5.2 跑快速指标：`python script/evaluation/ct_rate.py --metrics bleu rouge --max_samples 10`
- [ ] 5.3 跑 BERTScore / METEOR
- [ ] 5.4 跑 GREEN：需要 `green_score` 模块 + OpenAI/vLLM endpoint（见 `green_score/Dockerfile.green-scorer`）
- [ ] 5.5 全量评估 + 与论文表对照

## 🗺️ 阶段 6：移植到新 Qwen（待用户确认具体型号）

> ⚠️ 用户说的 "Qwen 3.5 / 3.6" 目前无官方对应；候选：Qwen2.5-{1.5B,3B,7B}-Instruct、Qwen3 后续版本。**先与用户确认**。

- [ ] 6.1 调研目标 Qwen 的 tokenizer 特殊 token（`<|im_start|>` 等）、chat template、`</think>` (151668) 是否仍有
- [ ] 6.2 参考 `src/model/language_model/u2qwen3.py` 新建 `u2qwenX.py`：
  - 继承新版 `QwenXForCausalLM`
  - 复用 `u2_arch.py` 中的 `u2MetaModel` / `u2MetaForCausalLM` 混入
  - 注意 `prepare_inputs_for_generation` 的签名变化（cache_position 等）
- [ ] 6.3 在 `src/model/language_model/__init__.py` 注册新类
- [ ] 6.4 在 `train_stage1.py` 的 model_type 分支加 `qwenX` 路径
- [ ] 6.5 调整 tokenizer 添加 `<im_patch>` 等特殊 token 的逻辑
- [ ] 6.6 写一份新 shell 脚本 `script/qwenX-ft.sh`
- [ ] 6.7 小规模 smoke test → 全量训练 → 评估

---

## ✅ 已确认决策

- **目标基座**：Qwen3.5-9B（首选）/ Qwen3.5-4B（备选）。Qwen3.6 无 <10B 版本，放弃。
- **复现范围**：SFT + DPO 都要。
- **GREEN 评估**：OpenAI API **和** 本地 vLLM **两条路径都要支持**，通过 `config/project.json` 的 `openai_server` 切换 `base_url`（OpenAI 官方 vs vLLM endpoint）。本地路径写一份 `script/slurm/vllm_green_server.sbatch` 拉起 `StanfordAIMI/GREEN-radllama2-7b` 服务。

- **DPO 偏好数据**：自己重新构造（论文同款流程）。
  - 流程：Stage1 模型对验证/训练样本多次采样 → GREEN 打分 → 高分作 chosen / 低分作 rejected → 落盘 JSONL → 喂给 `train_stage2.py`。
  - 工具复用 `src/preprocess/start_vllm_server.py` 起采样服务，`green_score/` 评分。
