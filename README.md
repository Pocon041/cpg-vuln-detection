# CPG 漏洞检测实验工程

本工程从 Joern 导出的 GraphML 构建函数级漏洞检测实验。课程基线比较 AST、CFG、PDG 三种视图，增强模型 `SelectiveFusionCPG` 使用核心 CPG 关系、冻结 CodeBERT 缓存和自适应门控融合。

## 环境

```powershell
conda activate EIT
.\scripts\setup_eit.ps1
```

当前工程按 Python 3.13、PyTorch 2.7、CUDA 12.6 和 RTX 4060 8 GB 验证。由于当前工作目录包含中文，建议直接激活 `EIT` 后执行 `python`，避免在脚本中使用 `conda run` 的 GBK 输出包装层。

## 数据审计

```powershell
python -m cpg_vuln audit
```

审计严格读取 `data/fq_graphml_dataset/metadata/labels.csv`，不会通过目录扫描误纳入 notebook checkpoint。默认记录 `3459_1` 的空 CPG，并排除四个规范化源码相同但标签冲突的样本：

```text
3569_0  6309_1  3835_0  955_1
```

审计结果、清洗后的 manifest 和两套 `8:1:1` split 写入 `artifacts/data/`：

- `course.json`：按标签分层随机切分。
- `strict.json`：按规范化源码哈希分组后切分，阻止重复代码跨集合泄漏。

## 运行顺序

```powershell
python -m cpg_vuln build-topologies
python -m cpg_vuln build-word2vec
python -m cpg_vuln build-codebert-cache
python -m cpg_vuln train-baselines
python -m cpg_vuln train-enhanced
python -m cpg_vuln summarize
python -m cpg_vuln explain
```

也可以运行：

```powershell
.\scripts\run_all.ps1
```

只运行 Word2Vec 路线时使用：

```powershell
.\scripts\run_all.ps1 -SkipCodeBert
```

训练命令默认检测已有 `metrics.json` 并跳过已完成组合。需要重新训练时显式加入 `--force`。也可以通过 `--views`、`--embeddings`、`--splits` 和 `--variants` 缩小实验矩阵。

训练过程中每个 epoch 会显示一条进度条，并打印训练损失、验证集 accuracy、precision、recall、F1、ROC-AUC、PR-AUC、最佳 F1 和剩余 patience。训练结束后会打印最终 validation 和 test 指标汇总。已经启动的训练进程不会动态加载代码修改，需要重新启动后才能看到新的输出。

GraphML 约 3.4 GB，CodeBERT 需要首次下载 `microsoft/codebert-base` 并离线编码唯一节点文本。拓扑和向量缓存均支持断点续跑。训练阶段只读取缓存，不加载 Transformer。

## 缓存恢复

特征构建会显示进度。请先等待 `build-topologies` 完成，再运行 `build-word2vec`。如果拓扑在 Word2Vec 构建后继续增长，旧模型和向量缓存会失效；使用以下命令重建：

```powershell
python -m cpg_vuln build-word2vec --force
```

如果 CodeBERT 首次下载被中断，后续运行提示缺少 `pytorch_model.bin`，需要移除该模型的不完整 Hugging Face 缓存后重新运行：

```powershell
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface\hub\models--microsoft--codebert-base"
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface\hub\.locks\models--microsoft--codebert-base" -ErrorAction SilentlyContinue
python -m cpg_vuln build-codebert-cache
```

或者预先下载好权重后运行 
```powershell
conda activate EIT
Remove-Item Env:HF_ENDPOINT -ErrorAction SilentlyContinue
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
python -m cpg_vuln build-codebert-cache
```

如果官方站点连接较慢，可以使用浏览器或 `curl` 从镜像下载 `pytorch_model.bin`。不要直接设置 `HF_ENDPOINT=https://hf-mirror.com`：当前 `huggingface_hub==0.36.2` 无法从该镜像的权重响应中读取完整元数据。下载完成后可设置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`，强制复用本地缓存。

## 已知环境提示

现有 `EIT` 环境中预先存在 `torchaudio==2.9.0`，但 PyTorch 为 `2.7.0+cu126`。本工程不导入 `torchaudio`，因此不修改这个与漏洞检测无关的既有包。如果后续需要音频任务，应单独将 `torchaudio` 与 PyTorch 版本对齐。

## 工程结构

```text
configs/                  实验配置
scripts/                  EIT 安装与一键运行脚本
src/cpg_vuln/data/        审计、解析、切图、split、Dataset、动态 batch
src/cpg_vuln/features/    Word2Vec 与 CodeBERT 离线缓存
src/cpg_vuln/models/      GCNClassifier 与 SelectiveFusionCPG
src/cpg_vuln/training/    指标、训练循环、实验矩阵
src/cpg_vuln/visualization/ 汇总图表与 Top-K 解释
tests/                    单元测试和小样本 smoke tests
artifacts/                可重建中间产物
outputs/                  checkpoint、预测、指标和图表
```

## 视图定义

| 视图 | 关系 |
|---|---|
| AST | `AST` |
| CFG | `CFG` |
| PDG | `CDG`, `REACHING_DEF` |
| core-CPG | `AST`, `CFG`, `CDG`, `REACHING_DEF` |
| dataflow-CPG | `CFG`, `CDG`, `REACHING_DEF` |

前三类视图用于课程基线。`core-CPG` 用于选择性融合模型，`dataflow-CPG` 用于消融。

## 测试

```powershell
python -m pytest
```

测试使用临时合成 GraphML，不依赖完整数据集，也不会触发 CodeBERT 下载。
