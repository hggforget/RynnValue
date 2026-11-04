# PKU Franka FR3 微调

真机推理服务和 Franka client 的启动步骤见 [README_inference.md](README_inference.md)。

2026-10-17 的双相机 collect-objects 数据集请使用
[独立配置及说明](collect_objects_20261017.md)。它使用绝对末端目标和 0/1 夹爪动作，
与本文旧单相机数据的动作契约不同。

2026-10-27 至 2026-10-30 合并的 pick-and-place 数据使用专用配置
`pi05_pku_franka_pickplace`。在 `pi-rl` 目录运行：

```bash
GPUS=0 bash scripts/train_sft_pku_franka_pickplace.sh
```

脚本默认读取 `/oss/embodied-ai-data/pku/fr3_real_pickplace_20261027_20261030_merged`，
保留全部 177 条轨迹和 44 个任务的独立指令，使用双相机、14 维实测状态和 7 维绝对动作。
默认每卡 batch 为 16、动作块长 30 帧、训练 20,000 步；启动前自动计算独立归一化统计。
通过 `GPUS=0,1` 选择多卡，`BATCH_SIZE_PER_GPU` 调整每卡 batch，`DRY_RUN=1` 仅打印命令。
覆盖数据路径使用 `REPO_ID`，同时作用于统计和训练；额外训练参数直接追加到脚本后。
续训设置原 `EXP_NAME` 并追加 `--resume`，统计有效时可设置 `SKIP_NORM=1`。
checkpoint 保存在 `checkpoints/pi05_pku_franka_pickplace/<EXP_NAME>/`。
无需设置 `PYTHON` 或激活环境：脚本会依次检查当前 virtualenv/Conda、项目 `.venv`、PATH，
以及已安装的 `pi-rl` / `pku-pi-rl` Conda 环境和 Conda 环境注册表，自动跳过依赖不全的解释器。
启动时会打印最终选中的 Python 路径；`DRY_RUN=1` 也会探测依赖，但不计算统计或训练。
也可通过 `PYTHON=/训练环境/bin/python` 显式指定，此时检查失败会直接报错。

pick-and-place 与 collect-objects **都做 SFT** 的联合版本使用
`pi05_pku_franka_pickplace_collect_objects`：

```bash
GPUS=0,1 bash scripts/train_sft_pku_franka_pickplace_collect_objects.sh
# 只检查环境和打印命令：
GPUS=0 DRY_RUN=1 bash scripts/train_sft_pku_franka_pickplace_collect_objects.sh
```

默认合并以下两路数据，按有效帧数混合采样，每个样本使用相同的 SFT 损失：

- pick-and-place：`/oss/embodied-ai-data/pku/fr3_real_pickplace_20261027_20261030_merged`，
  保留所有轨迹和 44 个任务各自的指令。
- collect-objects：`/oss/embodied-ai-data/pku/fr3_collect_objects_20261017_progress_expert_sup_step005000`，
  沿用已有 collect-objects SFT 的六条排除项 `0, 2, 14, 21, 47, 52`，指令为
  `Place each solid-colored block on the plate that matches its color. Leave all wooden-textured blocks on the table.`

两路任务索引分别解析，collect-objects 的任务 0 不会覆盖 pick-and-place 的任务 0。
均使用双相机、14 维实测状态、7 维绝对动作和 30 帧动作块；不读取 progress/reward 标注。
默认从 `pi05_base` 初始化，全参数训练 20,000 步，每卡 batch 为 16，学习率 `5e-5 → 5e-6`。

通过 `PICKPLACE_REPO_ID` 和 `COLLECT_REPO_ID` 分别覆盖数据路径，统计和训练会使用相同顺序与筛选。
启动时扫描两路所有有效帧，共用一套独立统计，写入
`assets/pi05_pku_franka_pickplace_collect_objects/pku_franka_pickplace_collect_objects/norm_stats.json`；
checkpoint 位于 `checkpoints/pi05_pku_franka_pickplace_collect_objects/<EXP_NAME>/`。
Python 自动发现、`BATCH_SIZE_PER_GPU`、`FSDP_DEVICES`、`WEIGHT_PARAMS_PATH`、`SKIP_NORM=1` 和
`EXP_NAME=已有实验名 ... --resume` 的用法与单数据集 SFT 相同；改动数据路径或筛选后应重新计算统计。
直接调用统计脚本时使用 `--repo-ids <pickplace路径> <collect路径>`，
直接训练时对应 `--data.repo-ids <pickplace路径> <collect路径>`。

三路数据共同做 SFT 使用 `pi05_pku_franka_pickplace_collect_drawer`：

```bash
GPUS=0,1 bash scripts/train_sft_pku_franka_pickplace_collect_drawer.sh
GPUS=0 DRY_RUN=1 bash scripts/train_sft_pku_franka_pickplace_collect_drawer.sh
```

第三路默认是 `/oss/embodied-ai-data/pku/fr3_drawer_20261102_03`；自动发现其中全部 18 个
LeRobot 子数据集，保留 125 条轨迹及开关抽屉、放置物块等任务各自的指令。无需预先合并或解压 `.tar`。
前两路路径和 collect-objects 筛选保持上述两路版本的约定，三路统一使用 30 Hz、双相机、
14 维实测状态、7 维绝对动作和 30 帧动作块，按有效帧均匀采样，共用 SFT 损失。
drawer 的深度、采集时间戳、相对动作以及 progress/reward 字段不作为模型输入。

通过 `DRAWER_ROOT` 覆盖第三路路径（也支持已经合并的单个 LeRobot 数据集）；
`PICKPLACE_REPO_ID`、`COLLECT_REPO_ID` 继续覆盖前两路。统计和训练采用同一套路径及筛选。
共享统计独立写入 `assets/pi05_pku_franka_pickplace_collect_drawer/pku_franka_pickplace_collect_drawer/norm_stats.json`，
checkpoint 位于 `checkpoints/pi05_pku_franka_pickplace_collect_drawer/<EXP_NAME>/`。
其余 GPU、batch、Python 自动发现和续训选项与两路版本一致。

星云入口支持三路版本，预览不提交作业：

```bash
bash submit_to_nebula_pku_franka_pickplace.sh --method cotrain3-sft --dry-run
```

实际提交去掉 `--dry-run`；可通过 `--drawer-root` 覆盖第三路根目录。

超级混训使用独立配置 `pi05_pku_franka_super`，默认包含五路数据：
上述 pick-and-place、collect-objects、`fr3_drawer_20261102_03`，以及新增的
`/oss/embodied-ai-data/pku/fr3_drawer_20261103` 和 `/oss/embodied-ai-data/pku/fr3_plate_20261103`。

```bash
# 不需要设置 Python 或先激活环境，默认使用可见 GPU：
bash scripts/train_sft_pku_franka_super.sh
# 指定 GPU，或只预览启动命令：
GPUS=0,1 bash scripts/train_sft_pku_franka_super.sh
DRY_RUN=1 bash scripts/train_sft_pku_franka_super.sh
# 星云预览（实际提交去掉 --dry-run）：
bash submit_to_nebula_pku_franka_pickplace.sh --method super-sft --dry-run
```

自动发现三份 collection 中的所有 LeRobot 子数据集，按有效帧均匀混合采样。
每个任务保留自己的指令，collect-objects 继续使用其固定指令和六条轨迹排除项；
均使用 30 Hz、双相机、14 维实测状态、7 维绝对动作和 30 帧动作块，全部执行 SFT。
默认从 `pi05_base` 初始化，全参数训练 20,000 步，每卡 batch 为 16；启动前重新计算五路共享统计。
深度归档不参与加载；相对动作和 progress/reward 不作为模型输入或训练标签。

通过 `PICKPLACE_REPO_ID`、`COLLECT_REPO_ID`、`DRAWER_ROOT`、`EXTRA_DRAWER_ROOT`、`PLATE_ROOT`
分别覆盖五路路径，统计与训练保持一致。星云对应参数为 `--repo-id`、`--collect-repo-id`、
`--drawer-root`、`--extra-drawer-root`、`--plate-root`。
直接调用统计脚本时，新增两路使用 `--additional-dataset-roots <新drawer路径> <新plate路径>`；
训练 CLI 对应 `--data.additional-dataset-roots`，也支持传入更多兼容的数据目录。
独立统计写入 `assets/pi05_pku_franka_super/pku_franka_super/norm_stats.json`，
checkpoint 位于 `checkpoints/pi05_pku_franka_super/<EXP_NAME>/`。
续训使用原 `EXP_NAME` 和 `--resume`，已有正确统计时可加 `SKIP_NORM=1`。

多粒度 Prompt 增强使用独立配置 `pi05_pku_franka_pickplace_prompt_aug`：

```bash
# 在 pi-rl 目录运行；自动发现训练 Python 和 GPU。
bash scripts/train_sft_pku_franka_pickplace_prompt_aug.sh
GPUS=0,1 DRY_RUN=1 bash scripts/train_sft_pku_franka_pickplace_prompt_aug.sh
# 完整细粒度 Prompt 对照组：相同数据、划分、采样器、模型和训练预算。
CONFIG_NAME=pi05_pku_franka_pickplace_prompt_baseline bash scripts/train_sft_pku_franka_pickplace_prompt_aug.sh
# 星云 8 卡启动（预览时追加 --dry-run）：
bash submit_to_nebula_pku_franka_pickplace.sh --method prompt-aug-sft --gpus 8
# 星云完整 Prompt 对照组：
bash submit_to_nebula_pku_franka_pickplace.sh --method prompt-baseline-sft --gpus 8
```

默认读取合并的 10 月 27—30 日放盘子数据、`fr3_plate_20261103` 和
`fr3_drawer_20261102_03`，只选择 `fine_task.kind=pick_place` 的盘子/抽屉轨迹。
不混入 collect-objects、开关抽屉和未切分的长交互记录。
当前挂载快照有 293 条盘子、65 条抽屉记录；方案提及的另外 3 条抽屉记录尚未包含其中。
默认排除非空 `note` 的 5 条盘子、9 条抽屉记录，剩余 344 条候选，再划分训练/验证集。
这是原始示教数；在线 Prompt 变体不计为独立示教，也不复制视频或动作。

每个样本先等概率选择一个原始 episode，再在该 episode 内均匀选择帧，并重新采样 Prompt：

- 盘子：按指定属性数量 0/1/2/3，以 20%/30%/30%/20% 采样，同一档的组合等概率，共 8 类。
- 抽屉：只对物体颜色和形状做省略，按 0/1/2 个属性以 20%/60%/20% 采样，共 4 类。
- 完整属性档保留原始完整 Prompt；其余英文模板只删除属性，颜色、形状和目标类型均来自 `fine_task`。
- `rectangular` 与 `flat_rectangular` 分开保留；抽屉轨迹始终描述为放入抽屉。

按 `(source_dataset, source_episode_index)` 和固定 seed 的哈希，在任何 Prompt 增强之前
划分 90%/10% 的训练/验证集；比例是确定性的哈希阈值，数量不保证精确等于 10%。
同一来源 episode 的全部帧和 Prompt 变体只属于一侧；重叠数据根目录或重复来源 episode 会报错。
统计只扫描训练侧所有帧，不包含验证集。验证侧默认使用固定的完整 Prompt，不进行随机增强。
模型仍是双相机、14 维实测状态、7 维绝对动作、30 帧动作块，默认从 `pi05_base` 训练 20,000 步。
当前管线要求这些输入数据共享 30 的标称视频 FPS；这不等于实际控制频率，
本配置不重写采集时间戳，也不对视频、状态或动作额外重采样。

通过 `PICKPLACE_REPO_ID`、`PLATE_ROOT`、`DRAWER_ROOT` 覆盖三路数据位置，
`VALIDATION_FRACTION`、`SPLIT_SEED` 调整 episode 划分。
人工复核后需要纳入带备注记录时，设置 `INCLUDE_NOTED_EPISODES=1`；统计与训练同步使用这些选项。
改变源数据、划分或质量筛选后应重新计算统计，不使用旧的 `SKIP_NORM=1` 结果。
独立统计和可追溯清单位于：

```text
assets/<配置名>/pku_franka_pickplace_prompt_aug/norm_stats.json
assets/<配置名>/pku_franka_pickplace_prompt_aug/prompt_episodes.json
```

清单保留来源数据集、日期、原 episode ID、原始 `fine_task`、完整 Prompt、场景和备注，
包括 `excluded_note` 和 `validation` 记录。缺少 `fine_task` 的记录不会从文字或外观中自动推断属性。
如有额外已挂载的兼容数据集，可直接通过训练 CLI 的 `--data.repo-ids` 提供全部根目录，
计算统计时使用相同的 `--repo-ids`，并保持 `--validation-fraction`、`--split-seed`、
`--include-noted-episodes` 与训练配置一致。

离线构建完整 Prompt 验证 loader（复用训练侧的归一化统计）：

```python
import dataclasses
from openpi.training import config, data_loader

cfg = config.get_config("pi05_pku_franka_pickplace_prompt_aug")
cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, split="validation"))
loader = data_loader.create_data_loader(cfg, shuffle=False, num_batches=1)
```

推理时直接传自由语言，训练数据根目录无需挂载，也不会随机改写用户指令。
实机评测应按完整、各类部分属性和通用指令分组，并按有无干扰物分别统计；
粗指令选择任意满足约束的物体和目标即符合语义，不要求复现示教中的唯一选择。

带奖励标注的 pick-and-place 使用 `pi05_pku_franka_pickplace_iql`：

```bash
GPUS=0,1 bash scripts/train_iql_pku_franka_pickplace.sh
# 只检查环境和打印命令：
GPUS=0 DRY_RUN=1 bash scripts/train_iql_pku_franka_pickplace.sh
# 从上次 IQL checkpoint 恢复策略、优化器、Q/V 和 target Q：
GPUS=0,1 EXP_NAME=已有实验名 SKIP_NORM=1 bash scripts/train_iql_pku_franka_pickplace.sh --resume
```

默认数据为
`/oss/embodied-ai-data/pku/fr3_real_pickplace_20261027_20261030_merged_progress_expert_sup_step025000_on_vlm4b90k`。
数据、相机、状态和绝对动作处理与上述 SFT 一致；保留全部轨迹及各自指令，
Q/V 额外输入 14 维实测状态和 44 维任务 one-hot，critic 只使用 7 个真实动作通道。
策略默认从 `pi05_base` 初始化，Q/V 随机初始化；可设置 `WEIGHT_PARAMS_PATH=/SFT检查点/params`
从已有 SFT 策略开始新的 IQL 实验。

标注 `progress = -remaining_time_s`，每步奖励为 `progress[t+1] - progress[t]`，
动作块奖励为 `sum(gamma**k * reward[t+k])`，Q 的目标为 `R + gamma**30 * mask * V(next)`。
默认 `gamma=0.99`，允许进度回退和负奖励，终止时关闭 bootstrap，padding 不产生奖励；
最后一帧没有后继转移，奖励为零。保留标注的终点预测，不将其强制置零，
也不把 episode 结束当作成功或套用旧 collect-objects 数据的失败编号。

默认训练 20,000 步，每卡 batch 为 16，策略学习率 `3e-5 → 3e-6`，
critic/value 学习率 `3e-4`，expectile 为 `0.8`，优势权重为 `min(exp(Q-V), 100)`。
前 1,000 步同时训练 Q/V 和普通 BC，之后策略采用优势加权 flow-matching 损失。
可追加 `--iql.expectile 0.7`、`--iql.A-scaling 2.0`、`--num-train-steps 10000` 等训练参数。
启动脚本支持与 SFT 相同的 Python 自动发现、`REPO_ID`、`BATCH_SIZE_PER_GPU`、
`FSDP_DEVICES`、`DRY_RUN` 和 `SKIP_NORM` 设置。
IQL 统计单独保存在 `assets/pi05_pku_franka_pickplace_iql/pku_franka_pickplace/norm_stats.json`，
checkpoint 保存在 `checkpoints/pi05_pku_franka_pickplace_iql/<EXP_NAME>/`。

以上 pick-and-place 版本都支持星云提交。在 `pi-rl` 目录运行：

```bash
# 预览源码快照、资源和训练参数，不创建作业：
bash submit_to_nebula_pku_franka_pickplace.sh --method sft --dry-run

# 各自提交一个独立作业：
bash submit_to_nebula_pku_franka_pickplace.sh --method sft
bash submit_to_nebula_pku_franka_pickplace.sh --method iql
bash submit_to_nebula_pku_franka_pickplace.sh --method cotrain-sft

# 资源和额外训练参数：
bash submit_to_nebula_pku_franka_pickplace.sh --method cotrain-sft \
    --gpus 2 --batch-size-per-gpu 16 --num-train-steps 20000 \
    -- --save-interval 1000 --log-interval 100

# 恢复原实验；IQL 同时恢复 Q/V、target Q 和优化器：
bash submit_to_nebula_pku_franka_pickplace.sh --method iql \
    --exp-name 已有实验名 --resume
```

默认使用队列 `damo_eai_rfm_l20z`、镜像 `pytorch271`、单 worker 八卡、每卡 batch 16。
`--gpus` 申请调度资源，worker 按实际 `CUDA_VISIBLE_DEVICES` 自动选择 GPU、FSDP 和全局 batch。
checkpoint 写入 `/oss/embodied-ai-data/pku/nebula_checkpoints/<配置名>/<EXP_NAME>/`，
环境安装日志写入 `/oss/embodied-ai-data/pku/nebula_setup_logs/<EXP_NAME>.log`。
作业默认在 worker 上重新计算归一化；`cotrain-sft` 会扫描两路有效帧，共同计算同一套统计。
已有有效统计时可用 `--skip-norm`，提交前会检查对应配置的统计并随源码打包。

`--repo-id` 覆盖 pick-and-place 数据，IQL 默认自动选择 expert 奖励版本；
`--collect-repo-id` 覆盖联合 SFT 的 collect-objects 数据。
`--weight-params-path`、`--ckpt-base-dir`、`--queue`、`--exp-name` 可覆盖权重、输出位置和调度配置。
默认使用 `--preflight-python /opt/conda/envs/pi-rl/bin/python` 在提交机检查真实训练参数和数据元信息。
提交机没有挂载数据或训练环境时，可显式增加 `--skip-local-preflight`，由 worker 启动时检查；
这也适用于 `--dry-run`，此时仅完成源码打包和配置预览。

认证与环境安装复用 [Lingdong 星云提交说明](../../README.md#lingdong-on-nebula)：
使用 `NEBULA_USER_CONFIG` 和 `NEBULA_OSS_ACCESS_ID/KEY/BUCKET/ENDPOINT`，
或 `--reference-root` 指向已有提交配置；挂载顺序为 `embodied-ai-data,damo-rynnvla-data-zjk`。
源码快照不包含 checkpoints、数据集和凭证，提交及 worker 均清除代理变量。
worker 复用 `setup_nebula.sh` 安装已验证依赖；`--wheelhouse`、`--pip-index-url` 和
`--skip-setup` 的含义与 Lingdong 提交入口一致，W&B 默认离线运行。

以下为旧单相机 `pi05_pku_franka` 配置说明。

数据集：`/home/hdc/datasets/fr3_put_the_cube_onto_the_plate_20261013`，
LeRobot v2.1，5 Hz，1 条轨迹、703 帧。

| 字段 | 内容 | 训练处理 |
| --- | --- | --- |
| `image` | 单相机 RGB，480×640 | 转为 HWC uint8，映射到 `base_0_rgb`，等比例补边至 224×224 |
| `state` | `[x,y,z,roll,pitch,yaw,gripper]` | 保留末端位姿与夹爪宽度，归一化后作为 π₀.₅ 离散状态输入 |
| `actions` | `[dx,dy,dz,droll,dpitch,dyaw,gripper]` | 保留原始增量和夹爪指令，归一化；不再次求差分 |
| `task_index` | `meta/tasks.jsonl` 中的任务索引 | 生成 prompt，忽略 `episodes.jsonl` 中的空 prompt |

夹爪动作是 `-1` 闭合、`+1` 张开，与状态中的米制夹爪宽度含义不同。
训练和推理都保留这个约定。当前任务文本沿用数据集：
`Pick up the object on the table and place it into the container.`

已注册配置：

| 配置 | 微调方式 | batch | horizon | steps | 学习率 |
| --- | --- | --- | --- | --- | --- |
| `pi05_pku_franka` | 全参数 | 8 | 10（2 秒） | 2000 | 5e-5 → 5e-6，warmup 200 |

配置从 `pi05_base` 初始化，模型内部保持 32 维以匹配预训练权重；
模型变换将 7 维 state/actions 补齐，推理输出再裁回 7 维。
配置显式使用 PyAV 解码 AV1 视频，使用其自带的 FFmpeg，避免当前环境中
TorchCodec 缺少系统 FFmpeg 动态库导致读帧失败。
全参数微调需要较多显存，可调整 batch 或使用多卡 FSDP。
目前数据量只适合检查训练链路和单轨迹拟合，2000 steps 是起始设置，需观察过拟合。

在 `pi-rl` 目录使用已经装好项目依赖的 Python 环境运行：

```bash
cd /home/hdc/RynnValue/pi-rl
source .venv/bin/activate

# batch-size=1 扫描全部 703 帧，避免统计脚本丢弃不足一个 batch 的尾帧。
JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python scripts/compute_norm_stats_fast.py \
  --config-name pi05_pku_franka --num-workers 0 --batch-size 1

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  python scripts/train.py pi05_pku_franka \
  --exp-name fr3_cube_plate
```

归一化统计写入 `<数据集目录>/norm_stats.json`。
checkpoint 保存在 `checkpoints/<配置名>/<exp-name>/`，每 500 步保存一次。
首次运行会下载 tokenizer 和 `pi05_base` 权重，也可通过
`--weight-loader.params-path /本地/pi05_base/params` 使用本地权重。
续训使用相同实验名并添加 `--resume`。

更换数据集时，统计和训练命令都需传入新路径，分别使用
`--repo-id /新数据集目录` 和 `--data.repo-id /新数据集目录`。
输入字段与动作语义应保持上表约定。

推理输入使用与数据集相同的原始键：

```python
observation = {
    "image": rgb_image,  # (480, 640, 3), uint8 RGB
    "state": state,      # (7,), [x,y,z,roll,pitch,yaw,gripper_width_m]
    "prompt": "Pick up the object on the table and place it into the container.",
}
actions = policy.infer(observation)["actions"]  # (10, 7)，未归一化的增量 + 夹爪指令
```

通过 `create_trained_policy` 加载对应配置和 checkpoint 即可使用这些变换。
输出每一步对应数据集的 0.2 秒；执行端自行选择一次执行多少步。
