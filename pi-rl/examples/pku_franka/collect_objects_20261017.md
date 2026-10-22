# FR3 collect-objects 数据集训练

数据位置：`pku-franka-hdc:/home/hdc/workspace/DATA/local/fr3_collect_objects_20261017`。
2026-10-19 检查到 LeRobot v2.1、64 条轨迹、46,590 帧、主相机和腕相机各 64 个视频。
全部 Parquet 时间戳与 128 个视频流均为 30 Hz；`video_info.video.fps` 中残留的 5 Hz
字段不作为训练时间轴。配置不修改原始数据和元数据。

| 微调方式 | 绝对动作配置 | 相对动作配置 | 学习率 |
| --- | --- | --- | --- |
| 全参数 SFT | `pi05_pku_franka_collect_objects` | `pi05_pku_franka_collect_objects_relative` | 5e-5 → 5e-6 |
| 全参数 IQL | `pi05_pku_franka_collect_objects_iql` | `pi05_pku_franka_collect_objects_relative_iql` | 3e-5 → 3e-6 |
| 全参数 Filter BC | `pi05_pku_franka_collect_objects_filter_bc` | `pi05_pku_franka_collect_objects_relative_filter_bc` | 5e-5 → 5e-6 |

以上六个离线配置均为 Batch 16、Horizon 30（1 秒）、20,000 步；从 `pi05_base` 初始化，
warmup 1,000 步，每 1,000 步保存 checkpoint。原配置名继续表示绝对动作，`_relative` 系列读取已录制的相对动作。
π0.5 保持内部 32 维以加载预训练权重，状态和动作分别补齐后输入模型。
20,000 步是起始训练预算，按 42,465 个保留帧和 Batch 16 约相当于 7.5 遍数据；
应结合独立保留轨迹和实际任务表现选择 checkpoint。当前配置沿用数据集的 train split，未建立验证集。

## 数据契约

| 原始字段 | 处理 |
| --- | --- |
| `observation.images.primary` | RGB，映射为 `base_0_rgb`，等比例补边到 224×224 |
| `observation.images.wrist` | RGB，映射为 `left_wrist_0_rgb`，等比例补边到 224×224 |
| `observation.state.cartesian` | 实测 `[x,y,z,roll,pitch,yaw]`，米/弧度 |
| `observation.state.gripper` | 实测夹爪开度，原数据已归一化到 0–1，保留该单位 |
| `observation.state.joints` | 实测 7 维关节角 |
| `action` | 7 维绝对末端目标 `[x,y,z,roll,pitch,yaw,gripper]`，机器人基座坐标系、Euler XYZ |
| `action.relative` | 相对版本使用：7 维 `[Δx,Δy,Δz,Δroll,Δpitch,Δyaw,gripper]`，机器人基座坐标系、米/弧度 |
| `task_index` | 从 `tasks.jsonl` 取得语言指令 |

模型状态为 `[实测 Cartesian(6), gripper(1), joints(7)]`，共 14 维，归一化后作为 π0.5
离散状态 token 输入。不要使用原始 20 维 `observation.state`：最后 6 维是
`observation.state.target`，在全部 64 条轨迹中都与当前 `action[:6]` 完全相同，属于标签泄漏。

绝对版本直接读取 `action`，相对版本直接读取 `action.relative`；两者均不做差分或累积。
相对字段是采集时保存的控制增量，不是用相邻绝对目标临时求差得到的标签。
两种版本的夹爪动作都保持 `0=闭合、1=张开`，夹爪这一维不取增量；图像处理与 14 维实测状态完全相同。
选中的动作字段缺失、元数据中的动作表示不匹配或标签包含 NaN/Inf 会报错，不会退回另一个字段。
两种动作版本的归一化统计和 checkpoint 不可混用；旧 `pi05_pku_franka` 的夹爪 `-1/+1` 约定也不同。

数据的两个 task 实际是同一任务，其中一个漏了首字母。适配器将
`lace each object on the plate that matches its color.` 统一为
`place each object on the plate that matches its color.`，推理时仍可传入其他指令。

SFT / Filter BC 默认排除 `episodes.jsonl` 备注中明确写有失败的轨迹 `0, 2, 14, 21, 47, 52`，
剩余 **58 条、42,465 帧**。IQL 保留全部 64 条，包括失败轨迹。备注为“最后有点瑕疵”和“resume 上条，红色没看见”的
轨迹 18、22 暂保留，未自行推断其成败。白名单从当前元数据读取，后续新增轨迹不会因为
硬编码总数而被遗漏。要使用全部轨迹，可在这两个预设中将 `excluded_episode_ids` 改成 `()`。

## 启动

在已安装 `pi-rl` 训练依赖、具备 NVIDIA GPU 的训练机器上运行；数据集需挂载或复制到
本机路径。SSH 路径本身不能直接传给 LeRobot。机器人机器当前已验证的采集环境有
NumPy / PyArrow / PyAV / LeRobot，但没有 JAX / Flax，且其 LeRobot 不提供本仓库使用的
`lerobot.common` 命名空间，因此不是兼容的 OpenPI 训练环境；不要混用两套环境的包。

从 `pi-rl` 项目目录启动，默认全参数 SFT、单 GPU：

```bash
bash scripts/train_pi05_pku_franka_collect_objects.sh

METHOD=iql bash scripts/train_pi05_pku_franka_collect_objects.sh
METHOD=filter_bc bash scripts/train_pi05_pku_franka_collect_objects.sh
```

相对动作版本，通过 `ACTION_REPRESENTATION=relative` 切换：

```bash
ACTION_REPRESENTATION=relative bash scripts/train_pi05_pku_franka_collect_objects.sh
ACTION_REPRESENTATION=relative METHOD=iql bash scripts/train_pi05_pku_franka_collect_objects.sh
ACTION_REPRESENTATION=relative METHOD=filter_bc bash scripts/train_pi05_pku_franka_collect_objects.sh
```

默认 `ACTION_REPRESENTATION=absolute`。若同时显式指定 CONFIG 和 ACTION_REPRESENTATION，二者必须一致。

根据显存调整 Batch / FSDP：

```bash
CONFIG=pi05_pku_franka_collect_objects GPUS=0,1 FSDP_DEVICES=2 \
  bash scripts/train_pi05_pku_franka_collect_objects.sh --batch-size 16
```

数据在另一台训练机器上的不同目录时：

```bash
REPO_ID=/本机数据路径/fr3_collect_objects_20261017 \
PYTHON=/已安装训练依赖的环境/bin/python \
  bash scripts/train_pi05_pku_franka_collect_objects.sh
```

脚本先用 CPU 扫描各方法实际使用的轨迹/帧，跳过视频解码，重新计算归一化统计，写入
`assets/<配置名>/<asset_id>/norm_stats.json`。绝对版本的 asset_id 为 `pku_franka_collect_objects`，
相对版本为 `pku_franka_collect_objects_relative`。统计只使用上述 14 维实测状态和所选字段的 7 维动作。
每个配置独立计算统计，避免动作表示或不同筛选集合覆盖彼此；确认统计正确且白名单未变后，可用 `SKIP_NORM=1` 跳过。
checkpoint 位于 `./checkpoints/<配置名>/<EXP_NAME>/`，默认实验名为
`fr3_collect_objects_20261017`。续训增加 `--resume`；新实验通过 `EXP_NAME` 更名。

## IQL 与 Filter BC 的 progress 接口

按“后续存在逐帧 progress 字段”的要求实现；原始数据当前仍没有此字段，没有写入虚构标注。
需要在 `meta/info.json` 中声明标量 float32 `progress`，并在每个 Parquet 中保存逐帧同名数值列。
约定越大越接近完成；如果标注是剩余时间/代价，应先转换为同方向的势能，例如取负值。
缺少字段或出现 NaN/Inf 会报错。合成 progress 仅用于接口测试。

IQL 请求两路 RGB、Cartesian、gripper、joints 在 `[t,t+H]` 的观测对，以及 `progress[t:t+H+1]`。
非终止转移的块奖励为 `gamma^H * progress[t+H] - progress[t]`；终止时后继势能设为 0、
mask 设为 0。已知失败轨迹在终止时额外施加折扣后的 -1 惩罚，其余轨迹未额外赋予“成功”标签。
预折扣奖励通过 `chunk_reward` 进入 trainer，避免重复 shaping；数据折扣必须等于 `iql.discount`。
策略执行优势加权 flow-matching BC，critic/value LR 为 3e-4、expectile 0.8、A scaling 10，
前 1,000 步策略使用均匀权重。策略未设置冻结过滤器，执行全参数更新。

Filter BC 对保留轨迹计算 `A_t = sum(k<W) lambda^k * (progress[t+k+1]-progress[t+k])`，
默认 `W=50`、`lambda=0.98`，保留 `A_t>0` 的全局帧索引，随后执行全参数普通 BC。
也支持 mean / median / quantile 阈值，不把筛选分数作为策略输入；每次重新计算索引，避免标签更新后误用旧缓存。

IQL 可从同一动作版本的 SFT checkpoint 初始化：

```bash
METHOD=iql INIT_CHECKPOINT=/训练输出/SFT实验/20000 \
  bash scripts/train_pi05_pku_franka_collect_objects.sh

ACTION_REPRESENTATION=relative METHOD=iql INIT_CHECKPOINT=/训练输出/相对动作SFT实验/20000 \
  bash scripts/train_pi05_pku_franka_collect_objects.sh
```

## DSRL 在线训练

`openpi.training.pku_franka_dsrl.DsrlConfig` 提供独立预设
`pi05_pku_franka_collect_objects_dsrl`。它冻结已训练的 VLA，由 SAC 学习噪声；
训练对象是噪声 actor 和 critic，基础 VLA 仅执行 `infer()`。
当前 DSRL 仍仅支持绝对动作；本次两种动作表示的扩展只覆盖离线训练。

链路：双相机 + 14 维实测状态 → SAC 输出 `(1,32)` 噪声 → 扩展到 `(30,32)` →
VLA 生成 `(30,7)` 绝对末端目标 → 每次执行 6 步、30 Hz → 环境奖励/progress → replay → SAC 更新。
夹爪预测在送往环境前限制到 `[0,1]`。默认 Batch 256、UTD 20、500 个查询样本后更新、
最多 500,000 次更新/500 个 episode。两路 SAC 图像缩放到 64×64，状态维度固定为 14。
bootstrap 折扣使用实际执行步数 `gamma^n`，终止 mask=0，时间截断 mask=1；replay 显式保存真实 next observation。

先检查配置，无需 JAX 或机器人：

```bash
python -m openpi.training.pku_franka_dsrl --print-config
```

在兼容训练环境中，以 fake 环境检查真实 VLA/SAC 运行路径：

```bash
bash scripts/train_dsrl_pku_franka_collect_objects.sh \
  --base-checkpoint /训练输出/SFT实验/20000 --fake-env --max-episodes 2 \
  --batch-size 4 --start-online-updates 4 --random-queries 1 --utd-ratio 1 --max-updates 4
```

真机通过 `--env-factory my_fr3_env:create_env --env-kwargs '{"url":"..."}'` 接入，工厂需返回：

- `reset() -> observation`：使用本文的两个 RGB 与三个实测状态字段；启用 shaping 时含有限标量 `progress`。
- `step_chunk(actions[N,7], publish_hz=30) -> result`：执行基座坐标系绝对 Cartesian 目标，
  返回 `{observation, rewards[N], terminated, truncated}`，夹爪 `0=关、1=开`。
- `close()`：释放环境资源。进度可由工厂封装的 RynnValue 服务计算；`--shaping-weight 0` 支持仅环境稀疏奖励。

当前 `pku-franka-hdc` 的旧 HTTP RobotClient 提供单相机/7 维状态和增量动作接口，缺少本配置的
腕相机与关节状态，因此不能直接用作这个工厂。此处完成训练入口和明确的环境契约，尚未实现或验收
该旧服务器到本契约的真机桥接。fake 环境的合成 progress 只用于协议测试，不代表任务成功率。
DSRL 输出目录保存配置和 SAC checkpoint；`--resume-agent` 恢复 SAC 权重，replay 从空缓存重新收集。

## 推理接口

通过 `create_trained_policy` 加载对应配置及其 checkpoint。输入只需两个 RGB 和实测状态：

```python
observation = {
    "observation.images.primary": primary_rgb,  # HWC uint8 RGB
    "observation.images.wrist": wrist_rgb,
    "observation.state.cartesian": measured_xyz_rpy,  # (6,), m/rad
    "observation.state.gripper": measured_gripper_0_to_1,  # scalar or (1,)
    "observation.state.joints": measured_joint_angles,  # (7,), rad
    "prompt": "place each object on the plate that matches its color.",
}
actions = policy.infer(observation)["actions"]  # (30, 7)，含义由加载的动作版本决定
```

输出裁回 7 维，并保持所选版本的绝对目标/相对增量与 0/1 夹爪约定。
执行端必须按 checkpoint 对应版本解释前 6 维，不能将绝对目标当增量或将增量当绝对目标。
一次执行多少步由执行端决定。此改动只增加离线训练配置和数据适配，未启动训练或机器人控制。

## 跨机推理

推理服务、SSH 隧道、replay / inference 客户端和验证程序均实现在本项目中。
完整命令见 [跨机推理说明](inference.md)。模型服务复用 `scripts/serve_policy.py`，
轻量客户端位于 `packages/openpi-client/src/openpi_client/franka_collect/`。

## 已完成的验证

- 10 项 NumPy/标准库适配器测试通过，覆盖绝对/相对动作字段选择、夹爪保持、缺少字段时拒绝回退、
  目标字段泄漏、两路图像、指令拼写、推理输入、字段形状，以及失败轨迹筛选和后续新增轨迹。
- 读取全部 64 个 Parquet，检查数值有限性、帧数、30 Hz 时间戳以及 target/action 一致性；
  核对全部 128 个视频的帧数和 30 Hz 视频流信息，未发现不一致。
- 两种动作适配器通过全部 64 条轨迹首/中/尾共 384 次真实动作块检查，包含轨迹末尾补齐；
  实际解码轨迹 1、22、63 的两路视频首帧，共 6 个视频。
- Python 语法、注册配置名唯一性及启动脚本语法检查通过。完整 OpenPI DataLoader 和训练更新尚未验证；
  tokenizer / JAX 模型前向已通过下述 checkpoint 推理验证。采集环境的 LeRobot 命名空间与本仓库不兼容。
- 新增 IQL 奖励/终止、Filter BC 索引，以及 DSRL 噪声到动作、显式 next observation、
  终止/截断和更新循环测试。progress 使用合成夹具；没有为原始数据伪造标注。
- 2026-10-20 在采集环境运行适配器、RL、启动脚本和已有 DSRL 回归共 26 项测试，全部通过。
  启动脚本测试覆盖 6 个配置的入口与独立统计路径、动作表示冲突和禁止误读另一种表示的统计。
  6 个配置工厂通过隔离构造验证；IQL/Filter BC 使用临时合成 progress 数据，模型/tokenizer 依赖被隔离。
  同时修复 macOS Bash 3 在未设置 INIT_CHECKPOINT 时展开空数组的兼容性问题。
- 2026-10-22：4090 加载绝对/相对动作两套 step 19999 checkpoint，机器人机器通过反向 SSH 隧道和
  原生 WebSocket 分别请求真实录制的双相机、实测状态观测，返回有限的 `(30, 7)` 动作；
  首次 JIT 约 11 秒，预热后模型推理约 80–90 ms，跨机往返约 250 ms。
- `replay` 模式已用轨迹 1 的连续帧 0、1、2 验证两套模型；`inference` 模式已用独立 ROS 测试
  namespace 中发布的录制观测完成真实模型跨机请求，验证关节顺序、采集夹爪状态约定、双相机、
  ROS 时间戳过期拒绝。未启动机械臂控制，真实硬件观测尚待驱动启动后验证。
- 客户端 8 项测试通过；轨迹 1、53、105 各抽取 10 帧，绝对/相对两套 checkpoint 共 60 次跨机
  replay 请求全部通过，往返时延中位数约 252 ms。
- 完整轨迹 1 的 556 帧分别请求两套模型，共保存 1112 个 `(30, 7)` 预测块。
  每帧第 1 步与同帧标签对齐后，绝对动作 XYZ MAE 分别为 4.69 / 4.39 / 3.87 mm，
  相对动作 XYZ MAE 为 1.63 / 1.28 / 1.49 mm。`compare_replay.py` 可复现采样、PNG/PDF 曲线、
  CSV 和误差统计；结果反映录制观测下的离线动作拟合。
