# PKU Franka FR3 微调

2026-10-17 的双相机 collect-objects 数据集请使用
[独立配置及说明](collect_objects_20261017.md)。它使用绝对末端目标和 0/1 夹爪动作，
与本文旧单相机数据的动作契约不同。

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
