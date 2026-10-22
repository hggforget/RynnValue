# Pi0.5 推理

pi-rl 内置的 Pi0.5 推理流程，直接使用本项目的模型加载和原生 OpenPI WebSocket 协议。
模型在 4090 上运行，Franka 机器负责提供观测、请求推理和保存预测动作。

| 模式 | 观测来源 | 行为 |
| --- | --- | --- |
| `replay` | LeRobot 录制数据或 NPZ 快照 | 回放观测并请求模型 |
| `inference` | 实时 ROS 位姿、关节、夹爪和双相机 | 读取新观测并请求模型 |

两个模式均只返回、保存模型输出，不执行机械臂动作。

## 启动模型及连接

在 4090 上，分别保持以下三个命令运行：

```bash
cd /home/hdc/RynnValue/pi-rl
bash examples/pku_franka/serve_collect_objects.sh
ACTION_REPRESENTATION=relative bash examples/pku_franka/serve_collect_objects.sh
bash examples/pku_franka/connect_inference.sh
```

`serve_collect_objects.sh` 使用 `.venv/bin/python`，默认读取
`~/MODELS/pi05-pku-franka-collect-objects/<配置名>/<实验>/19999`。
可通过 `PYTHON`、`MODEL_ROOT`、`CHECKPOINT`、`PORT`、`BIND_HOST` 覆盖默认值。

| 版本 | 配置名 | 实验 | 4090 监听 | Franka 端访问 |
| --- | --- | --- | --- | --- |
| 绝对 | `pi05_pku_franka_collect_objects` | `fr3_collect_objects_20261017_1020_190034` | `127.0.0.1:8000` | `ws://127.0.0.1:19080` |
| 相对 | `pi05_pku_franka_collect_objects_relative` | `fr3_collect_objects_20261017_1021_010646` | `127.0.0.1:8001` | `ws://127.0.0.1:19081` |

隧道使用 SSH 别名 `pku-franka-hdc`，可用 `ROBOT_HOST` 覆盖。机器人端执行
`curl http://127.0.0.1:19080/healthz` 返回 `OK` 表示服务可达。
两个 `1022` 实验仍缺权重分片，默认启动器使用上表中的完整权重及各自的归一化统计。

## replay：回放观测

在 Franka 机器上运行：

```bash
cd /home/hdc/workspace/RynnValue/pi-rl
bash examples/pku_franka/run_inference.sh --mode replay \
  --dataset ~/workspace/DATA/local/fr3_collect_objects_20261017 \
  --episode 1 --frame 0 --frame-step 1 --repeat 3 \
  --port 19080 --action-representation absolute \
  --output .runtime/pku_franka_collect_objects/results/absolute_actions.npy \
  --report .runtime/pku_franka_collect_objects/results/absolute.json
```

`--frame` 是起始帧，`--frame-step` 是帧间隔，`--repeat` 是请求数。
两路视频各解码一次，与 Parquet 的实测状态使用同一帧索引。
`--observation observation.npz` 可以替代 `--dataset`，重复使用一个已有观测快照。

相对版本使用 `--port 19081 --action-representation relative`，并更改输出文件名。
客户端核对服务配置名，检查每次输出为有限的 `(30, 7)` 动作块。
NPY 保存最后一次动作块；JSON 保存请求帧号、时延及服务元数据。

### 预测曲线与数据集动作对比

在 Franka 机器上逐帧请求两套模型，保存全部 `(帧数, 30, 7)` 预测块及同帧数据集标签：

```bash
cd /home/hdc/workspace/RynnValue/pi-rl
PYTHONPATH="$PWD/packages/openpi-client/src:$PWD/.runtime/pku_franka_collect_objects/vendor" \
  ~/workspace/data_collection_web/scripts/run.sh \
  python examples/pku_franka/compare_replay.py collect \
  --dataset ~/workspace/DATA/local/fr3_collect_objects_20261017 --episode 1 \
  --output .runtime/pku_franka_collect_objects/results/replay_curves_episode_000001/predictions.npz
```

默认使用每一帧观测；`--frame-step` 可调整采样间隔。绝对模型对应 `action`，相对模型对应
`action.relative`。复制结果到 4090 后绘图：

```bash
cd /home/hdc/RynnValue/pi-rl
mkdir -p .runtime/pku_franka_collect_objects/results/replay_curves_episode_000001
scp pku-franka-hdc:/home/hdc/workspace/RynnValue/pi-rl/.runtime/pku_franka_collect_objects/results/replay_curves_episode_000001/predictions.npz \
  .runtime/pku_franka_collect_objects/results/replay_curves_episode_000001/
.venv/bin/python examples/pku_franka/compare_replay.py plot \
  --input .runtime/pku_franka_collect_objects/results/replay_curves_episode_000001/predictions.npz \
  --output .runtime/pku_franka_collect_objects/results/replay_curves_episode_000001
```

生成 PNG、PDF、逐维 MAE / RMSE 的 `metrics.json` 和对齐数据 CSV。
曲线将每个观测产生的动作块第 1 步与同一帧标签比较；每次输入均来自数据集，属于离线动作拟合。
旋转角按 2π 等价对齐，夹爪保持原始连续预测值。

## inference：实时观测

机器人和相机驱动启动后，在 Franka 机器运行：

```bash
cd /home/hdc/workspace/RynnValue/pi-rl
bash examples/pku_franka/run_inference.sh --mode inference \
  --namespace left --repeat 100 \
  --port 19080 --action-representation absolute \
  --output .runtime/pku_franka_collect_objects/results/live_actions.npy \
  --report .runtime/pku_franka_collect_objects/results/live.json
```

实时读取器根据现有 `crisp_gym` 的 `fr3_dual_cam_640x480` 配置解析 ROS topics，
仅订阅，不创建机器人或夹爪控制发布器。每次请求等待新的观测，最大观测年龄为 0.25 秒，
各源时间戳差最大为 0.1 秒。缺少驱动或数据过期时会超时退出，
可用 `--observation-timeout` 调整等待时间。

| 字段 | 约定 |
| --- | --- |
| `observation.images.primary`、`observation.images.wrist` | 分别读取，HWC uint8 RGB |
| `observation.state.cartesian` | 实测 `(6,)` XYZ / Euler XYZ，米 / 弧度 |
| `observation.state.gripper` | `(1,)`，保持采集代码的 `1 - gripper.value` 状态变换 |
| `observation.state.joints` | `(7,)`，按配置中的关节名称排序，弧度 |
| `prompt` | 默认 `place each object on the plate that matches its color.` |

输出已经反归一化。绝对版本前六维是基座坐标系下的目标位姿，相对版本是采集时的控制增量；
两者输出的夹爪动作均为 `0=闭合、1=张开`。夹爪状态保持采集约定，不额外反转录制数据。

## 环境和验证

4090 使用 `.venv`；Franka 端复用现有 `jazzy-crisp` 环境，OpenPI client 源码位于 `packages/openpi-client/src/`，
`websockets==15.0.1` 放在 `.runtime/pku_franka_collect_objects/vendor/`，不修改共享 Pixi 环境。
可通过 `PYTHON`、`ROBOT_RUNTIME`、`PI05_VENDOR_DIR` 覆盖客户端运行环境。
模型权重、JAX 和 Torch 不需要安装到机器人端。

在硬件未启动时，可用隔离的 ROS namespace 发布录制观测并验证完整推理流程：

```bash
cd /home/hdc/workspace/RynnValue/pi-rl
PYTHONPATH="$PWD/packages/openpi-client/src:$PWD/.runtime/pku_franka_collect_objects/vendor" \
  ~/workspace/data_collection_web/scripts/run.sh \
  python -m openpi_client.franka_collect.check_live_inference
```

检查包括关节名称排序、采集夹爪状态约定、双相机、过期时间戳拒绝，以及实际 4090 模型的
WebSocket 推理。真实硬件观测待机器人驱动启动后验证。

运行日志和 PID 在 4090 的 `.runtime/pku_franka_collect_objects/logs/`，推理结果在机器人端的
`.runtime/pku_franka_collect_objects/results/`。`.runtime/` 不纳入 Git。

模型服务入口为 `scripts/serve_policy.py`；客户端和实时读取器实现位于
`packages/openpi-client/src/openpi_client/franka_collect/`，不依赖 RynnValue 的其他模块。
