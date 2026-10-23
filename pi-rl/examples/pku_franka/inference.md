# Pi0.5 推理

pi-rl 内置的 Pi0.5 推理流程，直接使用本项目的模型加载和原生 OpenPI WebSocket 协议。
模型在 4090（`219.223.195.77`）上运行，Franka（`219.223.195.140`）通过 IP 直接连接原生 WebSocket，
负责提供观测、请求推理和保存预测动作。

| 模式 | 观测来源 | 行为 |
| --- | --- | --- |
| `replay` | LeRobot 录制数据或 NPZ 快照 | 回放观测并请求模型 |
| `inference` | 实时 ROS 位姿、关节、夹爪和双相机 | 读取新观测并请求模型 |

两个模式默认只返回、保存模型输出。`inference` 可用 `EXECUTE=1` 启用机械臂和夹爪动作执行，
`replay` 始终只回放录制观测。

## Franka 端直接启动

4090 模型服务运行后，在 Franka 机器执行：

```bash
cd /home/hdc/workspace/RynnValue/pi-rl
bash examples/pku_franka/run_franka.sh
```

默认读取实时 ROS 观测，使用绝对动作模型，请求 100 次；需要机器人和双相机驱动已启动。
常用参数集中在 `run_franka.sh` 顶部，可以直接编辑。回放录制观测和切换模型只需：

```bash
bash examples/pku_franka/run_franka.sh replay
ACTION_REPRESENTATION=relative bash examples/pku_franka/run_franka.sh replay
```

`REPEAT`、`EPISODE`、`FRAME`、`FRAME_STEP`、`DATASET`、`TARGET_NAMESPACE` 等也可通过环境变量覆盖。
例如 `EPISODE=53 REPEAT=10 bash examples/pku_franka/run_franka.sh replay`。
默认连接 `219.223.195.77:8000`，相对模型自动选用端口 8001；`INFERENCE_HOST`、`PORT` 可覆盖连接地址。
每次运行在 `.runtime/pku_franka_collect_objects/results/` 下新建独立目录，
保存最后一个动作块 `actions.npy`、报告 `report.json` 和终端日志 `client.log`，启动时会打印完整路径；
`RUN_DIR` 可指定结果目录。即使启动报错，也能在 `client.log` 中查看原因。
`run_inference.sh` 保留为可传入完整客户端参数的入口。
`run_franka.sh` 仅接受可选的模式名，其余配置使用脚本顶部设置或环境变量，避免额外 CLI 参数
覆盖已打印的执行开关。

## 启动模型服务

在 4090 上启动绝对动作模型：

```bash
cd /home/hdc/RynnValue/pi-rl
bash examples/pku_franka/serve_collect_objects.sh
```

如需相对动作模型，在另一个终端启动：

```bash
cd /home/hdc/RynnValue/pi-rl
ACTION_REPRESENTATION=relative bash examples/pku_franka/serve_collect_objects.sh
```

`serve_collect_objects.sh` 使用 `.venv/bin/python`，默认读取
`~/MODELS/pi05-pku-franka-collect-objects/<配置名>/<实验>/19999`。
可通过 `PYTHON`、`MODEL_ROOT`、`CHECKPOINT`、`PORT`、`BIND_HOST` 覆盖默认值。

| 版本 | 配置名 | 实验 | Franka 端访问 |
| --- | --- | --- | --- |
| 绝对 | `pi05_pku_franka_collect_objects` | `fr3_collect_objects_20261017_1020_190034` | `ws://219.223.195.77:8000` |
| 相对 | `pi05_pku_franka_collect_objects_relative` | `fr3_collect_objects_20261017_1021_010646` | `ws://219.223.195.77:8001` |

服务默认绑定服务器地址 `219.223.195.77`。在其他服务器部署时，修改 `BIND_HOST`，并将客户端
`INFERENCE_HOST`（或 `--host`）设为相同服务器地址。Franka 端执行
`curl --noproxy '*' http://219.223.195.77:8000/healthz` 返回 `OK` 表示服务可达。
模型推理通过 IP 直连，不需要运行 `connect_inference.sh`；该隧道脚本已移除。
两个 `1022` 实验仍缺权重分片，默认启动器使用上表中的完整权重及各自的归一化统计。

## replay：回放观测

在 Franka 机器上运行：

```bash
cd /home/hdc/workspace/RynnValue/pi-rl
bash examples/pku_franka/run_inference.sh --mode replay \
  --dataset ~/workspace/DATA/local/fr3_collect_objects_20261017 \
  --episode 1 --frame 0 --frame-step 1 --repeat 3 \
  --host 219.223.195.77 --port 8000 --action-representation absolute \
  --output .runtime/pku_franka_collect_objects/results/absolute_actions.npy \
  --report .runtime/pku_franka_collect_objects/results/absolute.json
```

`--frame` 是起始帧，`--frame-step` 是帧间隔，`--repeat` 是请求数。
两路视频各解码一次，与 Parquet 的实测状态使用同一帧索引。
`--observation observation.npz` 可以替代 `--dataset`，重复使用一个已有观测快照。

相对版本使用 `--port 8001 --action-representation relative`，并更改输出文件名。
客户端核对服务配置名，检查每次输出为有限的 `(30, 7)` 动作块。
NPY 保存最后一次动作块；JSON 保存连接地址、请求帧号、时延及服务元数据。

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
  --host 219.223.195.77 --port 8000 --action-representation absolute \
  --output .runtime/pku_franka_collect_objects/results/live_actions.npy \
  --report .runtime/pku_franka_collect_objects/results/live.json
```

实时读取器根据现有 `crisp_gym` 的 `fr3_dual_cam_640x480` 配置解析 ROS topics，
每次请求等待新的观测，最大观测年龄为 0.25 秒，
各源时间戳差最大为 0.1 秒。缺少驱动或数据过期时会超时退出，
可用 `--observation-timeout` 调整等待时间。

### 从 4090 测试真机动作执行

**2026-10-23 实测状态：未通过。** 首次真机测试出现通信与关节速度保护，随后已恢复硬件。
发现并修复了初始化/退出保持消息中 float32 导致 ROS 坐标序列化错误的问题后，
在 `tmux test` 重测 3 mm 上移仍未达到预期，测试窗口没有新增 ERROR，脚本已停止 robot 驱动。
当前不能据隔离测试成功宣称真机执行正常。
检查记录见 [遥操与客户端执行对照](execution_review_20261023.md)。问题排除前仅使用干跑。

在 4090 运行以下命令，测试现有 client 是否能驱动真机：

```bash
cd /home/hdc/RynnValue/pi-rl
bash examples/pku_franka/test_robot_execution.sh --execute
```

脚本在 4090 的 `219.223.195.77:8002` 启动临时原生 WebSocket 服务，通过 SSH 启动 Franka 端
现有 `franka_collect.client`。Franka 的实时双相机与状态观测发到 4090，4090 返回确定的
`(30, 7)` 动作块，client 按正常执行路径发布 ROS 目标。测试服务使用独立端口，结束后关闭。
机器人和相机驱动需已启动，运行前停止其他控制客户端。

测试保持初始位姿，随后沿**基座 Z 轴上移 5 mm**、保持、返回初始位置并保持。
每个请求执行前 10 步，默认 10 Hz（5 mm 单程约 1 秒）；姿态固定，
`--disable-gripper` 禁止发送任何夹爪命令。
它检查真实反馈：上移、返回的末端误差均需小于 1.5 mm，实测上移量至少达到目标的 60%，
姿态和夹爪状态无明显变化，才能报告 `passed`。仅发送成功但机械臂不动会报告失败。
执行前默认倒计时 5 秒；远端设置独立监控，出现硬件 ERROR、客户端失败或超时时停止整个
robot 驱动。4090 终端按 Ctrl+C 也会请求停止 robot 驱动，停止结果会打印出来。
正常完成仍沿用执行器的保持逻辑。测试不执行自动故障恢复。
失败/中断后重跑前，先在 `data_collection_web` 启动 robot 服务；仅重跑测试不会自动启动驱动。
脚本现在在倒计时前检查服务，驱动未启动会直接提示 `Robot driver is stopped` 并退出。
原始客户端的 TimeoutError 等错误也会显示到测试终端，完整堆栈保存在 `client.log`。

```bash
# 只规划、不控制机器人（省略 --execute 时的默认行为）
bash examples/pku_franka/test_robot_execution.sh --dry-run
# 同一条真机链路测试相对动作，仍然上移 5 mm 后返回
bash examples/pku_franka/test_robot_execution.sh --execute --action-representation relative
```

`--distance-mm` 可在 3–100 mm 内调整幅度，`--robot-host`、`--remote-root`、`--host`、`--port`
可覆盖机器地址与部署路径。结果保存在 4090 的
`.runtime/pku_franka_collect_objects/results/<时间>_action_test_<动作类型>/`，
包括 `report.json`（各轮实测位姿、动作、运动判定）、`client_report.json`、`client.log`。
重复指定已有且非空的 `--output` 目录时，会在其中新建 `run_<时间>` 子目录，保留先前记录。
报告中 `policy_type=scripted_hardware_execution_test` 表示确定动作测试；模型预测执行仍使用
`run_franka.sh` 和 8000/8001 的 Pi0.5 服务。
`--control-hz` 可设为 5–30 Hz，`--countdown` 可设为 0–60 秒。
在 4090 的 `tmux test` 中受监督地重测可使用
`bash examples/pku_franka/test_robot_execution.sh --execute --distance-mm 3 --control-hz 10 --countdown 10`。
超过 10 mm 的行程要求使用 10 Hz，并拆成多轮请求，每步最多 1 mm。
例如 `--distance-mm 100 --control-hz 10` 上移和返回各分成 10 个动作块，单程至少 10 秒。
2026-10-23 的 100 mm 真机测试已观测到约 83.7 mm 上移，返回后距起点约 9 mm；
这确认了真实运动能力，但位置精度仍未通过 1.5 mm 的测试容差。

### 执行模型预测动作

`run_franka.sh` 启动时会打印 `Execute: 0/1`。默认值为 0，机器人不会因模型预测而运动。
先用真机观测规划 3 个目标，不发送任何控制命令：

```bash
DRY_RUN=1 REPEAT=1 EXECUTE_STEPS=3 bash examples/pku_franka/run_franka.sh
```

在机器人旁启用执行，先只运行一轮、3 步：

```bash
EXECUTE=1 REPEAT=1 EXECUTE_STEPS=3 bash examples/pku_franka/run_franka.sh
```

持续闭环执行使用 `EXECUTE=1 bash examples/pku_franka/run_franka.sh`。
默认每轮执行预测块的前 10 步，按 30 Hz 发布，再读取新观测请求模型；
用 `EXECUTE_STEPS`、`CONTROL_HZ` 修改这两个参数。
完整客户端参数对应 `--execute` / `--dry-run-execution`、`--execute-steps`、`--control-hz`。

执行器使用现有 CRISP 接口：启动执行时激活 `cartesian_impedance_controller`，
向 `/left/target_pose` 发布基座坐标系的 `PoseStamped`，夹爪通过
`/left/franka_gripper/gripper_action` 发送 `GripperCommand`。
绝对目标直接解读；相对版本按采集实现逐步累积平移、左乘旋转。
夹爪预测裁剪到 0–1，再按已有配置转换为手指位置和力。
夹爪同一时刻只保留一个未完成请求，检查最终执行结果；请求拒绝、执行中止、超过 2 秒未确认或
超过 10 秒未完成时，停止后续动作。每个动作块执行前重新确认 Cartesian 控制器处于 active 状态。

每步目标平移最多 5 mm、转动最多 0.03 rad；超出配置工作空间、远距离目标、观测过期、
实际跟踪误差过大或存在其他命令发布者时会拒绝继续。
工作空间读取采集端真正生效的 `config.safety_box`；报告的 `workspace_lower/upper` 中，
`null` 代表该方向未设置界限。现有采集数据包含合法负 Z，不能额外把基座 Z=0 当作桌面边界。
相对动作基于上一步实际发布的目标累积，限幅舍弃的增量不会在后续零动作中继续追赶。
退出或 `Ctrl+C` 时保持最新实测位姿，并取消尚未完成的夹爪目标。执行器不自动回 home。
`EXECUTE=1` 与 `DRY_RUN=1` 互斥，均不能用于 `replay`。

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
客户端会为选定的模型地址绕过继承的 HTTP/SOCKS 代理，保持 IP 直连。
模型权重、JAX 和 Torch 不需要安装到机器人端。

在硬件未启动时，可用隔离的 ROS namespace 发布录制观测并验证完整推理流程：

```bash
cd /home/hdc/workspace/RynnValue/pi-rl
PYTHONPATH="$PWD/packages/openpi-client/src:$PWD/.runtime/pku_franka_collect_objects/vendor" \
  ~/workspace/data_collection_web/scripts/run.sh \
  python -m openpi_client.franka_collect.check_live_inference
```

检查包括关节名称排序、采集夹爪状态约定、双相机、过期时间戳拒绝，以及实际 4090 模型的
WebSocket 推理。添加 `--with-execution` 可使用独立测试 namespace 内的模拟控制器和夹爪 action server，
验证控制器激活、目标发布及退出时保持位置，不接触 `/left` 的控制接口。
再添加 `--check-failures` 会注入夹爪执行失败和控制器中途失活，确认执行器明确报错并停止。
真实硬件观测和执行目标干跑已通过；真机动作执行待现场验证。

与采集端实际 `ManipulatorCartesianEnv.step()` 对照验证，无需创建机器人或 ROS 节点：

```bash
PYTHONPATH="$PWD/packages/openpi-client/src" \
  ~/workspace/data_collection_web/scripts/run.sh \
  python examples/pku_franka/check_collection_compatibility.py
```

验证绝对目标、相对平移/旋转、夹爪归一化、限幅后零动作保持以及有效工作空间。

运行日志和 PID 在 4090 的 `.runtime/pku_franka_collect_objects/logs/`，推理结果在机器人端的
`.runtime/pku_franka_collect_objects/results/`。`.runtime/` 不纳入 Git。

模型服务入口为 `scripts/serve_policy.py`；客户端和实时读取器实现位于
`packages/openpi-client/src/openpi_client/franka_collect/`，不依赖 RynnValue 的其他模块。
