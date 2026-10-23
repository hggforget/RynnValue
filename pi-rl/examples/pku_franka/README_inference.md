# Franka 真机推理

模型服务跑在 4090（`219.223.195.77`），Franka client 跑在机器人（`pku-franka-hdc`，`219.223.195.140`）。Client 用 IP 直连 WebSocket，不需要额外隧道。

| | 机器 | 代码目录 |
| --- | --- | --- |
| 推理服务 | 4090 | `/home/hdc/RynnValue/pi-rl` |
| Franka client | 机器人 | `/home/hdc/workspace/RynnValue/pi-rl` |

## 起推理服务

在 4090 上启动绝对动作模型，监听 `219.223.195.77:8000`：

```bash
cd /home/hdc/RynnValue/pi-rl
bash examples/pku_franka/serve_collect_objects.sh
```

相对动作模型另开一个终端，监听 `8001`：

```bash
cd /home/hdc/RynnValue/pi-rl
ACTION_REPRESENTATION=relative bash examples/pku_franka/serve_collect_objects.sh
```

脚本使用 `.venv/bin/python`，权重默认在 `~/MODELS/pi05-pku-franka-collect-objects/`。

| 动作 | 配置 | 实验 | 地址 |
| --- | --- | --- | --- |
| 绝对 | `pi05_pku_franka_collect_objects` | `fr3_collect_objects_20261017_1020_190034` | `ws://219.223.195.77:8000` |
| 相对 | `pi05_pku_franka_collect_objects_relative` | `fr3_collect_objects_20261017_1021_010646` | `ws://219.223.195.77:8001` |

checkpoint 取对应实验的 `19999` 步。换机器或换权重时覆盖 `BIND_HOST`、`PORT`、`MODEL_ROOT`、`CHECKPOINT`、`PYTHON`。

服务起来后，在 Franka 上确认可达：

```bash
curl --noproxy '*' http://219.223.195.77:8000/healthz
```

返回 `OK` 即可连 client。相对模型把端口换成 `8001`。

## 起 Franka client

先在采集页面启动 robot 和双相机，并关掉遥操采集器。不要同时开另一个会发布目标位姿的控制端。

在 Franka 上：

```bash
cd /home/hdc/workspace/RynnValue/pi-rl
bash examples/pku_franka/run_franka.sh
```

也可以从 4090 拉起。按键需要交互终端，SSH 要加 `-tt`：

```bash
ssh -tt pku-franka-hdc 'cd /home/hdc/workspace/RynnValue/pi-rl && bash examples/pku_franka/run_franka.sh'
```

不带参数时是实时评测：绝对动作、连 `8000`、执行机械臂和夹爪。默认 10 次尝试，每次最多 5 分钟；每次请求执行动作块前 10 步，30 Hz。任务文本是 `place each object on the plate that matches its color.`

按键不用回车：

| 按键 | 作用 |
| --- | --- |
| 空格 | 每轮开始前按下才开始推理和计时 |
| `s` / `f` | 本次成功 / 失败，保持当前位姿 |
| `r` | 打开夹爪，回到采集固定起点，然后等空格开下一轮 |
| `p` 或空格 | 运行中暂停。暂停时间计入这 5 分钟 |
| `c` | 继续 |
| `q` | 退出 client，保留 robot 驱动 |

`Ctrl+C` 同样只停 client，不关 robot 驱动。等待空格、等待 `r` 和复位本身不计时。

启动时会打印本次结果目录，位于 `.runtime/pku_franka_collect_objects/results/`。里面有 `actions.npy`、`report.json` 和 `client.log`。

常用覆盖都是环境变量，写在命令前面：

```bash
# 相对动作，自动改连 8001
ACTION_REPRESENTATION=relative bash examples/pku_franka/run_franka.sh

# 普通推理：不做逐次评分，请求次数由 REPEAT 控制，默认 10000
ATTEMPTS=0 bash examples/pku_franka/run_franka.sh

# 只向模型要动作，不驱动机械臂
EXECUTE=0 bash examples/pku_franka/run_franka.sh

# 用实时观测规划目标，不发送控制命令
DRY_RUN=1 bash examples/pku_franka/run_franka.sh

# 只复位：打开夹爪，回到采集起点。DRY_RUN=1 只打印目标
bash examples/pku_franka/run_franka.sh reset

# 回放数据集观测，不执行动作
bash examples/pku_franka/run_franka.sh replay
```

`INFERENCE_HOST` 和 `PORT` 用来改服务器地址。实时模式还可用 `EXECUTE_STEPS`、`CONTROL_HZ`、`ATTEMPTS`、`ATTEMPT_TIMEOUT`。参数集中在 `run_franka.sh` 顶部，也可以直接改脚本。

更细的观测、执行和报告字段见 [inference.md](inference.md)。
