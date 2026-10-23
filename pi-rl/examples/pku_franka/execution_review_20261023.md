# 2026-10-23 真机执行与遥操对照

本次真机动作测试失败，不能把隔离 ROS 测试通过视为真机可用。停止了测试服务与客户端，
未重启硬件、恢复错误或再次切换控制器。以下源码路径均为 Franka 机器上的实际运行环境。

## 遥操使用的执行路径

`data_collection_web/server.py:260` 启动
`/home/hdc/workspace/crisp_gym/crisp_gym/record/spacemouse_recorder.py`。

1. `spacemouse_recorder.py:304` 创建 Cartesian 环境，等待机器人、夹爪、相机与控制器服务就绪。
   数据集初始化等重 I/O 在切换控制器之前完成，源码明确说明这是为了避免控制循环超时。
2. `crisp_gym/record/teleop_control.py` 的 `prepare_cartesian_teleop()` 先重置目标，
   打开夹爪并确认，再将当前实测位姿设为目标，切换 Cartesian 控制器，等待 0.3 秒。
3. 遥操动作进入 `ManipulatorCartesianEnv.step()`。相对平移为
   `robot.target_pose.position + action[:3]`，旋转为 `Rotation(delta) * target_orientation`。
   `--action-representation` 指录制标签类型，不改变遥操本身采用相对控制的事实。
4. `env.step()` 调用 `robot.set_target()`，由 `crisp_py.Robot` 的 ROS 定时器按 50 Hz
   持续发布 `target_pose`，也持续发布初始化为当前关节值的 `target_joint`。
   目标更新频率与 ROS 发布频率分开。默认 `Robot` 在独立线程中运行 ROS executor。

实际配置为 `fr3_dual_cam_640x480`，`base_frame=base`，TCP 为 `fr3_hand_tcp`，
`cartesian_control_param_config=None`，没有额外控制参数文件需要加载。
`ManipulatorCartesianEnv.step()` 与 `Pose.to_ros_msg()` 都没有交换 XYZ 轴。
离线构造及 ROS 序列化往返检查也保留了目标 XYZ 与四元数，不涉及发布控制命令。

## 当前客户端的差异与缺口

| 项目 | 遥操 | 当前客户端 |
| --- | --- | --- |
| 更新目标 | `env.step()` → `Robot.set_target()` | 自行规划并发布 `PoseStamped` |
| ROS 发布 | 独立 executor、50 Hz 定时持续发布 | 动作块内 30 Hz；等待 WebSocket 期间停止发布 |
| 关节目标 | 初始化当前关节后持续发布 | 不发布；依赖控制器激活时初始化 |
| 初始化 | 当前目标、切换、等待 0.3 秒 | 当前目标、切换并检查 active，没有独立稳定等待 |
| 真机故障 | 本次未重新运行遥操验证 | 仅检查 active、反馈新鲜度及位姿误差，缺少硬件故障监控 |

控制器源码的 `on_activate()` 会用当前关节和末端状态初始化目标，因此仅凭没有
`target_joint` 发布不能断定它追踪零关节。30/50 Hz 是目标发布频率，Franka 硬件通信是另一层
1000 Hz 循环；不能直接用频率差异解释本次通信保护。上述差异尚不是已证实的根因。

## 本次设定与实际记录

- 14:08:03.829 激活原本 inactive 的 `cartesian_impedance_controller`。
- 初始目标 XYZ：`[0.351061434, 0.031948861, 0.227935821]` m。
- 上移目标 XYZ：`[0.351061434, 0.031948861, 0.232935821]` m。
- 姿态固定；禁止发送夹爪动作。上移和返回各 10 步，0.5 mm/步、30 Hz，单程约 0.33 秒。
- 初始保持 3 个请求，上移并保持 6 个请求，返回并保持 6 个请求，最后读取反馈并保持。
- 14:08:05.885 开始发送上移目标，但采样中未出现预期上移。
- 14:08:10.060 驱动首次报告 `communication_constraints_violation`。
- 客户端退出前最后一次采样为 14:08:14.159；相对起点约
  `[-0.556, -0.056, -0.541]` mm。这些采样不覆盖之后的硬件故障过程。
- 驱动随后反复报告 `joint_velocity_violation` 并自动恢复。
  14:08:21.008 自动恢复失败，停止硬件，并尝试停用依赖该硬件的控制器。

客户端在首次通信错误后仍继续下发目标，说明 active 与位姿误差检查不足以确认硬件健康。
退出时单次保持命令也不能确保底层驱动的自动恢复循环停止。需要补齐故障监控和停止行为，
并对照遥操的常驻控制执行路径验证；本次没有继续用真机试错。

4090 上的完整记录：
`.runtime/pku_franka_collect_objects/results/action_chain_real_absolute_20261023/`，
包含 `report.json`、`client_report.json`、`client.log` 和保存的 `robot.log`。
另有 36 项单元测试，以及隔离 ROS 下绝对/相对往返、禁用夹爪、故障注入测试通过。

## 随后按用户要求恢复故障

14:18 起，通过采集 Web 服务停止旧 robot 驱动并重新启动；确认四个运动控制器均 inactive、
`target_pose` 发布者为 0 后，调用 `/left/action_server/error_recovery`，结果为成功。
恢复后硬件状态为 active，两个状态广播器为 active，运动控制器仍为 inactive。
连续读取 5 份新鲜的位姿、关节、夹爪与双相机观测；本次重启之后日志没有 ERROR。
恢复后的末端 XYZ 约为 `[0.210801, 0.014200, 0.127268]` m；这个数值与动作客户端退出前不同，
无法据此重建未采样的故障阶段轨迹。此次仅清除故障，没有发送回 home 或测试运动。

## 随后确认并修复的消息类型错误

在实际 Franka Python/ROS 环境中，单独构造消息并执行序列化/反序列化（没有创建控制发布者）复现：

| `Pose.position` 输入 | 消息打印的 X | ROS 序列化后 X |
| --- | --- | --- |
| `numpy.float32(0.351061433)` | `0.35106143` | 约 `5.197e-315` |
| `numpy.float64(0.351061433)` | `0.351061433` | `0.351061433` |

客户端的观测是 float32，之前初始化保持和退出保持直接调用 `crisp_py.Pose.to_ros_msg()`，
该函数把 NumPy 值直接赋到 ROS double 字段，在此环境中产生了错误的坐标。
遥操从 `Pose.from_ros_msg()` 取得的位置默认是 float64，因此不走这个 float32 路径。
这证明此前客户端确实可能发布接近原点的保持目标，不能仅凭客户端动作数组声称 ROS 目标正确。

`ActionExecutor._publish_pose()` 已改为直接构造 `PoseStamped`，所有位置与四元数字段显式转换
为 Python `float`。真实 ROS 下 float32/float64 输入均已验证往返无误。
此前隔离夹具在初始化保持后突然移到原点，也与这个错误一致；修复后的隔离测试初始位置保持正常。

动作测试增加独立 ROS 错误监控和中断处理：当前硬件/控制器 ERROR、客户端失败、超时或
Ctrl+C 会调用已有 Web 服务停止整个 robot 驱动，防止客户端退出后底层仍持续自动恢复。
这只用于受监督的测试入口；正常推理入口的故障监控仍需进一步完善。

## 在 4090 的 tmux test 中重测

确认用户已连接该会话后，倒计时 10 秒，在姿态、夹爪保持不变的条件下执行基座 Z 轴
上移 3 mm 并返回，10 Hz、每程 10 步，单程约 1 秒。45 项单元测试、实际 ROS 序列化检查、
隔离 ROS 往返与监控的正常/故障/中断三种路径均先通过。

真机结果仍为失败：160 步已发布，实测未出现明显上移，上移阶段最大误差 3.035 mm，
返回阶段最大误差 0.137 mm。这一轮采样与之后 3 秒的驱动日志没有 ERROR。
失败判定后脚本停止了整个 robot 驱动，Web 服务与进程检查均确认已停止，没有自动重试或恢复。
因此序列化错误虽已修复，仍不能认定真机动作执行正常。

结果位于 4090：
`.runtime/pku_franka_collect_objects/results/tmux_test_serialization_fixed_20261023/`。

## 随后的重复启动报错

用户在同一 tmux 会话重跑后，实际错误是 `Missing/stale ROS observations after 10.0s`，
缺少位姿、关节与夹爪状态。上一轮因失败停止了 robot 驱动，重复启动时驱动尚未恢复，
因此这一次在首份实时观测处超时，没有下发动作。
同名输出目录导致旧 `report.json`/`client.log` 被覆盖，原 `client_report.json` 仍保留，
两轮终端历史另存为该目录的 `tmux_history.txt`。

已添加启动服务检查、原始错误摘要转发和重复输出目录隔离，48 项单元测试通过。
随后启动基础 robot 驱动，保持运动控制器 inactive，只做观测与干跑验证；16 轮干跑通过、
执行步数为 0。该干跑结果保存到原输出目录下的 `run_20261023_144635_673508/`。

## 按用户要求扩大至 100 mm

先确认驱动已启动、实时观测正常、无其他目标发布者，运动控制器 inactive。
实际参数为 `stop_commands=false`、`filter.target_pose=0.1`、Z 刚度 400 N/m、
`use_friction=false`。依用户明确要求，将脚本最大行程改为 100 mm；长行程拆分多轮请求，
每步最多 1 mm、10 Hz，上移和返回各 10 块，单程至少 10 秒。52 项测试通过后，在已连接的
4090 `tmux test` 会话倒计时启动，沿基座 Z 轴执行，姿态和夹爪目标固定。

本轮完成 34 次请求、340 步目标发布，机械臂实际运动：最大上移 83.715 mm；
上移保持阶段最大位置误差 16.617 mm，返回保持阶段最大位置误差 9.000 mm。
最终相对初始位置约 `[-3.651, -0.522, +8.201]` mm，最大姿态误差约 0.0366 rad。
这证明当前动作链路能驱动真机，仍存在明显跟踪偏差；不能用小幅测试的静止结果认定链路不通。
阻抗与摩擦等因素可能影响跟踪，需要单独验证，未擅自调整控制器参数。

由于位置精度未达到原来的 1.5 mm 容差，脚本仍报告 failed 并停止 robot 驱动。
记录位于 `.runtime/pku_franka_collect_objects/results/tmux_test_z100mm_20261023/`。
