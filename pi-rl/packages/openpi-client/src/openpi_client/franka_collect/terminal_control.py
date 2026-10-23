"""TTY controls serviced by the same thread that owns ROS execution."""

import os
import select
import sys
import termios
import tty
from collections import deque

from openpi_client.franka_collect.reset import START_POSE


class TerminalControl:
    def __init__(self, reader, *, stream=None, output=None, instructions=None):
        self.reader = reader
        self.stream = sys.stdin if stream is None else stream
        self.output = sys.stdout if output is None else output
        self.instructions = instructions or "控制：p / 空格 暂停；暂停后 r 复位、c 继续、q 退出"
        self.enabled = False
        self.paused = False
        self.saved = None
        self.keys = deque()

    def __enter__(self):
        if self.stream.isatty():
            self.fd = self.stream.fileno()
            self.saved = termios.tcgetattr(self.fd)
            try:
                tty.setcbreak(self.fd)
                self.enabled = True
                print(
                    self.instructions,
                    file=self.output,
                    flush=True,
                )
            except BaseException:
                self.__exit__()
                raise
        return self

    def _key(self):
        if not self.enabled:
            return None
        if not self.keys and select.select([self.fd], [], [], 0)[0]:
            data = os.read(self.fd, 1024)
            if not data:
                raise EOFError("Terminal input closed")
            self.keys.extend(data.decode(errors="ignore").lower())
        return self.keys.popleft() if self.keys else None

    def poll(self):
        if self.paused:
            return True
        while (key := self._key()) is not None:
            if key in ("q", "\x04"):
                raise KeyboardInterrupt("User quit inference")
            if key in ("p", " "):
                self.paused = True
                return True
        return False

    def wait(self, executor):
        if not self.enabled or not self.paused:
            return
        self._notify_supervisor(True)
        executor.hold()
        print("已暂停：r 复位到采集起点；c 继续；q 退出", file=self.output, flush=True)
        while self.paused:
            key = self._key()
            if key in ("q", "\x04"):
                raise KeyboardInterrupt("User quit paused inference")
            if key == "c":
                self.paused = False
                self._notify_supervisor(False)
                print("继续：重新读取观测并请求动作", file=self.output, flush=True)
            elif key == "r":
                if not getattr(executor, "control_gripper", True):
                    print(
                        "复位需要夹爪控制，请去掉 --disable-gripper 后重试；当前保持暂停。",
                        file=self.output,
                        flush=True,
                    )
                    continue
                executor.reset_to_pose(START_POSE)
                print(
                    "复位完成，保持暂停：c 继续；r 再次复位；q 退出", file=self.output, flush=True
                )
            else:
                self.reader.spin_for(0.05)

    def __exit__(self, *_):
        if self.saved is not None:
            termios.tcsetattr(self.fd, termios.TCSANOW, self.saved)
        self.enabled = False

    def _notify_supervisor(self, paused):
        channel = os.environ.get("PI05_SUPERVISOR_FD")
        if channel:
            try:
                os.write(int(channel), b"p" if paused else b"c")
            except BrokenPipeError:
                pass
