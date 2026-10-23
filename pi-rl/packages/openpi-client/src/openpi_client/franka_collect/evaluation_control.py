"""Single-key judgments and explicit retry commands for repeated robot evaluation."""

import termios

from openpi_client.franka_collect.terminal_control import TerminalControl


class EvaluationControl(TerminalControl):
    def __init__(self, reader, **kwargs):
        super().__init__(
            reader,
            instructions=(
                "评测：空格开始每轮；s 成功；f 失败；r 复位；运行中 p/空格暂停；c 继续；q 退出"
            ),
            **kwargs,
        )

    def discard_pending_input(self):
        """A new attempt/reset prompt requires input entered after the transition."""
        self.keys.clear()
        if self.enabled:
            termios.tcflush(self.fd, termios.TCIFLUSH)

    def next_event(self, phase):
        while (key := self._key()) is not None:
            if key in ("q", "\x04"):
                return "quit"
            if phase == "start":
                if key == " ":
                    return "start"
                continue
            if phase == "reset":
                if key == "r":
                    return "reset"
                continue
            if key in ("s", "f"):
                return "success" if key == "s" else "failure"
            if phase == "running" and key in ("p", " "):
                return "pause"
            if phase == "paused" and key == "c":
                return "resume"
            if key == "r":
                print("先按 s 判成功或 f 判失败，再按 r 复位/重试。", file=self.output, flush=True)
        return None
