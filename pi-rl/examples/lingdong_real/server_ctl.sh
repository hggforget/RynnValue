#!/bin/bash
# pi05 websocket inference server in tmux session "pi05-server" (no autostart on boot).
# usage: server_ctl.sh start|stop|status|log|attach      (detach from tmux: Ctrl+b then d)
HERE="$(cd "$(dirname "$0")" && pwd)"; S=pi05-server; PORT="${PORT:-8000}"
LOG=/mnt/hdc/lingdong_server/server.log; mkdir -p "$(dirname "$LOG")"
case "${1:-status}" in
  start)
    if tmux has-session -t $S 2>/dev/null; then echo "already running (tmux $S)"; exit 0; fi
    [ -f "$LOG" ] && mv "$LOG" "$LOG.prev"
    tmux new-session -d -s $S "PORT=$PORT $HERE/serve.sh 2>&1 | tee $LOG; echo; echo server exited, press Enter; read"
    echo "started in tmux session $S, port $PORT, log $LOG" ;;
  stop) tmux send-keys -t $S C-c 2>/dev/null; sleep 3; tmux kill-session -t $S 2>/dev/null && echo stopped || echo "not running" ;;
  status) tmux has-session -t $S 2>/dev/null && echo "running (tmux $S)" || echo "not running"; ss -ltn | grep ":$PORT " || echo "port $PORT not listening" ;;
  log) tail -n 40 "$LOG" ;;
  attach) exec tmux attach -t $S ;;
  *) echo "usage: $0 start|stop|status|log|attach"; exit 1 ;;
esac
