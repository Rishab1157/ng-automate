#!/usr/bin/env bash
# Live view for the test sandbox, started before the agent server:
#   Xvfb (virtual display $DISPLAY) + fluxbox (window manager) so headed browsers have a screen,
#   two x11vnc servers on 127.0.0.1 only (5900 interactive, 5901 view-only),
#   websockify on 0.0.0.0:8001: the only listener the host reaches (DockerWorkspace(extra_ports=True) publishes
#   8001 on the host's 127.0.0.1). It serves the noVNC files and lets a WebSocket through only with a token that
#   the API wrote into $TOKENS, each token mapped to one of the two VNC servers.
# Then the agent server ("$@") runs as the main process under tini. When the display cannot start, DISPLAY is
# unset (so browsers run headless) and the agent server starts anyway.
set -u

export DISPLAY="${DISPLAY:-:99}"
SCREEN="${NGAUTO_SCREEN:-1600x900x24}"
VNC_PORT="${NGAUTO_VNC_PORT:-5900}"            # interactive VNC, 127.0.0.1 only
VNC_VIEW_PORT="${NGAUTO_VNC_VIEW_PORT:-5901}"  # view-only VNC, 127.0.0.1 only
WS_PORT="${NGAUTO_WS_PORT:-8001}"              # websockify + noVNC files
NOVNC_DIR=/usr/share/novnc
STATE_DIR=/tmp/ngauto-live-view
TOKENS="$STATE_DIR/tokens"                     # "<token>: 127.0.0.1:<port>" lines, appended by the API
READY="$STATE_DIR/ready"                       # exists once the display stack is up; the API checks it

# Run "$@" in its own session and start it again whenever it exits. The launching subshell exits at once, so
# the loop is re-parented to tini (PID 1), which reaps it.
supervise() {
  local name=$1
  shift
  ( setsid bash -c 'while :; do "$@"; echo "[ngauto] exited with $?, restarting" >&2; sleep 1; done' "$name" "$@" \
      >>"$STATE_DIR/$name.log" 2>&1 </dev/null & )
}

start_live_view() {
  mkdir -p "$STATE_DIR" && chmod 700 "$STATE_DIR" || return 1
  rm -f "$READY"
  # Empty until the API writes a token: websockify refuses every connection until then.
  : >"$TOKENS" && chmod 600 "$TOKENS" || return 1
  local number="${DISPLAY#:}"
  rm -f "/tmp/.X${number}-lock" "/tmp/.X11-unix/X${number}"

  supervise xvfb Xvfb "$DISPLAY" -screen 0 "$SCREEN" -nolisten tcp -dpi 96 +extension RANDR
  local _
  for _ in $(seq 50); do xdpyinfo >/dev/null 2>&1 && break; sleep 0.2; done
  xdpyinfo >/dev/null 2>&1 || { echo "[ngauto] Xvfb did not start" >&2; return 1; }

  supervise wm fluxbox
  supervise vnc x11vnc -display "$DISPLAY" -rfbport "$VNC_PORT" -localhost -forever -shared -nopw -quiet -xkb
  supervise vnc-view x11vnc -display "$DISPLAY" -rfbport "$VNC_VIEW_PORT" -localhost -forever -shared -nopw -quiet \
    -viewonly
  # --file-only: noVNC's files but no folder listings.
  supervise websockify websockify --web "$NOVNC_DIR" --file-only \
    --token-plugin TokenFile --token-source "$TOKENS" "0.0.0.0:$WS_PORT"
  : >"$READY"
}

if [ "${NGAUTO_LIVE_VIEW:-1}" != "1" ]; then
  unset DISPLAY  # no display: browsers run headless
elif ! start_live_view; then
  echo "[ngauto] live view unavailable; the agent server starts anyway, browsers run headless" >&2
  unset DISPLAY
fi

exec "$@"
