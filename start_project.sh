#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

HEADLESS=0
DEGRADED=0
CALIBRATE=0
DURATION=30.0
PARTICLES=500
OUTPUT_DIR=""

PERCEPTION_PID=""
SAPF_PID=""

usage() {
  cat <<INNER_EOF
Usage: ./start_project.sh [options]

One-click startup for the Go1 AHPFF + SAPF closed loop.

Options:
  --degraded          Enable overexposure + motion blur + 50% depth dropout (6-20 s).
  --headless          Start Gazebo without the GUI (no rqt_image_view or rviz).
  --calibrate         Recollect the 351-sample calibration set and retrain before running.
  --duration SECONDS  Length of the SAPF run (default: 30.0).
  --particles N       Number of particles (default: 500).
  --output DIR        Output directory (default: figures/go1_live or figures/go1_degraded_live).
  -h, --help          Show this help.
INNER_EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --degraded) DEGRADED=1; shift ;;
    --headless) HEADLESS=1; shift ;;
    --calibrate) CALIBRATE=1; shift ;;
    --duration) DURATION="$2"; shift 2 ;;
    --particles) PARTICLES="$2"; shift 2 ;;
    --output) OUTPUT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
  if [[ "$DEGRADED" = "1" ]]; then
    OUTPUT_DIR="$PROJECT_DIR/figures/go1_degraded_live"
  else
    OUTPUT_DIR="$PROJECT_DIR/figures/go1_live"
  fi
fi

cleanup() {
  if [[ -n "$PERCEPTION_PID" ]] && kill -0 "$PERCEPTION_PID" 2>/dev/null; then
    kill -INT "$PERCEPTION_PID" 2>/dev/null || true
    wait "$PERCEPTION_PID" 2>/dev/null || true
  fi
  if [[ -n "$SAPF_PID" ]] && kill -0 "$SAPF_PID" 2>/dev/null; then
    kill -INT "$SAPF_PID" 2>/dev/null || true
    wait "$SAPF_PID" 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT

if [[ ! -f /opt/ros/noetic/setup.bash ]]; then
  echo "ROS Noetic was not found at /opt/ros/noetic/setup.bash"
  exit 1
fi
if [[ ! -f "$HOME/unitree_ws/devel/setup.bash" ]]; then
  echo "The Unitree workspace was not found at $HOME/unitree_ws/devel/setup.bash"
  exit 1
fi

source /opt/ros/noetic/setup.bash
source "$HOME/unitree_ws/devel/setup.bash"

python3 - <<'PY'
import importlib.util
import sys

packages = ["numpy", "sklearn", "joblib", "cv2", "cv_bridge", "rospy", "matplotlib"]
missing = [name for name in packages if importlib.util.find_spec(name) is None]
if missing:
    print("Missing Python modules: " + ", ".join(missing))
    sys.exit(1)
PY

echo "Installing the Go1 forward camera if needed..."
python3 "$PROJECT_DIR/ros_ws/install_go1_front_camera.py"

NEED_CALIBRATION=0
if [[ "$CALIBRATE" = "1" || ! -f "$PROJECT_DIR/data/ahpff_go1_model.joblib" ]]; then
  NEED_CALIBRATION=1
fi

if rosnode list 2>/dev/null | grep -qx "/gazebo"; then
  echo "Gazebo is already running; reusing it."
  if rosnode list 2>/dev/null | grep -qx "/gazebo_gui"; then
    echo "Gazebo GUI is available."
  elif [[ "$HEADLESS" = "0" ]]; then
    echo "Warning: the running Gazebo has no GUI. Use ./start_project.sh --headless or restart Gazebo with the GUI."
  fi
else
  GUI_ARG="gui:=true"
  if [[ "$HEADLESS" = "1" || -z "${DISPLAY:-}" ]]; then
    GUI_ARG="gui:=false"
  fi
  echo "Starting Gazebo and the Go1 street scene..."
  setsid nohup roslaunch "$PROJECT_DIR/ros_ws/go1_street.launch" "$GUI_ARG" > "$LOG_DIR/gazebo.log" 2>&1 < /dev/null &
  echo $! > "$LOG_DIR/gazebo.pid"
  disown || true
  echo "Waiting 18 s for Gazebo and the Go1 controllers..."
  sleep 18
  if ! rosnode list 2>/dev/null | grep -qx "/gazebo"; then
    echo "Gazebo did not become ready. See $LOG_DIR/gazebo.log"
    tail -20 "$LOG_DIR/gazebo.log" || true
    exit 1
  fi
  echo "Gazebo is ready."
fi

if [[ "$NEED_CALIBRATION" = "1" ]]; then
  echo "Running calibration and training (351 samples, cluster readout)..."
  "$PROJECT_DIR/ros_ws/run_ahpff_calibration_go1.sh"
fi

if [[ "$HEADLESS" = "0" && -n "${DISPLAY:-}" ]]; then
  if command -v rqt_image_view >/dev/null 2>&1; then
    if ! pgrep -f "[r]qt_image_view" >/dev/null 2>&1; then
      echo "Opening rqt_image_view for /camera_front/color/image_rectified..."
      nohup rqt_image_view /camera_front/color/image_rectified > "$LOG_DIR/rqt_image_view.log" 2>&1 &
    fi
  fi
  if command -v rviz >/dev/null 2>&1; then
    if ! pgrep -x "rviz" >/dev/null 2>&1; then
      echo "Opening RViz for SAPF particles and trajectories..."
      nohup rosrun rviz rviz -d "$PROJECT_DIR/ros_ws/rviz/sapf_particles.rviz" > "$LOG_DIR/rviz.log" 2>&1 &
    fi
  fi
fi

mkdir -p "$OUTPUT_DIR"
PERCEPTION_ARGS=(
  _mode:=infer
  _camera_topic:=/camera_front/color/image_raw
  _depth_topic:=/camera_front/depth/image_raw
  _vertical_flip:=true
  _publish_rectified:=true
  _degradation_mode:=none
  _model_path:="$PROJECT_DIR/data/ahpff_go1_model.joblib"
  _prediction_path:="$PROJECT_DIR/data/ahpff_go1_predictions_live.npz"
)
if [[ "$DEGRADED" = "1" ]]; then
  PERCEPTION_ARGS+=(
    _degradation_mode:=mixed
    _degradation_start:=6.0
    _degradation_end:=20.0
    _degradation_strength:=1.0
    _preview_path:="$OUTPUT_DIR/degraded_preview.png"
  )
fi

echo "Starting AHPFF perception..."
python3 "$PROJECT_DIR/ros_ws/ahpff_perception_node.py" "${PERCEPTION_ARGS[@]}" > "$LOG_DIR/perception.log" 2>&1 &
PERCEPTION_PID=$!
sleep 2

echo "Starting SAPF for ${DURATION}s with ${PARTICLES} particles..."
python3 "$PROJECT_DIR/ros_ws/sapf_gazebo_node.py" \
  _scenario:=go1_street \
  _model_name:=go1_gazebo \
  _spawn_environment:=false \
  _publish_joint_commands:=true \
  _use_external_labels:=true \
  _duration:="$DURATION" \
  _particles:="$PARTICLES" \
  _observation_rate:=5.0 \
  _observation_outages:=8:12,18:22 \
  _output_dir:="$OUTPUT_DIR" &
SAPF_PID=$!
wait "$SAPF_PID"
SAPF_PID=""

if [[ -n "$PERCEPTION_PID" ]] && kill -0 "$PERCEPTION_PID" 2>/dev/null; then
  kill -INT "$PERCEPTION_PID" 2>/dev/null || true
  wait "$PERCEPTION_PID" 2>/dev/null || true
fi
PERCEPTION_PID=""

echo ""
echo "Run complete. Results:"
python3 - "$OUTPUT_DIR/sapf_gazebo_result.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print(f"Result file not found: {path}")
    raise SystemExit(0)
result = json.loads(path.read_text())
print(f"Semantic accuracy: {result['semantic_label_accuracy'] * 100:.1f}%")
for method in ("LO", "SPF", "LPF", "EKF", "SAPF", "SAPF-Oracle", "SAPF-Shuffled"):
    metrics = result["metrics"][method]
    axis = metrics["axis_rmse"]
    print(/
        f"{method:14s} X={axis['x']:.3f} Y={axis['y']:.3f} "
        f"Z={axis['z']:.3f} Overall={metrics['position_rmse']:.3f} "
        f"StairsZ={metrics['stairs_z_rmse']:.3f} Orient={metrics['orientation_rmse']:.3f}"
    )
print(f"Saved: {path.parent}")
PY

echo ""
echo "Gazebo, rqt_image_view, and RViz remain open for inspection."
echo "Rerun with: ./start_project.sh"
echo "Degraded run: ./start_project.sh --degraded"
