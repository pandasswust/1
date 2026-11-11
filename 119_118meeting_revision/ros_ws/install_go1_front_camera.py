#!/usr/bin/env python3
"""Install a forward depth camera on the Go1 description used by this workspace."""

from __future__ import annotations

import shutil
from pathlib import Path

GO1_XACRO = Path.home() / "unitree_ws/src/unitree_ros/robots/go1_description/xacro"
ROBOT_XACRO = GO1_XACRO / "robot.xacro"
DEPTH_XACRO = GO1_XACRO / "depthCamera.xacro"

FRONT_BLOCK = """
    <xacro:depthCamera camID="6" name="front">
        <origin rpy="0 0 0" xyz="0.32 0.0 0.02"/>
    </xacro:depthCamera>
"""
REAR_DOWN_BLOCK = """    <xacro:depthCamera camID="5" name="rearDown">
        <origin rpy="${PI} ${PI/2} 0" xyz="-0.0825 0.0125 -0.04365"/>
    </xacro:depthCamera>
"""
SENSOR_TAG = '            <sensor name="camera_${name}_camera" type="depth">\n'
SENSOR_POSE = "                <pose>0 0 0 ${PI} 0 0</pose>\n"


def backup_once(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".bak_front_camera")
    if not backup.exists():
        shutil.copy2(path, backup)


def main() -> None:
    if not ROBOT_XACRO.exists() or not DEPTH_XACRO.exists():
        raise SystemExit(f"Go1 xacro files not found under {GO1_XACRO}")

    robot_text = ROBOT_XACRO.read_text(encoding="utf-8")
    if 'name="front"' not in robot_text:
        if REAR_DOWN_BLOCK not in robot_text:
            raise SystemExit("Could not find the rearDown camera block in robot.xacro")
        backup_once(ROBOT_XACRO)
        robot_text = robot_text.replace(REAR_DOWN_BLOCK, REAR_DOWN_BLOCK + FRONT_BLOCK)
        ROBOT_XACRO.write_text(robot_text, encoding="utf-8")
        print(f"Added front camera to {ROBOT_XACRO}")
    else:
        print(f"Front camera already present in {ROBOT_XACRO}")

    depth_text = DEPTH_XACRO.read_text(encoding="utf-8")
    if "<pose>0 0 0 ${PI} 0 0</pose>" not in depth_text:
        if SENSOR_TAG not in depth_text:
            raise SystemExit("Could not find the depth camera sensor block in depthCamera.xacro")
        backup_once(DEPTH_XACRO)
        depth_text = depth_text.replace(SENSOR_TAG, SENSOR_TAG + SENSOR_POSE, 1)
        DEPTH_XACRO.write_text(depth_text, encoding="utf-8")
        print(f"Added sensor orientation pose to {DEPTH_XACRO}")
    else:
        print(f"Sensor orientation pose already present in {DEPTH_XACRO}")


if __name__ == "__main__":
    main()
