# Rizon 4s Caricature Artist

A Flexiv Rizon 4s robotic arm that captures a person's face through a webcam, generates a caricature, and draws it on a paper pad mounted on an easel — using a marker pen held in a custom end-effector.

## Hardware

| Component | Spec |
|-----------|------|
| Robot arm | Flexiv Rizon 4s |
| End-effector | Custom pen gripper (ISO 9409-1-50-4-M6, Ø70 mm plate) |
| Pen bore | Ø14 mm (standard marker pen) |
| Pen tip offset | 150 mm from flange face along flange-Z |
| Canvas | 300 × 400 mm paper pad on easel |
| Camera | Any OpenCV-compatible webcam (default: device 0) |

## Software overview

```
main.py                        Entry point — wires components, starts state machine
config/config.yaml             All tunable parameters (IP, canvas geometry, speeds)
vision/
  camera.py                    OpenCV VideoCapture wrapper
  face_detector.py             MediaPipe Face Mesh → FaceLandmarks dataclass
  caricature.py                Exaggerate eyes / nose / mouth from landmarks
  stroke_generator.py          Convert landmarks to (u, v) canvas stroke paths
robot/
  pen_controller.py            Low-level Flexiv MoveL primitives (travel / draw / lift)
  stroke_executor.py           Iterate strokes, sequence pen-up / travel / pen-down / draw
  state_machine.py             Top-level state machine (see diagram below)
utils/
  geometry.py                  CanvasTransform: canvas ↔ robot world coords
```

## State machine

```
IDLE ──start()──► HOMING ──► WAIT_FOR_FACE
                                  │
                       face N consecutive frames
                                  │
                                  ▼
                            CAPTURE_FACE ──► GENERATE_STROKES ──► APPROACH_CANVAS
                                                                         │
                                                                         ▼
                                                                      DRAWING
                                                                         │
                                                                    all strokes done
                                                                         │
                                                                         ▼
                                                                  RETURN_HOME ──► DONE
                                                                                   │
                                                                            4 s pause
                                                                                   │
                                                                                   ▼
                                                                           WAIT_FOR_FACE
```

On any unhandled exception the machine transitions to **ERROR** and halts.

## Installation

```bash
# 1. Install the Flexiv RDK Python wheel (from the SDK package)
pip install path/to/flexivrdk-*.whl

# 2. Install remaining dependencies
pip install -r requirements.txt
```

## Configuration

Edit `config/config.yaml` before running:

- `robot.ip` / `robot.local_ip` — match your network setup
- `canvas.position` — X/Y/Z of the canvas centre in the robot base frame (metres)
- `canvas.width` / `canvas.height` — physical paper size (metres)
- `pen.tip_offset[2]` — distance from flange face to pen tip (metres)
- `motion.*_speed` — tune for your physical setup

## Running

```bash
python main.py
# or with a custom config:
python main.py --config config/config.yaml --log-level DEBUG
```

Stand in front of the camera.  Once the robot detects a stable face for
`camera.face_hold_frames` consecutive frames it will capture the image,
plan the strokes, and draw.

## Caricature parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `caricature_eye_scale` | 1.45 | Eye contours scaled 45 % larger from each pupil |
| `caricature_nose_scale` | 1.20 | Nose tip scaled 20 % larger from nasal root |
| `caricature_mouth_scale` | 1.25 | Lips stretched 25 % wider from lip centroid |

## End-effector CAD

The Blender script for the pen-gripper end-effector is located at
`meshes/generate_pen_gripper.py` (in the `cs225a` simulation project).
Open Blender → Scripting workspace → Run Script, or:

```bash
blender --background --python generate_pen_gripper.py
```

Output: `meshes/pen_gripper.blend`

## License

MIT
