# SO-101 Imitation Learning: ACT Policy on Edge Hardware

End-to-end imitation learning on the open-source **SO-101** robotic arm: teleoperated data collection, training an **ACT (Action Chunking with Transformers)** policy with [LeRobot](https://github.com/huggingface/lerobot), and deploying it on a **Synaptics Astra** edge board with a Torq NPU.

Built as the Synaptics-sponsored capstone at UC Irvine (MS, Embedded & Cyber-Physical Systems).

> **Demo:** _add a GIF or video link here (e.g. `media/demo.gif`)_

## Highlights

| | |
|---|---|
| Robot | SO-101 leader/follower arm pair |
| Policy | ACT (imitation learning, not a vision-language-action model) |
| Dataset | 200 teleoperated demonstrations |
| Result | **80% task success** on the real arm |
| Training | Linux workstation, NVIDIA RTX 4060 |
| Deployment | Synaptics Astra board, Torq NPU |

## Pipeline

```
 leader arm ──teleop──▶ follower arm + cameras ──▶ LeRobot dataset (200 episodes)
                                                         │
                                                         ▼
                                              ACT training (RTX 4060)
                                                         │
                                                         ▼
                                        export / quantize for the NPU
                                                         │
                                                         ▼
                                  Synaptics Astra board ──▶ SO-101 follower
```

1. **Data collection.** A human drives the leader arm while the follower arm and cameras are recorded with `lerobot-record`. Collection ran on a Windows laptop with no GPU.
2. **Training.** The dataset was moved to a Linux tower and an ACT policy trained with `lerobot-train`.
3. **Evaluation.** The policy was run on the real arm, giving 80% success on the task.
4. **Edge deployment.** The trained policy is deployed to a Synaptics Astra board, using its Torq NPU for inference, following Synaptics' [SO-101 vision robotic arm tutorial](https://developer.synaptics.com/blog/vision-robotic-arm).

## Task

_Describe the task the arm performs (objects, workspace layout, success criterion, how the 80% was measured, number of trials)._

## Results

| Metric | Value |
|---|---|
| Demonstrations | 200 |
| Task success rate | 80% |
| Trials | _fill in_ |
| Inference latency on board | _fill in_ |

_Add training-loss curves, failure-mode notes, and a short comparison of host vs. on-board inference here._

## Hardware

- SO-101 leader + follower arms (Feetech servos)
- USB cameras
- Windows laptop (data collection, ADB link to the board)
- Linux tower, RTX 4060 (training)
- Synaptics Astra board with Torq NPU (deployment)

## Reproducing

```bash
# 1. environment
python -m venv lerobot_env && source lerobot_env/bin/activate
pip install lerobot

# 2. calibrate and teleoperate
lerobot-calibrate ...
lerobot-teleoperate ...

# 3. record demonstrations
lerobot-record ...

# 4. train ACT
lerobot-train --policy.type=act --dataset.repo_id=<your-hf-dataset> ...
```

_Replace the `...` with the exact arguments you used (ports, camera indices, dataset id, steps, batch size)._

## Roadmap

- Voice-triggered tasks: a small language model on the board CPU launches the ACT policy on the NPU
- Pill-dispensing demo: after a voice-triggered dispense, the arm moves the medication to a tray
- Water-pouring loop: the arm checks whether a glass is empty and pours, with a 3D-printed jug and glass

## Acknowledgements

- [LeRobot](https://github.com/huggingface/lerobot) by Hugging Face
- [SO-101](https://github.com/TheRobotStudio/SO-ARM100) arm design by The Robot Studio
- Synaptics, for sponsoring the capstone and providing the hardware
- Prof. Fadi Kurdahi and the CECS lab, UC Irvine
