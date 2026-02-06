# DreamWaQ

A reinforcement learning framework for Unitree robots based on Isaac Lab and RSL-RL, with support for sim-to-sim (MuJoCo) and sim-to-real deployment.

---

## Table of Contents

- [Installation](#installation)
  - [Isaac Lab Environment Setup](#isaac-lab-environment-setup)
  - [Project Configuration](#project-configuration)
  - [Robot Description Files](#robot-description-files)
  - [RSL-RL Dependencies](#rsl-rl-dependencies)
- [Training and Evaluation](#training-and-evaluation)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
- [Sim-to-Sim Deployment (MuJoCo)](#sim-to-sim-deployment-mujoco)
  - [Python Implementation](#python-implementation)
  - [C++ Implementation](#c-implementation)
- [Sim-to-Real Deployment](#sim-to-real-deployment)

---

## Installation

### Isaac Lab Environment Setup

Create and configure the Isaac Lab environment following the official documentation:

```bash
# Reference: https://docs.robotsfan.com/isaaclab/source/setup/installation/pip_installation.html
conda create -n isaaclab python=3.11
conda activate isaaclab
```

### Project Configuration

Clone the repository and install dependencies:

```bash
git clone https://github.com/wx839/DreamWaQ.git
cd DreamWaQ/unitree_rl_lab
./unitree_rl_lab.sh -i  # Install unitree_rl_lab dependencies
```

### Robot Description Files

Download Unitree robot description files. You need to configure the corresponding robot's `UnitreeUrdfFileCfg` in the `unitree.py` file.

#### Method 1: Using USD Files

```bash
# Download Unitree USD files (maintaining folder structure)
git clone https://huggingface.co/datasets/unitreerobotics/unitree_model
```

Configure `UNITREE_MODEL_DIR` in `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`:

```python
UNITREE_MODEL_DIR = "/home/user/projects/unitree_usd"
```

#### Method 2: Using URDF Files (Recommended)

**Note:** Only for Isaac Sim >= 5.0

```bash
# Download Unitree robot URDF files
git clone https://github.com/unitreerobotics/unitree_ros.git
```

Configure `UNITREE_ROS_DIR` in `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`:

```python
UNITREE_ROS_DIR = "/home/user/projects/unitree_ros/unitree_ros"
```

**Optional:** Modify `robot_cfg.spawn` if you want to use URDF files.

### RSL-RL Dependencies

Install the RSL-RL reinforcement learning library:

```bash
cd DreamWaQ/rsl_rl
pip install -e .
```

---

## Training and Evaluation

### Model Training

Train the model using the following command. Results will be saved in `logs/rsl_rl/unitree_go2_velocity_dwaq/`:

```bash
cd DreamWaQ
./unitree_rl_lab/unitree_rl_lab.sh -t \
    --task Unitree-Go2-Velocity-DWAQ \
    --num_envs 4096 \
    --max_iterations 1000 \
    --headless
```

### Model Evaluation

Evaluate a trained model checkpoint:

```bash
./unitree_rl_lab/unitree_rl_lab.sh -p \
    --task Unitree-Go2-Velocity-DWAQ \
    --num_envs 10 \
    --checkpoint logs/rsl_rl/unitree_go2_velocity_dwaq/2026-01-16_18-25-16/model_20000.pt
```

---

## Sim-to-Sim Deployment (MuJoCo)

### Python Implementation

#### Step 1: Install System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y cmake g++ python3-dev
```

#### Step 2: Build and Install CycloneDDS C++ Library

```bash
cd ~
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=~/cyclonedds/install
cmake --build . --target install
```

#### Step 3: Set Environment Variables

```bash
export CYCLONEDDS_HOME=~/cyclonedds/install
export CMAKE_PREFIX_PATH=~/cyclonedds/install
```

#### Step 4: Install unitree_sdk2_python

```bash
cd ~/unitree_sdk2_python
pip3 install -e .
```

#### Step 5: Install Required Libraries

```bash
pip3 install mujoco pygame
```

#### Step 6: Testing

```bash
# Test MuJoCo simulator
cd ./unitree_mujoco/simulate_python
python3 ./unitree_mujoco.py

# TestControl program
cd ./unitree_mujoco/simulate_python/test
python3 ./go2_ctrl.py
```

---

### C++ Implementation

#### Step 1: Install Dependencies

```bash
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev

cd ~
git clone https://github.com/unitreerobotics/unitree_sdk2.git
cd unitree_sdk2
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF  # Install to /usr/local directory
sudo make install
```

#### Step 2: Build Robot Controller

```bash
cd unitree_rl_lab/deploy/robots/g1_29dof  # or other robot models
mkdir build && cd build
cmake .. && make
```

#### Step 3: Configure unitree_mujoco

Edit `/simulate/config.yaml` with the following settings:

- Set `robot` to `g1`
- Set `domain_id` to `0`
- Set `enable_elastic_hand` to `1`
- Set `use_joystick` to `1`

#### Step 4: Build unitree_mujoco

Download MuJoCo release from [https://github.com/google-deepmind/mujoco/releases](https://github.com/google-deepmind/mujoco/releases) and extract to `~/.mujoco` directory:

```bash
cd unitree_mujoco/simulate/
ln -s ~/.mujoco/mujoco-3.3.6 mujoco

cd unitree_mujoco/simulate
mkdir build && cd build
cmake ..
make -j4
```

#### Step 5: Run Sim-to-Sim

Open two terminal windows:

**Terminal 1 (MuJoCo Simulator):**
```bash
cd unitree_mujoco/simulate/build
./unitree_mujoco
# Or with custom robot and scene:
# ./unitree_mujoco -r go2 -s scene_terrain.xml
```

**Terminal 2 (Robot Controller):**
```bash
cd unitree_rl_lab/deploy/robots/g1_29dof/build
./g1_ctrl
```

---

## Sim-to-Real Deployment

Deploy the trained policy directly to the physical robot. **Ensure the on-board control program has been terminated before running.**

```bash
./g1_ctrl --network eth0  # Replace 'eth0' with your network interface name
```

---

## License

[Add your license information here]

## Acknowledgments

This project builds upon:
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)
- [Unitree Robotics](https://www.unitree.com/)

---

# 中文版 (Chinese Version)

## 安装 Isaac Lab

https://docs.robotsfan.com/isaaclab/source/setup/installation/pip_installation.html

conda create -n isaaclab python=3.11 (成功配置isaaclab环境)

## Sim 项目配置 

git clone https://github.com/wx839/DreamWaQ.git

conda activate isaaclab
cd DreamWaQ/unitree_rl_lab
./unitree_rl_lab.sh -i (配置 unitree_rl_lab 所需依赖)

Download unitree robot description files (配置机器人文件，修改unitree.py文件下对应机器人的UnitreeUrdfFileCfg)

  *Method 1: Using USD Files*
  - Download unitree usd files from [unitree_model](https://huggingface.co/datasets/unitreerobotics/unitree_model/tree/main), keeping folder structure
    ```bash
    git clone https://huggingface.co/datasets/unitreerobotics/unitree_model
    ```
  - Config `UNITREE_MODEL_DIR` in `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`.

    ```bash
    UNITREE_MODEL_DIR = "</home/user/projects/unitree_usd>"
    ```

  *Method 2: Using URDF Files [Recommended]* Only for Isaacsim >= 5.0
  -  Download unitree robot urdf files from [unitree_ros](https://github.com/unitreerobotics/unitree_ros)
      ```
      git clone https://github.com/unitreerobotics/unitree_ros.git
      ```
  - Config `UNITREE_ROS_DIR` in `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`.
    ```bash
    UNITREE_ROS_DIR = "</home/user/projects/unitree_ros/unitree_ros>"
    ```
  - [Optional]: change *robot_cfg.spawn* if you want to use urdf files


cd DreamWaQ/rsl_rl  (配置 rsl_rl 所需依赖)
pip install -e .

cd DreamWaQ
./unitree_rl_lab/unitree_rl_lab.sh -t \    (训练模型，结果保存在 logs/rsl_rl/unitree_go2_velocity_dwaq )
    --task Unitree-Go2-Velocity-DWAQ \
    --num_envs 4096 \
    --max_iterations 1000 \
    --headless

./unitree_rl_lab/unitree_rl_lab.sh -p --task Unitree-Go2-Velocity-DWAQ --num_envs 10 \
    --checkpoint logs/rsl_rl/unitree_go2_velocity_dwaq/2026-01-16_18-25-16/model_20000.pt


## Sim2sim mujoco配置（python）

1. 安装系统依赖
sudo apt-get update
sudo apt-get install -y cmake g++ python3-dev

2. 编译安装 cyclonedds C++ 库
cd ~
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=~/cyclonedds/install
cmake --build . --target install

3. 设置环境变量
export CYCLONEDDS_HOME=~/cyclonedds/install
export CMAKE_PREFIX_PATH=~/cyclonedds/install

4. 安装 unitree_sdk2_python
cd ~/unitree_sdk2_python
pip3 install -e .

5. 安装必要库
pip3 install mujoco
pip3 install pygame

6. 测试
cd ./unitree_mujoco/simulate_python
python3 ./unitree_mujoco.py

cd ./unitree_mujoco/simulate_python/test
python3 ./go2_ctrl.py


## Sim2sim mujoco配置（C++）

1. 安装依赖
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
cd ~
git clone https://github.com/unitreerobotics/unitree_sdk2.git
cd unitree_sdk2
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF # Install on the /usr/local directory
sudo make install

2. 编译 robot_controller
cd unitree_rl_lab/deploy/robots/g1_29dof # or other robots
mkdir build && cd build
cmake .. && make

3. 配置 unitree_mujoco
Set the `robot` at `/simulate/config.yaml` to g1
Set `domain_id` to 0
Set `enable_elastic_hand` to 1
Set `use_joystck` to 1

4. 编译 unitree_mujoco
Download the mujoco release, and extract it to the ~/.mujoco directory;
https://github.com/google-deepmind/mujoco/releases
cd unitree_mujoco/simulate/
ln -s ~/.mujoco/mujoco-3.3.6 mujoco
cd unitree_mujoco/simulate
mkdir build && cd build
cmake ..
make -j4

5. Sim2Sim
cd unitree_mujoco/simulate/build
./unitree_mujoco
(./unitree_mujoco -r go2 -s scene_terrain.xml)
cd unitree_rl_lab/deploy/robots/g1_29dof/build
./g1_ctrl

## Sim2Real
You can use this program to control the robot directly, but make sure the on-borad control program has been closed.
./g1_ctrl --network eth0 # eth0 is the network interface name.
