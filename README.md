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
