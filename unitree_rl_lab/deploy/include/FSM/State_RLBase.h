// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"
#include <mutex> 
#include <atomic> 

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);
    
    void enter()
    {
        // set gain
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[i].dq() = 0;
            lowcmd->msg_.motor_cmd()[i].tau() = 0;
        }

        env->robot->update();
        // Start policy thread
        policy_thread_running.store(true);      // 设置原子标志位，可以跑
        policy_thread = std::thread([this]{     // 创建新线程
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);       // deploy.yaml
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // Initialize timing
            auto sleepTill = clock::now() + dt;
            env->reset();     // 状态清零

            while (policy_thread_running.load())     // 检查线程 ture, false
            {
                {
                    std::lock_guard<std::mutex> lock(env_mutex_); 
                    env->step();
                    // 缓存 action 供 run() 使用 
                    cached_actions_ = env->action_manager->processed_actions(); 
                } 

                // Sleep
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            }
        });
    }

    void run();
    
    void exit()
    {
        policy_thread_running.store(false); 
        if (policy_thread.joinable()) {
            policy_thread.join();           // 杀死线程
        }
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    std::thread policy_thread;             // 线程句柄
    std::atomic<bool> policy_thread_running{false};
    
    // 用于线程间同步的互斥锁和缓存数据                        谁拿到了这把锁 mutex_，谁才能操作数据     
    mutable std::mutex env_mutex_;
    std::vector<float> cached_actions_;
};

REGISTER_FSM(State_RLBase)
