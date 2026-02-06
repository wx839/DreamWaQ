#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) //State_RLBase 构造函数 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        //这里是加载deploy.yaml的路径 
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    //创建 OrtRunner 实例，传入 ONNX 模型路径 
    //这里可以传入一个policy_dir / "exported" 的路径，然后进去自己指定.onnx 
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx"); 

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

//rl过程 - run() 由 FSM 线程调用，只负责将缓存的 action 发送给电机 
void State_RLBase::run()
{
    // 使用互斥锁安全地读取缓存的 actions 
    std::vector<float> action; 
    { 
        std::lock_guard<std::mutex> lock(env_mutex_); 
        action = cached_actions_; 
    } 
     
    // 检查 action 是否有效 
    if (action.empty()) { 
        return;  // 如果还没有有效的 action，跳过本次 
    } 
     
    for(size_t i(0); i < env->robot->data.joint_ids_map.size() && i < action.size(); i++) { 
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}
// void State_RLBase::run()
// {
//     // FSM tick dt（你已有 or 自己算）
//     double dt = this->fsm_dt_;   // 例如 0.002 / 0.005 / 0.01
//     time_acc_ += dt;

//     // ------------------------------------------------
//     // 1. 只在 policy_dt 到达时，才推进 RL
//     // ------------------------------------------------
//     if (time_acc_ >= policy_dt_)
//     {
//         time_acc_ -= policy_dt_;

//         // ---------- (A) 初始化 history（只做一次） ----------
//         if (!policy_initialized_)
//         {
//             // 让 env 先读取一次真实 LowState
//             env->sync_from_robot();   // ← 你 env 里等价接口
//             env->compute_observations();

//             // 用当前 obs 填满 history=5
//             env->observation_manager->reset_history_with_current();

//             // 计算一次 policy，作为初始 action
//             env->action_manager->compute();
//             last_action_ = env->action_manager->processed_actions();

//             policy_initialized_ = true;
//         }
//         else
//         {
//             // ---------- (B) 正常 RL step ----------
//             env->sync_from_robot();        // 读 LowState
//             env->compute_observations();   // obs + history 滚动
//             env->action_manager->compute();
//             last_action_ = env->action_manager->processed_actions();
//         }
//     }

//     // ------------------------------------------------
//     // 2. 每个 FSM tick 都下发 action（ZOH）
//     // ------------------------------------------------
//     for (int i = 0; i < env->robot->data.joint_ids_map.size(); i++)
//     {
//         lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q()
//             = last_action_[i];
//     }
// }
