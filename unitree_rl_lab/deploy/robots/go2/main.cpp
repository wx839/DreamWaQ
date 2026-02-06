#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;

void init_fsm_state()
{
    //lowcmd订阅者 
    auto lowcmd_sub = std::make_shared<unitree::robot::go2::subscription::LowCmd>();
    usleep(0.2 * 1e6);//给DDS通道初始化留出时间 
    if(!lowcmd_sub->isTimeout()) //如果订阅者收到数据（未超时） 说明已有其他进程在使用lowcmd通道 
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
        // exit(0);
    }
    //创建命令发布器 
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    //等待连接机器人 
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     Go2 Controller \n";

    // Unitree DDS Config 初始化DDS通信 
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());
    //机器人连接 
    init_fsm_state();

    // Initialize FSM  
    // make_unique是独占型指针 
    // 创建 FSM，传入配置。这将触发 CtrlFSM 构造函数 -> State_RLBase 构造函数 -> OrtRunner 构造函数 
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    std::cout << "Press [L2 + A] to enter FixStand mode.\n";
    std::cout << "And then press [Start] to start controlling the robot.\n";

    while (true)
    {
        sleep(1);
    }
    
    return 0;
}

