#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/common/time/time_tool.hpp>
#include <unitree/common/thread/thread.hpp>


using namespace unitree::common;
using namespace unitree::robot;

#define TOPIC_LOWCMD "rt/lowcmd"
#define TOPIC_LOWSTATE "rt/lowstate"

constexpr double PosStopF = (2.146E+9f);
constexpr double VelStopF = (16000.0f);

class Custom
{
public:
    Custom(){};
    ~Custom(){};
    void Init();


private:
    void InitLowCmd();
    void LowStateMessageHandler(const void *messages);
    void LowCmdWrite();

    void my_controller();


private:
    
    double dt = 0.001;
    double running_time = 0.0;

    // --- Internal state arrays ---
    double q[12]{};
    double dq[12]{};
    double quaternion[4]{};
    double gyroscope[3]{};
    double accelerometer[3]{};

    // --- Reference arrays ---
    double qref[12]{};
    double dqref[12]{};
    double kp[12]{};
    double kd[12]{};
    double tau[12]{};

    unitree_go::msg::dds_::LowCmd_ low_cmd{};
    unitree_go::msg::dds_::LowState_ low_state{};

    /*publisher*/
    ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_> lowcmd_publisher;
    /*subscriber*/
    ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> lowstate_subscriber;

    /*LowCmd write thread*/
    ThreadPtr lowCmdWriteThreadPtr;
};

#include "my_controller.cpp"


uint32_t crc32_core(uint32_t *ptr, uint32_t len)
{
    unsigned int xbit = 0;
    unsigned int data = 0;
    unsigned int CRC32 = 0xFFFFFFFF;
    const unsigned int dwPolynomial = 0x04c11db7;

    for (unsigned int i = 0; i < len; i++)
    {
        xbit = 1 << 31;
        data = ptr[i];
        for (unsigned int bits = 0; bits < 32; bits++)
        {
            if (CRC32 & 0x80000000)
            {
                CRC32 <<= 1;
                CRC32 ^= dwPolynomial;
            }
            else
            {
                CRC32 <<= 1;
            }

            if (data & xbit)
                CRC32 ^= dwPolynomial;
            xbit >>= 1;
        }
    }
    return CRC32;
}

void Custom::Init()
{
    InitLowCmd();
    lowcmd_publisher.reset(new ChannelPublisher<unitree_go::msg::dds_::LowCmd_>(TOPIC_LOWCMD));
    lowcmd_publisher->InitChannel();

    lowstate_subscriber.reset(new ChannelSubscriber<unitree_go::msg::dds_::LowState_>(TOPIC_LOWSTATE));
    lowstate_subscriber->InitChannel(std::bind(&Custom::LowStateMessageHandler, this, std::placeholders::_1), 1);

    lowCmdWriteThreadPtr = CreateRecurrentThreadEx("writebasiccmd", UT_CPU_ID_NONE, int(dt * 1000000), &Custom::LowCmdWrite, this);
}

void Custom::InitLowCmd()
{
    low_cmd.head()[0] = 0xFE;
    low_cmd.head()[1] = 0xEF;
    low_cmd.level_flag() = 0xFF;
    low_cmd.gpio() = 0;

    for (int i = 0; i < 20; i++)
    {
        low_cmd.motor_cmd()[i].mode() = (0x01);
        low_cmd.motor_cmd()[i].q() = (PosStopF);
        low_cmd.motor_cmd()[i].kp() = (0);
        low_cmd.motor_cmd()[i].dq() = (VelStopF);
        low_cmd.motor_cmd()[i].kd() = (0);
        low_cmd.motor_cmd()[i].tau() = (0);
    }
}


void Custom::LowStateMessageHandler(const void *message)
{
    low_state = *(unitree_go::msg::dds_::LowState_ *)message;

    for (int i = 0; i < 12; i++)
    {
        q[i] = low_state.motor_state()[i].q();
        dq[i] = low_state.motor_state()[i].dq();
    }
    for (int i = 0; i < 4; i++)
        quaternion[i] = low_state.imu_state().quaternion()[i];
    for (int i = 0; i < 3; i++)
    {
        gyroscope[i] = low_state.imu_state().gyroscope()[i];
        accelerometer[i] = low_state.imu_state().accelerometer()[i];
    }

    // for (int i = 0; i < 12; i++)
    // {
    //     std::cout << "motor state[" << i << "] q = " << q[i]
    //               << ", dq = " << dq[i] << std::endl;
    // }
}


void Custom::LowCmdWrite()
{
    running_time += dt;

    my_controller();

    for (int i = 0; i < 12; i++)
    {
        low_cmd.motor_cmd()[i].q() = qref[i];
        low_cmd.motor_cmd()[i].dq() = dqref[i];
        low_cmd.motor_cmd()[i].kp() = kp[i];
        low_cmd.motor_cmd()[i].kd() = kd[i];
        low_cmd.motor_cmd()[i].tau() = tau[i];
    }

    low_cmd.crc() = crc32_core((uint32_t *)&low_cmd, (sizeof(unitree_go::msg::dds_::LowCmd_) >> 2) - 1);
    lowcmd_publisher->Write(low_cmd);
}

int main(int argc, const char **argv)
{
    if (argc < 2)
    {
        ChannelFactory::Instance()->Init(1, "lo");
    }
    else
    {
        ChannelFactory::Instance()->Init(0, argv[1]);
    }
    std::cout << "Press enter to start **\n";
    std::cin.get();
    Custom custom;
    custom.Init();

    while (1)
    {
        sleep(10);
    }
    return 0;
}

