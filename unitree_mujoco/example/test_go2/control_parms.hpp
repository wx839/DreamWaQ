#pragma once

// Standing posture parameters
static const double stand_up_joint_pos[12] = {
    0.00571868, 0.608813, -1.21763,
   -0.00571868, 0.608813, -1.21763,
    0.00571868, 0.608813, -1.21763,
   -0.00571868, 0.608813, -1.21763
};

static const double stand_down_joint_pos[12] = {
    0.0473455, 1.22187, -2.44375,
   -0.0473455, 1.22187, -2.44375,
    0.0473455, 1.22187, -2.44375,
   -0.0473455, 1.22187, -2.44375
};

double phase = 0.0;

// You can easily extend this later:
//constexpr double KP_STAND = 50.0;
//constexpr double KD_STAND = 3.5;
