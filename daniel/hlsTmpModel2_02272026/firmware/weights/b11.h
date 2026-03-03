//Numpy array shape [14]
//Min -0.218750000000
//Max 0.187500000000
//Number of zeros 2

#ifndef B11_H_
#define B11_H_

#ifndef __SYNTHESIS__
// global extern pointer only - actual array allocated in myproject_test.cpp
extern bias11_t b11[14];
#else
bias11_t b11[14] = {0.00000, -0.03125, 0.06250, -0.06250, -0.21875, -0.03125, 0.09375, 0.15625, 0.03125, -0.03125, 0.18750, 0.09375, 0.03125, 0.00000};

#endif

#endif
