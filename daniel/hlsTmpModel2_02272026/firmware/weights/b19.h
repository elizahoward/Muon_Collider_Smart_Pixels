//Numpy array shape [42]
//Min -0.250000000000
//Max 0.375000000000
//Number of zeros 21

#ifndef B19_H_
#define B19_H_

#ifndef __SYNTHESIS__
// global extern pointer only - actual array allocated in myproject_test.cpp
extern bias19_t b19[42];
#else
bias19_t b19[42] = {0.000, -0.125, 0.000, -0.125, 0.000, 0.000, 0.000, 0.000, -0.125, 0.000, 0.000, 0.375, 0.000, 0.000, -0.250, 0.000, 0.125, -0.125, 0.000, 0.000, 0.000, 0.250, -0.125, 0.000, 0.250, -0.125, -0.250, -0.125, 0.000, -0.125, 0.125, -0.125, -0.125, 0.000, 0.000, 0.000, -0.125, 0.250, -0.125, -0.125, 0.000, 0.000};

#endif

#endif
