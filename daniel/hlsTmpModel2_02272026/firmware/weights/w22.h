//Numpy array shape [42, 1]
//Min -0.500000000000
//Max 0.500000000000
//Number of zeros 11

#ifndef W22_H_
#define W22_H_

#ifndef __SYNTHESIS__
// global extern pointer only - actual array allocated in myproject_test.cpp
extern weight22_t w22[42];
#else
weight22_t w22[42] = {0.125, 0.250, 0.000, -0.125, 0.000, 0.125, 0.000, 0.000, 0.000, 0.000, 0.000, 0.250, 0.000, 0.125, -0.500, 0.125, 0.125, -0.250, 0.125, 0.125, 0.000, 0.125, 0.125, 0.125, 0.500, 0.125, -0.250, -0.125, 0.125, -0.250, -0.375, -0.125, -0.250, -0.125, 0.125, 0.000, -0.375, 0.375, -0.125, -0.375, 0.000, 0.125};

#endif

#endif
