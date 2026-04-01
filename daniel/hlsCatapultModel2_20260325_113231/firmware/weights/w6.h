//Numpy array shape [10, 1]
//Min -0.750000000000
//Max 0.375000000000
//Number of zeros 2

#ifndef W6_H_
#define W6_H_

#ifndef __SYNTHESIS__
// global extern pointer only - actual array allocated in myproject_test.cpp
extern weight6_t w6[10];
#else
weight6_t w6[10] = {-0.750, -0.125, 0.000, -0.125, -0.125, -0.375, -0.250, 0.000, -0.750, 0.375};

#endif

#endif
