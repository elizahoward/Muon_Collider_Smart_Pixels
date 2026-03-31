//Numpy array shape [10]
//Min -0.625000000000
//Max 0.625000000000
//Number of zeros 0

#ifndef B2_H_
#define B2_H_

#ifndef __SYNTHESIS__
// global extern pointer only - actual array allocated in myproject_test.cpp
extern bias2_t b2[10];
#else
bias2_t b2[10] = {0.375, 0.375, 0.125, -0.250, -0.625, -0.250, 0.125, 0.250, -0.375, 0.625};

#endif

#endif
