//Numpy array shape [10]
//Min -0.500000000000
//Max 0.625000000000
//Number of zeros 0

#ifndef B3_H_
#define B3_H_

#ifndef __SYNTHESIS__
// global extern pointer only - actual array allocated in myproject_test.cpp
extern bias3_t b3[10];
#else
bias3_t b3[10] = {0.375, -0.375, -0.375, 0.125, -0.375, 0.500, 0.625, 0.375, 0.375, -0.500};

#endif

#endif
