//Numpy array shape [10]
//Min 0.000000000000
//Max 0.000000000000
//Number of zeros 10

#ifndef B4_H_
#define B4_H_

#ifndef __SYNTHESIS__
// global extern pointer only - actual array allocated in myproject_test.cpp
extern bias4_t b4[10];
#else
bias4_t b4[10] = {0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000};

#endif

#endif
