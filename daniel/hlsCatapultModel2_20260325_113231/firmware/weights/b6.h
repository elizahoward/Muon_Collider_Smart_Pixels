//Numpy array shape [1]
//Min 0.000000000000
//Max 0.000000000000
//Number of zeros 1

#ifndef B6_H_
#define B6_H_

#ifndef __SYNTHESIS__
// global extern pointer only - actual array allocated in myproject_test.cpp
extern bias6_t b6[1];
#else
bias6_t b6[1] = {0.000};

#endif

#endif
