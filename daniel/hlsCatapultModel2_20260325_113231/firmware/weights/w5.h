//Numpy array shape [10, 1]
//Min -0.500000000000
//Max 0.625000000000
//Number of zeros 3

#ifndef W5_H_
#define W5_H_

#ifndef __SYNTHESIS__
// global extern pointer only - actual array allocated in myproject_test.cpp
extern weight5_t w5[10];
#else
weight5_t w5[10] = {0.375, -0.375, -0.375, 0.000, -0.500, 0.000, 0.000, 0.625, 0.125, -0.250};

#endif

#endif
