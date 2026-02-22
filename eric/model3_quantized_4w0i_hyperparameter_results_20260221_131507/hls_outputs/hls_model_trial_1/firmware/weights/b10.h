//Numpy array shape [16]
//Min -0.125000000000
//Max 0.375000000000
//Number of zeros 5

#ifndef B10_H_
#define B10_H_

#ifndef __SYNTHESIS__
bias10_t b10[16];
#else
bias10_t b10[16] = {-0.125, 0.000, 0.000, 0.000, 0.125, 0.125, 0.250, 0.125, 0.125, 0.000, 0.000, 0.125, 0.125, -0.125, 0.125, 0.375};
#endif

#endif
