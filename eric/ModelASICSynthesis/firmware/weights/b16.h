//Numpy array shape [4]
//Min -0.250000000000
//Max 0.500000000000
//Number of zeros 0

#ifndef B16_H_
#define B16_H_

#ifndef __SYNTHESIS__
bias16_t b16[4];
#else
bias16_t b16[4] = {-0.125, -0.250, 0.500, 0.125};
#endif

#endif
