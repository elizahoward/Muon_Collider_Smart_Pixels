//Numpy array shape [4]
//Min -0.101562500000
//Max 0.539062500000
//Number of zeros 0

#ifndef B16_H_
#define B16_H_

#ifndef __SYNTHESIS__
bias16_t b16[4];
#else
bias16_t b16[4] = {0.5390625, -0.0234375, -0.1015625, -0.0312500};
#endif

#endif
