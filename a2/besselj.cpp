
#include <cmath>
#include "besselj.h"
#include "utils.h"

BesselJ::BesselJ(int radius) {
	int field_size = DIMENSION * DIMENSION;

    //Do this in a somewhat stupid way
    re = new float[field_size];
    w = 0.0;
    for(int i=0; i<DIMENSION; ++i) {
        for(int j=0; j<DIMENSION; ++j) {
            float ii = ((i + DIMENSION/2) % DIMENSION) - DIMENSION/2;
            float jj = ((j + DIMENSION/2) % DIMENSION) - DIMENSION/2;
            
            float r = sqrt(ii*ii + jj*jj) - radius;
            float v = 1.0 / (1.0 + exp(LOG_RES * r));
            
            w += v;
            re[i*DIMENSION+j] = v;
        }
    }
    
    im = new float[field_size];
    fft2(1, LOG_RES, re, im);
}