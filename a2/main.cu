
#include <iostream>
#include <algorithm>
#include <time.h>
#include "utils.h"
#include "besselj.h"
#include <cuda_runtime.h>


using namespace std;

//Precalculate multipliers for m,n
void initialize_MN(float* M_re, float* M_im, float* N_re, float* N_im, float inner_w, float outer_w){ 
    for(int i=0; i<DIMENSION; ++i) {
        for(int j=0; j<DIMENSION; ++j) {
            N_re[i*DIMENSION+j] = outer_w * (N_re[i*DIMENSION+j] - M_re[i*DIMENSION+j]);
            N_im[i*DIMENSION+j] = outer_w * (N_im[i*DIMENSION+j] - M_im[i*DIMENSION+j]);
            M_re[i*DIMENSION+j] *= inner_w;
            M_im[i*DIMENSION+j] *= inner_w;
        }
    }
}


float sigma(float x, float a, float alpha) {
    return (float)( 1.0 / (1.0 + exp(-4.0/alpha * (x - a))));
}

float sigma_2(float x, float a, float b) {
    return (float)( sigma(x, a, ALPHA_N) * (1.0 - sigma(x, b, ALPHA_N)));
}

float lerp(float a, float b, float t) {
    return (float)( (1.0-t)*a + t*b);
}

float S(float n,float m) {
    float alive = sigma(m, 0.5, ALPHA_M);
    return sigma_2(n, lerp(B1, D1, alive), lerp(B2, D2, alive));
}

void field_mult_original(float* a_r, float* a_i, float* b_r, float* b_i, float* c_r, float* c_i) {
    for(int i=0; i<DIMENSION; ++i) {
        // Host arrays
		float *Ar = &a_r[i*DIMENSION], *Ai = &a_i[i*DIMENSION];
        float *Br = &b_r[i*DIMENSION], *Bi = &b_i[i*DIMENSION];
		float *Cr = &c_r[i*DIMENSION], *Ci = &c_i[i*DIMENSION];

        // Original inner for-loop
			for(int j=0; j<DIMENSION; ++j) {
				float a = Ar[j];
				float b = Ai[j];
				float c = Br[j];
				float d = Bi[j];
				float t = a * (c + d);
				Cr[j] = t - d*(a+b);
				Ci[j] = t + c*(b-a);
			}
    }
}

__global__ void fieldMult(float* Ar, float* Br, float* Cr, float* Ai, float* Bi, float* Ci){
	float a = Ar[threadIdx.x];
    float b = Ai[threadIdx.x];
    float c = Br[threadIdx.x];
    float d = Bi[threadIdx.x];
    float t = a * (c + d);
    Cr[threadIdx.x] = t - d*(a+b);
    Ci[threadIdx.x] = t + c*(b-a);
}

bool serial = false;

// Parallelize this
void field_multiply(float* a_r, float* a_i, float* b_r, float* b_i, float* c_r, float* c_i) {
    // All parameters are arrays of length DIMENSION^2
	int arraySize = (DIMENSION * DIMENSION);
	
	if (serial){
		field_mult_original(a_r, a_i, b_r, b_i, c_r, c_i);
	}
	else{
		for(int i=0; i<DIMENSION; ++i) {
			// Host arrays (original code)
			float *Ar = &a_r[i*DIMENSION], *Ai = &a_i[i*DIMENSION];
			float *Br = &b_r[i*DIMENSION], *Bi = &b_i[i*DIMENSION];
			float *Cr = &c_r[i*DIMENSION], *Ci = &c_i[i*DIMENSION];
			// End of last bit of original function

			//cout << "DEBUG: &a_r[i*DIMENSION] is " << &a_r[i*DIMENSION] << endl;
			//cout << "DEBUG: Ar is " << Ar << endl;

			// Device arrays
			float *devAr, *devAi;
			float *devBr, *devBi;
			float *devCr, *devCi;

			cout << "Doing cudaMalloc" << endl;
			// Create counterparts for Ar, Br, Cr, Ai, Bi, Ci (cudaMalloc, cudaMemset)
			cudaMalloc((void**)&devAr, arraySize * sizeof(float));
			cudaMalloc((void**)&devBr, arraySize * sizeof(float));
			cudaMalloc((void**)&devCr, arraySize * sizeof(float));
			cudaMalloc((void**)&devAi, arraySize * sizeof(float));
			cudaMalloc((void**)&devBi, arraySize * sizeof(float));
			cudaMalloc((void**)&devCi, arraySize * sizeof(float));

			/*
			cudaMalloc(&devPtr,arraysize*sizeof(int));
			cudaMemcpy(devPtr,host_memory,arraysize*sizeof(int),cudaMemcpyHostToDevice);
			*/

			cout << "Doing cudaMemcpy for " << i << " out of " << DIMENSION << endl;
			// Move counterparts to device (cudaMemcpy)
			cout << "Ar" << endl;
			cudaMemcpy(devAr, Ar, arraySize * sizeof(float), cudaMemcpyHostToDevice);
			cout << "Ai" << endl;
			cudaMemcpy(devAi, Ai, arraySize * sizeof(float), cudaMemcpyHostToDevice);
			cout << "Br" << endl;
			cudaMemcpy(devBr, Br, arraySize * sizeof(float), cudaMemcpyHostToDevice);
			cout << "Bi" << endl;
			cudaMemcpy(devBi, Bi, arraySize * sizeof(float), cudaMemcpyHostToDevice);
			cout << "Cr" << endl;
			cudaMemcpy(devCr, Cr, arraySize * sizeof(float), cudaMemcpyHostToDevice);
			cout << "Ci" << endl;
			cudaMemcpy(devCi, Ci, arraySize * sizeof(float), cudaMemcpyHostToDevice);

			cout << "Launching grid" << endl;
			// Launch grid with fieldMult
			fieldMult<<<DIMENSION,1>>>(devAr, devBr, devCr, devAi, devBi, devCi);
		
			// Copy deviceCr and deviceCi back to host (cudaMemcpy)
			cudaMemcpy(Cr, devCr, arraySize * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(Ci, devCi, arraySize * sizeof(float), cudaMemcpyDeviceToHost);

			//cout << "DEBUG: Cr and Ci: " << Cr[0] << " " << Ci[0] << endl;

			// Deallocate all 6 counterparts (cudaFree)
			cudaFree(&devAr);
			cudaFree(&devBr);
			cudaFree(&devCr);
			cudaFree(&devAi);
			cudaFree(&devBi);
			cudaFree(&devCi);
		}
	}
}

//Applies the kernel to the image
void step(float** fields, int &current_field, float* imaginary_field, float* M_re, float* M_im, float* N_re, float* N_im, float* M_re_buffer, float* M_im_buffer, float* N_re_buffer, float* N_im_buffer) {
    
    //Read in fields
    float* cur_field = fields[current_field];
    current_field = (current_field + 1) % 2;
    float* next_field = fields[current_field];
    
    //Clear extra imaginary field
    for(int i=0; i<DIMENSION; ++i) {
        for(int j=0; j<DIMENSION; ++j) {
            imaginary_field[i*DIMENSION+j] = 0.0;
        }
    }
    
    //Compute m,n fields
    fft2(1, LOG_RES, cur_field, imaginary_field);
    field_multiply(cur_field, imaginary_field, M_re, M_im, M_re_buffer, M_im_buffer);
    fft2(-1, LOG_RES, M_re_buffer, M_im_buffer);
    field_multiply(cur_field, imaginary_field, N_re, N_im, N_re_buffer, N_im_buffer);
    fft2(-1, LOG_RES, N_re_buffer, N_im_buffer);
    
    //Step s
    for(int i=0; i<DIMENSION; ++i) {
        for(int j=0; j<DIMENSION; ++j) {
            next_field[i*DIMENSION+j] = S(N_re_buffer[i*DIMENSION+j], M_re_buffer[i*DIMENSION+j]);
        }
    }
}

//Extract image data
void draw_field(float** fields, int &current_field, int* color_shift, int* color_scale) {
    unsigned char data [DIMENSION * DIMENSION * 3];            
    int image_ptr = 0;
    
    float* cur_field = fields[current_field];
    
    for(int i=0; i<DIMENSION; ++i) {
        for(int j=0; j<DIMENSION; ++j) {
            float s = cur_field[i*DIMENSION+j];
        
            for(int k=0; k<3; ++k) {
                data[image_ptr++] = max(0, min(255, (int)(color_shift[k] + color_scale[k]*s)));
            }
        }
    }

    createBMP(DIMENSION, DIMENSION, data, DIMENSION*DIMENSION*4, "temp.bmp");
}

//Initialize field to x
void clear_field(float x, float** fields, int &current_field) {
    float* cur_field = fields[current_field];

    for(int i=0; i<DIMENSION; ++i) {
        for(int j=0; j<DIMENSION; ++j) {
          cur_field[i*DIMENSION+j] = x;
        }
    }
}


//Place a bunch of speckles on the field
void add_speckles(int count, int intensity, float **fields, int &current_field) {
    float* cur_field = fields[current_field];

	//THIS ISNT WORKING AND I DONT KNOW WHY!!!!!!
	srand(time(NULL));

    for(int i=0; i<count; ++i) {
        int u = (int)(rand() % (DIMENSION-INNER_RADIUS) + 1);
        int v = (int)(rand() % (DIMENSION-INNER_RADIUS) + 1);
        for(int x=0; x<INNER_RADIUS; ++x) {
            for(int y=0; y<INNER_RADIUS; ++y) {
                cur_field[(u+x)*DIMENSION+v+y] = intensity;
            }
        }
    }
}

int main() {
	//Coloring stuff
	int color_shift[3] = {0, 0, 0};
	int color_scale[3] = {256, 256, 256};

	//Buffers
	int field_dims[2] = {DIMENSION, DIMENSION};
	int field_size = field_dims[0] * field_dims[1];
	float* fields[2];

	for(int i=0; i < 2; ++i) {
		fields[i] = new float [field_size];
	}
	float* imaginary_field = new float[field_size];
	int current_field = 0;
	float* M_re_buffer = new float[field_size]; // old version was a two dimensional array [256, 256]
	float* M_im_buffer = new float[field_size];
	float* N_re_buffer = new float[field_size];
	float* N_im_buffer = new float[field_size];

	BesselJ inner_bessel(INNER_RADIUS);
    BesselJ outer_bessel(OUTER_RADIUS);
    
    float inner_w = (float)1.0 / inner_bessel.w;
    float outer_w = (float)1.0 / (outer_bessel.w - inner_bessel.w);
    
    float *M_re = inner_bessel.re;
    float *M_im = inner_bessel.im;
    float *N_re = outer_bessel.re;
    float *N_im = outer_bessel.im;

	initialize_MN(M_re, M_im, N_re, N_im, inner_w, outer_w);

	clear_field(0, fields, current_field);
	add_speckles(200, 1, fields, current_field);

	for (int i=0; i<10; i++) {
		step(fields, current_field, imaginary_field, M_re, M_im, N_re, N_im, M_re_buffer, M_im_buffer, N_re_buffer, N_im_buffer);
		draw_field(fields, current_field, color_shift, color_scale);
	}
	
	delete [] fields[0];
	delete [] fields[1];
	delete [] imaginary_field;
	delete [] M_re_buffer;
	delete [] M_im_buffer;
	delete [] N_re_buffer;
	delete [] N_im_buffer;
	delete [] M_re;
	delete [] M_im;
	delete [] N_re;
	delete [] N_im;

	return 1;
}