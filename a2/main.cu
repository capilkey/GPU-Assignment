
#include <iostream>
#include <algorithm>
#include <time.h>
#include <cuda_runtime.h>
#include "utils.h"
#include "besselj.h"

#define D_MEM_CHUNKS 2

using namespace std;

__global__ void draw_field(unsigned char *data, float *curr_field, int *color_shift, int *color_scale, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		int place = idx * 3;
		float s = curr_field[idx];
		for(int k=0; k<3; ++k) {
            data[place+k] = max(0, min(255, (int)(color_shift[k] + color_scale[k]*s)));
        }
	}
}

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


void fieldMultOriginal(float* a_r, float* a_i, float* b_r, float* b_i, float* c_r, float* c_i) {
    for(int i=0; i<DIMENSION; ++i) {
		// All arrays are 1D of length DIMENSION ^2, except a_r might be only DIMENSION
        float *Ar = &a_r[i*DIMENSION], *Ai = &a_i[i*DIMENSION];
        float *Br = &b_r[i*DIMENSION], *Bi = &b_i[i*DIMENSION];
        float *Cr = &c_r[i*DIMENSION], *Ci = &c_i[i*DIMENSION];
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

__global__ void fieldKernel(float* a_r, float* a_i, float* b_r, float* b_i, float* c_r, float* c_i)
{
	//float a = a_r[threadIdx.x];
    //float b = a_i[threadIdx.x];
    //float c = b_r[threadIdx.x];
    //float d = b_i[threadIdx.x];
    //float t = a * (c + d);
    //c_r[threadIdx.x] = t - d*(a+b);
    //c_i[threadIdx.x] = t + c*(b-a);
}

void field_multiply(float* a_r, float* a_i, float* b_r, float* b_i, float* c_r, float* c_i) {
	for(int i=0; i<DIMENSION; ++i) {
		cudaError_t error;

		// All arrays are 1D of length DIMENSION ^2, except a_r might be only DIMENSION
        float *Ar = &a_r[i*DIMENSION], *Ai = &a_i[i*DIMENSION];
        float *Br = &b_r[i*DIMENSION], *Bi = &b_i[i*DIMENSION];
        float *Cr = &c_r[i*DIMENSION], *Ci = &c_i[i*DIMENSION];
		
		float **devAr, **devAi, **devBr, **devBi, **devCr, **devCi;
		
		cout << "cudaMallocing" << endl;
		error = cudaMalloc((void**)&devAr, DIMENSION*sizeof(float));
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
		error = cudaMalloc((void**)&devAi, DIMENSION*sizeof(float));
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
		error = cudaMalloc((void**)&devBr, DIMENSION*sizeof(float));
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
		error = cudaMalloc((void**)&devBi, DIMENSION*sizeof(float));
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
		error = cudaMalloc((void**)&devCr, DIMENSION*sizeof(float));
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
		error = cudaMalloc((void**)&devCi, DIMENSION*sizeof(float));
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}

		cout << "cudaMemseting" << endl;
		error = cudaMemset(devAr, 0, DIMENSION*sizeof(float));
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
		error = cudaMemset(devAi, 0, DIMENSION*sizeof(float));
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
		error = cudaMemset(devBr, 0, DIMENSION*sizeof(float));
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
		error = cudaMemset(devBi, 0, DIMENSION*sizeof(float));
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
		error = cudaMemset(devCr, 0, DIMENSION*sizeof(float));
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
		error = cudaMemset(devCi, 0, DIMENSION*sizeof(float));
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}

		
		cout << "cudaMemcpying" << endl;
		error = cudaMemcpy(devAr, &a_r[i*DIMENSION], DIMENSION*sizeof(float), cudaMemcpyHostToDevice);
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
		error = cudaMemcpy(devAi, &a_i[i*DIMENSION], DIMENSION*sizeof(float), cudaMemcpyHostToDevice);
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
		error = cudaMemcpy(devBr, &b_r[i*DIMENSION], DIMENSION*sizeof(float), cudaMemcpyHostToDevice);
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
		error = cudaMemcpy(devBi, &b_i[i*DIMENSION], DIMENSION*sizeof(float), cudaMemcpyHostToDevice);
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
		error = cudaMemcpy(devCr, &c_r[i*DIMENSION], DIMENSION*sizeof(float), cudaMemcpyHostToDevice);
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
		error = cudaMemcpy(devCi, &c_i[i*DIMENSION], DIMENSION*sizeof(float), cudaMemcpyHostToDevice);
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
		
		cout << "Launching..." << endl;
		// Don't compile...
		// fieldKernel<<<DIMENSION,1>>>(devAr, devAi, devBr, devBi, devCr, devCi);
		// fieldKernel<<<DIMENSION,1>>>(**devAr, **devAi, **devBr, **devBi, **devCr, **devCi);
		// fieldKernel<<<DIMENSION,1>>>(&devAr, &devAi, &devBr, &devBi, &devCr, &devCi);
		// fieldKernel<<<DIMENSION,1>>>(&&devAr, &&devAi, &&devBr, &&devBi, &&devCr, &&devCi);

		// Compiles, but hangs
		fieldKernel<<<DIMENSION,1>>>(*devAr, *devAi, *devBr, *devBi, *devCr, *devCi);
				

		cudaDeviceSynchronize();
		cout << "Synchronized. cudaFreeing" << endl;
		cudaFree(&devAr);
		cudaFree(&devAi);
		cudaFree(&devBr);
		cudaFree(&devBi);
		cudaFree(&devCr);
		cudaFree(&devCi);

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
void draw_field(float** fields, int &current_field, int* color_shift, int* color_scale, int mThreads) {
    //unsigned char data [DIMENSION * DIMENSION * 3];            
    //int image_ptr = 0;
    cudaError_t error;

    float* cur_field = fields[current_field];
    
	int n = DIMENSION*DIMENSION;

	unsigned char *h_data = new unsigned char[n * 3];
	unsigned char *d_data;
	error = cudaMalloc((void**)&d_data, n*3*sizeof(unsigned char));
	if (error != cudaSuccess) {
		cout << cudaGetErrorString(error) << endl;
	}
	error = cudaMemset(d_data, 0, n*3*sizeof(unsigned char));
	if (error != cudaSuccess) {
		cout << cudaGetErrorString(error) << endl;
	}

	float *d_field;
	error = cudaMalloc((void**)&d_field, n*sizeof(float));
	if (error != cudaSuccess) {
		cout << cudaGetErrorString(error) << endl;
	}
	error = cudaMemcpy(d_field, cur_field, n*sizeof(float), cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		cout << cudaGetErrorString(error) << endl;
	}

	int *d_color_scale;
	error = cudaMalloc((void**)&d_color_scale, 3*sizeof(int));
	if (error != cudaSuccess) {
		cout << cudaGetErrorString(error) << endl;
	}
	error = cudaMemcpy(d_color_scale, color_scale, 3*sizeof(int), cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		cout << cudaGetErrorString(error) << endl;
	}

	int *d_color_shift;
	error = cudaMalloc((void**)&d_color_shift, 3*sizeof(int));
	if (error != cudaSuccess) {
		cout << cudaGetErrorString(error) << endl;
	}
	error = cudaMemcpy(d_color_shift, color_shift, 3*sizeof(int), cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		cout << cudaGetErrorString(error) << endl;
	}

	error = cudaGetLastError();
	draw_field<<<(n+mThreads-1) / mThreads, mThreads>>>(d_data, d_field, d_color_shift, d_color_scale, n);

	cudaDeviceSynchronize();

	error = cudaGetLastError();
	if (error != cudaSuccess) {
		cout << cudaGetErrorString(error) << endl;
	}

	error = cudaMemcpy(h_data, d_data, n*3*sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) {
		cout << cudaGetErrorString(error) << endl;
	}
	
	/*
    for(int i=0; i<DIMENSION; ++i) {
        for(int j=0; j<DIMENSION; ++j) {
            float s = cur_field[i*DIMENSION+j];
        
            for(int k=0; k<3; ++k) {
                data[image_ptr++] = max(0, min(255, (int)(color_shift[k] + color_scale[k]*s)));
            }
        }
    }
	*/
	
    createBMP(DIMENSION, DIMENSION, h_data, DIMENSION*DIMENSION*4, "temp.bmp");
	cudaFree(d_data);
	delete [] h_data;
	cudaFree(d_field);
	cudaFree(d_color_scale);
	cudaFree(d_color_shift);
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
	add_speckles(300, 1, fields, current_field);

	int d;
	cudaDeviceProp prop;
	cudaGetDevice(&d);
	cudaGetDeviceProperties(&prop, d);
	int mThreads = prop.maxThreadsDim[0];
	int mBlocks  = prop.maxGridSize[0];
	int mElemnts = prop.totalGlobalMem / (D_MEM_CHUNKS * sizeof(float));

	cout << "number of threads reduced to " << mThreads << endl;
	cout << "number of blocks reduced to " << mBlocks << endl;
	cout << "number of elements reduced to " << mElemnts << endl;

	for (int i=0; i<100; i++) {
		step(fields, current_field, imaginary_field, M_re, M_im, N_re, N_im, M_re_buffer, M_im_buffer, N_re_buffer, N_im_buffer);
		draw_field(fields, current_field, color_shift, color_scale, mThreads);
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