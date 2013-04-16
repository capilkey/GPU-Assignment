#pragma comment ( lib, "cufft.lib" )

#include <iostream>
#include <algorithm>
#include <time.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include "device_launch_parameters.h"
#include "utils.h"

#define D_MEM_CHUNKS 2
#define INNER_RADIUS 7
#define OUTER_RADIUS 3 * INNER_RADIUS
#define B1 0.238f
#define B2 0.365f
#define D1 0.267f
#define D2 0.445f
#define ALPHA_N 0.028f
#define ALPHA_M 0.147f
#define LOG_RES 8
#define DIMENSION 256
#define NUM_FIELDS 2;

using namespace std;

/* 
 * Complex type definition taken from the cuFFT samples.
 */
typedef float2 Complex;
static __device__ __host__ inline Complex ComplexScale(Complex, float);

void fft(int, int, Complex*);
void fft2(int, int, Complex*, cufftHandle);
Complex* besselJ(int, float&, cufftHandle);

__device__ float sigma(float x, float a, float alpha);
__device__ float sigma_2(float x, float a, float b);
__device__ float lerp(float a, float b, float t);

__global__ void draw_field(unsigned char *data, Complex *curr_field, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//Coloring stuff

	if (idx < n) {
		int place = idx * 3;
		float s = curr_field[idx].x;   // colourshift + colourscale
        data[place+0] = max(0, min(255, (int)(0 + 256*s)));
		data[place+1] = max(0, min(255, (int)(0 + 256*s)));
		data[place+2] = max(0, min(255, (int)(0 + 256*s)));
	}
}

__global__ void fieldKernel(Complex* a, Complex* b, Complex* c, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		float ax = a[idx].x;
		float ay = a[idx].y;
		float bx = b[idx].x;
		float by = b[idx].y;
		float t = ax * (bx + by);
		c[idx].x = t - by*(ax+ay);
		c[idx].y = t + bx*(ay-ax);
	}
}

__global__ void S(Complex* a, Complex* b, Complex* c, int n) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < n) {
		float alive = sigma(c[idx].x, 0.5f, ALPHA_M);
		a[idx].x = sigma_2(b[idx].x, lerp(B1, D1, alive), lerp(B2, D2, alive));
	}
}

__global__ void clearImaginary(Complex* a, int n) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < n) {
		a[idx].y = 0.0f;
	}
}

//Precalculate multipliers for m,n
void initialize_MN(Complex* M, Complex* N, float inner_w, float outer_w){ 
    for(int i=0; i<DIMENSION; ++i) {
        for(int j=0; j<DIMENSION; ++j) {
            N[i*DIMENSION+j].x = outer_w * (N[i*DIMENSION+j].x - M[i*DIMENSION+j].x);
            N[i*DIMENSION+j].y = outer_w * (N[i*DIMENSION+j].y - M[i*DIMENSION+j].y);
            M[i*DIMENSION+j].x *= inner_w;
            M[i*DIMENSION+j].y *= inner_w;
        }
    }
}


__device__ float sigma(float x, float a, float alpha) {
    return (float)( 1.0f / (1.0f + exp(-4.0f/alpha * (x - a))));
}

__device__ float sigma_2(float x, float a, float b) {
    return (float)( sigma(x, a, ALPHA_N) * (1.0f - sigma(x, b, ALPHA_N)));
}

__device__ float lerp(float a, float b, float t) {
    return (float)( (1.0f-t)*a + t*b);
}

__device__ float S(float n,float m) {
    float alive = sigma(m, 0.5f, ALPHA_M);
    return sigma_2(n, lerp(B1, D1, alive), lerp(B2, D2, alive));
}

void field_multiply(Complex* d_A, Complex* d_B, Complex* d_C, int mThreads) {
	cudaError_t error;

	error = cudaGetLastError();
	fieldKernel<<<(DIMENSION*DIMENSION+mThreads-1) / mThreads, mThreads>>>(d_A, d_B, d_C, DIMENSION*DIMENSION);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		cout << cudaGetErrorString(error) << endl;
	}

	cudaDeviceSynchronize();
}

//Applies the kernel to the image
void step(Complex** d_fields, int &current_field, Complex* d_M, Complex* d_N, Complex* d_M_buffer, Complex* d_N_buffer, int mThreads, cufftHandle plan) {
    cudaError_t error;
	
	//Read in fields
    Complex* d_cur_field = d_fields[current_field];
    current_field = (current_field + 1) % 2;
    Complex* next_field = d_fields[current_field];
    
    //Clear imaginary values from cur_field
    error = cudaGetLastError();
	clearImaginary<<<(DIMENSION*DIMENSION+mThreads-1) / mThreads, mThreads>>>(d_cur_field, DIMENSION*DIMENSION);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		cout << cudaGetErrorString(error) << endl;
	}

    //Compute m,n fields
    fft2(1, LOG_RES, d_cur_field, plan);
    field_multiply(d_cur_field, d_M, d_M_buffer, mThreads);
    fft2(-1, LOG_RES, d_M_buffer, plan);
    field_multiply(d_cur_field, d_N, d_N_buffer, mThreads);
    fft2(-1, LOG_RES, d_N_buffer, plan);

    //Step s
	error = cudaGetLastError();
	S<<<(DIMENSION*DIMENSION+mThreads-1) / mThreads, mThreads>>>(next_field, d_N_buffer, d_M_buffer, DIMENSION*DIMENSION);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		cout << cudaGetErrorString(error) << endl;
	}
}

//Extract image data
void draw_field(Complex** d_fields, int &current_field, int mThreads) {
    //unsigned char data [DIMENSION * DIMENSION * 3];            
    int n = DIMENSION * DIMENSION;
    cudaError_t error;
	unsigned char *h_data = new unsigned char[n * 3];
	unsigned char *d_data;

    Complex* d_cur_field = d_fields[current_field];

	error = cudaMalloc((void**)&d_data, n*3*sizeof(unsigned char));
	if (error != cudaSuccess) {
		cout << cudaGetErrorString(error) << endl;
	}
	error = cudaMemset(d_data, 0, n*3*sizeof(unsigned char));
	if (error != cudaSuccess) {
		cout << cudaGetErrorString(error) << endl;
	}

	error = cudaGetLastError();
	draw_field<<<(n+mThreads-1) / mThreads, mThreads>>>(d_data, d_cur_field, n);

	error = cudaMemcpy(h_data, d_data, n*3*sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) {
		cout << cudaGetErrorString(error) << endl;
	}
	
    createBMP(DIMENSION, DIMENSION, h_data, DIMENSION*DIMENSION*4, "temp.bmp");
	cudaFree(d_data);
	delete [] h_data;
}

//Initialize field to x
void clear_field(float x, Complex** fields, int &current_field) {
    Complex* cur_field = fields[current_field];

    for(int i=0; i<DIMENSION; ++i) {
        for(int j=0; j<DIMENSION; ++j) {
          cur_field[i*DIMENSION+j].x = x;
		  cur_field[i*DIMENSION+j].y = 0.0f;
        }
    }
}


//Place a bunch of speckles on the field
void add_speckles(int count, float intensity, Complex **fields, int &current_field) {
    Complex* cur_field = fields[current_field];

	srand(time(NULL));

    for(int i=0; i<count; ++i) {
        int u = (int)(rand() % (DIMENSION-INNER_RADIUS) + 1);
        int v = (int)(rand() % (DIMENSION-INNER_RADIUS) + 1);
        for(int x=0; x<INNER_RADIUS; ++x) {
            for(int y=0; y<INNER_RADIUS; ++y) {
                cur_field[(u+x)*DIMENSION+v+y].x = intensity;
            }
        }
    }
}

int main() {
	//Buffers
	int field_dims[2] = {DIMENSION, DIMENSION};
	int field_size = field_dims[0] * field_dims[1];
	Complex* fields[2];
	Complex *d_fields[2], *d_M, *d_N, *d_M_buffer, *d_N_buffer;
	cudaError_t error;

	for(int i=0; i < 2; ++i) {
		fields[i] = (Complex *)malloc(sizeof(Complex) * field_size);
	}
	int current_field = 0;
	Complex* M_buffer = (Complex *)malloc(sizeof(Complex) * field_size); // old version was a two dimensional array [256, 256]
	Complex* N_buffer = (Complex *)malloc(sizeof(Complex) * field_size);

	int d;
	cudaDeviceProp prop;
	cudaGetDevice(&d);
	cudaGetDeviceProperties(&prop, d);
	int mThreads = prop.maxThreadsDim[0];

	cufftHandle plan;
	if (cufftPlan2d(&plan, DIMENSION, DIMENSION, CUFFT_C2C) != CUFFT_SUCCESS){
		cout << "CUFFT Error: Unable to create plan" << endl;
	}
//	if (cufftSetCompatibilityMode(plan, CUFFT_COMPATIBILITY_NATIVE)!= CUFFT_SUCCESS){
//		cout << "CUFFT Error: Unable to set compatibility mode to native" << endl;	
//	}

	float inner_width;
	float outer_width;
	Complex* M = besselJ(INNER_RADIUS, inner_width, plan);
    Complex* N = besselJ(OUTER_RADIUS, outer_width, plan);
    
    float inner_w = (float)1.0f / inner_width;
    float outer_w = (float)1.0f / (outer_width - inner_width);

	initialize_MN(M, N, inner_w, outer_w);

	clear_field(0.0f, fields, current_field);
	add_speckles(300, 1.0f, fields, current_field);
	
	error = cudaMalloc((void**)&d_fields[0], DIMENSION*DIMENSION*sizeof(Complex));
	if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
	error = cudaMalloc((void**)&d_fields[1], DIMENSION*DIMENSION*sizeof(Complex));
	if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
	error = cudaMalloc((void**)&d_M, DIMENSION*DIMENSION*sizeof(Complex));
	if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
	error = cudaMalloc((void**)&d_N, DIMENSION*DIMENSION*sizeof(Complex));
	if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
	error = cudaMalloc((void**)&d_M_buffer, DIMENSION*DIMENSION*sizeof(Complex));
	if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
	error = cudaMalloc((void**)&d_N_buffer, DIMENSION*DIMENSION*sizeof(Complex));
	if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}

	error = cudaMemcpy(d_fields[0], fields[0], DIMENSION*DIMENSION*sizeof(Complex), cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
	error = cudaMemcpy(d_fields[1], fields[1], DIMENSION*DIMENSION*sizeof(Complex), cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
	error = cudaMemcpy(d_M, M, DIMENSION*DIMENSION*sizeof(Complex), cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
	error = cudaMemcpy(d_N, N, DIMENSION*DIMENSION*sizeof(Complex), cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
	error = cudaMemcpy(d_M_buffer, M_buffer, DIMENSION*DIMENSION*sizeof(Complex), cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
	error = cudaMemcpy(d_N_buffer, N_buffer, DIMENSION*DIMENSION*sizeof(Complex), cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}

	for (int i=0; i<1000; i++) {
		step(d_fields, current_field, d_M, d_N, d_M_buffer, d_N_buffer, mThreads, plan);
		draw_field(d_fields, current_field, mThreads);
		//cin >> inner_width;
	}
	
	cudaFree(d_fields[0]);
	cudaFree(d_fields[1]);
	cudaFree(d_M);
	cudaFree(d_N);
	cudaFree(d_M_buffer);
	cudaFree(d_N_buffer);

	free(fields[0]);
	free(fields[1]);
	free(M_buffer);
	free(N_buffer);
	free(M);
	free(N);
	
	cufftDestroy(plan);

	return 0;
}

//FFT
void fft(int dir, int m, Complex* a) {
	int nn,i,i1,j,k,i2,l,l1,l2;
	float c1,c2,t1,t2,u1,u2,z;
	Complex t;
    /* Calculate the number of points */
    nn = DIMENSION;
    
    /* Do the bit reversal */
    i2 = nn >> 1;
    j = 0;
    for (i=0;i<nn-1;i++) {
      if (i < j) {
         t = a[i];
         a[i] = a[j];
         a[j] = t;
      }
      k = i2;
      while (k <= j) {
         j -= k;
         k >>= 1;
      }
      j += k;
    }
    
    /* Compute the FFT */
    c1 = -1.0f;
    c2 = 0.0f;
    l2 = 1;
    for (l=0;l<m;l++) {
      l1 = l2;
      l2 <<= 1;
      u1 = 1.0f;
      u2 = 0.0f;
      for (j=0;j<l1;j++) {
         for (i=j;i<nn;i+=l2) {
            i1 = i + l1;
			t1 = u1 * a[i1].x - u2 * a[i1].y;
            t2 = u1 * a[i1].y + u2 * a[i1].x;
            a[i1].x = a[i].x - t1;
            a[i1].y = a[i].y - t2;
            a[i].x += t1;
            a[i].y += t2;
         }
         z =  u1 * c1 - u2 * c2;
         u2 = u1 * c2 + u2 * c1;
         u1 = z;
      }
      c2 = sqrt((1.0f - c1) / 2.0f);
      if (dir == 1)
         c2 = -c2;
      c1 = sqrt((1.0f + c1) / 2.0f);
    }
    
    /* Scaling for forward transform */
    if (dir == -1) {
      float scale_f = 1.0f / nn;        
      for (i=0;i<nn;i++) {
         a[i] = ComplexScale(a[i], scale_f);
      }
    }
}

//In place 2D fft
void fft2(int dir, int m, Complex* a, cufftHandle plan) {
	if (cufftExecC2C(plan, (cufftComplex*)a, (cufftComplex*)a, (dir==1 ? CUFFT_FORWARD : CUFFT_INVERSE)) != CUFFT_SUCCESS){
		cout << "CUFFT Error: Unable to execute plan" << endl;	
	}

	cudaError_t error = cudaDeviceSynchronize();
	if (error != cudaSuccess) {
		cout << "failed to synch: " << cudaGetErrorString(error) << endl;
	}
}

//In place 2D fft
void hostfft2(int dir, int m, Complex* a) {
  for(int i=0; i<DIMENSION; ++i) {
    fft(dir, m, &a[i*DIMENSION]);
  }
  for(int i=0; i<DIMENSION; ++i) {
    for(int j=0; j<i; ++j) {
      Complex t = a[i*DIMENSION+j];
      a[i*DIMENSION+j] = a[j*DIMENSION+i];
      a[j*DIMENSION+i] = t;
    }
  }

  for(int i=0; i<DIMENSION; ++i) {
    fft(dir, m, &a[i*DIMENSION]);
  }
}

Complex* besselJ(int radius, float &w, cufftHandle plan) {
	int field_size = DIMENSION * DIMENSION;

    //Do this in a somewhat stupid way
    Complex* a = (Complex *)malloc(sizeof(Complex) * field_size);;
    w = 0.0f;
    for(int i=0; i<DIMENSION; ++i) {
        for(int j=0; j<DIMENSION; ++j) {
            float ii = (float)((i + DIMENSION/2) % DIMENSION) - DIMENSION/2;
            float jj = (float)((j + DIMENSION/2) % DIMENSION) - DIMENSION/2;
            
            float r = sqrt(ii*ii + jj*jj) - radius;
            float v = 1.0f / (1.0f + exp(LOG_RES * r));
            
            w += v;
            a[i*DIMENSION+j].x = v;
			a[i*DIMENSION+j].y = 0.0f;
        }
    }
    
    hostfft2(1, LOG_RES, a);

	return a;
}

////////////////////////////////////////////////////////////////////////////////
// Complex operations
////////////////////////////////////////////////////////////////////////////////

// Complex scale
static __device__ __host__ inline Complex ComplexScale(Complex a, float s)
{
    Complex c;
    c.x = s * a.x;
    c.y = s * a.y;
    return c;
}

// Complex pointwise multiplication
/*
static __global__ void ComplexPointwiseMulAndScale(Complex *a, const Complex *b, int size, float scale)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = threadID; i < size; i += numThreads)
    {
        a[i] = ComplexScale(ComplexMul(a[i], b[i]), scale);
    }
}
*/
