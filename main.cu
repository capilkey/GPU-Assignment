
#include <iostream>
#include <algorithm>
#include <time.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "utils.h"
#include "besselj.h"

#define D_MEM_CHUNKS 2

using namespace std;

// Forward declaration of device version of FFT()
__device__ void devfft(int dir, int m, float* x, float* y);
// Forward declaration of field_multiply
void field_multiply(float* a_r, float* a_i, float* b_r, float* b_i, float* c_r, float* c_i, int mThreads);

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

__global__ void fieldKernel(float* a_r, float* a_i, float* b_r, float* b_i, float* c_r, float* c_i){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < DIMENSION) {
		float a = a_r[threadIdx.x];
		float b = a_i[threadIdx.x];
		float c = b_r[threadIdx.x];
		float d = b_i[threadIdx.x];
		float t = a * (c + d);
		c_r[threadIdx.x] = t - d*(a+b);
		c_i[threadIdx.x] = t + c*(b-a);
	}
}

__global__ void fft2Kernel1(int dir, int m, float* x, float* y){
	devfft(dir, m, &x[threadIdx.x*DIMENSION], &y[threadIdx.x*DIMENSION]);
}

__global__ void fft2Kernel2(int dir, int m, float* x, float* y){
	for(int j=0; j<threadIdx.x; ++j) {
		float a = x[threadIdx.x*DIMENSION+j];
		x[threadIdx.x*DIMENSION+j] = x[j*DIMENSION+threadIdx.x];
		x[j*DIMENSION+threadIdx.x] = a;
		
		float b = y[threadIdx.x*DIMENSION+j];
		y[threadIdx.x*DIMENSION+j] = y[j*DIMENSION+threadIdx.x];
		y[j*DIMENSION+threadIdx.x] = b;
	}
}

void stepField(float* cur_field, float* imaginary_field, float* M_re, float* M_im, float* M_re_buffer, float* M_im_buffer, float* N_re, float* N_im, float* N_re_buffer, float* N_im_buffer, int mThreads){
	cout << "DEBUG: Entering stepField" << endl;
	// Allocate device copies of each argument going to various fft2 calls
	cudaError_t error;
	
	// fields
	//cout << "DEBUG: Stepfield allocating fields" << endl;
	float* devCurField;
		error = cudaMalloc((void**)&devCurField, DIMENSION*sizeof(float));
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
		error = cudaMemcpy(devCurField, &cur_field, DIMENSION*sizeof(float), cudaMemcpyHostToDevice);
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
	float* devImaField;
		error = cudaMalloc((void**)&devImaField, DIMENSION*sizeof(float));
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
		error = cudaMemcpy(devImaField, &imaginary_field, DIMENSION*sizeof(float), cudaMemcpyHostToDevice);
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
	
	// buffers
	//cout << "DEBUG: Stepfield allocating buffers" << endl;
	float* devMREBuff;
		error = cudaMalloc((void**)&devMREBuff, DIMENSION*sizeof(float));
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
		error = cudaMemcpy(devMREBuff, &M_re_buffer, DIMENSION*sizeof(float), cudaMemcpyHostToDevice);
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
	float* devMIMBuff;
		error = cudaMalloc((void**)&devMIMBuff, DIMENSION*sizeof(float));
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
		error = cudaMemcpy(devMIMBuff, &M_im_buffer, DIMENSION*sizeof(float), cudaMemcpyHostToDevice);
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
	float* devNREBuff;
		error = cudaMalloc((void**)&devNREBuff, DIMENSION*sizeof(float));
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
		error = cudaMemcpy(devNREBuff, &N_re_buffer, DIMENSION*sizeof(float), cudaMemcpyHostToDevice);
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
	float* devNIMBuff;
		error = cudaMalloc((void**)&devNIMBuff, DIMENSION*sizeof(float));
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
		error = cudaMemcpy(devNIMBuff, &N_im_buffer, DIMENSION*sizeof(float), cudaMemcpyHostToDevice);
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
		
	// other
	//cout << "DEBUG: Stepfield allocating other" << endl;
	//int dimSquared = DIMENSION*DIMENSION;
	float* devMRE;
		//cout << "DEBUG: Stepfield allocating devMRE" << endl;
		error = cudaMalloc((void**)&devMRE, DIMENSION*sizeof(float));
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
		error = cudaMemcpy(devMRE, &M_re, DIMENSION*sizeof(float), cudaMemcpyHostToDevice);
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
	float* devMIM;
		//cout << "DEBUG: Stepfield allocating devMIM" << endl;
		error = cudaMalloc((void**)&devMIM, DIMENSION*sizeof(float));
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
		error = cudaMemcpy(devMIM, &M_im, DIMENSION*sizeof(float), cudaMemcpyHostToDevice);
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
	float* devNRE;
		//cout << "DEBUG: Stepfield allocating devNRE" << endl;
		error = cudaMalloc((void**)&devNRE, DIMENSION*sizeof(float));
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
		error = cudaMemcpy(devNRE, &N_re, DIMENSION*sizeof(float), cudaMemcpyHostToDevice);
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
	float* devNIM;
		//cout << "DEBUG: Stepfield allocating devNIM" << endl;
		error = cudaMalloc((void**)&devNIM, DIMENSION*sizeof(float));
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
		error = cudaMemcpy(devNIM, &N_im, DIMENSION*sizeof(float), cudaMemcpyHostToDevice);
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
	/*
	int* devMThreads = new int[1];
		error = cudaMalloc((void**)&devMThreads, sizeof(int));
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
		error = cudaMemcpy(devMThreads, &mThreads, sizeof(int), cudaMemcpyHostToDevice);
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
	*/
	error = cudaGetLastError();
	
	//cout << ".";
	
	// FFT2 process:
	// kernel1, kernel2, kernel1 again
	
	cout << "DEBUG: Stepfield allocation done, about to try to kernel" << endl;
	
	// First FFT2
	fft2Kernel1<<<(DIMENSION+mThreads-1) / mThreads, mThreads>>>(1, LOG_RES, devCurField, devImaField);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		cout << cudaGetErrorString(error) << endl;
	}
	cout << "DEBUG: First FFT part 1 done." << endl;
	cudaDeviceSynchronize();
	fft2Kernel2<<<(DIMENSION+mThreads-1) / mThreads, mThreads>>>(1, LOG_RES, devCurField, devImaField);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		cout << cudaGetErrorString(error) << endl;
	}
	cout << "DEBUG: First FFT part 2 done." << endl;
	cudaDeviceSynchronize();
	fft2Kernel1<<<(DIMENSION+mThreads-1) / mThreads, mThreads>>>(1, LOG_RES, devCurField, devImaField);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		cout << cudaGetErrorString(error) << endl;
	}
	cout << "DEBUG: First FFT part 3 done." << endl;
	cudaDeviceSynchronize();
	// /First FFT2
	// First FieldMultiply
	field_multiply(devCurField, devImaField, devMRE, devMIM, devMREBuff, devMIMBuff, mThreads);
	
	// Second FFT2
	fft2Kernel1<<<(DIMENSION+mThreads-1) / mThreads, mThreads>>>(-1, LOG_RES, devMREBuff, devMIMBuff);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		cout << cudaGetErrorString(error) << endl;
	}
	cout << "DEBUG: Second FFT part 1 done." << endl;
	cudaDeviceSynchronize();
	fft2Kernel2<<<(DIMENSION+mThreads-1) / mThreads, mThreads>>>(-1, LOG_RES, devMREBuff, devMIMBuff);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		cout << cudaGetErrorString(error) << endl;
	}
	cout << "DEBUG: Second FFT part 2 done." << endl;
	cudaDeviceSynchronize();
	fft2Kernel1<<<(DIMENSION+mThreads-1) / mThreads, mThreads>>>(-1, LOG_RES, devMREBuff, devMIMBuff);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		cout << cudaGetErrorString(error) << endl;
	}
	cout << "DEBUG: Second FFT part 3 done." << endl;
	cudaDeviceSynchronize();
	// /Second FFT2
	// Second FieldMultiply
	field_multiply(devCurField, devImaField, devNRE, devNIM, devNREBuff, devNIMBuff, mThreads);
	
	// Third FFT2
	fft2Kernel1<<<(DIMENSION+mThreads-1) / mThreads, mThreads>>>(-1, LOG_RES, devNREBuff, devNIMBuff);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		cout << cudaGetErrorString(error) << endl;
	}
	cout << "DEBUG: Third FFT part 1 done." << endl;
	cudaDeviceSynchronize();
	fft2Kernel2<<<(DIMENSION+mThreads-1) / mThreads, mThreads>>>(-1, LOG_RES, devNREBuff, devNIMBuff);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		cout << cudaGetErrorString(error) << endl;
	}
	cout << "DEBUG: Third FFT part 1 done." << endl;
	cudaDeviceSynchronize();
	fft2Kernel1<<<(DIMENSION+mThreads-1) / mThreads, mThreads>>>(-1, LOG_RES, devNREBuff, devNIMBuff);
	error = cudaGetLastError();
	if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
	cout << "DEBUG: Third FFT part 1 done." << endl;
	cudaDeviceSynchronize();
	// /Third FFT2
	
	cout << "DEBUG: Ready to copy back." << endl;
	// Copy back from device to host
	error = cudaMemcpy(&N_re_buffer, devNREBuff, DIMENSION*sizeof(float), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
	error = cudaMemcpy(&N_im_buffer, devNIMBuff, DIMENSION*sizeof(float), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
	cout << "DEBUG: Copying back done." << endl;
	
	/* //Original Logic
	fft2(1, LOG_RES, cur_field, imaginary_field);
    field_multiply(cur_field, imaginary_field, M_re, M_im, M_re_buffer, M_im_buffer, mThreads);
    fft2(-1, LOG_RES, M_re_buffer, M_im_buffer);
    field_multiply(cur_field, imaginary_field, N_re, N_im, N_re_buffer, N_im_buffer, mThreads);
    fft2(-1, LOG_RES, N_re_buffer, N_im_buffer);
	//Original Logic */
	
	// Deallocate everything
	cudaFree(devCurField);
	cudaFree(devImaField);	
	cudaFree(devMREBuff);
	cudaFree(devMIMBuff);
	cudaFree(devNREBuff);
	cudaFree(devNIMBuff);
	cudaFree(devMRE);
	cudaFree(devMIM);
	cudaFree(devNRE);
	cudaFree(devNIM);
	//cudaFree(devMThreads);
	cout << "DEBUG: Leaving stepField" << endl;
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
    return (float)( sigma(x, a, ALPHA_N) * (1.0f - sigma(x, b, ALPHA_N)));
}
float lerp(float a, float b, float t) {
    return (float)( (1.0f-t)*a + t*b);
}
float S(float n,float m) {
    float alive = sigma(m, 0.5f, ALPHA_M);
    return sigma_2(n, lerp(B1, D1, alive), lerp(B2, D2, alive));
}


__device__ float devSigma(float x, float a, float alpha) {
    return (float)( 1.0 / (1.0 + exp(-4.0/alpha * (x - a))));
}
__device__ float devSigma_2(float x, float a, float b) {
    return (float)( devSigma(x, a, ALPHA_N) * (1.0f - devSigma(x, b, ALPHA_N)));
}
__device__ float devLerp(float a, float b, float t) {
    return (float)( (1.0f-t)*a + t*b);
}
__device__ float devS(float n,float m) {
    float alive = devSigma(m, 0.5f, ALPHA_M);
    return devSigma_2(n, devLerp(B1, D1, alive), devLerp(B2, D2, alive));
}

/*
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
*/

// Rewrite to use the parameters that come in as if they were device arrays to begin with
void field_multiply(float* a_r, float* a_i, float* b_r, float* b_i, float* c_r, float* c_i, int mThreads) {
	//cudaError_t error;
	//float *devAr, *devAi, *devBr, *devBi, *devCr, *devCi;
	/*
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
	*/
	//cout << "DEBUG: Entering field_multiply" << endl;
	for(int i=0; i<DIMENSION; ++i) {
		/*
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
		error = cudaGetLastError();
		*/
		//fieldKernel<<<(DIMENSION+mThreads-1) / mThreads, mThreads>>>(devAr, devAi, devBr, devBi, devCr, devCi);
		cudaError_t error;
		fieldKernel<<<(DIMENSION+mThreads-1) / mThreads, mThreads>>>(a_r, a_i, b_r, b_i, c_r, c_i);
		error = cudaGetLastError();
		if (error != cudaSuccess) {
			cout << cudaGetErrorString(error) << endl;
		}

		cudaDeviceSynchronize();
		
		/*
		error = cudaMemcpy(&c_r[i*DIMENSION], devCr, DIMENSION*sizeof(float), cudaMemcpyDeviceToHost);
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
		error = cudaMemcpy(&c_i[i*DIMENSION], devCi, DIMENSION*sizeof(float), cudaMemcpyDeviceToHost);
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
		*/
	}
	/*
	cudaFree(devAr);
	cudaFree(devAi);
	cudaFree(devBr);
	cudaFree(devBi);
	cudaFree(devCr);
	cudaFree(devCi);
	*/
}

// __global__ void fft2Kernel1(int dir, int m, float* x, float* y){
//	devfft(dir, m, &x[threadIdx.x*DIMENSION], &y[threadIdx.x*DIMENSION]);
//}

__global__ void clearImaginary(float* field){
	for(int j=0; j<DIMENSION; ++j) {
		field[threadIdx.x*DIMENSION+j] = 0.0f;
	}
}

__global__ void stepKernel(float* next_field, float* N_re_buffer, float* M_re_buffer){
	for(int j=0; j<DIMENSION; ++j) {
		next_field[threadIdx.x*DIMENSION+j] = devS(N_re_buffer[threadIdx.x*DIMENSION+j], M_re_buffer[threadIdx.x*DIMENSION+j]);
	}
}

//Applies the kernel to the image
void step(float** fields, int &current_field, float* imaginary_field, float* M_re, float* M_im, float* N_re, float* N_im, float* M_re_buffer, float* M_im_buffer, float* N_re_buffer, float* N_im_buffer, int mThreads) {
    
    //Read in fields
    float* cur_field = fields[current_field];
    current_field = (current_field + 1) % 2;
    float* next_field = fields[current_field];
    
    //Clear extra imaginary field
	/*for(int i=0; i<DIMENSION; ++i) {
        for(int j=0; j<DIMENSION; ++j) {
            imaginary_field[i*DIMENSION+j] = 0.0;
        }
    }*/
	cudaError_t error;
	
	float* devImaginary;
	error = cudaMalloc((void**)&devImaginary, DIMENSION*sizeof(float));
	if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
	error = cudaMemcpy(devImaginary, &imaginary_field, DIMENSION*sizeof(float), cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
	
	cout << "DEBUG: About to clearImaginary" << endl;
	clearImaginary<<<(DIMENSION+mThreads-1) / mThreads, mThreads>>>(devImaginary);
	error = cudaGetLastError();
	if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
	cudaDeviceSynchronize();
	
	error = cudaMemcpy(&imaginary_field, devImaginary, DIMENSION*sizeof(float), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
	
	cout << "DEBUG: clearImaginary done" << endl;
	
	cout << "DEBUG: About to stepField" << endl;
    //Compute m,n fields
	stepField(cur_field, imaginary_field, M_re, M_im, M_re_buffer, M_im_buffer, N_re, N_im, N_re_buffer, N_im_buffer, mThreads);
    cout << "DEBUG: stepField done" << endl;
	
    //Step s
	// >>> GOOD KERNEL CANDIDATE <<<
    /*for(int i=0; i<DIMENSION; ++i) {
        for(int j=0; j<DIMENSION; ++j) {
            next_field[i*DIMENSION+j] = S(N_re_buffer[i*DIMENSION+j], M_re_buffer[i*DIMENSION+j]);
        }
    }*/
	
	cout << "Allocating and copying devNext" << endl;
	float* devNext;
		error = cudaMalloc((void**)&devNext, DIMENSION*sizeof(float));
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
		error = cudaMemcpy(devNext, &next_field, DIMENSION*sizeof(float), cudaMemcpyHostToDevice);
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
	
	cout << "Allocating and copying devNREBuff" << endl;
	float* devNREBuff;
		error = cudaMalloc((void**)&devNREBuff, DIMENSION*sizeof(float));
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
		error = cudaMemcpy(devNREBuff, &N_re_buffer, DIMENSION*sizeof(float), cudaMemcpyHostToDevice);
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
	
	cout << "Allocating and copying devMREBuff" << endl;
	float* devMREBuff;
		error = cudaMalloc((void**)&devMREBuff, DIMENSION*sizeof(float));
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
		error = cudaMemcpy(devMREBuff, &M_re_buffer, DIMENSION*sizeof(float), cudaMemcpyHostToDevice);
		if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
	
	cout << "DEBUG: About to stepkernel" << endl;
	stepKernel<<<(DIMENSION+mThreads-1) / mThreads, mThreads>>>(devNext, devNREBuff, devMREBuff);
	error = cudaGetLastError();
	if (error != cudaSuccess) {cout << cudaGetErrorString(error) << endl;}
	cudaDeviceSynchronize();
	cout << "DEBUG: stepkernel done" << endl;
	
	cudaFree(devImaginary);
	cudaFree(devNext);
	cudaFree(&devNREBuff);
	cudaFree(&devMREBuff);
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
void add_speckles(int count, float intensity, float **fields, int &current_field) {
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
	add_speckles(300, 1.0f, fields, current_field);

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
		step(fields, current_field, imaginary_field, M_re, M_im, N_re, N_im, M_re_buffer, M_im_buffer, N_re_buffer, N_im_buffer, mThreads);
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

	cout << "debug" << endl;
	return 0;
}

//FFT
__device__ void devfft(int dir, int m, float* x, float* y) {
   int nn,i,i1,j,k,i2,l,l1,l2;
   float c1,c2,tx,ty,t1,t2,u1,u2,z;
    
    // Calculate the number of points 
    nn = DIMENSION;
    
    // Do the bit reversal 
    i2 = nn >> 1;
    j = 0;
    for (i=0;i<nn-1;i++) {
      if (i < j) {
         tx = x[i];
         ty = y[i];
         x[i] = x[j];
         y[i] = y[j];
         x[j] = tx;
         y[j] = ty;
      }
      k = i2;
      while (k <= j) {
         j -= k;
         k >>= 1;
      }
      j += k;
    }
    
    // Compute the FFT 
    c1 = -1.0;
    c2 = 0.0;
    l2 = 1;
    for (l=0;l<m;l++) {
      l1 = l2;
      l2 <<= 1;
      u1 = 1.0;
      u2 = 0.0;
      for (j=0;j<l1;j++) {
         for (i=j;i<nn;i+=l2) {
            i1 = i + l1;
            t1 = u1 * x[i1] - u2 * y[i1];
            t2 = u1 * y[i1] + u2 * x[i1];
            x[i1] = x[i] - t1;
            y[i1] = y[i] - t2;
            x[i] += t1;
            y[i] += t2;
         }
         z =  u1 * c1 - u2 * c2;
         u2 = u1 * c2 + u2 * c1;
         u1 = z;
      }
      c2 = sqrt((1.0 - c1) / 2.0);
      if (dir == 1)
         c2 = -c2;
      c1 = sqrt((1.0 + c1) / 2.0);
    }
    
    // Scaling for forward transform
    if (dir == -1) {
      float scale_f = 1.0 / nn;        
      for (i=0;i<nn;i++) {
         x[i] *= scale_f;
         y[i] *= scale_f;
      }
    }
}