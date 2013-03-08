
#include <cmath>
#include <iostream>
#include "utils.h"

//FFT
void fft(int dir, int m, float* x, float* y) {
   int nn,i,i1,j,k,i2,l,l1,l2;
   float c1,c2,tx,ty,t1,t2,u1,u2,z;
    
    /* Calculate the number of points */
    nn = DIMENSION;
    
    /* Do the bit reversal */
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
    
    /* Compute the FFT */
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
    
    /* Scaling for forward transform */
    if (dir == -1) {
      float scale_f = 1.0 / nn;        
      for (i=0;i<nn;i++) {
         x[i] *= scale_f;
         y[i] *= scale_f;
      }
    }
}

//In place 2D fft
void fft2(int dir, int m, float* x, float* y) {
  for(int i=0; i<DIMENSION; ++i) {
    fft(dir, m, &x[i*DIMENSION], &y[i*DIMENSION]);
  }
  for(int i=0; i<DIMENSION; ++i) {
    for(int j=0; j<i; ++j) {
      float t = x[i*DIMENSION+j];
      x[i*DIMENSION+j] = x[j*DIMENSION+i];
      x[j*DIMENSION+i] = t;
    }
  }
  for(int i=0; i<DIMENSION; ++i) {
    for(int j=0; j<i; ++j) {
      float t = y[i*DIMENSION+j];
      y[i*DIMENSION+j] = y[j*DIMENSION+i];
      y[j*DIMENSION+i] = t;
    }
  }
  for(int i=0; i<DIMENSION; ++i) {
    fft(dir, m, &x[i*DIMENSION], &y[i*DIMENSION]);
  }  
}

/* 
	 * Function taken from http://stackoverflow.com/a/2654860
	 *   original author: deusmacabre
	 */
	void createBMP(int w, int h, const unsigned char* img, int s_data, const char* filename) {
		FILE *f;
		int filesize = 54 + 3*w*h;  //w is your image width, h is image height, both int

		unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
		unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};
		unsigned char bmppad[3] = {0,0,0};

		bmpfileheader[ 2] = (unsigned char)(filesize    );
		bmpfileheader[ 3] = (unsigned char)(filesize>> 8);
		bmpfileheader[ 4] = (unsigned char)(filesize>>16);
		bmpfileheader[ 5] = (unsigned char)(filesize>>24);

		bmpinfoheader[ 4] = (unsigned char)(       w    );
		bmpinfoheader[ 5] = (unsigned char)(       w>> 8);
		bmpinfoheader[ 6] = (unsigned char)(       w>>16);
		bmpinfoheader[ 7] = (unsigned char)(       w>>24);
		bmpinfoheader[ 8] = (unsigned char)(       h    );
		bmpinfoheader[ 9] = (unsigned char)(       h>> 8);
		bmpinfoheader[10] = (unsigned char)(       h>>16);
		bmpinfoheader[11] = (unsigned char)(       h>>24);

		f = fopen(filename,"wb");
		fwrite(bmpfileheader,1,14,f);
		fwrite(bmpinfoheader,1,40,f);
		for(int i=0; i<h; i++)
		{
			fwrite(img+(w*(h-i-1)*3),3,w,f);
			fwrite(bmppad,1,(4-(w*3)%4)%4,f);
		}
		fclose(f);
	}