

#define INNER_RADIUS 7
#define OUTER_RADIUS 3 * INNER_RADIUS
#define B1 0.278
#define B2 0.365
#define D1 0.267
#define D2 0.445
#define ALPHA_N 0.028
#define ALPHA_M 0.147
#define LOG_RES 8
#define DIMENSION 256
#define NUM_FIELDS 2;

void fft(int dir, int m, float* x, float* y);
void fft2(int dir, int m, float* x, float* y);
void createBMP(int w, int h, const unsigned char* img, int s_data, const char* filename);