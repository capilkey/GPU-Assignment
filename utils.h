

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

void fft(int dir, int m, float* x, float* y);
void fft2(int dir, int m, float* x, float* y);
void createBMP(int w, int h, const unsigned char* img, int s_data, const char* filename);