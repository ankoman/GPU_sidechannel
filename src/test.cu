#include<stdio.h>
#include<H5Cpp.h>

#define KEY_HYPOTHESIS 256
#define NUM_TRACES 1000
#define BLOCKSIZE 32

typedef struct ascad_metadata {
    unsigned char plaintext[16];
    unsigned char ciphertext[16];
    unsigned char key[16];
    unsigned char masks[16];
    unsigned int  desync;
} ascad_metadata;

__global__ void create_model(uint8_t *d_model, uint8_t *d_plaintexts){
    uint8_t Sbox[] = {  
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
        0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
        0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
        0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
        0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
        0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
        0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
        0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
        0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
        0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
        0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
        0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
        0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
        0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
        0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
        0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
        };
    
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = iy * KEY_HYPOTHESIS + ix;             //何スレッド目か

    if(ix < KEY_HYPOTHESIS && iy < NUM_TRACES)
        d_model[idx] = HW(Sbox[d_plaintexts[iy] ^ ix]);
}

__device__ uint8_t HW(uint8_t x)
{
    x = x - ((x >> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x << 2) & 0x33333333);

    return ((x + (x >> 4) & 0xF0F0F0F) * 0x1010101) >> 24;
}

__global__ void transpose_model(uint8_t *out, uint8_t *in, const int nx, const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if(ix < nx && iy << ny){
        out[ix * ny + iy] = in[iy * nx + ix];
    }
}

__global__ void correlation(float *d_corr, int8_t *d_trases_t, uint8_t *d_model_t, int T, int D){
    // T is n_pois, D is n_traces? 
    // listing 4.4: Computation of the correlation matrix for a first order CPA attack based on [CDOR09].
    __shared__ float Xs[BLOCKSIZE][BLOCKSIZE];
    __shared__ float Ys[BLOCKSIZE][BLOCKSIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int xBegin = bx * BLOCKSIZE * D;
    int yBegin = by * BLOCKSIZE * D;
    int yEnd = yBegin + D - 1;

    int x, y, k, o;
    float a1, a2, a3, a4, a5;
    float avgX, avgY, varX, varY, cov, rho;

    a1 = a2 = a3 = a4 = a5 = 0.0;

    for(y = yBegin, x = xBegin; y <= yEnd; y += BLOCKSIZE, x += BLOCKSIZE){
        Xs[tx][ty] = d_model_t[x + ty * D + tx];
        Ys[ty][tx] = d_trases_t[y + ty * D + tx];

        __syncthreads();

        for(k = 0; k < BLOCKSIZE; k++){
            a1 += Xs[k][tx];
            a2 += Ys[ty][k];
            a3 += Xs[k][tx] * Xs[k][tx];
            a4 += Ys[ty][k] * Ys[ty][k];
            a5 += Xs[k][tx] * Ys[ty][k];
        }

        __syncthreads();
    }
    
    avgX = a1 / D;
    avgY = a2 / D;

    varX = (a3 - avgX * avgX * D) / (D - 1);
    varY = (a4 - avgY * avgY * D) / (D - 1);
    cov = (a5 - avgX * avgY * D) / (D - 1);

    rho - cov / sqrtf(varX * varY);
    o = bx * BLOCKSIZE * T + tx * T + by * BLOCKSIZE + ty;

    d_corr[o] = rho;
}

__global__ void merge_sums(float *d_centr_sum_t_q1, float *d_centr_sum_1_q1, float *d_mean_t_q1, float *d_mean_1_q1,
                            float *d_centr_sum_t_q2, float *d_centr_sum_1_q2, float *d_mean_t_q2, float *d_mean_1_q2,
                            int D, int T, int K, int iteration){
                                
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;

    float mean_t_q1, mean_1_q1, centr_sum_t_q1, centr_sum_1_q1;
    float mean_t_q2, mean_1_q2, centr_sum_t_q2, centr_sum_1_q2;
    float mean_t_q, mean_1_q, centr_sum_t_q, centr_sum_1_q;
    float delta_1, delta_t, delta_n_1, delta_n_t;

    int n1, n2, n;

    n1 = D * iteration;
    n2 = D;
    n = n1 + n2;

    if(tidx < K){
        centr_sum_1_q1 = d_centr_sum_1_q1[tidx];
        centr_sum_1_q2 = d_centr_sum_1_q2[tidx];

        mean_1_q1 = d_mean_1_q1[tidx];
        mean_1_q2 = d_mean_1_q2[tidx];

        delta_1 = mean_1_q2 - mean_1_q1;
        delta_n_1 = delta_1 / n;

        centr_sum_1_q = centr_sum_1_q1 + centr_sum_1_q2 + n1 * n2 * delta_1 * delta_n_1;
        d_centr_sum_1_q1[tidx] = centr_sum_1_q;

        mean_1_q = mean_1_q1 + n2 * delta_n_1;
        d_mean_1_q1[tidx] = mean_1_q;
    }

    if(tidx < T){
        centr_sum_t_q1 = d_centr_sum_t_q1[tidx];
        centr_sum_t_q2 = d_centr_sum_t_q2[tidx];

        mean_t_q1 = d_mean_t_q1[tidx];
        mean_t_q2 = d_mean_t_q2[tidx];

        delta_t = mean_t_q2 - mean_t_q1;
        delta_n_t = delta_t / n;

        centr_sum_t_q = centr_sum_t_q1 + centr_sum_t_q2 + n1 * n2 * delta_t * delta_n_t;
        d_centr_sum_t_q1[tidx] = centr_sum_t_q;

        mean_t_q = mean_t_q1 + n2 * delta_n_t;
        d_mean_t_q1[tidx] = mean_t_q;
    }
}

__global__ void merge_adj_sum(float *d_adj_centr_sum_q1, float *d_mean_l_q1, float *d_mean_t_q1, 
                            float *d_adj_centr_sum_q2, float *d_mean_l_q2, float *d_mean_t_q2,
                            int D, int T, int K, int iteration)
{
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;

    float mean_t_q1, mean_l_q1, adj_centr_sum_q1;
    float mean_t_q2, mean_l_q2, adj_centr_sum_q2;
    float adj_centr_sum_q;
    float delta_t, delta_l;

    int n1, n2, n, index;

    if(tidx < T){
        n1 = D * iteration;
        n2 = D;
        n = n1 + n2;

        for(int k = 0; k < K; k++){
            index = k * T + tidx;

            mean_t_q1 = d_mean_t_q1[tidx];
            mean_t_q2 = d_mean_t_q2[tidx];

            mean_l_q1 = d_mean_l_q1[k];
            mean_l_q2 = d_mean_l_q2[k];

            adj_centr_sum_q1 = d_adj_centr_sum_q1[index];
            adj_centr_sum_q2 = d_adj_centr_sum_q2[index];

            delta_l = mean_l_q2 - mean_l_q1;
            delta_t = mean_t_q2 - mean_t_q1;

            adj_centr_sum_q = adj_centr_sum_q1 + adj_centr_sum_q2 + ((n1 * n2) / n) * delta_t * delta_l;
            d_adj_centr_sum_q1[index] = adj_centr_sum_q;
        }
    }
}


int main(int argc, char **argv){

	H5::H5File file("./my_ASCAD.h5", H5F_ACC_RDONLY);
    H5::Group group = file.openGroup("Attack_traces"); 
	H5::DataSet dataset = group.openDataSet("traces"); 
	H5::DataSpace dataspace = dataset.getSpace();
    int n_traces, n_pois;
    file.close();

    hsize_t dims[dataspace.getSimpleExtentNdims()];
    dataspace.getSimpleExtentDims( dims );
    n_traces = dims[0];
    n_pois = dims[1];

    char *traces = new char [n_traces * n_pois];
    printf("Traces dimension: %llu x %llu\n", dims[0], dims[1]);
    dataset.read( traces, dataset.getDataType() );

	dataset = group.openDataSet("metadata"); 
    ascad_metadata *metadata = new ascad_metadata [n_traces];
    unsigned char *plaintexts = new unsigned char[n_traces*16];

    dataset.read(metadata, dataset.getDataType() );
    for(int i=0; i<n_traces; i++){
        memcpy(plaintexts + 16*i, metadata[i].plaintext, 16);
    }

    for(int i = 0; i < 8; i++){
        for(int j = 0; j < 16; j++){
            printf("%d ", *(plaintexts + 16*i + j));
        }
        printf("\n");
    }

    // helloFromGPU<<<1, 10>>>();
    // cudaDeviceReset();
    delete traces, metadata;
    dataset.close();
    return 0;
}