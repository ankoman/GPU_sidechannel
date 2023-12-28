#include <stdio.h>
#include<H5Cpp.h>
#include<hdf5.h>

__global__ void helloFromGPU(){
    printf("Hello World rom GPU!\n");
}

int main(int argc, char **argv){
    printf("Hello World from CPU!\n");

	H5::H5File file("../ASCAD-master/ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/my_ASCAD.h5", H5F_ACC_RDONLY);
    H5::Group group= file.openGroup("Attack_traces"); 
	H5::DataSet dataset = group.openDataSet("traces"); 
	H5::DataSpace dataspace = dataset.getSpace();

    int n_traces = dataspace.getSimpleExtentNdims();

    printf("%d\n", n_traces);

    // helloFromGPU<<<1, 10>>>();
    // cudaDeviceReset();
    return 0;
}