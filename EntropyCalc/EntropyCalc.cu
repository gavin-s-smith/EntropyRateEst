
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include "npConvertion.h"

/**
 *
 *
 * @param offset
 * @param data
 * @param data_len
 * @param output
 *
 */
__global__ void process_i( int offset, int * data, int data_len, int * output )
{

	const int gpu_idx = blockDim.x*blockIdx.x + threadIdx.x;

	const int idx = gpu_idx + offset;

	if( idx >= data_len ) return;

	int start_idx = idx;


	int max_subsequence_matched = 0;

	for( int i = 0; i < start_idx; i++ )
	{
		int j = 0;

		while( start_idx+j < data_len && i+j < start_idx && data[i+j] == data[start_idx+j] )
		{
			j++;
		}

		if( j > max_subsequence_matched )
		{
			max_subsequence_matched = j;
		}
	}

	output[ idx ] = max_subsequence_matched + 1;

}


void EC( np::ndarray const & array_in, np::ndarray const & array_out )
{

	int * data = get_gpu_pointer_int_array( array_in );
	int data_len = array_in.shape(0);

	int * output = get_pointer_int_array( array_out );

	int * d_out;
	checkCudaErrors( cudaMalloc((void**)&d_out, array_out.shape(0) * sizeof(int)) );

	// now do the real processing
	int offset = 0;

	int threads_per_block = 512;
	int blocks = ( array_in.shape(0) / threads_per_block) + 1 ;
	while ( blocks > 65535 )
	{
		printf("Threads per block: %i, blocks %i",threads_per_block, blocks);
		process_i<<<65535,threads_per_block>>>( offset, data, data_len, d_out );
		checkCudaErrors( cudaDeviceSynchronize() );
		offset += 65535*threads_per_block;
		blocks -= 65535;
	}
	process_i<<<blocks,threads_per_block>>>( offset, data, data_len, d_out );


	checkCudaErrors( cudaDeviceSynchronize() );

	// get the data back from the GPU directly into dest_out, the numpy array
	checkCudaErrors(cudaMemcpy(output, d_out, array_out.shape(0)*sizeof(int), cudaMemcpyDeviceToHost ));

	checkCudaErrors( cudaDeviceSynchronize() );




}


BOOST_PYTHON_MODULE(libEntropyCalc)
{
	np::initialize();
	def("EC", EC);
}
