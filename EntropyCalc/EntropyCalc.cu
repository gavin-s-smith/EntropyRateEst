
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
__global__ void process_i( int offset, int64_t * data, int data_len, int64_t * output )
{

	const int gpu_idx = blockDim.x*blockIdx.x + threadIdx.x;

	const int idx = gpu_idx + offset;

	if( idx >= data_len )
	{
		return; // we have over-allocated threads and this one will be off the end of the array so do nothing
	}

	if( data[idx] == -1 )
	{
		// this means we are at a trajectory breakpoint.
		// put 0 in as the result since we will take the mean of these
		// by the sum of the array divide by the number of non -1 numbers.
		// therefore putting 0 here is correct.
		output[idx] = 0;
		return;
	}


	int start_idx = idx;


	int max_subsequence_matched = 0;

	// Using L_{n} definition from
	//"Nonparametric Entropy Estimation for Stationary Process and Random Fields, with Applications to English Text"
	// by Kontoyiannis et. al.
	// $L_{n} = 1 + max \{l :0 \leq l \leq n, X^{l-1}_{0} = X^{-j+l-1}_{-j} \text{ for some } l \leq j \leq n \}$

	// for each position, i, in the sub-sequence that occurs before the current position, start_idx
	// check to see the maximum continuously equal string we can make by simultaneously extending from i and start_idx
	// If we hit a -1 then we no longer have a match, since -1 delineates trajectories, and we are looking for
	// the longest match *within* trajectories.
	// This is implemented by forcing -1 != -1. If we were using floating point numbers we could
	// simply use NaN for this, however, the arrays are int64 since this is the returned type from healpix.

	for( int i = 0; i < start_idx; i++ )
	{
		int j = 0;

		// increase the length of the substring starting at j and start_idx
		// while they are the same keeping track of the length
		while( start_idx+j < data_len
				&& i+j < start_idx
				&& data[i+j] == data[start_idx+j]
				&& data[i+j] != -1
				&& data[start_idx+j] != -1
			)
		{
			j++;
		}



		if( j > max_subsequence_matched )
		{
			max_subsequence_matched = j;
		}
	}

	//L_{n} is obtained by adding 1 to the longest match-length
	output[ idx ] = max_subsequence_matched + 1;


}



/**
 * Calculates the L_{n} counts for a given input sequence for the entropy calculation
 * @param array_in A 1D input array. Trajectories are delineated by -1 values.
 * @param array_out. An array of the same length as array_in.
 */
void EC( np::ndarray const & array_in, np::ndarray const & array_out )
{

	if( array_in.shape(0) != array_out.shape(0) )
	{
		PyErr_SetString(PyExc_TypeError, "\n\nError in EC (cuda code). Two passed arrays must be equal length.");
		p::throw_error_already_set();
	}

	int64_t * data = get_gpu_pointer_int64_array( array_in );

	//print_gpu_int64_array<<<1,1,1>>>(data,array_out.shape(0));

	int data_len = array_in.shape(0);

	int64_t * output = get_pointer_int64_array( array_out );

	int64_t * d_out;
	checkCudaErrors( cudaMalloc((void**)&d_out, array_out.shape(0) * sizeof(int64_t)) );

	// now do the real processing
	int offset = 0;

	int threads_per_block = 512;
	int blocks = ( array_in.shape(0) / threads_per_block) + 1 ;

	while ( blocks > 65535 )
	{
		printf("\nThreads per block: %i, blocks %i",threads_per_block, blocks);
		process_i<<<65535,threads_per_block>>>( offset, data, data_len, d_out );
		checkCudaErrors( cudaDeviceSynchronize() );
		offset += 65535*threads_per_block;
		blocks -= 65535;
	}
	process_i<<<blocks,threads_per_block>>>( offset, data, data_len, d_out );


	checkCudaErrors( cudaDeviceSynchronize() );

	// get the data back from the GPU directly into dest_out, the numpy array
	checkCudaErrors(cudaMemcpy(output, d_out, array_out.shape(0)*sizeof(int64_t), cudaMemcpyDeviceToHost ));

	checkCudaErrors( cudaDeviceSynchronize() );


	cudaFree(d_out);
	cudaFree(data);
	// do not free "output" as this is a pointer to the numpy memory!

}


BOOST_PYTHON_MODULE(libEntropyCalc)
{
	np::initialize();
	def("EC", EC);
}
