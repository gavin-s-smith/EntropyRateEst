/*
 * npConvertion.h
 *
 *  Created on: 9 Jun 2013
 *      Author: gavin
 */

#ifndef NPCONVERTION_H_
#define NPCONVERTION_H_

#include <stdio.h>
#include <math.h>
#include <boost/unordered_map.hpp>
#include <boost/program_options.hpp>
#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include "helper_cuda.h"

namespace p = boost::python;
namespace np = boost::numpy;

double * get_pointer_double_array( np::ndarray const & array )
{
	if (array.get_dtype() != np::dtype::get_builtin<double>()) {
	        PyErr_SetString(PyExc_TypeError, "Incorrect array data type. Make sure you create numpy arrays with numpy.float64");
	        p::throw_error_already_set();
	    }
	    if (array.get_nd() != 1) {
	        PyErr_SetString(PyExc_TypeError, "Incorrect number of dimensions. This method is for 1D arrays only.");
	        p::throw_error_already_set();
	    }

	    if (!(array.get_flags() & np::ndarray::C_CONTIGUOUS)) {
	        PyErr_SetString(PyExc_TypeError, "Array must be row-major contiguous. Use numpy.require in Python to ensure this before calling this method.");
	        p::throw_error_already_set();
	    }

	    return reinterpret_cast<double*>(array.get_data());
}

double * get_gpu_pointer_double_array( np::ndarray const & array )
{
	if (array.get_dtype() != np::dtype::get_builtin<double>()) {
	        PyErr_SetString(PyExc_TypeError, "Incorrect array data type. Make sure you create numpy arrays with numpy.float64");
	        p::throw_error_already_set();
	    }
	    if (array.get_nd() != 1) {
	        PyErr_SetString(PyExc_TypeError, "Incorrect number of dimensions. This method is for 1D arrays only.");
	        p::throw_error_already_set();
	    }

	    if (!(array.get_flags() & np::ndarray::C_CONTIGUOUS)) {
	        PyErr_SetString(PyExc_TypeError, "Array must be row-major contiguous. Use numpy.require in Python to ensure this before calling this method.");
	        p::throw_error_already_set();
	    }

	    double * cpu_pointer = reinterpret_cast<double*>(array.get_data());

	    double * gpu_pointer;
	    checkCudaErrors( cudaMalloc((void**)&gpu_pointer, array.shape(0) * sizeof(double)) );
	    checkCudaErrors( cudaMemcpy(gpu_pointer, cpu_pointer, array.shape(0) * sizeof(double), cudaMemcpyHostToDevice ) );

	    return gpu_pointer;
}

unsigned * get_pointer_unsigned_array( np::ndarray const & array )
{
	if (array.get_dtype() != np::dtype::get_builtin<uint>()) {
	        PyErr_SetString(PyExc_TypeError, "Incorrect array data type. Make sure you create numpy arrays with numpy.uint32");
	        p::throw_error_already_set();
	    }
	    if (array.get_nd() != 1) {
	        PyErr_SetString(PyExc_TypeError, "Incorrect number of dimensions. This method is for 1D arrays only.");
	        p::throw_error_already_set();
	    }

	    if (!(array.get_flags() & np::ndarray::C_CONTIGUOUS)) {
	        PyErr_SetString(PyExc_TypeError, "Array must be row-major contiguous. Use numpy.require in Python to ensure this before calling this method.");
	        p::throw_error_already_set();
	    }

	    return reinterpret_cast<unsigned*>(array.get_data());
}


int * get_pointer_int_array( np::ndarray const & array )
{
	if (array.get_dtype() != np::dtype::get_builtin<int>()) {
			PyErr_SetString(PyExc_TypeError, "Incorrect array data type. Make sure you create numpy arrays with numpy.int32");
	        p::throw_error_already_set();
	    }
	    if (array.get_nd() != 1) {
	        PyErr_SetString(PyExc_TypeError, "Incorrect number of dimensions. This method is for 1D arrays only.");
	        p::throw_error_already_set();
	    }

	    if (!(array.get_flags() & np::ndarray::C_CONTIGUOUS)) {
	        PyErr_SetString(PyExc_TypeError, "Array must be row-major contiguous. Use numpy.require in Python to ensure this before calling this method.");
	        p::throw_error_already_set();
	    }

	    return reinterpret_cast<int*>(array.get_data());
}

int * get_gpu_pointer_int_array( np::ndarray const & array )
{
	if (array.get_dtype() != np::dtype::get_builtin<int>()) {
	        PyErr_SetString(PyExc_TypeError, "Incorrect array data type. Make sure you create numpy arrays with numpy.int32");
	        p::throw_error_already_set();
	    }
	    if (array.get_nd() != 1) {
	        PyErr_SetString(PyExc_TypeError, "Incorrect number of dimensions. This method is for 1D arrays only.");
	        p::throw_error_already_set();
	    }

	    if (!(array.get_flags() & np::ndarray::C_CONTIGUOUS)) {
	        PyErr_SetString(PyExc_TypeError, "Array must be row-major contiguous. Use numpy.require in Python to ensure this before calling this method.");
	        p::throw_error_already_set();
	    }


	    int * cpu_pointer = reinterpret_cast<int*>(array.get_data());

	    int * gpu_pointer;
	    checkCudaErrors( cudaMalloc((void**)&gpu_pointer, array.shape(0) * sizeof(int)) );
	    checkCudaErrors( cudaMemcpy(gpu_pointer, cpu_pointer, array.shape(0) * sizeof(int), cudaMemcpyHostToDevice ) );

	    return gpu_pointer;

}



#endif /* NPCONVERTION_H_ */
