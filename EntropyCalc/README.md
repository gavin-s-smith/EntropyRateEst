GPU based Entropy Rate Calculation

This is a small library for python which computes the entropy rate of a symbolic sequence on a NVIDIA GPU.

It has been compiled and tested on Ubuntu 13.10 with a GeForce GTX 580 (CUDA 5.5).

Dependencies:
CUDA
Boost.Numpy

Compilation is via the makefile in the Release directory.

Some install notes for Ubuntu 13.10
===================================
Installing CUDA:
Install via the .deb package provided by NVIDIA (tested version 5.5, used .deb for Ubuntu 12.10)

Installing Boost.Numpy:
sudo apt-get install scons
Download zip from this site
Extract and from within the ./Boost.NumPy-master directory:
scons
sudo scons install
NOTE: If you get errors surrounding libbost_numpy.so it may help to make a symlink in /usr/lib, i.e.
sudo ln -s /usr/local/lib/libboost_numpy.so /usr/lib/libboost_numpy.so