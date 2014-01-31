This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

==============================================================================

Author: Gavin Smith
Organization: Horizon Digital Economy Institute, The University of Nottingham.
Contact: See www.cs.nott.ac.uk/~gss

This code was originally developed to support work on determining a refined
upper bound on the limit of human predictability: http://www.cs.nott.ac.uk/~gss/mobility.php

===============================================================================

# GPU based Entropy Rate Calculation

This is a small library for python which computes the entropy rate of a symbolic sequence on a NVIDIA GPU.

It has been compiled and tested on Ubuntu 13.10 with a GeForce GTX 580 (CUDA 5.5).

Dependencies:
CUDA
Boost.Numpy

Compilation is via the makefile in the Release directory.

### Some install notes for Ubuntu 13.10

Installing CUDA:
```
Install via the .deb package provided by NVIDIA (tested version 5.5, used .deb for Ubuntu 12.10)
```

Installing Boost.Numpy:
```
sudo apt-get install scons
Download zip from this site
Extract and from within the ./Boost.NumPy-master directory:
scons
sudo scons install
NOTE: If you get errors surrounding libbost_numpy.so it may help to make a symlink in /usr/lib, i.e.
sudo ln -s /usr/local/lib/libboost_numpy.so /usr/lib/libboost_numpy.so
```