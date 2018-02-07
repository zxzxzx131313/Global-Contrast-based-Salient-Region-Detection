# Global Contrast based Salient Region Detection

This is an implementation of Global Contrast based Salient Region Detection by Mingming Cheng et al. CVPR 2011. The graph segmentation part of this implementation is modified from David Stutz's graph-based image segmentation(https://github.com/davidstutz/graph-based-image-segmentation) see license below.

## Building

The implementation is based on [CMake](https://cmake.org/), [OpenCV](http://opencv.org/) and [Boost](http://www.boost.org/). The following steps have been tested on Ubuntu 12.04:

    $ sudo apt-get install build-essential
    $ sudo apt-get install cmake
    $ sudo apt-get install libboost-all-dev

OpenCV can either be installed following [these instructions](http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html#linux-installation), or using:

    $ sudo apt-get install libopencv-dev

With all requirements installed, run:

    $ mkdir build
    $ cd build
    $ cmake ..
    $ make

## Usage

The provided tool can easily be used as follows (from within the `build` directory):

    # Show a help message.
    $ ../bin/refh_cli --help
    Allowed options:
      -h [ --help ]            produce help message
      --input arg              folder containing the images to process
      --threshold arg (=1000)    constant for threshold function
      --minimum-size arg (=200) minimum component size
      --region-distance arg (=500) weight for regional distance
      --output arg (=output)   save segmentation as CSV file and contour images
    # Oversegment the provided examples:
    $ ../bin/refh_cli ../data/ ../output

The latter command will create the `output` directory containing the oversegmentations as `.csv` files and visualizations as `.png` files.

## License

Note that the two provided imags are taken from the [BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html) [2].

    [2] P. Arbelaez, M. Maire, C. Fowlkes and J. Malik.
        Contour Detection and Hierarchical Image Segmentation
        IEEE TPAMI, Vol. 33, No. 5, pp. 898-916, May 2011.

Copyright (c) 2016, David Stutz
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
