#!/bin/bash

# Remember repository path
# This path pulled with docker image running
REPO_PATH=$(pwd)

# Build OpenCV from source code
# Install OpenCV dependencies
cd $REPO_PATH

# System dependencies
apt-get update
apt-get install -y git-lfs
apt-get install -y curl wget nano bzip2 libfreetype6 libgl1-mesa-dev libglu1-mesa libxi6 libxrender1
apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev
apt-get install -y unzip
apt-get install -y zip
apt-get install -y python3-dev python3-pip
pip3 install --upgrade pip

apt -y remove x264 libx264-dev

## Install dependencies
apt -y install build-essential checkinstall cmake pkg-config yasm
apt -y install git gfortran
apt -y install libjpeg8-dev libpng-dev

apt -y install software-properties-common
add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
apt -y update

apt -y install libjasper1
apt -y install libtiff-dev

apt -y install libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev
apt -y install libxine2-dev libv4l-dev
cd /usr/include/linux
ln -s -f ../libv4l1-videodev.h videodev.h
cd /tmp/

apt -y install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
apt -y install libgtk2.0-dev libtbb-dev qt5-default
apt -y install libatlas-base-dev
apt -y install libfaac-dev libmp3lame-dev libtheora-dev
apt -y install libvorbis-dev libxvidcore-dev
apt -y install libopencore-amrnb-dev libopencore-amrwb-dev
apt -y install libavresample-dev
apt -y install x264 v4l-utils

# Optional dependencies
apt -y install libprotobuf-dev protobuf-compiler
apt -y install libgoogle-glog-dev libgflags-dev
apt -y install libgphoto2-dev libeigen3-dev libhdf5-dev doxygen

# Download OpenCV source code
mkdir build_opencv; cd build_opencv
wget https://github.com/opencv/opencv/archive/4.5.1.zip -O opencv-4.5.1.zip
wget https://github.com/opencv/opencv_contrib/archive/4.5.1.zip -O opencv_contrib-4.5.1.zip
unzip opencv-4.5.1.zip
unzip opencv_contrib-4.5.1.zip
cd opencv-4.5.1/;mkdir build; cd build


#  Build OpenCV
# Simple building:
### Section 1
cmake -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.5.1/modules ..
### End of Section 1

make -j$(nproc)
make install
ldconfig

cd ../../../
rm -r build_opencv

# Then return to repository folder
cd $REPO_PATH
