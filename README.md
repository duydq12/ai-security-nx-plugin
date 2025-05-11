# Project: Network Optix Meta Plugin for Camera Security

## Description

This project is a C++ analytics plugin for the Network Optix video management system. It is designed to perform real-time object detection and tracking on video streams. The plugin utilizes the YOLOv11 model for object detection and classification, and OpenCV for image processing and tracking functionalities. It integrates with the Network Optix SDK to process video frames and push metadata back to the system.

## Dependencies
* **Base Operating System:**
    * Ubuntu 22.04

* **System Packages (installed via APT):**
    * `build-essential` (for compiling software)
    * `cmake` (build system generator)
    * `g++-12` (C++ compiler, configured as default `g++` and `gcc`)
    * `unzip` (for extracting zip files)
    * `python3.10`
    * `python3-pip` (Python package installer)
    * `wget` (for downloading files)
    * `sudo` (for privilege escalation)
    
* **Software Development Kits (SDKs):**
    * Network Optix Metavms Metadata SDK (version 5.1.5.39242, installed in `/opt`)

* **Python Packages (typically installed via pip post-container startup for development):**
    * `conan<2` (C/C++ package manager - installation step is manual within the container as per previous instructions)

* **Key Configurations:**
    * Default C/C++ compiler set to GCC/G++ version 12.

## Development Mode Setup and Usage

### 1. Building plugin

```bash
mkdir build && cd build
cmake -DmetadataSdkDir=/opt/metadata_sdk .. 
cmake --build . 
```

### Install plugin

```bash
sudo systemctl stop networkoptix-metavms-mediaserver
sudo mkdir $SERVER_DIR/bin/plugins/opencv_object_detection_analytics_plugin
sudo rm $SERVER_DIR/bin/plugins/libopencv_object_detection_analytics_plugin.so
sudo cp $BUILD_DIR/libopencv_object_detection_analytics_plugin.so $SERVER_DIR/bin/plugins/opencv_object_detection_analytics_plugin
sudo cp $BUILD_DIR/MobileNetSSD.caffemodel $BUILD_DIR/MobileNetSSD.prototxt $SERVER_DIR/bin/plugins/opencv_object_detection_analytics_plugin
sudo systemctl start networkoptix-metavms-mediaserver
```
