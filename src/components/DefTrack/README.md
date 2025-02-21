# DeformationTracking

SOFA plugin to track the surface of non-rigid objects using RGB-D camera.

## Dependencies

* ViSP 3.2.0 
* PCL 1.9
* Boost 1.57.0
* CMake 3.15
* VTK 6.2.0
* OpenCV 3.0


## Build Steps

1. Ensure that the DeformationTracking source code is in /sofa/applications/plugins/DeformationTracking/ directory
2. Copy contents of 'data' directory to the build directory
3. Move the .vtk and .obj files to sofa/share/mesh/ directory
4. Check the paths specified in: sofa/applications/plugins/DeformationTracking/scene/parameter.properties
5. Recompile Sofa
