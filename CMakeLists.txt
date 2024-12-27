cmake_minimum_required(VERSION 3.12)
project(MyPlugin VERSION 1.0)

include(cmake/environment.cmake)

find_package(Sofa.Config REQUIRED)
sofa_find_package(Sofa.Core REQUIRED)
sofa_find_package(Sofa.Component.Controller REQUIRED)
find_package(Sofa.Helper REQUIRED)

set(SRC_DIR "binding")

set(HEADER_FILES
    src/initMyPlugin.h
    src/components/MyComponent/MyComponent.h
    src/components/DeformationTracking/config.h
    src/components/DeformationTracking/JacobianFEM.h
    src/components/DeformationTracking/RigidTracking.h
    src/components/DeformationTracking/OcclusionCheck.h
    src/components/DeformationTracking/NonRigidMatching.h
    src/components/DeformationTracking/DeformationTracking.h
)
set(SOURCE_FILES
    src/initMyPlugin.cpp
    src/components/MyComponent/MyComponent.cpp
    src/components/DeformationTracking/JacobianFEM.cpp
    src/components/DeformationTracking/RigidTracking.cpp
    src/components/DeformationTracking/OcclusionCheck.cpp
    src/components/DeformationTracking/NonRigidMatching.cpp
    src/components/DeformationTracking/DeformationTracking.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

find_package(PCL 1.9 REQUIRED COMPONENTS common io search visualization filters segmentation)

if(PCL_FOUND)
    include_directories(${PCL_INCLUDE_DIRS})
    add_definitions(${PCL_DEFINITIONS})
    message(${PCL_LIBRARY_DIRS})
    link_directories(${PCL_LIBRARY_DIRS})
    target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES} ${REALSENSE_LIBRARY})
else(PCL_FOUND)
    message(FATAL_ERROR "The Point Cloud Library (PCL) is required to compile. Please install it.")
endif(PCL_FOUND)

find_package(OpenCV REQUIRED)

if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
else(OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV is required to compile. Please install it.")
endif(OpenCV_FOUND)

find_package(VTK REQUIRED)

if(VTK_FOUND)
    include_directories(${VTK_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${VTK_LIBRARIES})
else(VTK_FOUND)
    message(FATAL_ERROR "VTK is required to compile. Please install it.")
endif(VTK_FOUND)

find_package(VISP REQUIRED)

if(VISP_FOUND)
    include_directories(${VISP_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${VISP_LIBRARIES})
else(VISP_FOUND)
    message(FATAL_ERROR "ViSP is required to compile. Please install it.")
endif(VISP_FOUND)

target_include_directories(${PROJECT_NAME} PUBLIC " $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src> ")
target_include_directories(${PROJECT_NAME} PUBLIC " $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include> ")
target_include_directories(${PROJECT_NAME} PUBLIC " $<INSTALL_INTERFACE:include> ")

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS " -DSOFA_BUILD_MYPLUGIN ")
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
target_link_libraries(${PROJECT_NAME}
    Sofa.Core
    Sofa.Component.Controller

    # Sofa.Helper
)
