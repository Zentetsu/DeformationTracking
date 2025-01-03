cmake_minimum_required(VERSION 3.12)
project(Tracking VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)

# Define folders
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

set(HEADER_FILES

    # ${INCLUDE_DIR}/Tracking.hpp
)
set(SOURCE_FILES
    ${SRC_DIR}/Tracking.cpp
)

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})

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

find_package(VISP REQUIRED visp_core visp_mbt visp_io visp_gui visp_sensor)

if(VISP_FOUND)
    include_directories(${VISP_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${VISP_LIBRARIES})
else(VISP_FOUND)
    message(FATAL_ERROR "ViSP is required to compile. Please install it.")
endif(VISP_FOUND)

if(VISP_HAVE_PCL)
    list(APPEND CXX_FLAGS_MUTE_WARNINGS "-Wno-missing-field-initializers")
    list(APPEND CXX_FLAGS_MUTE_WARNINGS "-Wno-shadow")
endif()

target_include_directories(${PROJECT_NAME} PUBLIC " $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src> ")
target_include_directories(${PROJECT_NAME} PUBLIC " $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include> ")
target_include_directories(${PROJECT_NAME} PUBLIC " $<INSTALL_INTERFACE:include> ")

# set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS " -DSOFA_BUILD_MYPLUGIN ")
# set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
# target_link_libraries(${PROJECT_NAME}
# Sofa.Core
# Sofa.Component.Controller

# # Sofa.Helper
# )
