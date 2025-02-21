////////////////////////////////////////////////////////////////////////////////
// Copyright 2017 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License.  You may obtain a copy
// of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
// License for the specific language governing permissions and limitations
// under the License.
////////////////////////////////////////////////////////////////////////////////
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>

#ifdef _WIN32
#include <intrin.h>
#elif defined(__i386__) || defined(__x86_64__)
#include <immintrin.h>
#elif defined(__ARM_FEATURE_SIMD32) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

// #include "MaskedOcclusionCulling.h"
#include "OcclusionCheck.h"

OcclusionCheck::OcclusionCheck() : _A(3, 3), _b(3, 1), _transform_point_P(4, 1), _transform_point_T(4, 4) {}

float OcclusionCheck::getTriangleArea(point_2d &A, point_2d &B, point_2d &C) {
    float ab = EUCLEDIAN_DIST_2D(A.x, A.y, B.x, B.y);
    float bc = EUCLEDIAN_DIST_2D(C.x, C.y, B.x, B.y);
    float ac = EUCLEDIAN_DIST_2D(A.x, A.y, C.x, C.y);

    float semi_perimeter = (ab + bc + ac) / 2.0f;

    return sqrt(semi_perimeter * (semi_perimeter - ab) * (semi_perimeter - bc) * (semi_perimeter - ac));
}

/**
Bilinear interpolation of values in a 2D image

@param x : position of the value at which the interpolation must be done at - X   Jaguar1##

@param y : position of the value at which the interpolation must be done at - Y coordinates
@param A : value at left-upper corner
@param B : value at right-upper corner
@param C : value at left-lower corner
@param D : value at right-lower corner


  A++++++B
  +      +
  +      +
  +      +
  C++++++D

  Orientation for interpolation. The values in the argument are interpreted as the figure shown above.


@returns float : interpolated value*/
float OcclusionCheck::bilinearInterpolationAtValues(float x, float y, float A, float B, float C, float D) {
    int x_ = static_cast<int>(floor(x));
    int y_ = static_cast<int>(floor(y));
    float del_x = x - static_cast<float>(x_);
    float del_y = y - static_cast<float>(y_);

    float intensity = 0.0f;

    intensity = (A * (1 - del_x) * (1 - del_y)) + (C * (del_x) * (1 - del_y)) + (B * (1 - del_x) * (del_y)) + (D * (del_x) * (del_y));

    return A;
}

/**
Read raw depth data

@param path : path to depth data


@returns the loaded depth file in OpenCV's Mat format*/
cv::Mat OcclusionCheck::read_raw_depth_data(string path) {
    cv::Mat img;
    FILE *fp = NULL;
    char *imagedata = NULL;
    int framesize = IMAGE_WIDTH * IMAGE_HEIGHT;

    // Open raw Bayer image.
    fp = fopen(path.c_str(), "rb");

    // Memory allocation for bayer image data buffer.
    imagedata = (char *)malloc(sizeof(char) * framesize);

    // Read image data and store in buffer.
    fread(imagedata, sizeof(char), framesize, fp);

    // Create Opencv mat structure for image dimension. For 8 bit bayer, type should be CV_8UC1.
    img.create(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8U);

    memcpy(img.data, imagedata, framesize);

    free(imagedata);

    fclose(fp);

    return img;
}

/**
Loads a depth map from a PNG file

@param path : path to depth data in PNG format


@returns the loaded depth file in OpenCV's Mat format*/
cv::Mat OcclusionCheck::read_png_depth_data(string path) {
    _log_("Trying to load: " << path) cv::Mat img = cv::imread(path);

    return img;
}

/**
Loads a depth map from a CSV file

@param path : path to depth data in csv format


@returns the loaded depth file in OpenCV's Mat format*/
cv::Mat OcclusionCheck::read_csv_depth_data(string path) {
    cv::Mat img;
    img.create(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32FC1);

    int row = 0;
    string delimiter = ",";

    ifstream file(path);
    string str;
    while (std::getline(file, str)) {
        int col = 0;
        size_t pos = 0;
        string token;
        while ((pos = str.find(delimiter)) != string::npos) {
            token = str.substr(0, pos);
            str.erase(0, pos + delimiter.length());
            double depth = stod(token);

            if (depth > MAX_DEPTH) {
                img.at<float>(row, col) = 0.0f;
            } else {
                img.at<float>(row, col) = depth;
            }

            col++;
        }
        row++;
    }

    return img;
}

/**
Transforms an organized pointcloud by a transformation matrix

@param cloud : input - 3D point in PCL's PointXYZRGB format
@param depth_image :
@param transform : tranformation matrix
@param fx : camera intrinsics
@param fy : camera intrinsics
@param cx : camera intrinsics
@param cy : camera intrinsics

Transforming organized pointcloud requires re-projection of the transformed points using pinhole camera model

@returns the transformed pointcloud in XYZRGB format*/
PointCloud<PointXYZRGB>::Ptr OcclusionCheck::transformPointcloudXYZRGB(PointCloud<PointXYZRGB>::Ptr &cloud, cv::Mat &depth_image, Eigen::Matrix4f &transform, float fx, float fy, float cx, float cy) {
    PointCloud<PointXYZRGB>::Ptr transformed_cloud(new PointCloud<PointXYZRGB>());
    cv::Mat colored_depth;
    colored_depth.create(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32FC1);
    transformed_cloud->height = cloud->height;
    transformed_cloud->width = cloud->width;
    transformed_cloud->is_dense = false;
    transformed_cloud->points.resize(cloud->width * cloud->height);

    PointXYZ pt;
    PointXYZRGB pt_rgb;
    point_2d pt_img;

    int debug_flag = 0;
    int i = 0;
    int j = 0;
    int _X;
    int _Y;

    try {
        for (i = 0; i < transformed_cloud->width; i++) {
            for (j = 0; j < transformed_cloud->height; j++) {
                debug_flag = 0;
                pt_rgb = cloud->at(i, j);
                if (fabs(pt_rgb.z) > FLOAT_ZERO) {
                    debug_flag++;
                    pt = transform_point(pt_rgb, transform);
                    debug_flag++;
                    pt_img = projection_raw(pt, fx, fy, cx, cy);
                    debug_flag++;
                    _X = (int)round(pt_img.x);
                    _Y = (int)round(pt_img.y);
                    if ((_X > 0.0f) && (_X < transformed_cloud->width) && (_Y > 0.0f) && (_Y < transformed_cloud->height)) {
                        transformed_cloud->at(_X, _Y).x = pt.x;
                        transformed_cloud->at(_X, _Y).y = pt.y;
                        transformed_cloud->at(_X, _Y).z = pt.z;
                        debug_flag++;
                        transformed_cloud->at(_X, _Y).r = pt_rgb.r;
                        transformed_cloud->at(_X, _Y).g = pt_rgb.g;
                        transformed_cloud->at(_X, _Y).b = pt_rgb.b;
                        debug_flag++;

                        colored_depth.at<float>((int)round(pt_img.y), (int)round(pt_img.x)) = (float)depth_image.at<float>(j, i);
                    }
                }
            }
        }
    } catch (const std::out_of_range &oor) {
        _hline_ _error_("Out of range error: " << oor.what() << endl
                                               << "at point: " << pt_rgb << " with transformed point: " << pt << " and debug status: " << debug_flag << " at loop: (" << i << "," << j << ")" << endl
                                               << " with projected points: [" << pt_img.x << "," << pt_img.y << "]" << endl) _hline_
    }

    depth_image = colored_depth;

    return transformed_cloud;
}

/**
Transforms a 3D point with RGB value by a given transformation matrix

@param p : input - 3D point in PCL's PointXYZRGB format
@param transform : input - transformation matrix in Eigen's format

This method strips the RGB value from the input point 'p'

@returns the transformed 3D point in PCL's PointXYZ format*/
PointXYZ OcclusionCheck::transform_point(PointXYZRGB &p, Eigen::Matrix4f &transform) {
    PointXYZ _transform_point_p;

    _transform_point_P(0, 0) = p.x;
    _transform_point_P(1, 0) = p.y;
    _transform_point_P(2, 0) = p.z;
    _transform_point_P(3, 0) = 1.0f;

    _transform_point_T(0, 0) = transform(0, 0);
    _transform_point_T(0, 1) = transform(0, 1);
    _transform_point_T(0, 2) = transform(0, 2);
    _transform_point_T(0, 3) = transform(0, 3);
    _transform_point_T(1, 0) = transform(1, 0);
    _transform_point_T(1, 1) = transform(1, 1);
    _transform_point_T(1, 2) = transform(1, 2);
    _transform_point_T(1, 3) = transform(1, 3);
    _transform_point_T(2, 0) = transform(2, 0);
    _transform_point_T(2, 1) = transform(2, 1);
    _transform_point_T(2, 2) = transform(2, 2);
    _transform_point_T(2, 3) = transform(2, 3);
    _transform_point_T(3, 0) = transform(3, 0);
    _transform_point_T(3, 1) = transform(3, 1);
    _transform_point_T(3, 2) = transform(3, 2);
    _transform_point_T(3, 3) = transform(3, 3);

    _transform_point_P = _transform_point_T * _transform_point_P;

    _transform_point_p.x = _transform_point_P(0, 0);
    _transform_point_p.y = _transform_point_P(1, 0);
    _transform_point_p.z = _transform_point_P(2, 0);

    return _transform_point_p;
}

/**
Transforms a 3D point by a given transformation matrix

@param p : input - 3D point in PCL's PointXYZ format
@param transform : input - transformation matrix in Eigen's format


@returns the transformed 3D point in PCL's PointXYZ format*/
PointXYZ OcclusionCheck::transform_point_XYZ(PointXYZ &p, Eigen::Matrix4f &transform) {
    PointXYZ _transform_point_p;

    _transform_point_P(0, 0) = p.x;
    _transform_point_P(1, 0) = p.y;
    _transform_point_P(2, 0) = p.z;
    _transform_point_P(3, 0) = 1.0f;

    _transform_point_T(0, 0) = transform(0, 0);
    _transform_point_T(0, 1) = transform(0, 1);
    _transform_point_T(0, 2) = transform(0, 2);
    _transform_point_T(0, 3) = transform(0, 3);
    _transform_point_T(1, 0) = transform(1, 0);
    _transform_point_T(1, 1) = transform(1, 1);
    _transform_point_T(1, 2) = transform(1, 2);
    _transform_point_T(1, 3) = transform(1, 3);
    _transform_point_T(2, 0) = transform(2, 0);
    _transform_point_T(2, 1) = transform(2, 1);
    _transform_point_T(2, 2) = transform(2, 2);
    _transform_point_T(2, 3) = transform(2, 3);
    _transform_point_T(3, 0) = transform(3, 0);
    _transform_point_T(3, 1) = transform(3, 1);
    _transform_point_T(3, 2) = transform(3, 2);
    _transform_point_T(3, 3) = transform(3, 3);

    _transform_point_P = _transform_point_T * _transform_point_P;

    _transform_point_p.x = _transform_point_P(0, 0);
    _transform_point_p.y = _transform_point_P(1, 0);
    _transform_point_p.z = _transform_point_P(2, 0);

    return _transform_point_p;
}

/**
Load a color image from path and, given the intrinsics and extrinsics, register this image with a pointcloud

@param color_image_path : input - path to color image
@param cloud : output - the pointcloud to register, should have valid depth data
@param data_map : the camera intrinsics and extrinsics are stored in this structure
@param color_image : output - the loaded color image
@param colored_depth_image : output - depth map registered with the color image in cv:Mat format
@param count : file number (unused variable), could be useful while logging debug messages etc.

Note: Any depth value over 'MAX_DEPTH' is discarded, the image must be of dimension IMAGE_HEIGHT x IMAGE_WIDTH - else it crashes (check the value of these macros in the header file)
The 'BLUR_IMAGE' macro, if enabled, will apply a Gaussian blur of kernel sized 3x3 on the colored image before registering it

@returns void */
void OcclusionCheck::load_image_and_register(string color_image_path, PointCloud<PointXYZRGB> &cloud, offline_data_map &data_map, cv::Mat &color_image, cv::Mat &colored_depth_image, int count) {
    PointCloud<PointXYZRGB>::Ptr transformed_cloud(new PointCloud<PointXYZRGB>());

    color_image = cv::imread(color_image_path, cv::IMREAD_GRAYSCALE);
    cv::Mat float_color_image;
    color_image.convertTo(float_color_image, CV_32F);

    cv::Mat colored_depth;
    colored_depth.create(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32FC1);

    cv::Mat aligned_depth;
    aligned_depth.create(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32FC1);

#ifdef BLUR_IMAGE
    cv::GaussianBlur(float_color_image, float_color_image, cv::Size(3, 3), 0, 0);
#endif
    transformPointCloud(cloud, *transformed_cloud, data_map.depth2color_extrinsics);
    PointXYZ pt;

    float A;
    float B;
    float C;
    float D;
    try {
        for (int i = 0; i < transformed_cloud->width; i++) {
            for (int j = 0; j < transformed_cloud->height; j++) {
                pt.x = transformed_cloud->at(i, j).x;
                pt.y = transformed_cloud->at(i, j).y;
                pt.z = transformed_cloud->at(i, j).z;

                if ((pt.z > 0.0f) && (pt.z < MAX_DEPTH)) {
                    point_2d point = projection_raw(pt, data_map.color_Fx, data_map.color_Fy, data_map.color_Cx, data_map.color_Cy);

                    if ((point.x > 0) && (point.x < transformed_cloud->width) && (point.y > 0) && (point.y < transformed_cloud->height)) {
                        float x_overflow = fabs(point.x - ((float)((int)point.x)));
                        float y_overflow = fabs(point.y - ((float)((int)point.y)));
                        if ((x_overflow > 0.001f) || (y_overflow > 0.001f)) {
                            A = (float)float_color_image.at<float>(floor(point.y),
                                                                   floor(point.x));  //((float)color_image.at<cv::Vec3b>(floor(point.y),floor(point.x))[2]+(float)color_image.at<cv::Vec3b>(floor(point.y),floor(point.x))[1]+(float)color_image.at<cv::Vec3b>(floor(point.y),floor(point.x))[0])/3.0f ;
                            B = (float)float_color_image.at<float>(floor(point.y),
                                                                   ceil(point.x));  // ((float)color_image.at<cv::Vec3b>(floor(point.y),ceil(point.x))[2] +(float)color_image.at<cv::Vec3b>(floor(point.y),ceil(point.x))[1] +(float)color_image.at<cv::Vec3b>(floor(point.y),ceil(point.x))[0] )/3.0f ;
                            C = (float)float_color_image.at<float>(ceil(point.y),
                                                                   floor(point.x));  // ((float)color_image.at<cv::Vec3b>(ceil(point.y),floor(point.x))[2] +(float)color_image.at<cv::Vec3b>(ceil(point.y),floor(point.x))[1] +(float)color_image.at<cv::Vec3b>(ceil(point.y),floor(point.x))[0] )/3.0f ;
                            D = (float)float_color_image.at<float>(ceil(point.y),
                                                                   ceil(point.x));  //  ((float)color_image.at<cv::Vec3b>(ceil(point.y),ceil(point.x))[2]  +(float)color_image.at<cv::Vec3b>(ceil(point.y),ceil(point.x))[1]  +(float)color_image.at<cv::Vec3b>(ceil(point.y),ceil(point.x))[0]  )/3.0f ;

                            float pixel_intensity = bilinearInterpolationAtValues(point.x, point.y, A, B, C, D);

                            uint32_t rgb = ((uint32_t)pixel_intensity << 16 | (uint32_t)pixel_intensity << 8 | (uint32_t)pixel_intensity);
                            cloud.at(i, j).rgb = *reinterpret_cast<float *>(&rgb);
                            colored_depth.at<float>(j, i) = pixel_intensity;
                            aligned_depth.at<float>(j, i) = pt.z;
                        } else {
                            float pixel_intensity = (float)float_color_image.at<float>((int)round(point.y), (int)round(point.x));
                            uint32_t rgb = ((uint32_t)pixel_intensity << 16 | (uint32_t)pixel_intensity << 8 | (uint32_t)pixel_intensity);
                            cloud.at(i, j).rgb = *reinterpret_cast<float *>(&rgb);

                            colored_depth.at<float>(j, i) = pixel_intensity;
                            aligned_depth.at<float>(j, i) = pt.z;
                        }
                    }
                }
            }
        }
    } catch (cv::Exception &e) {
        _error_("Error zhile loqding file")
    }

    colored_depth.convertTo(colored_depth_image, CV_8UC1, 1.0);
}

/**
Converts a depth image to XYZRGB pointcloud

@param depth_image : input - the depth image in OpenCV's Mat format
@param cloud : output - the empty pointcloud
@param data_map : the camera intrinsics are stored in this structure, only the values corresponding to the 'depth' camera are used

The RGB value in the XYZRGB pointcloud is set to an arbitrary (100, 149, 237) [cornflowerblue].\n
To assign actual RGB value from depth image, check the method 'load_image_and_register'

@returns void */
void OcclusionCheck::depth_to_pcl(cv::Mat &depth_image, PointCloud<PointXYZRGB> &cloud, offline_data_map &data_map) {
    PointCloud<PointXYZRGB> cloud_;

    cloud_.width = IMAGE_WIDTH;
    cloud_.height = IMAGE_HEIGHT;
    cloud_.is_dense = false;
    cloud_.points.resize(cloud_.width * cloud_.height);

    PointXYZ pt;

    uint8_t r = 100, g = 149, b = 237;
    uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);

    for (int i = 1; i < (IMAGE_WIDTH - 1); i++) {
        for (int j = 1; j < (IMAGE_HEIGHT - 1); j++) {
#ifdef DEPTH_FORMAT_PNG
            double depth_value = (float)depth_image.at<cv::Vec3b>(j, i)[0] / 10.0;
#else
            double depth_value = depth_image.at<float>(j, i);
#endif
            inverse_projection_raw(i, j, depth_value, data_map.depth_Fx, data_map.depth_Fy, data_map.depth_Cx, data_map.depth_Cy, pt);
            cloud_.at(i, j).x = pt.x;
            cloud_.at(i, j).y = pt.y;
            cloud_.at(i, j).z = pt.z;

            cloud_.at(i, j).rgb = *reinterpret_cast<float *>(&rgb);
        }
    }

    cloud = cloud_;
}

/**
Picks out 3 consecutive columns from an EigenMatrix and creates a vpMatrix out of it

@param E : input - Eigen matrix
@param V : output - ViSP Matrix
@param start_col_index : index number of column from which the three consecutive columns are picked up from


@returns void */
void OcclusionCheck::eigen_to_visp(EigenMatrix &E, vpMatrix &V) {
    V.resize(E.rows(), E.cols(), true);
    memcpy(V.data, E.data(), E.cols() * E.rows() * sizeof(double));
}

/**
Picks out 3 consecutive columns from an EigenMatrix and creates a vpMatrix out of it

@param E : input - Eigen matrix
@param V : output - ViSP Matrix
@param start_col_index : index number of column from which the three consecutive columns are picked up from


@returns void */
void OcclusionCheck::eigen_to_visp_3col(EigenMatrix &E, vpMatrix &V, int start_col_index) {
    V.resize(E.rows(), 3, true);
    for (int i = 0; i < E.rows(); i++) {
        for (int j = start_col_index; j < (start_col_index + 3); j++) {
            V[i][(j - start_col_index)] = E(i, j);
        }
    }
}

/**
Convert EigenMatrix to ViSP Matrix with additional weights

@param E : input - Eigen matrix (\f$ \mathbf{E} \f$)
@param V : output - ViSP Matrix (\f$ \mathbf{V} \f$)
@param W : a column vector containing the weights that are to be multiplied to the diagonals of E (\f$ \mathbf{W} \f$)

The function computes: \f$ \mathbf{V}_{(i,j)} = \mathbf{W}_i \mathbf{V}_{(i,j)}\f$

@returns void */
void OcclusionCheck::eigen_to_visp_weighted(EigenMatrix &E, vpMatrix &V, vpColVector &W) {
    V.resize(E.rows(), E.cols(), true);
    for (int i = 0; i < E.rows(); i++) {
        for (int j = 0; j < E.cols(); j++) {
            V[i][j] = W[i] * E(i, j);
        }
    }
}

/**
Read calibration file

@param I : output - calibration matrix
@param path : path to calibration file
@param vertex_size : number of vertices in the model


@returns void */
void OcclusionCheck::read_calibration_file(EigenMatrix &I, string path, int vertex_size) {
    ifstream file_path(path);
    string str;
    string delimiter = ",";
    double num = 0.0f;

    vector<vector<double>> matrix;

    while (std::getline(file_path, str)) {
        vector<double> row;
        while (str.length() > 0) {
            num = stof(str.substr(0, str.find(delimiter)));
            str.erase(0, str.find(delimiter) + delimiter.length());
            row.push_back(num);
        }

        matrix.push_back(row);
    }

    if ((matrix.size() != (3 * vertex_size)) || (matrix.size() != matrix[0].size())) {
        _warn_("Calibration matrix row/column size does not match 3 x 'no. of vertices' of the model (" << vertex_size << ")") _warn_("Read no. of lines: " << matrix.size() << " and no. of columns in the first row: " << matrix[0].size() << ", while no. of vertices of the model: #" << vertex_size)

            if (matrix.size() != matrix[0].size()) _fatal_("Calibration matrix is non-square! Exiting.")
    }

    I.resize(matrix.size(), matrix.size());

    for (int i = 0; i < I.rows(); i++) {
        for (int j = 0; j < I.cols(); j++) {
            I(i, j) = matrix[i][j];
        }
    }
}

/**
Change a 4x4 homogeneous transformation matrix from ViSP's format to Eigen

@param V : input - 4x4 ViSP homogeneous transformation matrix
@param E : output - 4x4 homogeneous transformation matrix in Eigen's format


@returns void */
void OcclusionCheck::visp_to_eigen(const vpMatrix &V, EigenMatrix &E) {
    E.resize(V.getRows(), V.getCols());
    memcpy(E.data(), V.data, V.getCols() * V.getRows() * sizeof(double));
}
/**
Load a 4x4 transformation matrix from a flat text file

@param path : path to text file

The transformation matrix should be encoded in the text file in the following format:

\f$R_{11}\f$ \f$R_{12}\f$ \f$R_{13}\f$ \f$t_x\f$\n
\f$R_{21}\f$ \f$R_{22}\f$ \f$R_{23}\f$ \f$t_y\f$\n
\f$R_{31}\f$ \f$R_{32}\f$ \f$R_{33}\f$ \f$t_z\f$\n
\f$0\f$ \f$0\f$ \f$0\f$ \f$1\f$\n

(every row should be followed by a newline character)

@returns double** a 2D double array containing the transformation matrix */
double **OcclusionCheck::load_pose(char *path) {
    double **transform = new double *[4];

    std::ifstream file(path);
    for (unsigned int i = 0; i < 4; i++) {
        transform[i] = new double[4];
        for (unsigned int j = 0; j < 4; j++) {
            file >> transform[i][j];
        }
    }
    return transform;
}
/**
Given a 4x4 homogeneous transformation matrix in Eigen's format, compute its inverse

@param E : input - 4x4 homogeneous transformation matrix
@param iE : output - 4x4 homogeneous transformation matrix, but inverted

Given \f$ \mathbf{T} \f$, this function returns \f$ \mathbf{T}^{-1} \f$

@returns void */
void OcclusionCheck::eigen_to_inv_eigen_4x4(Matrix4f E, Matrix4f &iE) {
    vpHomogeneousMatrix V;
    eigen_to_visp_4x4(E, V);
    V = V.inverse();
    visp_to_eigen_4x4(V, iE);
}
/**
Extract the 3x3 rotation matrix from a 4x4 gomogeneous transformation matrix

@param transform : input - homogeneous transformation matrix
@param rotation : output - extracted 3x3 rotation matrix

@returns void */
void OcclusionCheck::extract_rotation(const Eigen::Matrix4f &transform, Eigen::Matrix3f &rotation) {
    rotation(0, 0) = transform(0, 0);
    rotation(0, 1) = transform(0, 1);
    rotation(0, 2) = transform(0, 2);
    rotation(1, 0) = transform(1, 0);
    rotation(1, 1) = transform(1, 1);
    rotation(1, 2) = transform(1, 2);
    rotation(2, 0) = transform(2, 0);
    rotation(2, 1) = transform(2, 1);
    rotation(2, 2) = transform(2, 2);
}
/**
Convert a 4x4 homogeneous matrix from Eigen's format to ViSP's format

@param E : input - EigenMatrix of 4x4 dimension
@param V : output - vpHomogeneousMatrix denoting the input homogeneous matrix in ViSP format

@returns void */
void OcclusionCheck::eigen_to_visp_4x4(const Matrix4f E, vpHomogeneousMatrix &V) {
    V[0][0] = E(0, 0);
    V[0][1] = E(0, 1);
    V[0][2] = E(0, 2);
    V[0][3] = E(0, 3);
    V[1][0] = E(1, 0);
    V[1][1] = E(1, 1);
    V[1][2] = E(1, 2);
    V[1][3] = E(1, 3);
    V[2][0] = E(2, 0);
    V[2][1] = E(2, 1);
    V[2][2] = E(2, 2);
    V[2][3] = E(2, 3);
    V[3][0] = E(3, 0);
    V[3][1] = E(3, 1);
    V[3][2] = E(3, 2);
    V[3][3] = E(3, 3);
}
/**
Convert a 4x4 homogeneous matrix from ViSP's format to Eigen's format

@param V : vpHomogeneousMatrix denoting the input homogeneous matrix in ViSP format
@param E : output - EigenMatrix of 4x4 dimension

@returns void */
void OcclusionCheck::visp_to_eigen_4x4(const vpHomogeneousMatrix V, Matrix4f &E) {
    E(0, 0) = V[0][0];
    E(0, 1) = V[0][1];
    E(0, 2) = V[0][2];
    E(0, 3) = V[0][3];
    E(1, 0) = V[1][0];
    E(1, 1) = V[1][1];
    E(1, 2) = V[1][2];
    E(1, 3) = V[1][3];
    E(2, 0) = V[2][0];
    E(2, 1) = V[2][1];
    E(2, 2) = V[2][2];
    E(2, 3) = V[2][3];
    E(3, 0) = V[3][0];
    E(3, 1) = V[3][1];
    E(3, 2) = V[3][2];
    E(3, 3) = V[3][3];
}

/**
Load a PCD file in XYZ format (without the RGB information)

@param cloud : output pointcloud in PointCloud<pcl::PointXYZRGB> format
@param path : input path

@returns void */
void OcclusionCheck::load_pcd(pcl::PointCloud<pcl::PointXYZ> &cloud, std::string path) {
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(path, cloud) == -1) {
        (PCL_ERROR("Couldn't read file pcd"));
    }
}

/**
Load a PCD file in XYZRGB format

@param cloud : output pointcloud in PointCloud<pcl::PointXYZRGB> format
@param path : input path

@returns void */
void OcclusionCheck::load_pcd_rgb(pcl::PointCloud<pcl::PointXYZRGB> &cloud, std::string path) {
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(path, cloud) == -1) {
        (PCL_ERROR("Couldn't read file pcd"));
    }
}

/**
Tranforms a Polygonal mesh

@param inMesh : pcl::PolygonMesh representing the object model (the output overwrites this mesh)
@param transform : the \f$4 \times 4\f$ homogeneous transformation matrix (say \f$ \mathbf{T} \f$)

If the 3D vertices of the mesh can be arranged as\f$ {}^{\mathrm{A}}\mathbf{P} = \begin{bmatrix} \mathbf{p}_1 & \mathbf{p}_2 & \cdots \mathbf{p}_n \end{bmatrix}\f$ w.r.t arbitrary reference frame \f$ \mathrm{A} \f$
This method computes \f$ {}^{\mathrm{B}}\mathbf{P} = {}^{\mathrm{B}}\mathbf{T}_{\mathrm{A}} {}^{\mathrm{A}}\mathbf{P}\f$ and feeds it back into the polygonal mesh, without affecting the vertex connectivity

@returns void */
void OcclusionCheck::transformPolygonMesh(PolygonMesh &inMesh, const Matrix4f &transform) {
    PointCloud<pcl::PointXYZ> cloud;
    fromPCLPointCloud2(inMesh.cloud, cloud);
    transformPointCloud(cloud, cloud, transform);
    toPCLPointCloud2(cloud, inMesh.cloud);
}

/**
Computes the cross product of two vectors

@param a : pcl::PointXYZ denotes the first 3-dimensional vector \f$\mathbf{a}\f$
@param b : pcl::PointXYZ denotes the first 3-dimensional vector \f$\mathbf{b}\f$

This method computes \f$ \mathbf{p} = \mathbf{a} \times \mathbf{b}\f$

@returns PointXYZ the output \f$ \mathbf{p} \f$ */
PointXYZ OcclusionCheck::cross_product_raw(const PointXYZ &a, const PointXYZ &b) {
    _product.x = (a.y * b.z) - (a.z * b.y);
    _product.y = (a.z * b.x) - (a.x * b.z);
    _product.z = (a.x * b.y) - (a.y * b.x);

    return _product;
}
/**
Computes the dot product of two vectors

@param a : pcl::PointXYZ denotes the first 3-dimensional vector \f$\mathbf{a}\f$
@param b : pcl::PointXYZ denotes the first 3-dimensional vector \f$\mathbf{b}\f$

This method computes \f$ \mathbf{p} = \mathbf{a} \cdot \mathbf{b}\f$

@returns double scalar output of the dot product  */
double OcclusionCheck::dot_product_raw(PointXYZ &a, PointXYZ &b) {
    _product.x = a.x * b.x;
    _product.y = a.y * b.y;
    _product.z = a.z * b.z;

    return (_product.x + _product.y + _product.z);
}

/**
Computes the normalized cross product of two vectors

@param a : pcl::PointXYZ denotes the first 3-dimensional vector \f$\mathbf{a}\f$
@param b : pcl::PointXYZ denotes the first 3-dimensional vector \f$\mathbf{b}\f$
@param product (output) : pcl::PointXYZ denotes the value of the cross product (let's say \f$\mathbf{p}\f$)

This method computes \f$ \mathbf{p}^{r} = \mathbf{a} \times \mathbf{b}\f$
and the output is given as: \f$ \mathbf{p} = \frac{\mathbf{p}^{r}}{\| \mathbf{p}^{r} \|} \f$

@returns void  */
void OcclusionCheck::cross_product(const PointXYZ &a, const PointXYZ &b, PointXYZ &product) {
    product.x = (a.y * b.z) - (a.z * b.y);
    product.y = (a.z * b.x) - (a.x * b.z);
    product.z = (a.x * b.y) - (a.y * b.x);

    double norm = sqrt((product.x * product.x) + (product.y * product.y) + (product.z * product.z));

    if ((norm > 0) || (norm < 0)) {
        product.x /= norm;
        product.y /= norm;
        product.z /= norm;
    }
}

/**
Computes a barycentric coordinate of P w.r.t the three sides of the triangle a, b, c.

@param a : pcl::PointXYZ denoting the 3D cartesian coordinate of the first vertex of the triangle
@param b : pcl::PointXYZ denoting the 3D cartesian coordinate of the second vertex of the triangle
@param c : pcl::PointXYZ denoting the 3D cartesian coordinate of the third vertex of the triangle
@param P : pcl::PointXYZ denoting the 3D cartesian coordinate of the point for which the barycentric coordinate is to be computed
@param P_res : pcl::PointXYZ denoting the return value (barycentric coordinate)

@returns void  */
void OcclusionCheck::compute_barycentric_coordinates(PointXYZ a, PointXYZ b, PointXYZ c, PointXYZ P, PointXYZ &P_res) {
#ifdef BARYCENTRIC_IN_IMAGE_PLANE
    point_2d a_ = projection(a);
    point_2d b_ = projection(b);
    point_2d c_ = projection(c);
    point_2d P_ = projection(P);

    a.x = a_.x;
    a.y = a_.y;
    a.z = 1.0f;
    b.x = b_.x;
    b.y = b_.y;
    b.z = 1.0f;
    c.x = c_.x;
    c.y = c_.y;
    c.z = 1.0f;
    P.x = P_.x;
    P.y = P_.y;
    P.z = 1.0f;

#endif
    _v2_1.x = b.x - a.x;
    _v2_1.y = b.y - a.y;
    _v2_1.z = b.z - a.z;

    _v2_3.x = b.x - c.x;
    _v2_3.y = b.y - c.y;
    _v2_3.z = b.z - c.z;

    _v2_P.x = b.x - P.x;
    _v2_P.y = b.y - P.y;
    _v2_P.z = b.z - P.z;

    float d00 = dot_product_raw(_v2_1, _v2_1);
    float d01 = dot_product_raw(_v2_1, _v2_3);
    float d11 = dot_product_raw(_v2_3, _v2_3);

    float denom = d00 * d11 - d01 * d01;

    float d20 = dot_product_raw(_v2_P, _v2_1);
    float d21 = dot_product_raw(_v2_P, _v2_3);
    P_res.x = (d11 * d20 - d01 * d21) / denom;
    P_res.y = (d00 * d21 - d01 * d20) / denom;
    P_res.z = 1.0f - P_res.x - P_res.y;
}

/**
Computes the 3D cartesian coordinates of a point, when the barycentric coordinates w.r.t the three sides of a triangle a, b, c is given.

@param a : pcl::PointXYZ denoting the 3D cartesian coordinate of the first vertex of the triangle
@param b : pcl::PointXYZ denoting the 3D cartesian coordinate of the second vertex of the triangle
@param c : pcl::PointXYZ denoting the 3D cartesian coordinate of the third vertex of the triangle
@param P : pcl::PointXYZ denoting the barycentric coordinate of the point for which the cartesian coordinate is to be computed
@param P_res : pcl::PointXYZ denoting the return value (cartesian coordinate)

@returns void  **/
void OcclusionCheck::compute_inverse_barycentric_coordinates(PointXYZ a, PointXYZ b, PointXYZ c, PointXYZ &P, PointXYZ &P_res) {
#ifdef BARYCENTRIC_IN_IMAGE_PLANE
    point_2d a_ = projection(a);
    point_2d b_ = projection(b);
    point_2d c_ = projection(c);

    a.x = a_.x;
    a.y = a_.y;
    a.z = 1.0f;
    b.x = b_.x;
    b.y = b_.y;
    b.z = 1.0f;
    c.x = c_.x;
    c.y = c_.y;
    c.z = 1.0f;

#endif
    _A(0, 0) = a.x;
    _A(0, 1) = c.x;
    _A(0, 2) = b.x;
    _A(1, 0) = a.y;
    _A(1, 1) = c.y;
    _A(1, 2) = b.y;
    _A(2, 0) = a.z;
    _A(2, 1) = c.z;
    _A(2, 2) = b.z;

    _b(0, 0) = P.x;
    _b(1, 0) = P.y;
    _b(2, 0) = P.z;

    _cartesian = _A * _b;

    P_res.x = _cartesian(0, 0);
    P_res.y = _cartesian(1, 0);
    P_res.z = _cartesian(2, 0);
}

/**
Given a vector of residuals, use Tukey's m-estimator to create a column vector of weights

@param residual : vector of input residuals/errors
@param W : output weight vector, expressed as a column matrix
@param threshold : threshold for Tukey based m-estimator

The m-estimator has been implemented using ViSP. For more details for the <threshold> and other parameters,
check the documentation of ViSP for vpRobust module, currently available online at: https://visp-doc.inria.fr/doxygen/visp-2.9.0/classvpRobust.html

@returns void  **/
void OcclusionCheck::assign_weights_to_matrix_visp(EigenMatrix &residual, vpColVector &W, float threshold) {
    vpColVector error(residual.rows());

    int n = error.size();

    memcpy(error.data, residual.data(), n * sizeof(double));

    W = 1;

    vpRobust robust(n);

    robust.setIteration(0);
    robust.setThreshold(threshold);
    robust.MEstimator(vpRobust::TUKEY, error, W);  // it is also possible to use 'vpRobust::HUBER'
}

/**
Coverts a 2D matrix in OpenCV's cv::Mat format to a column vector in Eigen's format

@param data : matrix of type cv::Mat
@param output : column matrix of type EigenMatrix
@param datatype : integer variable designating the datatype, two options: 0 : uchar, 1: float


@returns void  **/
void OcclusionCheck::cvMat_to_eigenMatrix_row(cv::Mat &data, EigenMatrix &output, int datatype) {
    output.resize(data.rows * data.cols, 1);

    int count = 0;

    for (int i = 0; i < data.rows; i++) {
        for (int j = 0; j < data.cols; j++) {
            if (datatype == 0) {
                output(count++, 0) = data.at<uchar>(i, j);
            } else {
                output(count++, 0) = data.at<float>(i, j);
            }
        }
    }
}

/**
Coverts a column vector in Eigen's format to 2D matrix in OpenCV's cv::Mat format

@param data : column matrix of type EigenMatrix should be of length (ROWxCOLUMN)
@param output : matrix of type cv::Mat, should be resized to 2D matrix of dimension [ROWxCOLUMN]
@param datatype : integer variable designating the datatype, two options: 0 : uchar, 1: float

Ensure length of <data> is equal to [ROWxCOLUMN] of <output>, or the function crashes (with an error message)

@returns void  **/
void OcclusionCheck::eigenMatrix_to_cvMat_row(EigenMatrix &data, cv::Mat &output, int datatype) {
    int count = 0;

    if ((output.rows * output.cols) != data.rows()) {
        _fatal_("cvMat and EigenMatrix has a dimension mismatch!")
    }

    for (int i = 0; i < output.rows; i++) {
        for (int j = 0; j < output.cols; j++) {
            if (datatype == 0) {
                output.at<uchar>(i, j) = data(count++, 0);
            } else {
                output.at<float>(i, j) = data(count++, 0);
            }
        }
    }
}

/**
Set the depth camera parameters (intrinsics)

@returns void  **/
void OcclusionCheck::set_camera_params(float F_x, float F_y, float C_x, float C_y, float d_1, float d_2, float d_3, float d_4) {
    rvecP.create(3, 1, cv::DataType<double>::type);
    tvecP.create(3, 1, cv::DataType<double>::type);
    F_xP = F_x;
    C_xP = C_x;
    F_yP = F_y;
    C_yP = C_y;
    _camParam.initPersProjWithDistortion(F_x, F_y, C_x, C_y, d_1, d_2);
}

/**
Set the color camera parameters (intrinsics)

@returns void  **/
void OcclusionCheck::set_color_camera_params(float F_x, float F_y, float C_x, float C_y, float d_1, float d_2, float d_3, float d_4) { _colorCamParam.initPersProjWithDistortion(F_x, F_y, C_x, C_y, d_1, d_2); }

/**
Projects a 3D point with the pinhole camera model. Uses the intrinsics set using the <set_camera_params> method

@param pt : input 3D point

Uses ViSP internally

@returns point_2d representing the projected 2D point  **/
point_2d OcclusionCheck::projection(const pcl::PointXYZ &pt) {
    double x_normalized = pt.x / pt.z;
    double y_normalized = pt.y / pt.z;

    vpMeterPixelConversion::convertPoint(_camParam, x_normalized, y_normalized, pt_P.x, pt_P.y);

    pt_P.x = std::round(pt_P.x);
    pt_P.y = std::round(pt_P.y);

    return pt_P;
}

/**
Projects a 3D point with the pinhole camera model. Uses the intrinsics set using the <set_camera_params> method

@param pt : input 3D point

Uses ViSP internally. Does not round off the projected points

@returns point_2d representing the projected 2D point  **/
point_2d OcclusionCheck::projection_raw(const pcl::PointXYZ &pt) {
    double x_normalized = pt.x / pt.z;
    double y_normalized = pt.y / pt.z;

    vpMeterPixelConversion::convertPoint(_camParam, x_normalized, y_normalized, pt_P.x, pt_P.y);

    return pt_P;
}

/**
Projects a 3D point with the pinhole camera model. Uses the intrinsics provided in the argument

@param pt : input 3D point

Does not round off the projected points

@returns point_2d representing the projected 2D point  **/
point_2d OcclusionCheck::projection_raw(PointXYZ &pt, float Fx, float Fy, float Cx, float Cy) {
    point_2d point;
    point.x = ((pt.x * Fx / pt.z) + Cx);
    point.y = ((pt.y * Fy / pt.z) + Cy);

    return point;
}

/**
Projects a 2D image point back to 3D using the pinhole camera model. Uses the intrinsics provided in the argument

@param x : image coordinates
@param y : image coordinates
@param depth : depth corresponding to the image coordinates
@param pt : output 3D point in PointXYZ format


@returns void **/
void OcclusionCheck::inverse_projection_raw(int x, int y, double depth, float Fx, float Fy, float Cx, float Cy, PointXYZ &pt) {
    pt.x = ((float)x - Cx) * (depth / Fx);
    pt.y = ((float)y - Cy) * (depth / Fy);
    pt.z = depth;
}

/**
Projects a 2D image point back to 3D using the pinhole camera model. Uses the intrinsics provided in the argument

@param x : image coordinates
@param y : image coordinates
@param depth : depth corresponding to the image coordinates
@param pt : output 3D point in PointXYZRGB format


@returns void **/
void OcclusionCheck::inverse_projection_floating(double x, double y, double depth, float Fx, float Fy, float Cx, float Cy, PointXYZRGB &pt) {
    pt.x = ((float)x - Cx) * (depth / Fx);
    pt.y = ((float)y - Cy) * (depth / Fy);
    pt.z = depth;
}

float sign(point_2d &p1, point_2d &p2, point_2d &p3) { return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y); }

bool OcclusionCheck::point_in_triangle(point_2d &pt, point_2d &v1, point_2d &v2, point_2d &v3) {
    bool b1, b2, b3;

    b1 = sign(pt, v1, v2) < 0.0f;
    b2 = sign(pt, v2, v3) < 0.0f;
    b3 = sign(pt, v3, v1) < 0.0f;

    return ((b1 == b2) && (b2 == b3));
}

bool OcclusionCheck::point_in_triangle(Point2f &pt, point_2d &v1, point_2d &v2, point_2d &v3) {
    bool b1, b2, b3;

    point_2d pt_;
    pt_.x = pt.x;
    pt_.y = pt.y;

    b1 = sign(pt_, v1, v2) < 0.0f;
    b2 = sign(pt_, v2, v3) < 0.0f;
    b3 = sign(pt_, v3, v1) < 0.0f;

    return ((b1 == b2) && (b2 == b3));
}

point_2d OcclusionCheck::projection_gl(pcl::PointXYZ pt, float F_x, float F_y, float C_x, float C_y) {
    point_2d pt_;
    std::vector<cv::Point3f> objectPoints;
    cv::Mat A(3, 3, cv::DataType<double>::type);

    cv::Point3f p(pt.x, pt.y, pt.z);

    objectPoints.push_back(p);

    A.at<double>(0, 0) = F_x;
    A.at<double>(0, 1) = 0;
    A.at<double>(0, 2) = 0.0;
    A.at<double>(1, 0) = 0;
    A.at<double>(1, 1) = F_y;
    A.at<double>(1, 2) = 0.0;
    A.at<double>(2, 0) = 0;
    A.at<double>(2, 1) = 0;
    A.at<double>(2, 2) = 1;

    cv::Mat distCoeffs(4, 1, cv::DataType<double>::type);
    distCoeffs.at<double>(0) = 0;
    distCoeffs.at<double>(1) = 0;
    distCoeffs.at<double>(2) = 0;
    distCoeffs.at<double>(3) = 0;

    std::vector<cv::Point2f> projectedPoints;

    cv::projectPoints(objectPoints, cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(3, 1, CV_64F), A, distCoeffs, projectedPoints);  //---> error at this line. More details below.

    for (std::vector<cv::Point2f>::iterator it = projectedPoints.begin(); it != projectedPoints.end(); (++it)) {
        double x_ = round(it.base()->x);  //+ cam_depth.get_u0();
        double y_ = round(it.base()->y);  // + cam_depth.get_v0();
        pt_.x = x_;
        pt_.y = y_;
    }

    return pt_;
}

PointCloud<PointXYZ>::Ptr OcclusionCheck::load_vtk_mesh(string filename) {
    vtkSmartPointer<vtkXMLUnstructuredGridReader> reader = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
    reader->SetFileName(filename.c_str());
    reader->Update();

    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputConnection(reader->GetOutputPort());

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    vtkSmartPointer<vtkUnstructuredGrid> mesh = reader->GetOutput();
    vtkSmartPointer<vtkPoints> points = mesh->GetPoints();
    vtkSmartPointer<vtkDataArray> dataArray = points->GetData();

    PointCloud<PointXYZ>::Ptr mech_cloud(new PointCloud<PointXYZ>);

    mech_cloud->width = dataArray->GetNumberOfTuples();
    mech_cloud->height = 1;
    mech_cloud->is_dense = true;
    mech_cloud->points.resize(mech_cloud->width * mech_cloud->height);

    float x_max = -1000.0f;
    float x_min = 1000.0f;
    float y_max = -1000.0f;
    float y_min = 1000.0f;
    float z_max = -1000.0f;
    float z_min = 1000.0f;

    for (int j = 0; j < dataArray->GetNumberOfTuples(); j++) {
        double *value = dataArray->GetTuple(j);
        mech_cloud->points[j].x = value[0];
        mech_cloud->points[j].y = value[1];
        mech_cloud->points[j].z = value[2];

        (value[0] > x_max) ? (x_max = value[0]) : (0);
        (value[0] < x_min) ? (x_min = value[0]) : (0);
        (value[1] > y_max) ? (y_max = value[1]) : (0);
        (value[1] < y_min) ? (y_min = value[1]) : (0);
        (value[2] > z_max) ? (z_max = value[2]) : (0);
        (value[2] < z_min) ? (z_min = value[2]) : (0);
    }

    return mech_cloud;
}

void OcclusionCheck::get_transform(const string &path, Matrix4f &pose) {
    char *cstr = new char[path.length() + 1];
    strcpy(cstr, path.c_str());
    double **transform = load_pose(cstr);

    pose = Eigen::Matrix4f::Identity();
    pose(0, 0) = transform[0][0];
    pose(0, 1) = transform[0][1];
    pose(0, 2) = transform[0][2];
    pose(0, 3) = transform[0][3];
    pose(1, 0) = transform[1][0];
    pose(1, 1) = transform[1][1];
    pose(1, 2) = transform[1][2];
    pose(1, 3) = transform[1][3];
    pose(2, 0) = transform[2][0];
    pose(2, 1) = transform[2][1];
    pose(2, 2) = transform[2][2];
    pose(2, 3) = transform[2][3];
    pose(3, 0) = transform[3][0];
    pose(3, 1) = transform[3][1];
    pose(3, 2) = transform[3][2];
    pose(3, 3) = transform[3][3];
}

/**
Count the number of files in a given folder

@param path : path to folder

@returns number of files **/
int OcclusionCheck::count_files_in_folder(string path) {
    DIR *dp;
    int i = 0;
    struct dirent *ep;
    dp = opendir(path.c_str());

    if (dp != NULL) {
        while (ep = readdir(dp)) i++;

        (void)closedir(dp);
    } else {
        _error_("Couldn't open the directory: " << path);
        return 0;
    }

    return i;
}

/**
Populate the <offline_data_map> structure with the offline data available in the paths specified in the arguments

@param color_directory : path to folder containing color images, should be named sequentially in PNG format
@param depth_directory : path to folder containing depth images, should be named sequentially in PNG/CSV format
@param offset : number of files to skip from the begining of the sequence of images

The file sequence must begin from one, i.e., color image folder should contain 1.png, 2.png, ...... etc.
Note that the data is not actually loaded at this point.

@returns offline_data_map **/
offline_data_map OcclusionCheck::check_offline_data(string color_directory, string depth_directory, int offset) {
    int count_color = count_files_in_folder(color_directory);
    int count_depth = count_files_in_folder(depth_directory);

    offline_data_map data;
    vector<int> color_file_ids;
    vector<int> depth_file_ids;

    if (count_color == count_depth) {
        for (int i = offset; i < count_color; i++) {
            color_file_ids.push_back(i);
            depth_file_ids.push_back(i);
        }

        data.color_files = color_file_ids;
        data.depth_files = depth_file_ids;
    } else if (count_color < count_depth) {
        float progress_frac = (float)count_depth / (float)count_color;
        float start = 0.0f;

        for (int i = 0; i < count_color; i++) {
            color_file_ids.push_back(i);
            depth_file_ids.push_back(round(start));
            start += progress_frac;
        }

        data.color_files = color_file_ids;
        data.depth_files = depth_file_ids;
    } else {
        float progress_frac = (float)count_color / (float)count_depth;
        float start = 0.0f;

        for (int i = 0; i < count_depth; i++) {
            color_file_ids.push_back(round(start));
            depth_file_ids.push_back(i);
            start += progress_frac;
        }

        data.color_files = color_file_ids;
        data.depth_files = depth_file_ids;
    }

    return data;
}

/**
Get the visible points in a polygon mesh using the normal direction w.r.t the origin

@param mesh : the polygon mesh for which the visibility is to be determined

The input mesh must be maintained in the camera coordinate, such that the point (0,0,0) can denote the camera position


@returns mesh_map **/
mesh_map OcclusionCheck::get_visibility_normal(pcl::PolygonMesh &mesh) {
    pcl::PointCloud<pcl::PointXYZ> objCloud;
    pcl::PCLPointCloud2 ptCloud2 = mesh.cloud;
    pcl::fromPCLPointCloud2(ptCloud2, objCloud);

    PointXYZ pt1;
    PointXYZ pt2;
    PointXYZ pt3;

    PointXYZ centroid;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    vector<Vertices> polys;
    pcl::PolygonMesh mesh_vis;
    int vis_count = 0;
    mesh_map map_;
    std::vector<Eigen::Vector3f> visible_normal;
    std::vector<Eigen::Vector3f> visible_un_normal;

    std::vector<pcl::Vertices, std::allocator<pcl::Vertices>>::iterator face;

    for (face = mesh.polygons.begin(); face != mesh.polygons.end(); ++face) {
        pt1 = objCloud[face->vertices[0]];
        pt2 = objCloud[face->vertices[1]];
        pt3 = objCloud[face->vertices[2]];

        centroid.x = (pt1.x + pt2.x + pt3.x) / 3.0f;
        centroid.y = (pt1.y + pt2.y + pt3.y) / 3.0f;
        centroid.z = (pt1.z + pt2.z + pt3.z) / 3.0f;

        _vec12(0) = pt2.x - pt1.x;
        _vec12(1) = pt2.y - pt1.y;
        _vec12(2) = pt2.z - pt1.z;
        _vec23(0) = pt3.x - pt2.x;
        _vec23(1) = pt3.y - pt2.y;
        _vec23(2) = pt3.z - pt2.z;
        _vecNorm = _vec12.cross(_vec23);
        _vecUnNorm = _vecNorm;
        _vecNorm.normalize();
        _vec_centroid(0) = centroid.x;
        _vec_centroid(1) = centroid.y;
        _vec_centroid(2) = centroid.z;
        _vec_centroid.normalize();

        float res = _vecNorm.transpose() * _vec_centroid;

        if (res < 0.0f) {
            cloud->push_back(pt1);
            cloud->push_back(pt2);
            cloud->push_back(pt3);

            pcl::Vertices v;
            v.vertices.push_back(vis_count);
            v.vertices.push_back(vis_count + 1);
            v.vertices.push_back(vis_count + 2);
            vis_count += 3;

            map_.visible_indices.push_back(face->vertices[0]);
            map_.visible_indices.push_back(face->vertices[1]);
            map_.visible_indices.push_back(face->vertices[2]);

            polys.push_back(v);
            visible_normal.push_back(_vecNorm);
            visible_un_normal.push_back(_vecUnNorm);
        }
    }

    mesh_vis.polygons = polys;
    pcl::PCLPointCloud2::Ptr visible_blob(new pcl::PCLPointCloud2);
    pcl::toPCLPointCloud2(*cloud, *visible_blob);
    mesh_vis.cloud = *visible_blob;
    map_.mesh = mesh_vis;
    map_.visible_normal = visible_normal;
    map_.visible_normal_unnormalized = visible_un_normal;
    map_.visible_vertices = cloud;

    return map_;
}
