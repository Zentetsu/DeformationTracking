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

// ////////////////////////////////////////////////////////////////////////////////////////
// // Image utility functions, minimal BMP writer and depth buffer tone mapping
// ////////////////////////////////////////////////////////////////////////////////////////

// static void WriteBMP(const char* filename, const unsigned char* data, int w, int h) {
//     short header[] = { 0x4D42, 0, 0, 0, 0, 26, 0, 12, 0, (short)w, (short)h, 1, 24 };
//     FILE* f = fopen(filename, "wb");
//     fwrite(header, 1, sizeof(header), f);
// #if USE_D3D == 1
//     // Flip image because Y axis of Direct3D points in the opposite direction of bmp. If the library
//     // is configured for OpenGL (USE_D3D 0) then the Y axes would match and this wouldn't be required.
//     for (int y = 0; y < h; ++y)
//         fwrite(&data[(h - y - 1) * w * 3], 1, w * 3, f);
// #else
//     fwrite(data, 1, w * h * 3, f);
// #endif
//     fclose(f);
// }

// static void TonemapDepth(float* depth, unsigned char* image, int w, int h) {
//     // Find min/max w coordinate (discard cleared pixels)
//     float minW = FLT_MAX, maxW = 0.0f;
//     for (int i = 0; i < w * h; ++i) {
//         if (depth[i] > 0.0f) {
//             minW = std::min(minW, depth[i]);
//             maxW = std::max(maxW, depth[i]);
//         }
//     }

//     // Tonemap depth values
//     for (int i = 0; i < w * h; ++i) {
//         int intensity = 0;
//         if (depth[i] > 0)
//             intensity = (unsigned char)(223.0 * (depth[i] - minW) / (maxW - minW) + 32.0);

//         image[i * 3 + 0] = intensity;
//         image[i * 3 + 1] = intensity;
//         image[i * 3 + 2] = intensity;
//     }
// }

void OcclusionCheck::eigen_to_visp(MatrixXd& E, vpMatrix& V) {
    V.resize(E.rows(), E.cols(), true);
    for (int i = 0; i < E.rows(); i++) {
        for (int j = 0; j < E.cols(); j++) {
            V[i][j] = E(i, j);
        }
    }
}

void OcclusionCheck::visp_to_eigen(vpMatrix& V, MatrixXd& E) {
    E.resize(V.getRows(), V.getCols());

    for (int i = 0; i < V.getRows(); i++) {
        for (int j = 0; j < V.getCols(); j++) {
            E(i, j) = V[i][j];
        }
    }
}

double** OcclusionCheck::load_pose(char* path) {
    double** transform = new double*[4];

    std::ifstream file(path);
    for (unsigned int i = 0; i < 4; i++) {
        transform[i] = new double[4];
        for (unsigned int j = 0; j < 4; j++) {
            file >> transform[i][j];
        }
    }

    return transform;
}

void OcclusionCheck::eigen_to_visp_4x4(Matrix4f E, vpHomogeneousMatrix& V) {  // this is specific to blender simulated data
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

void OcclusionCheck::visp_to_eigen_4x4(vpHomogeneousMatrix V, Matrix4f& E) {
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

void OcclusionCheck::load_pcd(pcl::PointCloud<pcl::PointXYZ>& cloud, std::string path) {
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(path, cloud) == -1) {
        (PCL_ERROR("Couldn't read file pcd"));
    }
}

void OcclusionCheck::transformPolygonMesh(pcl::PolygonMesh& inMesh, Eigen::Matrix4f& transform) {
    // Important part starts here
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromPCLPointCloud2(inMesh.cloud, cloud);
    pcl::transformPointCloud(cloud, cloud, transform);
    pcl::toPCLPointCloud2(cloud, inMesh.cloud);
}

vector<Eigen::Vector3d> OcclusionCheck::nearest_neighbor(PointCloud<PointXYZ> cloud_, PointCloud<PointXYZ>::Ptr cluster, float dist) {
    PointCloud<PointXYZ>::Ptr cloud(new PointCloud<PointXYZ>);
    *cloud = cloud_;

    MatrixXd A = MatrixXd::Random(cluster->size(), 3);

    vector<Vector3d> normals;

    int K = 1;

    KdTreeFLANN<PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    vector<int> pointIdxNKNSearch(K);
    vector<float> pointNKNSquaredDistance(dist);

    int num_points = 0;

    for (int i = 0; i < cluster->size(); i++) {
        pcl::PointXYZ p = cluster->points[i];

        if (kdtree.nearestKSearch(p, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
            for (size_t j = 0; j < pointIdxNKNSearch.size(); ++j) {
                pcl::PointXYZ pt;
                pt.x = cloud->points[pointIdxNKNSearch[j]].x;
                pt.y = cloud->points[pointIdxNKNSearch[j]].y;
                pt.z = cloud->points[pointIdxNKNSearch[j]].z;

                A(j, 0) = -pt.x + p.x;
                A(j, 1) = -pt.y + p.y;
                A(j, 2) = -pt.z + p.z;

                // normals.push_back(n);
            }
        }
    }

    Vector3d v = A.colwise().mean();
    v.normalize();

    // cout<<"normal: "<<v<<endl;

    return normals;
}

MatrixXd OcclusionCheck::nearest_neighbor_list(PointCloud<PointXYZ>& cloud_, PointCloud<PointXYZ>::Ptr cluster, float dist) {
    PointCloud<PointXYZ>::Ptr cloud(new PointCloud<PointXYZ>);
    *cloud = cloud_;

    MatrixXd A = MatrixXd::Zero(cluster->size(), 3);

    vector<Vector3d> normals;

    int K = 1;

    KdTreeFLANN<PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    vector<int> pointIdxNKNSearch(K);
    vector<float> pointNKNSquaredDistance(dist);

    int num_points = 0;

    for (int i = 0; i < cluster->size(); i++) {
        pcl::PointXYZ p = cluster->points[i];

        if (kdtree.nearestKSearch(p, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
            pcl::PointXYZ pt;
            A(i, 0) = cloud->points[pointIdxNKNSearch[0]].x;
            A(i, 1) = cloud->points[pointIdxNKNSearch[0]].y;
            A(i, 2) = cloud->points[pointIdxNKNSearch[0]].z;
        }
    }

    // cout<<"normal: "<<v<<endl;

    return A;
}

void OcclusionCheck::load_pcd_as_text_rgbd(pcl::PointCloud<pcl::PointXYZRGB>& cloud, std::string path, bool isStructured) {
    std::ifstream file(path.c_str());
    std::vector<double> points;
    vector<uint32_t> rgb_points;

    try {
        int i = 0;
        long long int rgb_f;
        if (file.is_open()) {
            std::string line;
            while (getline(file, line)) {
                i++;
                int j = 0;
                if (i > 11) {
                    char* pch;
                    pch = strtok(const_cast<char*>(line.c_str()), " ");

                    while (pch != NULL) {
                        if (j < 3) {
                            points.push_back(strtof(pch, NULL));
                        } else if (j == 3) {
                            rgb_f = stoll(pch);
                            rgb_points.push_back((uint32_t)rgb_f);
                        }
                        pch = strtok(NULL, " ");
                        j++;
                    }
                }
            }
            file.close();
        } else {
            std::cerr << "Failed to open: " << path << std::endl;
        }

    } catch (const char* exception) {
        std::cerr << exception << std::endl;
        std::cerr << "ERROR: The pointcloud from " << path << " did not finish loading properly." << std::endl;
    }

    pcl::PointCloud<pcl::PointXYZRGB> cloud_;

    if (isStructured) {
        cloud_.width = IMAGE_WIDTH;
        cloud_.height = IMAGE_HEIGHT;
    } else {
        cloud_.width = points.size() / 3;
        cloud_.height = 1;
    }
    cloud_.is_dense = false;
    cloud_.points.resize(cloud_.width * cloud_.height);

    int k = 0;
    int l = 0;

    if (isStructured) {
        for (int i = 0; i < 640; i++) {
            for (int j = 0; j < 480; j++) {
                cloud_.at(i, j).x = points[k];
                cloud_.at(i, j).y = points[k + 1];
                cloud_.at(i, j).z = points[k + 2];
                cloud_.at(i, j).rgb = rgb_points[l++];
                k += 3;
            }
        }
    } else {
        cout << "currently not working for unstructured cloud" << endl;
        exit(0);
    }

    cloud = cloud_;
}

vector<Eigen::Vector3d> OcclusionCheck::nearest_neighbor_generalized(PointCloud<PointXYZ> cloud_, PointCloud<PointXYZ>::Ptr cluster, float dist) {
    PointCloud<PointXYZ>::Ptr cloud(new PointCloud<PointXYZ>);
    *cloud = cloud_;

    vector<Vector3d> normals;

    int K = 1;

    KdTreeFLANN<PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    vector<int> pointIdxNKNSearch(K);
    vector<float> pointNKNSquaredDistance(dist);

    int num_points = 0;

    Vector3d v;

    for (int i = 0; i < cluster->size(); i++) {
        pcl::PointXYZ p = cluster->points[i];

        if (kdtree.nearestKSearch(p, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
            for (size_t j = 0; j < pointIdxNKNSearch.size(); ++j) {
                v(0) = cloud->points[pointIdxNKNSearch[j]].x;
                v(1) = cloud->points[pointIdxNKNSearch[j]].y;
                v(2) = cloud->points[pointIdxNKNSearch[j]].z;

                normals.push_back(v);
            }
        }
    }

    // cout<<"normal: "<<v<<endl;

    return normals;
}

void OcclusionCheck::set_camera_params(float F_x, float F_y, float C_x, float C_y, float d_1, float d_2, float d_3, float d_4) {
    rvecP.create(3, 1, cv::DataType<double>::type);
    tvecP.create(3, 1, cv::DataType<double>::type);
    AP.create(3, 3, cv::DataType<double>::type);
    distCoeffsP.create(4, 1, cv::DataType<double>::type);

    distCoeffsP.at<double>(0) = d_1;
    distCoeffsP.at<double>(1) = d_2;
    distCoeffsP.at<double>(2) = d_3;
    distCoeffsP.at<double>(3) = d_4;
    F_xP = F_x;
    C_xP = C_x;
    F_yP = F_y;
    C_yP = C_y;

    // divide by height and width. seemed better
    AP.at<double>(0, 0) = F_xP;
    AP.at<double>(0, 1) = 0;
    AP.at<double>(0, 2) = C_xP;
    AP.at<double>(1, 0) = 0;
    AP.at<double>(1, 1) = F_yP;
    AP.at<double>(1, 2) = C_yP;
    AP.at<double>(2, 0) = 0;
    AP.at<double>(2, 1) = 0;
    AP.at<double>(2, 2) = 1;
}

point_2d OcclusionCheck::projection(pcl::PointXYZ& pt) {
    pP.x = pt.x;
    pP.y = pt.y;
    pP.z = pt.z;

    objectPointsP.push_back(pP);

    cv::projectPoints(objectPointsP, cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(3, 1, CV_64F), AP, distCoeffsP, projectedPointsP);  //---> error at this line. More details below.

    for (std::vector<cv::Point2f>::iterator it = projectedPointsP.begin(); it != projectedPointsP.end(); (++it)) {
        pt_P.x = round(it.base()->x);
        pt_P.y = round(it.base()->y);
    }

    return pt_P;
}

float sign(point_2d& p1, point_2d& p2, point_2d& p3) { return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y); }

bool OcclusionCheck::point_in_triangle(point_2d& pt, point_2d& v1, point_2d& v2, point_2d& v3) {
    ////log("^");
    bool b1, b2, b3;

    b1 = sign(pt, v1, v2) < 0.0f;
    b2 = sign(pt, v2, v3) < 0.0f;
    b3 = sign(pt, v3, v1) < 0.0f;

    ////log("v");
    return ((b1 == b2) && (b2 == b3));
}

point_2d OcclusionCheck::projection_gl(pcl::PointXYZ pt, float F_x, float F_y, float C_x, float C_y) {
    point_2d pt_;
    std::vector<cv::Point3f> objectPoints;
    cv::Mat rvec(3, 1, cv::DataType<double>::type);
    cv::Mat tvec(3, 1, cv::DataType<double>::type);
    cv::Mat A(3, 3, cv::DataType<double>::type);

    cv::Point3f p(pt.x, pt.y, pt.z);

    objectPoints.push_back(p);

    // divide by height and width. seemed better
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

// float x_map(float X)
//{
//     return (((2*X)/(2*C_x)) - 1);
// }
//
// float y_map(float Y)
//{
//     return (((2*Y)/(2*C_y)) - 1);
// }

// mesh_map OcclusionCheck::get_visibility(pcl::PolygonMesh mesh, float F_x, float F_y, float C_x, float C_y)
//{
//     // Flush denorms to zero to avoid performance issues with small values
//     _mm_setcsr(_mm_getcsr() | 0x8040);
//
//     mesh_map map_;
//
//     MaskedOcclusionCulling *moc = MaskedOcclusionCulling::Create();
//
//     ////////////////////////////////////////////////////////////////////////////////////////
//     // Print which version (instruction set) is being used
//     ////////////////////////////////////////////////////////////////////////////////////////
//
//     MaskedOcclusionCulling::Implementation implementation = moc->GetImplementation();
//     switch (implementation) {
//     case MaskedOcclusionCulling::SSE2: printf("Using SSE2 version\n"); break;
//     case MaskedOcclusionCulling::SSE41: printf("Using SSE41 version\n"); break;
//     case MaskedOcclusionCulling::AVX2: printf("Using AVX2 version\n"); break;
//     case MaskedOcclusionCulling::AVX512: printf("Using AVX-512 version\n"); break;
//     }
//
//     ////////////////////////////////////////////////////////////////////////////////////////
//     // Setup and state related code
//     ////////////////////////////////////////////////////////////////////////////////////////
//
//     // Setup a 1920 x 1080 rendertarget with near clip plane at w = 1.0
//     // width of the buffer in pixels, must be a multiple of 8,  height of the buffer in pixels, must be a multiple of 4
//     const int width = IMAGE_WIDTH, height = IMAGE_HEIGHT;
//     moc->SetResolution(width, height);
//     moc->SetNearClipPlane(0.0f);
//
//     // Clear the depth buffer
//     moc->ClearBuffer();
//
//     ////////////////////////////////////////////////////////////////////////////////////////
//     // Render some occluders
//     ////////////////////////////////////////////////////////////////////////////////////////
//
//     struct ClipspaceVertex { float x, y, z, w; };
//
//     pcl::PointCloud<pcl::PointXYZ>::Ptr vertices_visible(new pcl::PointCloud<pcl::PointXYZ>);
//     pcl::fromPCLPointCloud2( mesh.cloud, *vertices_visible );
//
//     std::string ply_filename1("/home/agniv/Code/registration_iProcess/scripts/SOFA_minimization/build/mesh2.ply");
//     pcl::io::savePLYFile(ply_filename1, mesh);
//
//
//     std::vector<pcl::Vertices, std::allocator<pcl::Vertices>>::iterator face;
//     std::vector<unsigned int> vertex_indices;
//     int count_face = 0;
//     for(face = mesh.polygons.begin(); face != mesh.polygons.end(); ++face)
//     {
//         unsigned int v1 = face->vertices[0];
//         unsigned int v2 = face->vertices[1];
//         unsigned int v3 = face->vertices[2];
//
//         pcl::PointXYZ p1 = vertices_visible->points.at(v1);
//         pcl::PointXYZ p2 = vertices_visible->points.at(v2);
//         pcl::PointXYZ p3 = vertices_visible->points.at(v3);
//
//         vertex_indices.push_back(v1);
//         vertex_indices.push_back(v2);
//         vertex_indices.push_back(v3);
//
//         ////std::cout<<p1.x<<","<<p1.y<<","<<p1.z<<" "<<p2.x<<","<<p2.y<<","<<p2.z<<" "<<p3.x<<","<<p3.y<<","<<p3.z<<std::endl;
//
//         float dist_p1 = sqrt(pow(p1.x,2) + pow(p1.y,2) + pow(p1.z,2));
//         float dist_p2 = sqrt(pow(p2.x,2) + pow(p2.y,2) + pow(p2.z,2));
//         float dist_p3 = sqrt(pow(p3.x,2) + pow(p3.y,2) + pow(p3.z,2));
//
//         map_.distance_array.push_back(dist_p1);
//         map_.distance_array.push_back(dist_p2);
//         map_.distance_array.push_back(dist_p3);
//
//         point_2d p1_ = projection_gl(p1, F_x, F_y, C_x, C_y);
//         point_2d p2_ = projection_gl(p2, F_x, F_y, C_x, C_y);
//         point_2d p3_ = projection_gl(p3, F_x, F_y, C_x, C_y);
//         //        std::cout<<"**"<<std::endl;
//
//         map_.projected_points.push_back(p1_);
//         map_.projected_points.push_back(p2_);
//         map_.projected_points.push_back(p3_);
//
//         map_.mesh_points.push_back(p1);
//         map_.mesh_points.push_back(p2);
//         map_.mesh_points.push_back(p3);
//
//         ClipspaceVertex triVerts[] = { {p1_.x, -p1_.y, 0, 20*p1.z }, { p2_.x, -p2_.y, 0, 20*p2.z }, { p3_.x, -p3_.y, 0, 20*p3.z } };
//         unsigned int triIndices[] = { 0, 1, 2 };
//
//         //ClipspaceVertex triVerts[] = { { -10, 0, 0, 10 }, { 10, 0, 0, 10 }, { 0, 10, 0, 11 } };
//         //unsigned int triIndices[] = { 0, 1, 2 };
//
//         // Render the triangle
//         moc->RenderTriangles((float*)triVerts, triIndices, 1 , nullptr, MaskedOcclusionCulling::BACKFACE_NONE);
//
//     }
//
//     int counter = 0;
//     int vis_count = 0;
//     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
//     std::vector<pcl::Vertices> polys;
//     pcl::PolygonMesh mesh;
//
//     std::vector<unsigned int> visible_indices_;
//
//     for(int i = 0; i<map_.mesh_points.size(); i+=3)
//     {
//         point_2d p1_ = map_.projected_points[i];
//         point_2d p2_ = map_.projected_points[i+1];
//         point_2d p3_ = map_.projected_points[i+2];
//
//         float d1 = map_.distance_array[i];
//         float d2 = map_.distance_array[i+1];
//         float d3 = map_.distance_array[i+2];
//
//         pcl::PointXYZ p1 = map_.mesh_points[i+0];
//         pcl::PointXYZ p2 = map_.mesh_points[i+1];
//         pcl::PointXYZ p3 = map_.mesh_points[i+2];
//
//
//         ClipspaceVertex triVerts[] = { {p1_.x, -p1_.y, 0, 20*p1.z }, { p2_.x, -p2_.y, 0, 20*p2.z }, { p3_.x, -p3.y, 0, 20*p3.z } };
//         //ClipspaceVertex triVerts[] = { { -10, 0, 0, 10 }, { 10, 0, 0, 10 }, { 0, 10, 0, 10.5 } };
//         unsigned int triIndices[] = { 0, 1, 2 };
//
//         MaskedOcclusionCulling::CullingResult result;
//         result = moc->TestTriangles((float*)triVerts, triIndices, 1);
//
//         if (result == MaskedOcclusionCulling::VISIBLE)
//         {
//
//             //pcl::PointXYZ P1 = map_.mesh_points[i];
//             //pcl::PointXYZ P2 = map_.mesh_points[i+1];
//             //pcl::PointXYZ P3 = map_.mesh_points[i+2];
//
//             cloud->push_back(map_.mesh_points[i]);
//             cloud->push_back(map_.mesh_points[i+1]);
//             cloud->push_back(map_.mesh_points[i+2]);
//
//             pcl::Vertices v;
//             v.vertices.push_back(vis_count);
//             v.vertices.push_back(vis_count+1);
//             v.vertices.push_back(vis_count+2);
//             vis_count+=3;
//
//             polys.push_back(v);
//
//             map_.visible_indices.push_back(i);
//             map_.visible_indices.push_back(i+1);
//             map_.visible_indices.push_back(i+2);
//
//             visible_indices_.push_back(vertex_indices[i]);
//             visible_indices_.push_back(vertex_indices[i+1]);
//             visible_indices_.push_back(vertex_indices[i+2]);
//
//             //printf("Tested triangle is VISIBLE\n");
//             ////std::cout<<P1.x<<" "<<P1.y<<" "<<P1.z<<std::endl<<P2.x<<" "<<P2.y<<" "<<P2.z<<std::endl<<P3.x<<" "<<P3.y<<" "<<P3.z<<std::endl;
//         }
//
//     }
//
//     mesh.polygons = polys;
//     pcl::PCLPointCloud2::Ptr visible_blob(new pcl::PCLPointCloud2);
//     pcl::toPCLPointCloud2(*cloud, *visible_blob);
//
//     mesh.cloud = *visible_blob;
//
//     map_.mesh = mesh;
//
// #ifdef DEBUG_DUMP
//     std::string ply_filename("/home/agniv/Code/registration_iProcess/scripts/SOFA_minimization/build/mesh.ply");
//     pcl::io::savePLYFile(ply_filename, mesh);
//
//
//     //std::cout<<"Visibility testing done"<<std::endl;
//
//
////	// Compute a per pixel depth buffer from the hierarchical depth buffer, used for visualization.
//    float *perPixelZBuffer = new float[width * height];
//    moc->ComputePixelDepthBuffer(perPixelZBuffer, false);
//
//    // Tonemap the image
//    unsigned char *image = new unsigned char[width * height * 3];
//    TonemapDepth(perPixelZBuffer, image, width, height);
//    WriteBMP("image.bmp", image, width, height);
//    delete[] image;
//
// #endif
////
////	// Destroy occlusion culling object and free hierarchical z-buffer
//    MaskedOcclusionCulling::Destroy(moc);
//
//
//    return map_;
//
//}

PointCloud<PointXYZ>::Ptr OcclusionCheck::load_vtk_mesh(string filename) {  /// process this file now : 17:15 on 29 Octobre
    // read all the data from the file
    vtkSmartPointer<vtkXMLUnstructuredGridReader> reader = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
    reader->SetFileName(filename.c_str());
    reader->Update();

    // Create a mapper and actor
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
    mapper->SetInputConnection(reader->GetOutputPort());

    vtkSmartPointer<vtkActor> actor =  //  <--figure out ways to iterate over this shit!
        vtkSmartPointer<vtkActor>::New();
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
        double* value = dataArray->GetTuple(j);
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

    float x_span = fabs(x_max - x_min);
    float y_span = fabs(y_max - y_min);
    float z_span = fabs(z_max - z_min);

    return mech_cloud;
}

void OcclusionCheck::get_transform(const string& path, Matrix4f& pose) {
    char* cstr = new char[path.length() + 1];
    strcpy(cstr, path.c_str());
    double** transform = load_pose(cstr);

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

mesh_map OcclusionCheck::get_visibility_vtk(pcl::PolygonMesh mesh) {
    std::cout << "C++: " << "Visibility testing started" << std::endl;
    mesh_map map_;

    // std::string ply_filename1("/home/agniv/Code/registration_iProcess/scripts/SOFA_minimization/build/mesh_init.ply");
    // pcl::io::savePLYFile(ply_filename1, mesh);

    // Add the polygon to a list of polygons
    vtkSmartPointer<vtkCellArray> polygons = vtkSmartPointer<vtkCellArray>::New();

    // Setup three points
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();

    // Create the polygon
    vtkSmartPointer<vtkPolygon> polygon = vtkSmartPointer<vtkPolygon>::New();

    int count = 0;

    std::cout << "C++: " << "debug 1" << std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr vertices_visible(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(mesh.cloud, *vertices_visible);

    std::cout << "C++: " << "debug 2" << std::endl;
    std::vector<pcl::Vertices, std::allocator<pcl::Vertices>>::iterator face;
    for (face = mesh.polygons.begin(); face != mesh.polygons.end(); ++face) {
        // cout << "face: " << face->vertices[0] << " " << face->vertices[1] << " " << face->vertices[2] << endl;
        unsigned int v1 = face->vertices[0];
        unsigned int v2 = face->vertices[1];
        unsigned int v3 = face->vertices[2];

        pcl::PointXYZ p1 = vertices_visible->points.at(v1);
        pcl::PointXYZ p2 = vertices_visible->points.at(v2);
        pcl::PointXYZ p3 = vertices_visible->points.at(v3);

        points->InsertNextPoint(p1.x, p1.y, p1.z);
        points->InsertNextPoint(p2.x, p2.y, p2.z);
        points->InsertNextPoint(p3.x, p3.y, p3.z);

        // Create the polygon
        vtkSmartPointer<vtkPolygon> polygon = vtkSmartPointer<vtkPolygon>::New();

        polygon->GetPointIds()->SetNumberOfIds(3);  // make a quad
        polygon->GetPointIds()->SetId(0, count++);
        polygon->GetPointIds()->SetId(1, count++);
        polygon->GetPointIds()->SetId(2, count++);

        polygons->InsertNextCell(polygon);
    }

    std::cout << "C++: " << "debug 3" << std::endl;
    // Initialize the representation
    vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
    polydata->SetPoints(points);
    polydata->SetPolys(polygons);
    // obbTree->GenerateRepresentation(0, polydata);

    std::cout << "C++: " << "debug 4" << std::endl;
    // Create the tree
    std::cout << "C++: " << "debug 4.1" << std::endl;
    vtkSmartPointer<vtkOBBTree> obbTree = vtkSmartPointer<vtkOBBTree>::New();
    std::cout << "C++: " << "debug 4.2" << std::endl;
    obbTree->SetDataSet(polydata);
    std::cout << "C++: " << "debug 4.3" << std::endl;
    obbTree->BuildLocator();
    std::cout << "C++: " << "debug 4.4" << std::endl;

    double lineP0[3] = {0.0, 0.0, 0.0};
    float thresh = 0.01f;

    std::cout << "C++: " << "debug 5" << std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<pcl::Vertices> polys;
    pcl::PolygonMesh mesh_vis;
    int vis_count = 0;

    for (face = mesh.polygons.begin(); face != mesh.polygons.end(); ++face) {
        // cout << "face: " << face->vertices[0] << " " << face->vertices[1] << " " << face->vertices[2] << endl;
        int visible = 0;
        std::vector<unsigned int> ids;

        for (int i = 0; i < 3; i++) {
            unsigned int v = face->vertices[i];
            pcl::PointXYZ p = vertices_visible->points.at(v);
            double lineP1[3] = {p.x, p.y, p.z};
            ids.push_back(v);

            vtkSmartPointer<vtkPoints> intersectPoints = vtkSmartPointer<vtkPoints>::New();
            obbTree->IntersectWithLine(lineP0, lineP1, intersectPoints, NULL);

            // Display list of intersections
            double intersection[3];
            // for(int i = 0; i < intersectPoints->GetNumberOfPoints(); i++ )
            if (intersectPoints->GetNumberOfPoints()) {
                intersectPoints->GetPoint(0, intersection);
                if ((abs(p.x - intersection[0]) < thresh) && (abs(p.y - intersection[1]) < thresh) && (abs(p.z - intersection[2]) < thresh)) {
                    visible++;
                }
            }
        }

        if (visible == 3) {
            cloud->push_back(vertices_visible->points.at(ids[0]));
            cloud->push_back(vertices_visible->points.at(ids[1]));
            cloud->push_back(vertices_visible->points.at(ids[2]));

            pcl::Vertices v;
            v.vertices.push_back(vis_count);
            v.vertices.push_back(vis_count + 1);
            v.vertices.push_back(vis_count + 2);
            vis_count += 3;

            polys.push_back(v);
        }
    }

    std::cout << "C++: " << "debug 6" << std::endl;
    mesh_vis.polygons = polys;
    pcl::PCLPointCloud2::Ptr visible_blob(new pcl::PCLPointCloud2);
    pcl::toPCLPointCloud2(*cloud, *visible_blob);

    mesh_vis.cloud = *visible_blob;

    map_.mesh = mesh_vis;

    std::cout << "C++: " << "debug 7" << std::endl;
    // std::string ply_filename("/home/agniv/Code/registration_iProcess/scripts/SOFA_minimization/build/mesh3.ply");
    // pcl::io::savePLYFile(ply_filename, mesh_vis);

    std::cout << "C++: " << "END: Visibility testing started" << std::endl;
    return map_;
}
