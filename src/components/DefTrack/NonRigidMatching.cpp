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
#include <omp.h>
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

#include "NonRigidMatching.h"
#include "OcclusionCheck.h"

float min_span = 0.0f;

float signedVolumeOfTriangle(pcl::PointXYZ p1, pcl::PointXYZ p2, pcl::PointXYZ p3) {
    float v321 = p3.x * p2.y * p1.z;
    float v231 = p2.x * p3.y * p1.z;
    float v312 = p3.x * p1.y * p2.z;
    float v132 = p1.x * p3.y * p2.z;
    float v213 = p2.x * p1.y * p3.z;
    float v123 = p1.x * p2.y * p3.z;
    return (1.0f / 6.0f) * (-v321 + v231 + v312 - v132 - v213 + v123);
}

float volumeOfMesh(pcl::PolygonMesh mesh) {
    float vols = 0.0;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(mesh.cloud, *cloud);
    for (int triangle = 0; triangle < mesh.polygons.size(); triangle++) {
        pcl::PointXYZ pt1 = cloud->points[mesh.polygons[triangle].vertices[0]];
        pcl::PointXYZ pt2 = cloud->points[mesh.polygons[triangle].vertices[1]];
        pcl::PointXYZ pt3 = cloud->points[mesh.polygons[triangle].vertices[2]];
        vols += signedVolumeOfTriangle(pt1, pt2, pt3);
    }
    return abs(vols);
}

void divideToConquereForClustering(const vector<PointXYZ> &point_source, PointCloud<PointXYZ>::Ptr &cloud, int istart, int ipoints) {
    for (int i = istart; i < istart + ipoints; i++) {
        cloud->points[i].x = point_source[i].x;
        cloud->points[i].y = point_source[i].y;
        cloud->points[i].z = point_source[i].z;
    }
}

int clustering(vector<PointXYZ> &point_source, vector<PointCloud<PointXYZ>::Ptr> &output_clusters, vector<int> &sizes) {
    int max = -99999;
    int index = -1;
    float cluster_tolerance = 0.002f;
    int min_cluster_size = point_source.size() / 5;
    int max_cluster_size = point_source.size();

    PointCloud<PointXYZ>::Ptr cloud(new PointCloud<PointXYZ>);

    cloud->width = point_source.size();
    cloud->height = 1;
    cloud->is_dense = false;
    cloud->points.resize(cloud->width * cloud->height);

    int iam, nt, ipoints, istart, npoints(cloud->points.size());
#pragma omp parallel num_threads(4) default(shared) private(iam, nt, ipoints, istart)
    {
        iam = omp_get_thread_num();
        nt = omp_get_num_threads();
        ipoints = npoints / nt;
        istart = iam * ipoints;
        if (iam == nt - 1) ipoints = npoints - istart;
        divideToConquereForClustering(point_source, cloud, istart, ipoints);
    }

    output_clusters.push_back(cloud);
    return 0;  // hack to disable clustering

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(cluster_tolerance);
    ec.setMinClusterSize(min_cluster_size);
    ec.setMaxClusterSize(max_cluster_size);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    int count = 0;

    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it) {
        PointCloud<PointXYZ>::Ptr cloud_cluster(new PointCloud<PointXYZ>);

        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit) cloud_cluster->points.push_back(cloud->points[*pit]);

        cloud_cluster->width = cloud_cluster->points.size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;
        output_clusters.push_back(cloud_cluster);
        sizes.push_back(cloud_cluster->points.size());

        if (max > cloud_cluster->points.size()) {
            max = cloud_cluster->points.size();
            index = count;
        }

        count++;
    }

    return index;
}

int get_nearest_node(Vector4f centroid, PointCloud<PointXYZ>::Ptr points) {
    int K = 1;
    float dist = 0.2;

    KdTreeFLANN<PointXYZ> kdtree;
    kdtree.setInputCloud(points);

    vector<int> pointIdxNKNSearch(K);
    vector<float> pointNKNSquaredDistance(dist);

    PointXYZ p;
    p.x = centroid[0];
    p.y = centroid[1];
    p.z = centroid[2];

    int return_index = -1;

    if (kdtree.nearestKSearch(p, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
        return_index = pointIdxNKNSearch[0];
    }

    return return_index;
}

void debug_polygon(PointXYZ p, PointXYZ q, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, std::vector<pcl::Vertices> &polys, int &vis_c) {
    PointXYZ p1;
    PointXYZ p2;
    PointXYZ p3;

    p1.x = p.x - 0.005;
    p1.y = p.y;
    p1.z = p.z;

    p2.x = p.x + 0.005;
    p2.y = p.y;
    p2.z = p.z;

    p3.x = q.x;
    p3.y = q.y;
    p3.z = q.z;

    cloud->push_back(p1);
    cloud->push_back(p2);
    cloud->push_back(p3);

    pcl::Vertices v;
    v.vertices.push_back(vis_c);
    v.vertices.push_back(vis_c + 1);
    v.vertices.push_back(vis_c + 2);

    vis_c += 3;

    polys.push_back(v);
}

void create_directory(string path) {
    if (mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1) {
        if (errno == EEXIST) {
        } else {
            cerr << "Cannot create folder: " << path << ", error:" << strerror(errno) << endl;
        }
    }
}

void NonRigidMatching::log_output(PointCloud<PointXYZRGB>::Ptr &frame, PolygonMesh &model, int frame_num, vpHomogeneousMatrix &cMo, vector<PointCloud<PointXYZRGB>> debugCloudList, vector<string> debugCloudLabel, cv::Mat depth_image, double residual, string data_path, string opfilepath,
                                  bool needs_initialization, string &image_debug_path) {
    static int prev_frame_number = -1;

    _log_("step 1")

        if (needs_initialization) {
        try {
            create_directory(opfilepath + "cpp_model/");
            create_directory(opfilepath + "cpp_cMo/");
            create_directory(opfilepath + "cpp_errmap/");
            create_directory(opfilepath + "cpp_img/");
            create_directory(opfilepath + "cpp_depth_map/");
            create_directory(opfilepath + "cpp_pcd/");
        } catch (const std::exception &e) {
            cerr << e.what() << " , " << "Exiting!" << endl;
            exit(0);
        }
    }
    try {
        // STEP 1:
        image_debug_path = opfilepath + "cpp_img/";
        pcl::io::savePLYFileBinary(opfilepath + "cpp_model/" + to_string(frame_num) + ".ply", model);

        _log_("step 2")

            std::ofstream cMo_file;
        cMo_file.open(opfilepath + "cpp_cMo/" + to_string(frame_num) + ".txt", std::ios_base::app);
        for (int i = 0; i < cMo.getRows(); i++) {
            for (int j = 0; j < cMo.getCols(); j++) {
                cMo_file << cMo[i][j] << " ";
            }
            cMo_file << endl;
        }
        cMo_file.close();

        // STEP 2:
        try {
            std::ofstream newFile2(opfilepath + "/R.txt", std::ios_base::app);
            if (newFile2.is_open()) {
                newFile2 << frame_num << "," << residual << "\n";
            }
            newFile2.close();
        } catch (const std::exception &e) {
            cerr << e.what() << " , " << "Exiting!" << endl;
            exit(0);
        }

        _log_("step 3")
            // STEP 3:
            /*for(int i = 0; i < debugCloudList.size(); i++)
            {
                PointCloud<PointXYZRGB> cloud = debugCloudList[i];
                string label = debugCloudLabel[i];
                pcl::io::savePCDFileASCII (opfilepath+"cpp_errmap/"+label+".pcd", cloud);
            }*/

            // PointCloud<PointXYZRGB> cloud = debugCloudList[debugCloudList.size()-1];
            // string label = debugCloudLabel[debugCloudList.size()-1];
            // pcl::io::savePCDFileASCII (opfilepath+"cpp_errmap/"+label+".pcd", cloud);

            /** Toggle comment the following four lines, if PCD needs to be logged to the <cpp_pcd/> directory **/
            /// if(prev_frame_number != frame_num){
            ///   pcl::io::savePCDFileASCII (opfilepath+"cpp_pcd/"+to_string(frame_num)+".pcd", *frame);
            ///   prev_frame_number = frame_num;
            ///  }

            // STEP 4:
            std::ofstream outfile;
        outfile.open(opfilepath + "cpp_log.txt", std::ios_base::app);
        outfile << to_string(frame_num) << " " << "/cpp_model/" + to_string(frame_num) + ".ply" << " " << "/cpp_errmap/" + to_string(frame_num) + ".txt" << " " << "/cpp_cMo/" + to_string(frame_num) + ".txt" << " " << "/cpp_errmap/" << debugCloudLabel[0] + ".pcd" << endl;
        outfile.close();

        _log_("step 4")

            // STEP 5:
            std::ofstream addnl_path;
        addnl_path.open(opfilepath + "additional_info.txt");
        addnl_path << "Data read from path: " << data_path << endl;
        time_t now = time(0);
        string dt = ctime(&now);
        addnl_path << "Timestamp: " << dt << endl;
        addnl_path.close();

        _log_("step 5")

            // STEP 6:
            double min;
        double max;
        cv::minMaxIdx(depth_image, &min, &max);
        cv::Mat adjMap;
        cv::convertScaleAbs(depth_image, adjMap, 255 / max);
        cv::imwrite(opfilepath + "cpp_depth_map/" + to_string(frame_num) + ".png", adjMap);

        _log_("step 6")

    } catch (const std::exception &e) {
        cerr << e.what() << " , " << "Exiting!" << endl;
        exit(0);
    }
}

float eucledian_dist(PointXYZ a, PointXYZ b) { return sqrt(pow((a.x - b.x), 2) + pow((a.y - b.y), 2) + pow((a.z - b.z), 2)); }

ModelCoefficients::Ptr get_plane_equation(PointXYZRGBA point_a, PointXYZRGBA point_b, PointXYZRGBA point_c) {
    pcl::ModelCoefficients::Ptr plane(new pcl::ModelCoefficients);

    Hyperplane<float, 3> eigen_plane = Hyperplane<float, 3>::Through(point_a.getArray3fMap(), point_b.getArray3fMap(), point_c.getArray3fMap());
    plane->values.resize(4);

    for (int i = 0; i < plane->values.size(); i++) {
        plane->values[i] = eigen_plane.coeffs()[i];
    }

    return plane;
}

PointXYZ transformed_point(const PointXYZ p, Matrix4f &transform) {
    PointXYZ p_;
    Vector3f origin_point;
    origin_point << p.x, p.y, p.z;

    Vector4f transformed_origin = transform * (origin_point.homogeneous());

    p_.x = transformed_origin[0];
    p_.y = transformed_origin[1];
    p_.z = transformed_origin[2];

    return p_;
}

PointCloud<PointXYZ> outlier_removel(PointCloud<PointXYZ> cloud) {
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(cloud, cloud, indices);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPTR(new pcl::PointCloud<pcl::PointXYZ>);
    *cloudPTR = cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
    outrem.setInputCloud(cloudPTR);
    outrem.setRadiusSearch(1.1);
    outrem.setMinNeighborsInRadius(4);
    outrem.filter(*cloud_filtered);
    return *cloud_filtered;
}

bool depth_range(PointXYZ target, PointXYZ p1, PointXYZ p2, PointXYZ p3) {
    float min_z = 10000.0f;
    float max_z = -1000.0f;

    float buffer_range = Z_BUFFER;

    if (p1.z < min_z) {
        min_z = p1.z;
    }
    if (p2.z < min_z) {
        min_z = p2.z;
    }
    if (p3.z < min_z) {
        min_z = p3.z;
    }

    if (p1.z > max_z) {
        max_z = p1.z;
    }
    if (p2.z > max_z) {
        max_z = p2.z;
    }
    if (p3.z > max_z) {
        max_z = p3.z;
    }

    max_z += buffer_range;
    min_z -= buffer_range;

    if ((target.z < max_z) && (target.z > min_z)) {
        return true;
    } else {
        return false;
    }
}

bool check_rejection(vector<int> rejection_list, int index) {
    for (int i = 0; i < rejection_list.size(); i++) {
        if (index == rejection_list[i]) {
            return false;
        }
    }

    return true;
}

void process_normal(vector<vector<Vector3d>> normal, vector<int> rejection_list) {
    for (int i = 0; i < normal.size(); i++) {
        if (check_rejection(rejection_list, i)) {
            vector<Vector3d> n = normal[i];
            MatrixXd A = MatrixXd::Random(n.size(), 3);

            for (int j = 0; j < n.size(); j++) {
                A(j, 0) = n[j](0);
                A(j, 1) = n[j](1);
                A(j, 2) = n[j](2);
            }

            Vector3d v = A.colwise().mean();
            v.normalize();

        } else {
            cout << "noraml@" << i << " **has been cutoff" << endl;
        }
    }
}

bool check_sanity(PointCloud<PointXYZ>::Ptr &cloud) {
    int count_null = 0;

    for (int i = 0; i < cloud->size(); i++) {
        PointXYZ p = cloud->points[i];

        if ((p.x == 0.0f) && (p.y == 0.0f) && (p.z = 0.0f)) {
            count_null++;
        }
    }

    float null_fraction = (float)count_null / (float)cloud->size();

    if (null_fraction > 0.5f) {
        return false;
    } else {
        return true;
    }
}

vector<int> post_process_cluster(vector<PointCloud<PointXYZ>::Ptr> &clusters) {
    vector<int> size_of_clusters;
    vector<int> reject_index;
    for (int i = 0; i < clusters.size(); i++) {
        size_of_clusters.push_back(clusters[i]->size());
    }

    sort(size_of_clusters.begin(), size_of_clusters.end(), greater<int>());
    int cutoff_size = -1;

    for (int i = 0; i < (clusters.size() - 1); i++) {
        float fraction_drop = ((float)(size_of_clusters[i] - size_of_clusters[i + 1])) / (float)size_of_clusters[i];

        if (fraction_drop > CLUSTER_SIZE_CUTOFF) {
            cutoff_size = size_of_clusters[i + 1];
        }
    }

    if (cutoff_size != -1) {
        for (int i = 0; i < clusters.size(); i++) {
            if ((clusters[i]->size() <= cutoff_size) && (check_sanity(clusters[i]))) {
                reject_index.push_back(i);
            }
        }
    }
    return reject_index;
}

vector<PointCloud<PointXYZ>::Ptr> eucledian_cluster(PointCloud<PointXYZ>::Ptr &cloud, vector<Vector3d> &normals) {
    search::KdTree<PointXYZ>::Ptr tree(new search::KdTree<PointXYZ>);
    tree->setInputCloud(cloud);

    vector<PointIndices> cluster_indices;
    EuclideanClusterExtraction<PointXYZ> ec;
    ec.setClusterTolerance(CLUSTER_TOLERANCE);
    ec.setMinClusterSize(MIN_CLUSTER_SIZE);
    ec.setMaxClusterSize(cloud->size());
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    vector<PointCloud<PointXYZ>::Ptr> cluster_list;
    vector<vector<Vector3d>> normal_list;

    int j = 0;

    for (vector<PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it) {
        PointCloud<PointXYZ>::Ptr cloud_cluster(new PointCloud<PointXYZ>);
        vector<Vector3d> normal_;

        for (vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit) {
            if ((fabs(cloud->points[*pit].x) > 0.001f) && (fabs(cloud->points[*pit].y) > 0.001f) && (fabs(cloud->points[*pit].z) > 0.001f)) {
                cloud_cluster->points.push_back(cloud->points[*pit]);
                normal_.push_back(normals[*pit]);
            }
        }

        if (cloud_cluster->size() > 0) {
            cloud_cluster->width = cloud_cluster->points.size();
            cloud_cluster->height = 1;
            cloud_cluster->is_dense = true;

            cluster_list.push_back(cloud_cluster);
            normal_list.push_back(normal_);
        } else {
            cerr << "[NonRigidMatching->eucledian_cluster] WARNING:: Zero size cloud detected" << endl;
        }
        j++;
    }

    vector<int> rejection_list = post_process_cluster(cluster_list);
    process_normal(normal_list, rejection_list);

    return cluster_list;
}

bool check_if_already_exists(vector<int> &indices, int index) {
    bool ret_flag = false;
    for (int i = 0; i < indices.size(); i++) {
        if (indices[i] == index) {
            ret_flag = true;
        }
    }

    return ret_flag;
}

void NonRigidMatching::set_valid_index() {
    valid_indices.push_back(1);
    valid_indices.push_back(2);
    valid_indices.push_back(3);

    last_index = -1;
    second_last_index = -1;
}

bool NonRigidMatching::verify_index(int index) {
    if (valid_indices.size() == 0) {
        cerr << "WARNING: NonRigidMatching::verify_index: valid indices not set" << endl;
    } else {
        for (int i = 0; i < valid_indices.size(); i++) {
            if ((index == valid_indices[i]) && (index != last_index) && (index != second_last_index)) return true;
        }
    }

    return false;
}
vector<PointXYZ> NonRigidMatching::nearest_neighbor_search(Eigen::Vector4f &source, PointCloud<PointXYZ>::Ptr target, vector<int> &indices, int index, mesh_map &visible_mesh) {
    KdTreeFLANN<PointXYZ> kdtree;
    kdtree.setInputCloud(target);
    int K = ((int)(target->size() / 10) == 0) ? 1 : (int)(target->size() / 10);

    vector<PointXYZ> matched_mesh_points;
    indices.clear();

    vector<int> pointIdxNKNSearch(K);
    vector<float> pointNKNSquaredDistance(K);

    PointXYZ p;
    p.x = source[0];
    p.y = source[1];
    p.z = source[2];

    if (kdtree.nearestKSearch(p, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
        bool match_not_found = true;
        for (int i = 0; i < pointIdxNKNSearch.size() && match_not_found; i++) {
            if (verify_index(pointIdxNKNSearch[i])) {
                PointXYZ m = target->points[pointIdxNKNSearch[i]];
                indices.push_back(pointIdxNKNSearch[i]);
                matched_mesh_points.push_back(m);
                match_not_found = false;
                second_last_index = last_index;
                last_index = pointIdxNKNSearch[i];
            }
        }

        if (match_not_found) {
            cerr << "WARNING: NonRigidMatching:nearest_neighbor_search: no matching and valid node found for tracking deformation" << endl;
        }
    }

    return matched_mesh_points;
}

bool cluster_error_map(vector<vpColVector> &err_map, int count, Eigen::Vector4f &centroid, vpHomogeneousMatrix &cMo, residual_statistics &res_stat) {
    vector<PointXYZ> candidate_points;
    float thresh1, thresh2;

    if ((res_stat.average < 0.000000001f) && (res_stat.average > -0.000000001f)) {
        float range = (CLUSTER_TOLERANCE * fabs(res_stat.maximum - res_stat.minimum));
        thresh1 = res_stat.average + range;
        thresh2 = res_stat.average - range;
    } else {
        thresh1 = res_stat.unweighted_average + 0.0001f;
        thresh2 = res_stat.unweighted_average - 0.0001f;
    }
    for (int i = 0; i < err_map.size(); i++) {
        if ((err_map[i][3] > thresh1) || (err_map[i][3] < thresh2)) {
            PointXYZ p;
            p.x = err_map[i][0];
            p.y = err_map[i][1];
            p.z = err_map[i][2];
            candidate_points.push_back(p);
        }
    }

    if (candidate_points.size() > MIN_CLUSTER_SIZE) {
        vector<PointCloud<PointXYZ>::Ptr> clusters;
        vector<int> sizes;

        int index = clustering(candidate_points, clusters, sizes);
        pcl::compute3DCentroid(*(clusters[index]), centroid);
        return true;
    } else {
        return false;
    }
}

vector<Vector3d> nearest_neighbor(PointCloud<PointXYZ> cloud_, PointCloud<PointXYZ>::Ptr cluster, float dist) {
    PointCloud<PointXYZ>::Ptr cloud(new PointCloud<PointXYZ>);
    *cloud = cloud_;

    MatrixXd A = MatrixXd::Random(cluster->size(), 3);
    vector<Vector3d> normals;

    int K = 1;

    KdTreeFLANN<PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    vector<int> pointIdxNKNSearch(K);
    vector<float> pointNKNSquaredDistance(dist);

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
            }
        }
    }

    Vector3d v = A.colwise().mean();
    v.normalize();

    return normals;
}

void NonRigidMatching::read_frame(PointCloud<PointXYZRGB>::Ptr &frame, int count) {
    OcclusionCheck ocl;
    ocl.load_pcd_rgb(*frame, data_folder + to_string(count + 1) + ".pcd");
}

void NonRigidMatching::read_frame_and_register(PointCloud<PointXYZRGB>::Ptr &frame, cv::Mat &img, cv::Mat &depth_image, offline_data_map data_map, cv::Mat &colored_depth, int count, int offset) {
    OcclusionCheck ocl;

#ifdef DEPTH_FORMAT_CSV
    _log_("Reading depth file: " << depth_data_folder + to_string(data_map.depth_files[offset + count]) + ".csv and image file: " << color_data_folder + to_string(data_map.color_files[offset + count]) + ".png (relative paths)") depth_image =
        ocl.read_csv_depth_data(depth_data_folder + to_string(data_map.depth_files[offset + count]) + ".csv");
    _log_("finished reqding depth")
#elif defined(DEPTH_FORMAT_PNG)
    _log_("Reading depth file: " << to_string(data_map.depth_files[offset + count]) + ".png and image file: " << to_string(data_map.color_files[offset + count]) + ".png (relative paths)") depth_image =
        ocl.read_png_depth_data(depth_data_folder + to_string(data_map.depth_files[offset + count]) + ".png");

#else
    _log_("Reading depth file: " << to_string(data_map.depth_files[offset + count]) + ".raw and image file: " << to_string(data_map.color_files[offset + count]) + ".png (relative paths)") depth_image =
        ocl.read_raw_depth_data(depth_data_folder + to_string(data_map.depth_files[offset + count]) + ".raw");
#endif
        _log_("converting rqz depth to pointcloud") ocl.depth_to_pcl(depth_image, *frame, data_map);
    _log_("registering pointcloud zith i,qge") ocl.load_image_and_register(color_data_folder + to_string(data_map.color_files[offset + count]) + ".png", *frame, data_map, img, colored_depth, (offset + count));
    _log_("done registrqtion")
}

void NonRigidMatching::initialize_rigid_tracking(vpMbGenericTracker &tracker) {
    map<std::string, int> tracker_map;
    tracker_map["Camera1"] = vpMbGenericTracker::KLT_TRACKER;
    tracker_map["Camera2"] = vpMbGenericTracker::DEPTH_DENSE_TRACKER;
    tracker.setTrackerType(tracker_map);
}

void NonRigidMatching::initialize(PointCloud<PointXYZRGB>::Ptr &frame_, Matrix4f &transform_init, RigidTracking &rTrack, vpMbGenericTracker &tracker, string config_path, string cao_model_path, vpHomogeneousMatrix &cMo, offline_data_map &data_map, cv::Mat &image, cv::Mat &depth_image,
                                  cv::Mat &colored_depth, int pcd_count, int node_count, int data_offset, bool is_initialized, bool is_registration_required) {
#ifndef SENSOR_INPUT_ACTIVE
    PointCloud<PointXYZRGB>::Ptr frame(new PointCloud<PointXYZRGB>);
    if (is_registration_required) {
        if ((node_count == 0) || (pcd_count < VISP2CUSTOM_TRACKING_SWITCH)) {
            _log_("Registration is required") read_frame_and_register(frame, image, depth_image, data_map, colored_depth, pcd_count, data_offset);
        } else {
            frame = frame_;
        }
    } else {
        _log_("Registration is not required") read_frame(frame, pcd_count);
    }
#endif

    ocl.eigen_to_visp_4x4(transform_init, cMo);
    vector<vpColVector> err_map;

    if (!is_initialized) {
#ifdef SENSOR_INPUT_ACTIVE
        rTrack.initialize(config_path, cao_model_path, frame_, cMo, tracker, true);
        rTrack.track(frame_, cMo, tracker, err_map, pcd_count, true, true);
#else
        // rTrack.initialize(config_path, cao_model_path, frame, cMo, tracker, false);
        // rTrack.track(frame, cMo, tracker, err_map, pcd_count, true, false);
#endif
    } else {
#ifdef SENSOR_INPUT_ACTIVE
        rTrack.track(frame_, cMo, tracker, err_map, pcd_count, true, true);
#else
        rTrack.track(frame, cMo, tracker, err_map, pcd_count, true, false);
#endif
    }

#ifndef SENSOR_INPUT_ACTIVE
    frame_ = frame;
#endif
    if (is_registration_required) {
        vpImageConvert::convert(image, rTrack.I_color);
    }
}

void clean_mechanical_mesh(PointCloud<PointXYZ>::Ptr &mechanical_mesh_points, vector<int> &fixed_constraints) {
    for (int i = 0; i < fixed_constraints.size(); i++) {
        mechanical_mesh_points->at(fixed_constraints[i]).x = EXTREMELY_DISTANT_POINTS;  // setting the fixed-constrained points to a very far off point,
        mechanical_mesh_points->at(fixed_constraints[i]).y = EXTREMELY_DISTANT_POINTS;  // so that nearest neighbor does not pick it up
        mechanical_mesh_points->at(fixed_constraints[i]).z = EXTREMELY_DISTANT_POINTS;
    }
}

// status : 0 - 'ready'
void NonRigidMatching::align_and_cluster(PointCloud<PointXYZ>::Ptr &mechanical_mesh_points, vpHomogeneousMatrix &transform, vector<PointXYZ> &matched_points, vector<int> &indices, vector<vpColVector> &err_map, residual_statistics &res_stat, int status, int count, mesh_map &visible_mesh) {
    if (status == 0) {
        PointCloud<PointXYZ>::Ptr local_mesh(new PointCloud<PointXYZ>);
        copyPointCloud(*mechanical_mesh_points, *local_mesh);

        indices.clear();
        matched_points.clear();

        OcclusionCheck ocl;
        Eigen::Matrix4f T;
        ocl.visp_to_eigen_4x4(transform, T);
        pcl::transformPointCloud(*local_mesh, *local_mesh, T);

        matched_points.clear();
        Eigen::Vector4f centroid;

        if (cluster_error_map(err_map, count, centroid, transform, res_stat)) {
            matched_points = nearest_neighbor_search(centroid, local_mesh, indices, count, visible_mesh);
        }
    }
}

bool not_in_list(int index, vector<int> &data_list) {
    bool ret_val = true;

    for (int i = 0; i < data_list.size(); i++) {
        if (data_list[i] == index) ret_val = false;
    }

    return ret_val;
}

void NonRigidMatching::m_estimator_mat(cv::Mat &data, float threshold, int datatype) {
    vpColVector W;

    EigenMatrix E;
    ocl.cvMat_to_eigenMatrix_row(data, E, datatype);
    //_info_("Norm: "<<E.colwise().mean())

    W.resize(E.rows(), false);
    ocl.assign_weights_to_matrix_visp(E, W, threshold);

    for (int i = 0; i < E.rows(); i++) {
        E(i, 0) *= W[i];
    }

    ocl.eigenMatrix_to_cvMat_row(E, data, datatype);
}

void NonRigidMatching::align_and_cluster_2(cv::Mat &prev_image, cv::Mat &curr_image, PointCloud<PointXYZ>::Ptr &mechanical_mesh_points, mesh_map &visible_mesh, vector<int> &fixed_constraints, Matrix4f &transform, vector<int> &active_indices, vector<PointXYZ> &active_indices_point,
                                           cv::Mat geometric_error, string additionalDebugFolder, int debugFlag, int num_clusters, int pcd_count) {
    _log_("Trying to compute " << num_clusters << " active points")

        cv::absdiff(prev_image, curr_image, diff_image);  // difference between previous and current image

    cv::Rect crop(visible_mesh.roi.min_x, visible_mesh.roi.min_y, (visible_mesh.roi.max_x - visible_mesh.roi.min_x), (visible_mesh.roi.max_y - visible_mesh.roi.min_y));

    diff_image = diff_image(crop);
    geometric_error = geometric_error(crop);

    m_estimator_mat(diff_image, 6.5f, 0);
    m_estimator_mat(geometric_error, 0.06f, 1);

    if (debugFlag) {
        cv::Mat debug_image = geometric_error.clone();
        cv::normalize(geometric_error, debug_image, 0, 255, cv::NORM_MINMAX);
        cv::imwrite(additionalDebugFolder + "/geometric_image" + to_string(pcd_count) + ".png", debug_image);
    }

    for (int i = 0; i < geometric_error.rows; i++) {
        for (int j = 0; j < geometric_error.cols; j++) {
            diff_image.at<uchar>(i, j) = geometric_error.at<float>(i, j) * (float)diff_image.at<uchar>(i, j);
        }
    }

    cv::medianBlur(diff_image, diff_image, 3);

    if (debugFlag) {
        cv::Mat debug_image = diff_image.clone();
        cv::normalize(diff_image, debug_image, 0, 255, cv::NORM_MINMAX);
        cv::imwrite(additionalDebugFolder + "/difference_image" + to_string(pcd_count) + ".png", debug_image);
    }

    // exit(0);

    cv::Mat cropped_image = diff_image;

    //    cv::Rect crop(visible_mesh.roi.min_x, visible_mesh.roi.min_y, (visible_mesh.roi.max_x - visible_mesh.roi.min_x), (visible_mesh.roi.max_y-visible_mesh.roi.min_y));
    //    cv::Mat cropped_image;
    //    cropped_image = diff_image(crop); //crop to RoI
    cv::adaptiveThreshold(cropped_image, cropped_image, 255.0f, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 3, 0);  // adaptive threshold to binary
    int size = 0;

    cv::Mat data_vec = cv::Mat::zeros(cv::Size(1, 49), CV_64FC1);

    for (int i = 0; i < geometric_error.rows; i++) {
        for (int j = 0; j < geometric_error.cols; j++) {
            diff_image.at<uchar>(i, j) = geometric_error.at<float>(i, j) * (float)diff_image.at<uchar>(i, j);
        }
    }

    // cv::PCA pt_pca(cropped_image, cv::Mat(), CV_PCA_DATA_AS_ROW, 0);
    //
    // cv::Mat pt_mean = pt_pca.mean;
    //
    // cv::Mat pt_eig_vals = pt_pca.eigenvalues;
    //
    // ofstream my_file;
    // my_file.open("/media/agniv/f9826023-e8c9-47ab-906c-2cbd7ccf196a/home/agniv/Documents/data/real_data/pizza_new/staging_area/eigen.csv",ios::app);
    // for (int i = 0; i < 6; ++i)
    //     my_file<<setprecision(10)<<i<<"-th eigen value::, " << pt_eig_vals.at<float>(i, 0) << std::endl;
    //
    // my_file<<"***"<<endl;
    // my_file.close();

    // count number of pixels with valid data after thresholding
    for (int y = 0; y < cropped_image.rows; y++) {
        for (int x = 0; x < cropped_image.cols; x++) {
            if (cropped_image.at<uchar>(y, x) > 0.01f) {
                size++;
            }
        }
    }

    // creating a Mat of size 'size' and populating it with thresholded pixel coordinates
    cv::Mat samples(size, 2, CV_32F);
    int count = 0;
    for (int y = 0; y < cropped_image.rows; y++) {
        for (int x = 0; x < cropped_image.cols; x++) {
            if (cropped_image.at<uchar>(y, x) > 0.05f) {
                samples.at<float>(count, 0) = y;
                samples.at<float>(count, 1) = x;
                count++;
            }
        }
    }

    // splitting the thresholded pixels into clusters
    cv::Mat labels, centers;
    cv::kmeans(samples, num_clusters, labels, cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10, 0.1), 3, cv::KMEANS_PP_CENTERS, centers);
    pcl::transformPointCloud(*mechanical_mesh_points, *mechanical_mesh_points, transform);

    // centroiding...
    float x_sum = 0.0f;
    float y_sum = 0.0f;

    for (int i = 0; i < num_clusters; i++) {
        x_sum += centers.at<float>(i, 0);
        y_sum += centers.at<float>(i, 1);
    }

    x_sum /= (float)num_clusters;
    y_sum /= (float)num_clusters;

    centers.at<float>(0, 0) = x_sum;
    centers.at<float>(0, 1) = y_sum;

    num_clusters = 1;
    // .....centroiding ends

    if (debugFlag) {
        cv::Mat debug_image = curr_image.clone();
        for (int i = 0; i < num_clusters; i++) {
            cv::Point cp;
            cp.x = visible_mesh.roi.min_x + centers.at<float>(i, 1);
            cp.y = visible_mesh.roi.min_y + centers.at<float>(i, 0);
            cv::Scalar black(0, 0, 0);
            cv::circle(debug_image, cp, 10, black);
            cv::drawMarker(debug_image, cp, black, cv::MARKER_CROSS, 10, 1);
        }

        cv::imwrite(additionalDebugFolder + "/clusters_raw_" + to_string(pcd_count) + ".png", debug_image);
    }

    /*vector<int> additional_constraints;
    additional_constraints.push_back(1);
    additional_constraints.push_back(24);
    additional_constraints.push_back(8);
    additional_constraints.push_back(21);
    additional_constraints.push_back(0);
    additional_constraints.push_back(44);
    additional_constraints.push_back(14);
    additional_constraints.push_back(31);
    additional_constraints.push_back(4);
    additional_constraints.push_back(20);
    additional_constraints.push_back(11);
    additional_constraints.push_back(27);
    additional_constraints.push_back(5);
    additional_constraints.push_back(34);
    additional_constraints.push_back(15);
    additional_constraints.push_back(41);*/

    active_indices.clear();
    active_indices_point.clear();

    for (int i = 0; i < num_clusters; i++) {
        int min_index = -1;
        float min_dist = 10000.0f;

        // finding the projected mesh vertex closest to the cluster center
        for (int j = 0; j < visible_mesh.projected_points_5d.size(); j++) {
            float dist = sqrt((visible_mesh.roi.min_x + centers.at<float>(i, 1) - visible_mesh.projected_points_5d[j].u) * (visible_mesh.roi.min_x + centers.at<float>(i, 1) - visible_mesh.projected_points_5d[j].u) +
                              (visible_mesh.roi.min_y + centers.at<float>(i, 0) - visible_mesh.projected_points_5d[j].v) * (visible_mesh.roi.min_y + centers.at<float>(i, 0) - visible_mesh.projected_points_5d[j].v));

            if (dist < min_dist) {
                min_dist = dist;
                min_index = j;
            }
        }

        if (min_index != -1) {
            int min_index_cloud = -1;
            float min_dist_cloud = 100000.0f;
            float x, y, z;
            PointXYZ best_point;

            for (int j = 0; j < mechanical_mesh_points->size(); j++) {
                float dist = EUCLEDIAN_DIST(visible_mesh.projected_points_5d[min_index].X, visible_mesh.projected_points_5d[min_index].Y, visible_mesh.projected_points_5d[min_index].Z, mechanical_mesh_points->points[j].x, mechanical_mesh_points->points[j].y, mechanical_mesh_points->points[j].z);

                if ((dist < min_dist_cloud) && not_in_list(j, fixed_constraints) /*&& not_in_list(j, prev_active_points)*/) {
                    min_dist_cloud = dist;
                    min_index_cloud = j;
                    x = mechanical_mesh_points->points[j].x;
                    y = mechanical_mesh_points->points[j].y;
                    z = mechanical_mesh_points->points[j].z;
                    best_point.x = x;
                    best_point.y = y;
                    best_point.z = z;
                }
            }
            active_indices.push_back(min_index_cloud);
            active_indices_point.push_back(best_point);

            _log_("Computed Active Point: " << min_index_cloud)
        }
    }

    prev_active_points.clear();
    prev_active_points = active_indices;
}

void NonRigidMatching::align_and_cluster_3(cv::Mat &prev_image, cv::Mat &curr_image, PointCloud<PointXYZ>::Ptr &mechanical_mesh_points, mesh_map &visible_mesh, vector<int> &fixed_constraints, Matrix4f &transform, vector<int> &active_indices, vector<PointXYZ> &active_indices_point,
                                           cv::Mat geometric_error, string additionalDebugFolder, int debugFlag, int num_clusters, int pcd_count) {
    cv::absdiff(prev_image, curr_image, diff_image);

    cv::Rect crop(visible_mesh.roi.min_x, visible_mesh.roi.min_y, (visible_mesh.roi.max_x - visible_mesh.roi.min_x), (visible_mesh.roi.max_y - visible_mesh.roi.min_y));

    diff_image = diff_image(crop);
    geometric_error = geometric_error(crop);

    bool do_not_update = false;

    if (debugFlag) {
        cv::Mat debug_image = geometric_error.clone();
        cv::normalize(geometric_error, debug_image, 0, 255, cv::NORM_MINMAX);
        cv::imwrite(additionalDebugFolder + "/geometric_image_3_" + to_string(pcd_count) + ".png", debug_image);
    }

    for (int i = 0; i < geometric_error.rows; i++) {
        for (int j = 0; j < geometric_error.cols; j++) {
            diff_image.at<uchar>(i, j) = 1.0f * (float)diff_image.at<uchar>(i, j);
        }
    }

    cv::medianBlur(diff_image, diff_image, 7);

    if (debugFlag) {
        cv::Mat debug_image = diff_image.clone();
        cv::normalize(diff_image, debug_image, 0, 255, cv::NORM_MINMAX);
        cv::imwrite(additionalDebugFolder + "/difference_image_3_" + to_string(pcd_count) + ".png", debug_image);
    }

    cv::Mat cropped_image = diff_image;

    float min_value = 2.0f;
    float max_value = 150.0f;

    int size = 0;

    for (int y = 0; y < cropped_image.rows; y++) {
        for (int x = 0; x < cropped_image.cols; x++) {
            if ((cropped_image.at<float>(y, x) > min_value) && (cropped_image.at<float>(y, x) < max_value)) {
                _info_(" ") size++;
            }
        }
    }

    cv::Mat samples(size, 2, CV_32F);
    int count = 0;
    for (int y = 0; y < cropped_image.rows; y++) {
        for (int x = 0; x < cropped_image.cols; x++) {
            if ((cropped_image.at<float>(y, x) > min_value) && (cropped_image.at<float>(y, x) < max_value)) {
                samples.at<float>(count, 0) = y;
                samples.at<float>(count, 1) = x;
                count++;
            }
        }
    }

    cv::Mat labels, centers;
    cv::kmeans(samples, ELEMENTARY_CLUSTER_COUNT, labels, cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 3, 0.1), 3, cv::KMEANS_PP_CENTERS, centers);

    if (centers.rows > 0) {
        pcl::transformPointCloud(*mechanical_mesh_points, *mechanical_mesh_points, transform);

        // centroiding...
        float x_sum = 0.0f;
        float y_sum = 0.0f;

        for (int i = 0; i < ELEMENTARY_CLUSTER_COUNT; i++) {
            x_sum += centers.at<float>(i, 0);
            y_sum += centers.at<float>(i, 1);
        }

        x_sum /= (float)ELEMENTARY_CLUSTER_COUNT;
        y_sum /= (float)ELEMENTARY_CLUSTER_COUNT;

        if (pcd_count > CLUSTERING_HOLDBACK_CUTOFF) {
            float dist = sqrt(pow((x_sum - prev_centroid.x), 2) + pow((y_sum - prev_centroid.y), 2));

            if (dist < CLUSTERING_DISTANCE_THRESHOLD) {
                x_sum = prev_centroid.x;
                y_sum = prev_centroid.y;
                do_not_update = true;
            } else {
                prev_centroid.x = x_sum;
                prev_centroid.y = y_sum;
            }
        } else {
            prev_centroid.x = x_sum;
            prev_centroid.y = y_sum;
        }

        centers.at<float>(0, 0) = x_sum;
        centers.at<float>(0, 1) = y_sum;

        if (debugFlag) {
            cv::Mat debug_image = curr_image.clone();
            for (int i = 0; i < ELEMENTARY_CLUSTER_COUNT; i++) {
                cv::Point cp;
                cp.x = visible_mesh.roi.min_x + centers.at<float>(i, 1);
                cp.y = visible_mesh.roi.min_y + centers.at<float>(i, 0);
                cv::Scalar black(0, 0, 0);
                cv::circle(debug_image, cp, 10, black);
                cv::drawMarker(debug_image, cp, black, cv::MARKER_CROSS, 10, 1);
            }
            cv::imwrite(additionalDebugFolder + "/clusters_raw_3_" + to_string(pcd_count) + ".png", debug_image);
        }

        if (!do_not_update) {
            active_indices.clear();
            active_indices_point.clear();

            for (int i = 0; i < num_clusters; i++) {
                int min_index = -1;
                float min_dist = 10000.0f;

                for (int j = 0; j < visible_mesh.projected_points_5d.size(); j++) {
                    int c = 0;
                    float dist = sqrt((visible_mesh.roi.min_x + centers.at<float>(c, 1) - visible_mesh.projected_points_5d[j].u) * (visible_mesh.roi.min_x + centers.at<float>(c, 1) - visible_mesh.projected_points_5d[j].u) +
                                      (visible_mesh.roi.min_y + centers.at<float>(c, 0) - visible_mesh.projected_points_5d[j].v) * (visible_mesh.roi.min_y + centers.at<float>(c, 0) - visible_mesh.projected_points_5d[j].v));

                    if (dist < min_dist) {
                        min_dist = dist;
                        min_index = j;
                    }
                }

                if (min_index != -1) {
                    int min_index_cloud = -1;
                    float min_dist_cloud = 100000.0f;
                    float x, y, z;
                    PointXYZ best_point;

                    for (int j = 0; j < mechanical_mesh_points->size(); j++) {
                        float dist =
                            EUCLEDIAN_DIST(visible_mesh.projected_points_5d[min_index].X, visible_mesh.projected_points_5d[min_index].Y, visible_mesh.projected_points_5d[min_index].Z, mechanical_mesh_points->points[j].x, mechanical_mesh_points->points[j].y, mechanical_mesh_points->points[j].z);

                        if ((dist < min_dist_cloud) && not_in_list(j, fixed_constraints) && not_in_list(j, active_indices)) {
                            min_dist_cloud = dist;
                            min_index_cloud = j;
                            x = mechanical_mesh_points->points[j].x;
                            y = mechanical_mesh_points->points[j].y;
                            z = mechanical_mesh_points->points[j].z;
                            best_point.x = x;
                            best_point.y = y;
                            best_point.z = z;
                        }
                    }
                    active_indices.push_back(min_index_cloud);
                    active_indices_point.push_back(best_point);
                }
            }

            prev_active_points.clear();
            prev_active_points = active_indices;
        }
    } else {
        active_indices = prev_active_points;
        _info_("Not enough active points detected!")
    }
}

void NonRigidMatching::format_deformed_polygons(PolygonMesh model, vector<vector<Vec3>> deformed_meshes, vector<PolygonMesh> &formatted_meshes) {
    _log_("trying final formatting ....") int size = deformed_meshes[0].size();

    _log_("trying final formatting") int count = 0;
    for (int i = 0; i < 6; i++) {
        _log_("Loop #" << i) if ((i == 0) || (i == 2) || (i == 4)) {
            PolygonMesh model_ = model;
            pcl::PointCloud<pcl::PointXYZ>::Ptr deformed(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::fromPCLPointCloud2(model_.cloud, *deformed);

            if (size != deformed->size()) {
                cerr << "[NonRigidMatching::format_deformed_polygons] WARNING: Wrong sized model received from SOFA simulation. Expecting model with " << deformed->size() << " vertices, but received " << size << " vertices, for Jacobian no.:" << i << endl;
            }

            for (int j = 0; j < size; j++) {
                deformed->at(j).x = deformed_meshes[count][j][0];
                deformed->at(j).y = deformed_meshes[count][j][1];
                deformed->at(j).z = deformed_meshes[count][j][2];
            }
            pcl::toPCLPointCloud2(*deformed, model.cloud);
            formatted_meshes.push_back(model_);

            count++;
        }
        else {
            formatted_meshes.push_back(model);
        }
    }
    _log_("final formatting done")
}

void NonRigidMatching::update_polygon(PolygonMesh &model, vector<vector<Vec3>> deformed_mesh) {
    int size = deformed_mesh[0].size();

    _log_("Update polygon size: " << size)

        PolygonMesh model_ = model;
    pcl::PointCloud<pcl::PointXYZ>::Ptr deformed(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(model_.cloud, *deformed);
    for (int j = 0; j < size; j++) {
        deformed->at(j).x = deformed_mesh[0][j][0];
        deformed->at(j).y = deformed_mesh[0][j][1];
        deformed->at(j).z = deformed_mesh[0][j][2];
    }
    pcl::toPCLPointCloud2(*deformed, model_.cloud);
    model = model_;
}
