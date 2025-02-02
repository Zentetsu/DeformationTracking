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

vector<PointCloud<PointXYZ>::Ptr> clustering(vector<PointXYZ> point_source, float max_size) {
    float leaf_size = 0.01f;
    float cluster_tolerance = 0.2f;
    max_size * 2;
    int min_cluster_size = point_source.size() / 20;
    int max_cluster_size = point_source.size();

    PointCloud<PointXYZ>::Ptr cloud(new PointCloud<PointXYZ>);
    vector<PointCloud<PointXYZ>::Ptr> clusters;

    cloud->width = point_source.size();
    cloud->height = 1;
    cloud->is_dense = false;
    cloud->points.resize(cloud->width * cloud->height);

    int count = 0;
    for (int i = 0; i < cloud->points.size(); i++) {
        cloud->points[i].x = point_source[count].x;
        cloud->points[i].y = point_source[count].y;
        cloud->points[i].z = point_source[count].z;

        count++;
    }

    // Create the filtering object: downsample the dataset using a leaf size of 1cm
    // VoxelGrid<PointXYZ> vg;
    // PointCloud<PointXYZ>::Ptr cloud_filtered (new PointCloud<PointXYZ>);
    // vg.setInputCloud (cloud);
    // vg.setLeafSize (leaf_size, leaf_size, leaf_size);
    // vg.filter (*cloud_filtered);

    // Creating the KdTree object for the search method of the extraction
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

    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it) {
        PointCloud<PointXYZ>::Ptr cloud_cluster(new PointCloud<PointXYZ>);

        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit) cloud_cluster->points.push_back(cloud->points[*pit]);

        cloud_cluster->width = cloud_cluster->points.size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;
        clusters.push_back(cloud_cluster);
    }

    return clusters;
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
        // cout<<"Index of force: "<<pointIdxNKNSearch[0]<<", and coordinates are: ("<<points->points[pointIdxNKNSearch[0]].x<<","<<points->points[pointIdxNKNSearch[0]].y<<","<<points->points[pointIdxNKNSearch[0]].z<<")"<<endl;
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

// #ifdef DEBUG_DUMP
void NonRigidMatching::log_output(PolygonMesh &model, int frame_num, vpHomogeneousMatrix &cMo, double residual, string data_path, string opfilepath) {
    try {
        // STEP 1:
        pcl::io::savePLYFileBinary(opfilepath + "cpp_model/" + to_string(frame_num) + ".ply", model);

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

        // STEP 4:
        std::ofstream outfile;
        outfile.open(opfilepath + "cpp_log.txt", std::ios_base::app);
        outfile << to_string(frame_num) << " " << "/cpp_model/" + to_string(frame_num) + ".ply" << " " << "/cpp_errmap/" + to_string(frame_num) + ".txt" << " " << "/cpp_cMo/" + to_string(frame_num) + ".txt" << endl;
        outfile.close();

        // STEP 5:
        std::ofstream addnl_path;
        addnl_path.open(opfilepath + "additional_info.txt");
        addnl_path << "Data read from path: " << data_path << endl;
        time_t now = time(0);
        string dt = ctime(&now);
        addnl_path << "Timestamp: " << dt << endl;
        addnl_path.close();
    } catch (const std::exception &e) {
        // this executes if f() throws std::logic_error (base class rule)
        cerr << e.what() << " , " << "Exiting!" << endl;
        exit(0);
    }
}
// #endif

float eucledian_dist(PointXYZ a, PointXYZ b) { return sqrt(pow((a.x - b.x), 2) + pow((a.y - b.y), 2) + pow((a.z - b.z), 2)); }

ModelCoefficients::Ptr get_plane_equation(PointXYZRGBA point_a, PointXYZRGBA point_b, PointXYZRGBA point_c) {
    pcl::ModelCoefficients::Ptr plane(new pcl::ModelCoefficients);

    // cout<<"in here"<<endl;

    Hyperplane<float, 3> eigen_plane = Hyperplane<float, 3>::Through(point_a.getArray3fMap(), point_b.getArray3fMap(), point_c.getArray3fMap());
    // cout<<"done"<<endl;
    // cout<<eigen_plane<<endl;
    plane->values.resize(4);

    // cout<<"size: "<<plane->values.size()<<endl;

    for (int i = 0; i < plane->values.size(); i++) {
        // cout<<"ho"<<endl;
        plane->values[i] = eigen_plane.coeffs()[i];
        // cout<<"val: "<<eigen_plane.coeffs()[i]<<endl;
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
    // build the filter
    outrem.setInputCloud(cloudPTR);
    outrem.setRadiusSearch(1.1);
    outrem.setMinNeighborsInRadius(4);
    // apply filter
    cout << "filtering..." << endl;
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

    // cout<<max_z<<","<<min_z<<","<<target.z<<endl;

    if ((target.z < max_z) && (target.z > min_z)) {
        // cout<<"true"<<endl;
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
        if (true /*check_rejection(rejection_list,i)*/)  // set this to true
        {
            vector<Vector3d> n = normal[i];
            MatrixXd A = MatrixXd::Random(n.size(), 3);

            cout << "Cluster " << i << " finally on" << endl;
            for (int j = 0; j < n.size(); j++) {
                A(j, 0) = n[j](0);
                A(j, 1) = n[j](1);
                A(j, 2) = n[j](2);
            }

            Vector3d v = A.colwise().mean();
            v.normalize();

            cout << "normal@" << i << ": " << endl << v << endl;
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
    cout << "post processing clusters" << endl;
    vector<int> size_of_clusters;
    vector<int> reject_index;
    for (int i = 0; i < clusters.size(); i++) {
        size_of_clusters.push_back(clusters[i]->size());
    }

    sort(size_of_clusters.begin(), size_of_clusters.end(), greater<int>());
    int cutoff_size = -1;

    for (int i = 0; i < (clusters.size() - 1); i++) {
        float fraction_drop = ((float)(size_of_clusters[i] - size_of_clusters[i + 1])) / (float)size_of_clusters[i];

        cout << size_of_clusters[i] << "," << size_of_clusters[i + 1] << " ..,.. " << fraction_drop << endl;

        if (fraction_drop > CLUSTER_SIZE_CUTOFF) {
            cutoff_size = size_of_clusters[i + 1];
            cout << "cutoff size: " << cutoff_size << endl;
        }
    }

    if (cutoff_size != -1) {
        for (int i = 0; i < clusters.size(); i++) {
            if ((clusters[i]->size() <= cutoff_size) && (check_sanity(clusters[i]))) {
                reject_index.push_back(i);
                cout << "reject index: " << i << endl;
            }
        }
    }
    return reject_index;
}

vector<PointCloud<PointXYZ>::Ptr> eucledian_cluster(PointCloud<PointXYZ>::Ptr &cloud, vector<Vector3d> &normals) {
    cout << "start..." << endl;
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

    cout << "extracted" << endl;

    vector<PointCloud<PointXYZ>::Ptr> cluster_list;
    vector<vector<Vector3d>> normal_list;

    int j = 0;

    for (vector<PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it) {
        cout << "list " << j << endl;
        PointCloud<PointXYZ>::Ptr cloud_cluster(new PointCloud<PointXYZ>);
        vector<Vector3d> normal_;

        for (vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit) {
            if ((fabs(cloud->points[*pit].x) > 0.001f) && (fabs(cloud->points[*pit].y) > 0.001f) && (fabs(cloud->points[*pit].z) > 0.001f)) {
                cloud_cluster->points.push_back(cloud->points[*pit]);  //*
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
            cout << "zero size cloud detected" << endl;
        }
        j++;
        cout << "list end" << endl;
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

vector<PointXYZ> nearest_neighbor_search(vector<vpColVector> source, PointCloud<PointXYZ>::Ptr target, vector<int> &indices, int index) {
    vector<PointXYZ> matched_mesh_points;
    //    KdTreeFLANN<PointXYZ> kdtree;
    //    kdtree.setInputCloud (target);
    //    int K = 1; float dist = 0.08;
    //
    //    vector<PointXYZ> matched_mesh_points;
    //
    //    vector<int> pointIdxNKNSearch(K);
    //    vector<float> pointNKNSquaredDistance(0.0001);
    //
    //    PointXYZ p;
    //
    //    for (int i = 0; i < source.size(); i++)
    //    {
    //        p.x = source[i][0];p.y = source[i][1];p.z = source[i][2];
    //        if (kdtree.nearestKSearch (p, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
    //        {
    //            PointXYZ m = target->points[pointIdxNKNSearch[0]];
    //            if(!check_if_already_exists(indices,pointIdxNKNSearch[0]))
    //            {
    //                indices.push_back(pointIdxNKNSearch[0]);
    //                matched_mesh_points.push_back(m);
    //            }
    //        }
    //    }

    indices.push_back(17);
    PointXYZ p;
    p.x = 10.0f;
    p.z = 10.0f;
    p.z = 10.0f;
    matched_mesh_points.push_back(p);

    return matched_mesh_points;
}

void cluster_error_map(vector<vpColVector> &err_map, int count, vector<vpColVector> &centroid, vpHomogeneousMatrix &cMo) {
    float max = -1000.0f;
    float min = 1000.0f;
    float x_min, y_min, z_min, x_max, y_max, z_max;
    float error_cutoff_threshold = 0.7f;

    centroid.clear();
    vector<PointCloud<PointXYZ>::Ptr> clusters;

    if ((count > 0)) {
        vpColVector p(4);
        p[0] = -0.028288;
        p[1] = -0.006f;
        p[2] = 0.142213f;
        p[3] = 1.0f;

        p = cMo * p;
        centroid.push_back(p);
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

void NonRigidMatching::read_frame(PointCloud<PointXYZRGB>::Ptr &frame, int count) {
    cout << data_folder + to_string(count + 1) + ".pcd" << endl;

    OcclusionCheck ocl;
    ocl.load_pcd_as_text_rgbd(*frame, data_folder + to_string(count + 1) + ".pcd", true);
}

vpMbGenericTracker NonRigidMatching::initialize_rigid_tracking() {
    std::vector<int> tracker_type(2);
    tracker_type[0] = vpMbGenericTracker::EDGE_TRACKER;
    tracker_type[1] = vpMbGenericTracker::DEPTH_DENSE_TRACKER;
    vpMbGenericTracker tracker(tracker_type);

    return tracker;
}

void NonRigidMatching::initialize(PointCloud<PointXYZRGB>::Ptr &frame_, Matrix4f &transform_init, RigidTracking &rTrack, vpMbGenericTracker &tracker, string config_path, string cao_model_path, vpHomogeneousMatrix &cMo, int pcd_count) {
    PointCloud<PointXYZRGB>::Ptr frame(new PointCloud<PointXYZRGB>);
    read_frame(frame, pcd_count);
#ifdef RIGID_TRACKING
    ocl.eigen_to_visp_4x4(transform_init, cMo);
    rTrack.initialize(config_path, cao_model_path, frame, cMo, tracker);
#endif
    frame_ = frame;
}

// status : 0 - 'ready'
void NonRigidMatching::align_and_cluster(PointCloud<PointXYZRGB>::Ptr &frame, PointCloud<PointXYZ>::Ptr &mechanical_mesh_points, Matrix4f &transform_init, PolygonMesh &model, RigidTracking &rTrack, vpMbGenericTracker &tracker, vector<PointXYZ> &matched_points, vector<int> &indices, int status,
                                         int count) {
    cout << "good to go for matching" << endl;
    if (status == 0) {
        vpHomogeneousMatrix cMo;
        ocl.eigen_to_visp_4x4(transform_init, cMo);
        std::vector<vpColVector> err_map;
        vector<vpColVector> centroid;

#ifdef RIGID_TRACKING
        rTrack.track(frame, cMo, tracker, err_map, count, true);
#endif
        cluster_error_map(err_map, count, centroid, cMo);
        matched_points = nearest_neighbor_search(centroid, mechanical_mesh_points, indices, count);
    }
}

void NonRigidMatching::format_deformed_polygons(PolygonMesh model, vector<vector<Vec3>> deformed_meshes, vector<PolygonMesh> &formatted_meshes) {
    int size = deformed_meshes[0].size();

    for (int i = 0; i < 6; i++) {
        PolygonMesh model_ = model;
        pcl::PointCloud<pcl::PointXYZ>::Ptr deformed(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromPCLPointCloud2(model_.cloud, *deformed);

        for (int j = 0; j < size; j++) {
            deformed->at(j).x = deformed_meshes[i][j][0];
            deformed->at(j).y = deformed_meshes[i][j][1];
            deformed->at(j).z = deformed_meshes[i][j][2];
        }
        pcl::toPCLPointCloud2(*deformed, model.cloud);

        formatted_meshes.push_back(model_);
    }
}

void NonRigidMatching::update_polygon(PolygonMesh &model, vector<vector<Vec3>> deformed_mesh) {
    int size = deformed_mesh[0].size();

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
