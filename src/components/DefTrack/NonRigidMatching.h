#ifndef NONRIGIDMATCHING_H
#define NONRIGIDMATCHING_H

#include <sofa/component/controller/Controller.h>
#include <sofa/core/behavior/BaseController.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/SolidTypes.h>
#include <sofa/defaulttype/VecTypes.h>

#include "OcclusionCheck.h"
#include "RigidTracking.h"

class NonRigidMatching {
   public:
    typedef sofa::type::Vec<3, double> Vec3;
    typedef sofa::type::Vec<4, double> Vec4;

    void align_and_cluster(PointCloud<PointXYZ>::Ptr &mechanical_mesh_points, vpHomogeneousMatrix &transform, vector<PointXYZ> &matched_points, vector<int> &indices, vector<vpColVector> &err_map, residual_statistics &res_stat, int status, int count, mesh_map &visible_mesh);
    void initialize_rigid_tracking(vpMbGenericTracker &tracker);
    void initialize(PointCloud<PointXYZRGB>::Ptr &frame_, Matrix4f &transform_init, RigidTracking &rTrack, vpMbGenericTracker &tracker, string config_path, string cao_model_path, vpHomogeneousMatrix &cMo, offline_data_map &data_map, cv::Mat &image, cv::Mat &depth_image, cv::Mat &colored_depth,
                    int pcd_count, int node_count, int data_offset, bool is_initialized, bool is_registration_required);
    string data_folder;
    string color_data_folder;
    string depth_data_folder;
    void format_deformed_polygons(PolygonMesh model, vector<vector<Vec3>> deformed_meshes, vector<PolygonMesh> &formatted_meshes);
    void update_polygon(PolygonMesh &model, vector<vector<Vec3>> deformed_mesh);

    void align_and_cluster_2(cv::Mat &prev_image, cv::Mat &curr_image, PointCloud<PointXYZ>::Ptr &mechanical_mesh_points, mesh_map &visible_mesh, vector<int> &fixed_constraints, Matrix4f &transform, vector<int> &active_indices, vector<PointXYZ> &active_indices_point, cv::Mat geometric_error,
                             string additionalDebugFolder, int debugFlag, int num_clusters, int pcd_count);

    void align_and_cluster_3(cv::Mat &prev_image, cv::Mat &curr_image, PointCloud<PointXYZ>::Ptr &mechanical_mesh_points, mesh_map &visible_mesh, vector<int> &fixed_constraints, Matrix4f &transform, vector<int> &active_indices, vector<PointXYZ> &active_indices_point, cv::Mat geometric_error,
                             string additionalDebugFolder, int debugFlag, int num_clusters, int pcd_count);

    // #ifdef DEBUG_DUMP
    void log_output(PointCloud<PointXYZRGB>::Ptr &frame, PolygonMesh &model, int frame_num, vpHomogeneousMatrix &cMo, vector<PointCloud<PointXYZRGB>> debugCloudList, vector<string> debugCloudLabel, cv::Mat depth_image, double residual, string data_path, string opfilepath, bool needs_initialization,
                    string &image_debug_path);
    // #endif

    vector<int> valid_indices;
    void set_valid_index();
    int last_index;
    int second_last_index;

    void m_estimator_mat(cv::Mat &data, float threshold, int datatype);

    point_2d prev_centroid;

   protected:
    void read_frame(PointCloud<PointXYZRGB>::Ptr &frame, int count);
    void read_frame_and_register(PointCloud<PointXYZRGB>::Ptr &frame, cv::Mat &img, cv::Mat &depth_image, offline_data_map data_map, cv::Mat &colored_depth, int count, int offset);
    vector<PointXYZ> nearest_neighbor_search(Eigen::Vector4f &source, PointCloud<PointXYZ>::Ptr target, vector<int> &indices, int index, mesh_map &visible_mesh);
    bool verify_index(int index);
    OcclusionCheck ocl;

    cv::Mat diff_image;
    vector<int> prev_active_points;
};

#endif
