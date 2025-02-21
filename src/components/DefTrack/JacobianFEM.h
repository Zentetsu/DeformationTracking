#ifndef JACOBIANFEM_H
#define JACOBIANFEM_H

#include "OcclusionCheck.h"

class JacobianFEM {
   public:
    static bool normal_estimation(const PointXYZ &a, const PointXYZ &b, const PointXYZ &c, const PointXYZ &centroid, pcl::PointXYZ &normal, bool should_debug);
    double compute_update(PointCloud<PointXYZRGB>::Ptr &pointcloud, PolygonMesh base_model, Eigen::Matrix4f &transform, vpHomogeneousMatrix &cMo, EigenMatrix &update, cv::Mat &colored_depth);
    void get_residual_simple(PolygonMesh poly_mesh, PointCloud<PointXYZRGB>::Ptr &pcl_cloud, vpHomogeneousMatrix &cMo, int iteration, vector<vpColVector> &err_map, bool create_error_map, residual_statistics &res_stats, mesh_map &visible_mesh, vector<vector<double>> &correspond,
                             EigenMatrix &residual, Mat &geometric_error_map);
    void set_iteration(int iter);
    void set_pcd_count(int count);
    void set_file_num(int _file_num);
    void set_op_file_path(string op_file_path);
    void set_camera_parameters(float Fx, float Fy, float Cx, float Cy);
    void occlusion_init();
    int iteration;
    int pcd_count;
    int node_count;
    int file_num;
    float F_x;
    float F_y;
    float C_x;
    float C_y;
    int debugFlag;
    void set_debug_flag(int value);
    string additionalDebugFolder;
    void set_additional_debug_folder(const string &value);

    vector<PointCloud<PointXYZRGB>> getDebugCloudList();
    vector<string> getDebugCloudLabel();
    bool get_photometric_residual_secondary(cv::Mat &colored_depth, vector<vector<double>> &_correspondence_photo, vector<PointXYZ> &_corresponding_model_list, vector<PointXYZ> &_correspondence_barycentric_photo, vector<double> &residual, int index);
    bool get_sparse_residual_secondary(vector<PointXYZ> &_corresponding_model_list, map<int, double> &residual);

    void setInitialLambda(float value);

    JacobianFEM();

    void setJacobian_displacement(float value);

    PointCloud<PointXYZRGB>::Ptr get_prev_pointcloud();

    void set_prev_pointcloud(const PointCloud<PointXYZRGB>::Ptr &prev_pointcloud);
    void set_is_photo_initialized(bool is_photo_initialized);

    void setCurr_image(const cv::Mat &curr_image);

    void setData_map(const offline_data_map &value);
    vector<int> active_point;
    bool is_first_node;  // this flag has been compromised and has been used to represent different things at different points in the code, highly problematic! Do not modify.
    // P.S: too lazy to fix this properly
    bool is_first_iteration;
    int num_vertices;

    void set_prev_colored_depth(const cv::Mat &value);
    cv::Mat get_prev_colored_depth();

    bool safe_to_proceed;
    bool undef_mesh_initialized;

    void decrementLmLambda();
    void setLmLambda(double value);

    void setPrev_residual(double value);

    EigenMatrix inflMat;
    EigenMatrix inflMat_a;

    void augment_transformation(const Eigen::Matrix4f &transform);
    void clear_sparse_buffer();
    void clear_sparse_buffer_partial();

    bool no_more_updates;

   private:
    sparse_map sparse_correspondence;
    bool is_sparse_prepared;

    double lmLambda;
    double prev_residual;

    vector<int> visible_full_map;

    pcl::PointCloud<pcl::PointXYZ>::Ptr undef_mesh_original;

    void get_influence_matrix(EigenMatrix &influence_mat, int index_0, int index_1, int index_2, int control_point_index, int option);

    void regularize_sparse_buffer(int size);
    void feature_matching(cv::Mat img1, cv::Mat img2);
    PointXYZRGB get_depth_for_sparse(PointCloud<PointXYZRGB>::Ptr &cloud, double x, double y);

    void levenberg_marquadt_visp(EigenMatrix &residual, EigenMatrix &J, vpColVector &update, double lmLambda, double alpha);

    int global_counter;
    void debug_residual(vector<Eigen::Vector4f> &residual_map, string &debug_path, string &file_name, int iteration);
    void resetDebugCloudList();
    void resetDebugCloudLabel();
    EigenMatrix get_residual_depth(pcl::PolygonMesh &poly_mesh, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pcl_cloud, bool is_first, std::string file_name, mesh_map &mesh, int iteration);
    EigenMatrix get_residual_photometric(cv::Mat &colored_depth, pcl::PolygonMesh poly_mesh, pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud, Eigen::Matrix4f transform, bool is_first, std::string file_name, mesh_map mesh, vector<int> observability_indices, int iteration, int balancer);
    EigenMatrix get_residual_sparse(cv::Mat &colored_depth, pcl::PolygonMesh poly_mesh, pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud, Eigen::Matrix4f transform, bool is_first, std::string file_name, mesh_map mesh, int iteration);
    double compute_jacobian(int num_control_points, pcl::PolygonMesh &visible_points, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pcl_cloud, Eigen::Matrix4f &transform, EigenMatrix &update, cv::Mat &colored_depth, int iteration, int pcd_count);
    void preformat_mesh(PolygonMesh &mesh, mesh_map &mesh_struct, pcl::PointCloud<pcl::PointXYZ>::Ptr &vertices_visible);
    bool preformat_all_models(PolygonMesh &base_model, Eigen::Matrix4f &transform, pcl::PointCloud<pcl::PointXYZ>::Ptr &vertices_visible, vector<mesh_map> &meshes);
    bool compute_analytic_part_jacobian(vector<mesh_map> meshes_full, EigenMatrix &J, Eigen::Matrix4f &transform, int num_forces);

#ifdef WEIGHT_VISUALIZATION
    vpColVector gauss_newton_visp_rigid(EigenMatrix &residual, EigenMatrix &J, vpColVector &update, double rigid_lambda, double alpha, int cutpoint);
#else
    void gauss_newton_visp_rigid(EigenMatrix &residual, EigenMatrix &J, vpColVector &update, double rigid_lambda, double alpha, int cutpoint);
#endif

    float get_inter_triangular_distance(PointXYZ &A1, PointXYZ &A2, PointXYZ &A3, PointXYZ &B1, PointXYZ &B2, PointXYZ &B3);

    void compute_analytic_part_jacobian_photometric(vector<mesh_map> meshes_full, EigenMatrix &J, Eigen::Matrix4f &transform, cv::Mat image, int num_forces);
    Eigen::Matrix4f inverse_transform_mesh_complete(vector<mesh_map> &meshes_full, Eigen::Matrix4f &transform);

    Eigen::Matrix4f inverse_transform_mesh(vector<mesh_map> &meshes_full, Eigen::Matrix4f &transform);

    vector<PointCloud<PointXYZRGB>> debugCloudList;
    vector<string> debugCloudLabel;
    vector<Eigen::Vector4f> residual_map;

    PointCloud<PointXYZ>::Ptr inverse_transform_mesh_single(mesh_map &mesh, Eigen::Matrix4f &transform);

    cv::Mat _prev_image;
    cv::Mat _curr_image;
    cv::Mat debug_img;
    cv::Mat prev_colored_depth;
    PointCloud<PointXYZRGB>::Ptr _prev_pointcloud;
    bool _is_photo_initialized;

    float initial_lambda;
    Eigen::Matrix<double, Dynamic, Dynamic, ColMajor> point_stack;
    Eigen::Matrix<double, Dynamic, Dynamic, ColMajor> normal_stack;
    double residual[BUFFER_SIZE];
    // std::vector<double[BUFFER_SIZE]> v_residuals; ak
    int normal_vector_size;

    vector<PointXYZ> corresponding_model_list_depth_first;

    vpHomogeneousMatrix oMc;
    vpHomogeneousMatrix init_cMo;  // debug variable
    offline_data_map data_map;

    int first_mesh_size;
    vector<std::vector<double>> correspondence_photo_first;
    vector<PointXYZ> correspondence_barycentric_photo_first;

    bool mesh_deform_regular;

#ifdef USE_CUBLAS
    CublasUtility cublas;
#endif

    float jacobian_displacement;
};

#endif
