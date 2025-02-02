#ifndef JACOBIANFEM_H
#define JACOBIANFEM_H

#include "OcclusionCheck.h"

class JacobianFEM {
   public:
    pcl::PointXYZ normal_estimation(pcl::PointXYZ& a, pcl::PointXYZ& b, pcl::PointXYZ& c, pcl::PointXYZ& centroid);
    double compute_update(PointCloud<PointXYZ>::Ptr& pointcloud, PolygonMesh base_model, vector<PolygonMesh>& deformed_models, Eigen::Matrix4f& transform, Eigen::MatrixXd& update);
    void set_iteration(int iter);
    void set_pcd_count(int count);
    void set_file_num(int _file_num);
    void set_op_file_path(string op_file_path);
    void set_camera_parameters(float Fx, float Fy, float Cx, float Cy);
    int iteration;
    int pcd_count;
    int file_num;
    float F_x;
    float F_y;
    float C_x;
    float C_y;
};

#endif
