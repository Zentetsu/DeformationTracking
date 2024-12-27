#ifndef OCCLUSIONCHECK_H
#define OCCLUSIONCHECK_H

#define BOOST_FILESYSTEM_NO_DEPRECATED
#define BOOST_FILESYSTEM_VERSION 3

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <algorithm>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem.hpp>
#include <boost/thread/thread.hpp>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <dirent.h>
#include "/opt/homebrew/opt/libomp/include/omp.h"
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <pcl/common/common.h>
#include <pcl/common/common_headers.h>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/features/board.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/io.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/pcl_base.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>
#include <visp3/core/vpConfig.h>
#include <visp3/core/vpIoTools.h>
#include <visp3/core/vpRobust.h>
#include <visp3/gui/vpDisplayD3D.h>
#include <visp3/gui/vpDisplayGDI.h>
#include <visp3/gui/vpDisplayGTK.h>
#include <visp3/gui/vpDisplayOpenCV.h>
#include <visp3/gui/vpDisplayX.h>
#include <visp3/io/vpImageIo.h>
#include <visp3/io/vpParseArgv.h>
#include <visp3/mbt/vpMbGenericTracker.h>
#include <vtkActor.h>
#include <vtkCallbackCommand.h>
#include <vtkCellArray.h>
#include <vtkCommand.h>
#include <vtkDataSetMapper.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkMath.h>
#include <vtkOBBTree.h>
#include <vtkPLYReader.h>
#include <vtkPointSource.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolygon.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkSliderRepresentation2D.h>
#include <vtkSliderWidget.h>
#include <vtkSmartPointer.h>
#include <vtkSphereSource.h>
#include <vtkUnstructuredGrid.h>
#include <vtkVersion.h>
#include <vtkWidgetEvent.h>
#include <vtkWidgetEventTranslator.h>
#include <vtkXMLUnstructuredGridReader.h>


namespace fs = ::boost::filesystem;
using namespace std::chrono;
using namespace pcl;
using namespace Eigen;
using namespace std;
using namespace std::chrono;

#define die(e) do { fprintf(stderr, "%s\n", e); exit(EXIT_FAILURE); } while (0);


#define IMAGE_WIDTH 640
#define IMAGE_HEIGHT 480

#define FRAME_RATE 60.0

#define FORCE_INCREMENT 2000.000

#define FORCE_GAIN_COMPUTATION

#define CAO_HEADER_LENGTH 4

//#define RIGID_TRACKING

#define MAX_DEPTH 0.7f

#define DEBUG_DUMP


#define FRAME_RATE 60.0
//#define FORCE_INCREMENT 1000.000
#define FORCE_GAIN_COMPUTATION

#define Z_BUFFER 1.0f
//#define ERROR_BAND_FRACTION 10.0f //for plank
#define ERROR_BAND_FRACTION 15.0f //for torus  //this is '20.0' for sequential OBJs
#define MIN_CLUSTER_SIZE 100
//#define CLUSTER_TOLERANCE 0.5f  //for plank
#define CLUSTER_TOLERANCE 0.1f  //for torus
#define CLUSTER_SIZE_CUTOFF 0.6

using namespace std::chrono;

struct point_2d {
    double x;
    double y;
};

struct mesh_map {
    std::vector<float> distance_array;
    std::vector<pcl::PointXYZ> mesh_points;
    std::vector<point_2d> projected_points;
    std::vector<int> visible_indices;
    pcl::PolygonMesh mesh;
};

struct force_data {
    Eigen::MatrixXf force_vector;
    Eigen::VectorXf nodes;
    int num_of_nodes;
};

class OcclusionCheck {

public:
    point_2d projection(pcl::PointXYZ& pt);
    point_2d projection_gl(pcl::PointXYZ pt, float F_x, float F_y, float C_x, float C_y);
    bool point_in_triangle(point_2d& pt, point_2d& v1, point_2d& v2, point_2d& v3);

    void eigen_to_visp(MatrixXd& E, vpMatrix& V);
    void visp_to_eigen(vpMatrix& V, MatrixXd& E);
    void eigen_to_visp_4x4(Matrix4f E, vpHomogeneousMatrix& V);
    void visp_to_eigen_4x4(vpHomogeneousMatrix V, Matrix4f& E);

    //mesh_map get_visibility(pcl::PolygonMesh mesh, float F_x, float F_y, float C_x, float C_y);
    mesh_map get_visibility_vtk(pcl::PolygonMesh mesh);

    void transformPolygonMesh(pcl::PolygonMesh& inMesh, Eigen::Matrix4f& transform);
    double** load_pose(char* path);
    void load_pcd(pcl::PointCloud<pcl::PointXYZ>& cloud, std::string path);

    force_data prep_force_data(vector<double> Fx, vector<double> Fy, vector<double> Fz, vector<int> nodes);

    vector<Eigen::Vector3d> nearest_neighbor(PointCloud<PointXYZ> cloud_, PointCloud<PointXYZ>::Ptr cluster, float dist);
    vector<Eigen::Vector3d> nearest_neighbor_generalized(PointCloud<PointXYZ> cloud_, PointCloud<PointXYZ>::Ptr cluster, float dist);
    MatrixXd nearest_neighbor_list(PointCloud<PointXYZ>& cloud_, PointCloud<PointXYZ>::Ptr cluster, float dist);

    void load_pcd_as_text_rgbd(pcl::PointCloud<pcl::PointXYZRGB>& cloud, std::string path, bool isStructured);

    PointCloud<PointXYZ>::Ptr load_vtk_mesh(string filename);
    void get_transform(const string& path, Matrix4f& pose);

    void set_camera_params(float F_x, float F_y, float C_x, float C_y, float d_1, float d_2, float d_3, float d_4);

    point_2d pt_P;
    std::vector<cv::Point3f> objectPointsP;
    cv::Mat rvecP;
    cv::Mat tvecP;
    cv::Mat AP;
    cv::Mat distCoeffsP;

    float F_xP, F_yP, C_xP, C_yP;


    std::vector<cv::Point2f> projectedPointsP;

    cv::Point3f pP;


};

#endif
