#ifndef OCCLUSIONCHECK_H
#define OCCLUSIONCHECK_H

#define BOOST_FILESYSTEM_NO_DEPRECATED
#define BOOST_FILESYSTEM_VERSION 3
#include <dirent.h>
#include <math.h>
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

#include <Eigen/Dense>
#include <algorithm>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem.hpp>
#include <boost/thread/thread.hpp>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
/// #include <pcl/filters/uniform_sampling.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
// #include <pcl/io/io.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/pcl_base.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/surface/vtk_smoothing/vtk_utils.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <visp3/core/vpCameraParameters.h>
#include <visp3/core/vpConfig.h>
#include <visp3/core/vpFeatureDisplay.h>
#include <visp3/core/vpIoTools.h>
#include <visp3/core/vpPoint.h>
#include <visp3/core/vpRobust.h>
#include <visp3/gui/vpDisplayD3D.h>
#include <visp3/gui/vpDisplayGDI.h>
#include <visp3/gui/vpDisplayGTK.h>
#include <visp3/gui/vpDisplayOpenCV.h>
#include <visp3/gui/vpDisplayX.h>
#include <visp3/io/vpImageIo.h>
#include <visp3/io/vpParseArgv.h>
#include <visp3/mbt/vpMbGenericTracker.h>
#include <visp3/sensor/vpRealSense.h>
#include <visp3/sensor/vpRealSense2.h>

#include <string>  //ak
#include <vector>
// For pose estimation
#include <visp3/blob/vpDot2.h>
#include <visp3/core/vpImageConvert.h>
#include <visp3/core/vpPixelMeterConversion.h>
#include <visp3/vision/vpPose.h>
#include <vtkActor.h>
#include <vtkCallbackCommand.h>
#include <vtkCellArray.h>
#include <vtkCommand.h>
#include <vtkDataSetMapper.h>
#include <vtkGenericCell.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkMath.h>
#include <vtkModifiedBSPTree.h>
#include <vtkOBBTree.h>
#include <vtkPLYReader.h>
#include <vtkPointSource.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolygon.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSliderRepresentation2D.h>
#include <vtkSliderWidget.h>
#include <vtkSmartPointer.h>
#include <vtkSphereSource.h>
#include <vtkUnstructuredGrid.h>
#include <vtkVersion.h>
#include <vtkWidgetEvent.h>
#include <vtkWidgetEventTranslator.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkXMLUnstructuredGridReader.h>

#define ALLOW_LOGS

#define EUCLEDIAN_NORM(x, y, z) sqrtf((float)((x) * (x) + (y) * (y) + (z) * (z)))
#define EUCLEDIAN_DIST(x1, y1, z1, x2, y2, z2) sqrtf(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)) + ((z2 - z1) * (z2 - z1)))
#define EUCLEDIAN_DIST_2D(x1, y1, x2, y2) sqrtf(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

namespace fs = ::boost::filesystem;
using namespace std::chrono;
using namespace pcl;
using namespace Eigen;
using namespace std;
using namespace std::chrono;

typedef Matrix<double, Dynamic, Dynamic, RowMajor> EigenMatrix;

#define die(e)                      \
    do {                            \
        fprintf(stderr, "%s\n", e); \
        exit(EXIT_FAILURE);         \
    } while (0);

#define RESET "\033[0m"
#define BLACK "\033[30m"              /* Black */
#define RED "\033[31m"                /* Red */
#define GREEN "\033[32m"              /* Green */
#define YELLOW "\033[33m"             /* Yellow */
#define BLUE "\033[34m"               /* Blue */
#define MAGENTA "\033[35m"            /* Magenta */
#define CYAN "\033[36m"               /* Cyan */
#define WHITE "\033[37m"              /* White */
#define BOLDBLACK "\033[1m\033[30m"   /* Bold Black */
#define BOLDRED "\033[1m\033[31m"     /* Bold Red */
#define BOLDGREEN "\033[1m\033[32m"   /* Bold Green */
#define BOLDYELLOW "\033[1m\033[33m"  /* Bold Yellow */
#define BOLDBLUE "\033[1m\033[34m"    /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m" /* Bold Magenta */
#define BOLDCYAN "\033[1m\033[36m"    /* Bold Cyan */
#define BOLDWHITE "\033[1m\033[37m"   /* Bold White */

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#ifdef ALLOW_LOGS
#define _log_(m) cout << "[" << __FILENAME__ << ":" << __func__ << ":" << __LINE__ << "] " << m << endl << flush;
#define _warn_(m) cout << "[" << __FILENAME__ << ":" << __func__ << ":" << __LINE__ << "] " << BOLDWHITE << m << RESET << endl << flush;
#define _info_(m) cout << "[" << __FILENAME__ << ":" << __func__ << ":" << __LINE__ << "] " << BOLDGREEN << m << RESET << endl << flush;
#define _error_(m) cout << "[" << __FILENAME__ << ":" << __func__ << ":" << __LINE__ << "] " << BOLDRED << "ERROR: " << BOLDMAGENTA << m << RESET << endl << flush;
#else
#define _log_(m) ;
#define _warn_(m) ;
#define _info_(m) cout << "[" << __FILENAME__ << ":" << __func__ << ":" << __LINE__ << "] " << BOLDGREEN << m << RESET << endl << flush;
#define _error_(m) ;
#endif
#define _fatal_(m)                                                                                                                           \
    cout << "[" << __FILENAME__ << ":" << __func__ << ":" << __LINE__ << "] " << BOLDRED << "FATAL: " << RED << m << RESET << endl << flush; \
    exit(0);
#define _hline_ cout << BOLDMAGENTA << "***************************************" << endl << flush;

#define IMAGE_WIDTH 640
#define IMAGE_HEIGHT 480
#define FRAME_RATE 60.0
#define FORCE_INCREMENT 2000.000
#define FORCE_GAIN_COMPUTATION
#define CAO_HEADER_LENGTH 4
// #define RIGID_TRACKING
// #define CONTINIOUS_RIGID_TRACKING_VISP
#define DEBUG_DUMP
// #define CONTINIOUS_RIGID_TRACKING
// #define RIGID_SPARSE_TRACKING
#define VISP2CUSTOM_TRACKING_SWITCH 2

#define FLOAT_ZERO 0.00000000001f
#define VERY_SMALL_DISTANCE 0.001f

// #define ARAP_ON_DEPTH

// #define RAYTRACING_SIZE_THRESHOLD 0.01f   //for larger, simualated scenes
// #define RAYTRACING_SIZE_THRESHOLD 0.014f  //for smaller, real-sized data
//^the above commented thresholds can be removed

#define FRAME_RATE 60.0
// #define FORCE_INCREMENT 1000.000
#define FORCE_GAIN_COMPUTATION

#define CLUSTERING_INITIALIZATION_POINT 1

#define CLUSTERING_DISTANCE_THRESHOLD 70.0f
#define CLUSTERING_HOLDBACK_CUTOFF 4
#define ELEMENTARY_CLUSTER_COUNT 1
#define CLUSTERING_KERNEL_SIZE 7
#define CLUSTERING_MIN 2.0f
#define CLUSTERING_MAX 150.0f

#define Z_BUFFER 1.0f
// #define ERROR_BAND_FRACTION 10.0f //for plank
#define ERROR_BAND_FRACTION 15.0f  // for torus  //this is '20.0' for sequential OBJs

// #define DEPTH_FORMAT_RAW
#define DEPTH_FORMAT_CSV
// #define DEPTH_FORMAT_PNG

#define _DEPTH 0
#define _PHOTO 1

#define DEPTH_JACOBIAN_BINARIZED

#define EXTREMELY_DISTANT_POINTS 1000.0f

// #define COMMUNICATION
// #define SENSOR_INPUT_ACTIVE
#define VISUALIZER

// #define USE_CUBLAS
// #define USE_OPENMP

// #define JACOBIAN_FILTER

// #define JACOBIAN_AXIS_6

#define PHOTOMETRIC_MINIMIZATION
// #define PURELY_PHOTOMETRIC

// #define SPARSE_MINIMIZATION
// #define PURELY_SPARSE
// #define L2_NORM_SPARSE
#define SSD_SPARSE  // --> only this option actually works!

#define ENABLE_AKAZE
// #define ENABLE_ORB

#define AKAZE_DISTANCE_THRESHOLD 10.0f
#define ORB_QUALITY_THRESHOLD 0.2f
#define ORB_MAX_FEATURES 5000

#define INTERPOLATE_SPARSE_DEPTH

#define BARYCENTRIC_IN_IMAGE_PLANE

// #define NULLIFY_MOTION_MATRIX

#define DEPTH_JACOBIAN_ANALYTIC
#define PHOTO_JACOBIAN_ANALYTIC

// #define REORIENT_FORCE_VECTOR

// #define BLUR_IMAGE

#ifndef PHOTOMETRIC_MINIMIZATION
#define FATALIZE_PHOTOMETRIC
#endif

#ifndef SPARSE_MINIMIZATION
#define FATALIZE_SPARSE
#endif

//////***********Important Thresholds***********////////
#define RAYTRACING_SIZE_THRESHOLD 0.0001f  // universal size
#define MIN_CLUSTER_SIZE 100
// #define CLUSTER_TOLERANCE 0.5f  //for plank
#define CLUSTER_TOLERANCE 0.4f  // for torus
#define CLUSTER_SIZE_CUTOFF 0.4
#define MAX_DEPTH 40.0f
// #define TRIANGLE_AREA_THRESHOLD  180.0f --> for sponge
#define TRIANGLE_AREA_THRESHOLD 3.0f
#define BOUNDING_BOX_PADDING 0.008f
#define POLYGON_DISPLACEMENT_THRESHOLD 0.001f
#define MAX_DIMENSION_CHECK 100.0f
#define MAX_RIGID_TRACKING_ITERATION 7
//////***********Important Thresholds***********////////

// For time improvements using Cublas in JacobianFEM::get_residual
#define BUFFER_SIZE 30000
#define NORMAL_SIZE 250
#define NB_MP_THREADS 6

#define OPENCV_WINDOW_1 "Sparse Correspondence"
#define OPENCV_WINDOW_2 "Sparse Optical Flow"
#define OPENCV_WINDOW_3 "XXXXXXX"

#define STATUS_0 "ready"
#define STATUS_1 "jacobian_ready"
#define STATUS_2 "update_ready"
#define STATUS_3 "applying_J"
#define STATUS_4 "applying_update"

using namespace std::chrono;
using namespace cv;

struct point_2d {
    double x;
    double y;
};

struct triangle_3D {
    PointXYZ P1;
    PointXYZ P2;
    PointXYZ P3;
    int index1;
    int index2;
    int index3;
};

/*5d points contains the map (X, Y, Z) |-> (u,v);
 * i.e., mapping between 3D cartesian coordinates and the corresponding 2D point on the image plane (obtained via perspective projection)
   X, Y, Z : 3D coordinates; u,v : image coordinates*/
struct point_5d {
    double u;
    double v;
    double X;
    double Y;
    double Z;
};

struct roi_2d {
    double min_x;
    double min_y;
    double max_x;
    double max_y;
};

struct mesh_map {
    std::vector<float> distance_array;
    std::vector<pcl::PointXYZ> mesh_points;
    std::vector<point_2d> projected_points;
    std::vector<point_5d> projected_points_5d;
    roi_2d roi;
    std::vector<int> visible_indices;
    pcl::PolygonMesh mesh;
    PointCloud<PointXYZ>::Ptr visible_vertices;
    std::vector<Eigen::Vector3f> visible_normal;
    std::vector<Eigen::Vector3f> visible_normal_unnormalized;
};

/*  ---> this was used in a previous build
struct triangle
{
  PointXYZ A;
  PointXYZ B;
  PointXYZ C;
  int index_A;
  int index_B;
  int index_C;
};*/

struct less_than_point_5d {
    inline bool operator()(const point_5d &p1, const point_5d &p2) { return ((p1.u == p2.u) ? (p1.v < p2.v) : (p1.u < p2.u)); }
};

struct sparse_map {
    vector<KeyPoint> keypoints1;
    vector<KeyPoint> keypoints2;
    vector<KeyPoint> keypoints2_rigid;
    vector<PointXYZ> keypoints3D_1;
    vector<PointXYZ> keypoints3D_2;
    vector<PointXYZ> barycentric_1;
    vector<PointXYZ> barycentric_2;  // reserved --> would possibly remain unused
    vector<triangle_3D> triangle_container_1;
    vector<triangle_3D> triangle_container_2;  // reserved --> would possibly remain unused
    vector<bool> valid_points;
};

struct force_data {
    Eigen::MatrixXf force_vector;
    Eigen::VectorXf nodes;
    int num_of_nodes;
};

struct residual_statistics {
    float minimum;
    float maximum;
    float average;
    float unweighted_average;
};

struct offline_data_map {
    vector<int> color_files;
    vector<int> depth_files;
    float color_Cx;
    float color_Cy;
    float color_Fx;
    float color_Fy;
    float depth_Cx;
    float depth_Cy;
    float depth_Fx;
    float depth_Fy;
    Eigen::Matrix4f depth2color_extrinsics;
};

class OcclusionCheck {
   public:
    OcclusionCheck();

    point_2d projection(const PointXYZ &pt);

    point_2d projection_raw(const pcl::PointXYZ &pt);
    point_2d projection_raw(PointXYZ &pt, float Fx, float Fy, float Cx, float Cy);
    point_2d projection_gl(pcl::PointXYZ pt, float F_x, float F_y, float C_x, float C_y);
    void inverse_projection_raw(int x, int y, double depth, float Fx, float Fy, float Cx, float Cy, PointXYZ &pt);
    void inverse_projection_floating(double x, double y, double depth, float Fx, float Fy, float Cx, float Cy, PointXYZRGB &pt);
    bool point_in_triangle(point_2d &pt, point_2d &v1, point_2d &v2, point_2d &v3);
    bool point_in_triangle(Point2f &pt, point_2d &v1, point_2d &v2, point_2d &v3);
    float bilinearInterpolationAtValues(float x, float y, float A, float B, float C, float D);

    void assign_weights_to_matrix_visp(EigenMatrix &residual, vpColVector &W, float threshold);

    PointXYZ transform_point_XYZ(PointXYZ &p, Eigen::Matrix4f &transform);

    void read_calibration_file(EigenMatrix &I, string path, int vertex_size);
    void cvMat_to_eigenMatrix_row(cv::Mat &data, EigenMatrix &output, int datatype);
    void eigenMatrix_to_cvMat_row(EigenMatrix &data, cv::Mat &output, int datatype);

    void load_image_and_register(string color_image_path, PointCloud<PointXYZRGB> &cloud, offline_data_map &data_map, cv::Mat &color_image, cv::Mat &colored_depth_image, int count);

    cv::Mat read_raw_depth_data(string path);
    cv::Mat read_csv_depth_data(string path);
    cv::Mat read_png_depth_data(string path);

    void eigen_to_visp(EigenMatrix &E, vpMatrix &V);
    void eigen_to_visp_3col(EigenMatrix &E, vpMatrix &V, int start_col_index);
    void visp_to_eigen(const vpMatrix &V, EigenMatrix &E);
    void eigen_to_visp_weighted(EigenMatrix &E, vpMatrix &V, vpColVector &W);
    void eigen_to_visp_4x4(const Matrix4f E, vpHomogeneousMatrix &V);
    void visp_to_eigen_4x4(const vpHomogeneousMatrix V, Matrix4f &E);
    void eigen_to_inv_eigen_4x4(Matrix4f E, Matrix4f &iE);
    void depth_to_pcl(cv::Mat &depth_image, PointCloud<PointXYZRGB> &cloud, offline_data_map &data_map);

    mesh_map get_visibility_normal(pcl::PolygonMesh &mesh);
    float getTriangleArea(point_2d &A, point_2d &B, point_2d &C);

    int count_files_in_folder(string path);

    offline_data_map check_offline_data(string color_directory, string depth_directory, int offset);

    PointCloud<PointXYZRGB>::Ptr transformPointcloudXYZRGB(PointCloud<PointXYZRGB>::Ptr &cloud, Mat &depth_image, Eigen::Matrix4f &transform, float fx, float fy, float cx, float cy);
    void transformPolygonMesh(pcl::PolygonMesh &inMesh, const Matrix4f &transform);
    double **load_pose(char *path);
    void load_pcd(pcl::PointCloud<pcl::PointXYZ> &cloud, std::string path);
    void load_pcd_rgb(pcl::PointCloud<pcl::PointXYZRGB> &cloud, std::string path);
    void extract_rotation(const Eigen::Matrix4f &transform, Eigen::Matrix3f &rotation);

    force_data prep_force_data(vector<double> Fx, vector<double> Fy, vector<double> Fz, vector<int> nodes);

    PointXYZ transform_point(PointXYZRGB &p, Eigen::Matrix4f &transform);

    Eigen::MatrixXd _A;
    Eigen::MatrixXd _b;
    Eigen::MatrixXd _cartesian;
    PointXYZ _v2_1;
    PointXYZ _v2_3;
    PointXYZ _v2_P;

    void compute_barycentric_coordinates(PointXYZ a, PointXYZ b, PointXYZ c, PointXYZ P, PointXYZ &P_res);
    void compute_inverse_barycentric_coordinates(PointXYZ a, PointXYZ b, PointXYZ c, PointXYZ &P, PointXYZ &P_res);

    PointCloud<PointXYZ>::Ptr load_vtk_mesh(string filename);
    void get_transform(const string &path, Matrix4f &pose);

    void set_camera_params(float F_x, float F_y, float C_x, float C_y, float d_1, float d_2, float d_3, float d_4);
    void set_color_camera_params(float F_x, float F_y, float C_x, float C_y, float d_1, float d_2, float d_3, float d_4);

    point_2d pt_P;

    cv::Mat rvecP;
    cv::Mat tvecP;

    float F_xP, F_yP, C_xP, C_yP;

    PointXYZ _product;
    void cross_product(const PointXYZ &a, const PointXYZ &b, PointXYZ &product);
    double dot_product_raw(pcl::PointXYZ &a, pcl::PointXYZ &b);
    PointXYZ cross_product_raw(const PointXYZ &a, const PointXYZ &b);

    vpCameraParameters _camParam;
    vpCameraParameters _colorCamParam;

    ofstream testfile;

   protected:
    Eigen::Vector3f _vec12;
    Eigen::Vector3f _vec23;
    Eigen::Vector3f _vecNorm;
    Eigen::Vector3f _vecUnNorm;
    Eigen::Vector3f _vec_centroid;

    EigenMatrix _transform_point_P;
    EigenMatrix _transform_point_T;
    // PointXYZ _transform_point_p;
};

#endif
