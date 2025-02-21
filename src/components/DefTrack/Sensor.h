#ifndef SENSOR_H
#define SENSOR_H


#include "OcclusionCheck.h"


class Sensor
{


public:
	Sensor();
#ifdef SENSOR_INPUT_ACTIVE
 	void initialize_d435();
	void acquire_d435(vpImage<vpRGBa> &color, PointCloud<PointXYZRGB>::Ptr &pointcloud);
    bool trackScene(vpColVector& robotObservation);
#else
    void init_visualizer(vpImage<vpRGBa> &_I);
#endif
#ifdef VISUALIZER
    #ifdef SENSOR_INPUT_ACTIVE
        void updateScreen(mesh_map &visible_mesh, vector<vpColVector> &err_map, residual_statistics &res_stats,vector<PointXYZ> matched_points, bool display_error_map, bool display_centroids);
    #else
        void updateScreen(mesh_map &visible_mesh, vector<vpColVector> &err_map, residual_statistics &res_stats, vector<PointXYZ> matched_points, vpCameraParameters &param, vpImage<vpRGBa> &I, bool display_error_map, bool display_centroids, bool debug_folder_active, string &debug_path, int debug_flag, int pcd_count);
    #endif

#endif



	vpHomogeneousMatrix get_extrinsics_depth2color();
    void set_extrinsics_depth2color(vpHomogeneousMatrix &extrinsicMat);
	vpCameraParameters get_intrinsics_color_d435();
	vpCameraParameters get_intrinsics_depth_d435();

    void extract_boundingbox(PointCloud<PointXYZ>::Ptr &visible_vertices);

    vpHomogeneousMatrix extrinsics;
   
        

protected:

#ifdef SENSOR_INPUT_ACTIVE
    vpRealSense2 _grabber; ///< Grabber to acquire images from an Intel RealSense camera
    rs2::config _config; ///< To configure the RealSense camera
    rs2::align*  _align; ///< To align the depth stream on the color stream
#endif

    std::vector<vpColVector> _vsharedMarkersPositions;
    vpDisplayX _display; ///< To display what the camera sees
    vpCameraParameters _param; ///< Intrisics parameters of the colour stream
    vpCameraParameters _paramDepth; ///< Intrisics parameters of the depth stream
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr _ppointCloud; ///< Stores the homogeneous coordinates of each pixel


    vpImage<vpRGBa> _I; ///< Color image from the sensor
    vpImage<unsigned char> _Igrey; ///< Grey-scale version of m_I needed for the tracking functions.

    float x_max, x_min, y_max, y_min, z_max, z_min;

    bool _is_d435_initialized;

#ifdef SENSOR_INPUT_ACTIVE
    int initialize_markers(bool is_robot_marker);
    bool keypointTracking();
    void convert_to_image(float x, float y, float z, vpImagePoint &ip);
#else
    void convert_to_image(float x, float y, float z, vpImagePoint &ip, vpCameraParameters &param);
#endif
    bool computePose(vpHomogeneousMatrix &cMq);




    vpHomogeneousMatrix _qMeff;



    PCLPointCloud2::Ptr filter_cloud;
    PCLPointCloud2::Ptr cloud_pass;


    std::vector<vpDot2*> _vpVisualFeatures; ///< The visual features which are observed on the object.
    std::vector<vpImagePoint> _vcog; ///< Vector which contains the CoG of the visual features to make possible to retrieve the pose from the markers.

    double _markerConfigurationSize; ///< The size of the markers configuration for display purpose.
    vpHomogeneousMatrix _cMlastMarkers; ///>Remember the last pose of the markers for the next estimation.
    std::vector<vpDot2*> _vpRobotMarkers; ///< The markers which are observed on the robot.
    std::vector<vpImagePoint> _vRobotCog; ///< Vector which contains the CoG of the robot markers to make possible to retrieve its pose.
    std::vector<vpPoint> _vMarkerRobotConfigModel; ///< Model of the configuration of the marker on the object containing the 3D positions of the markers

    vpHomogeneousMatrix _cMlastRobot; ///>Remember the last pose of the robot markers for the next estimation.
    bool _isFirstRobotMarkerPoseEstimation; ///< Indicates if it is the first pose estimation iteration for the markers.

    vpColVector pointHomogeneousCoord;
    vpColVector cMpoint;
   

	
};

#endif 
