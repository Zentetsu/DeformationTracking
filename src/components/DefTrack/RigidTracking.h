#ifndef RIGIDTRACKING_H
#define RIGIDTRACKING_H

#include "OcclusionCheck.h"

#define GETOPTARGS "i:dclt:e:Dmh"

class RigidTracking {
   public:
    RigidTracking();

    void initialize(std::string config_file_path, std::string cao_model_path, pcl::PointCloud<pcl::PointXYZRGB>::Ptr init_frame, vpHomogeneousMatrix &cMo, vpMbGenericTracker &tracker, bool is_sensor_active);
    void track(pcl::PointCloud<pcl::PointXYZRGB>::Ptr frame, vpHomogeneousMatrix &cMo, vpMbGenericTracker &tracker, std::vector<vpColVector> &err_map, int count, bool should_track, bool is_sensor_active);
    void reset_tracker(vpMbGenericTracker &tracker, pcl::PointCloud<pcl::PointXYZRGB>::Ptr frame, vpHomogeneousMatrix &cMo, bool should_reset);
    void initialize_camera(vpCameraParameters &color_params, vpCameraParameters &depth_params, string tracker_init);
    bool read_data_batch(pcl::PointCloud<pcl::PointXYZRGB>::Ptr init_frame, vpImage<unsigned char> &image, vpImage<uint16_t> &depth, std::vector<vpColVector> &pointcloud, vpImage<vpRGBa> &color);

#if defined VISP_HAVE_X11
    vpDisplayX display1, display2, display3;
#elif defined VISP_HAVE_GDI
    vpDisplayGDI display1, display2, display3;
#elif defined VISP_HAVE_OPENCV
    vpDisplayOpenCV display1, display2, display3;
#elif defined VISP_HAVE_D3D9
    vpDisplayD3D display1, display2, display3;
#elif defined VISP_HAVE_GTK
    vpDisplayGTK display1, display2, display3;
#endif

    vpImage<unsigned char> I, I_depth;
    vpImage<uint16_t> I_depth_raw;
    vpImage<vpRGBa> I_color;

    vpCameraParameters cam_color, cam_depth;

    bool is_camera_initialized;

    string tracker_init_file;

    vpDisplayOpenCV display;

   private:
    void parse_pointcloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, std::vector<vpColVector> &pointcloud, vpImage<unsigned char> &I_, vpImage<uint16_t> &I_depth_, vpImage<vpRGBa> &I_color);
};

#endif
