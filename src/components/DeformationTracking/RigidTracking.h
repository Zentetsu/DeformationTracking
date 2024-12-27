#ifndef RIGIDTRACKING_H
#define RIGIDTRACKING_H

#include "OcclusionCheck.h"


#define GETOPTARGS "i:dclt:e:Dmh"


class RigidTracking {

public:

    void initialize(std::string config_file_path, std::string cao_model_path, pcl::PointCloud<pcl::PointXYZRGB>::Ptr init_frame, vpHomogeneousMatrix cMo_truth, vpMbGenericTracker& tracker);
    double track(pcl::PointCloud<pcl::PointXYZRGB>::Ptr frame, vpHomogeneousMatrix& cMo, vpMbGenericTracker& tracker, std::vector<vpColVector>& err_map, int count, bool should_track);
    void reset_tracker(vpMbGenericTracker& tracker, pcl::PointCloud<pcl::PointXYZRGB>::Ptr frame, vpHomogeneousMatrix& cMo, bool should_reset);


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



    /*private:
        bool read_data_batch(pcl::PointCloud<pcl::PointXYZRGB>::Ptr init_frame, vpImage<unsigned char> &I, vpImage<uint16_t> &I_depth, std::vector<vpColVector> &pointcloud);
        bool read_data(const std::string &input_directory, const int cpt, const vpCameraParameters &cam_depth,
                       vpImage<unsigned char> &I, vpImage<uint16_t> &I_depth,
                       std::vector<vpColVector> &pointcloud, vpHomogeneousMatrix &cMo);
        void parse_pointcloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, std::vector<vpColVector> &pointcloud,
                              vpImage<unsigned char> &I, vpImage<uint16_t> &I_depth, const vpCameraParameters &cam_depth);
    */

};

#endif
