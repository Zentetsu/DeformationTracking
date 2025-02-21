
#include "RigidTracking.h"

#ifdef _WIN32
#include <intrin.h>
#elif defined(__i386__) || defined(__x86_64__)
#include <immintrin.h>
#elif defined(__ARM_FEATURE_SIMD32) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include "OcclusionCheck.h"

RigidTracking::RigidTracking() : is_camera_initialized(false) {}

void RigidTracking::initialize_camera(vpCameraParameters &color_params, vpCameraParameters &depth_params, string tracker_init) {
    cam_color = color_params;
    cam_depth = depth_params;
    tracker_init_file = tracker_init;
    is_camera_initialized = true;
}

void RigidTracking::parse_pointcloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, std::vector<vpColVector> &pointcloud, vpImage<unsigned char> &image, vpImage<uint16_t> &depth, vpImage<vpRGBa> &color) {
    float depth_scaling = 1.0f;
    float depth_visualization_scaling = 1.0f;

    vpImage<vpRGBa> I_color_(480, 640);
    I_color_.resize(480, 640);
    image.resize(480, 640);
    color.resize(480, 640);
    depth.resize(480, 640);
    pointcloud.resize((480 * 640));

    for (int i = 0; i < 640; i++) {
        for (int j = 0; j < 480; j++) {
            pcl::PointXYZRGB p = cloud->at(i, j);
            I_color_[j][i].R = p.r;
            I_color_[j][i].G = p.g;
            I_color_[j][i].B = p.b;

            vpColVector pt3d(4, 1.0);
            pt3d[0] = p.x;
            pt3d[1] = p.y;
            pt3d[2] = p.z;
            pointcloud[(j * 640) + i] = pt3d;

            depth[j][i] = p.z * depth_scaling * depth_visualization_scaling;
        }
    }

    vpImageConvert::convert(I_color_, image);
    color = I_color_;
}

bool RigidTracking::read_data_batch(pcl::PointCloud<pcl::PointXYZRGB>::Ptr init_frame, vpImage<unsigned char> &image, vpImage<uint16_t> &depth, std::vector<vpColVector> &pointcloud, vpImage<vpRGBa> &color) {
    parse_pointcloud(init_frame, pointcloud, image, depth, color);

    return true;
}

#ifdef DEBUG
void debug_log_img(vpImage<unsigned char> I_, vpImage<unsigned char> I_depth_, int count) {
    std::string debug_dir(DEBUG_DIR);
    vpImageIo::write(I_, debug_dir + "images/I_" + std::to_string(count) + ".pgm");
    vpImageIo::write(I_depth_, debug_dir + "images/depth_" + std::to_string(count) + ".pgm");

    vpImage<vpRGBa> Ioverlay;
    vpDisplay::getImage(I_depth_, Ioverlay);
    std::string ofilename(debug_dir + "images/img_" + std::to_string(count) + ".png");
    vpImageIo::write(Ioverlay, ofilename);
}
#endif

void RigidTracking::track(pcl::PointCloud<pcl::PointXYZRGB>::Ptr frame, vpHomogeneousMatrix &cMo, vpMbGenericTracker &tracker, std::vector<vpColVector> &err_map, int count, bool should_track, bool is_sensor_active) {
    std::vector<vpColVector> pointcloud;
    read_data_batch(frame, I, I_depth_raw, pointcloud, I_color);

#ifdef RIGID_TRACKING
    vpImageConvert::createDepthHistogram(I_depth_raw, I_depth);
    std::map<std::string, const vpImage<unsigned char> *> mapOfImages;
    mapOfImages["Camera1"] = &I;
    std::map<std::string, pcl::PointCloud<pcl::PointXYZ>::ConstPtr> mapOfPointCloudsPCL;
    PointCloud<PointXYZ>::Ptr input(new pcl::PointCloud<PointXYZ>);
    copyPointCloud(*frame, *input);
    mapOfPointCloudsPCL["Camera2"] = input;

#ifdef SENSOR_INPUT_ACTIVE
#ifdef VISUALIZER
    vpDisplay::display(I);
#endif
#endif

#ifdef CONTINIOUS_RIGID_TRACKING_VISP
    tracker.track(mapOfImages, mapOfPointCloudsPCL);
    cMo = tracker.getPose();
#else
    if (count < VISP2CUSTOM_TRACKING_SWITCH) {
        tracker.track(mapOfImages, mapOfPointCloudsPCL);
        cMo = tracker.getPose();
    } /*else{
     _log_("third iterations reached")
     _log_(cMo)
     }*/
#endif

#ifdef SENSOR_INPUT_ACTIVE
#ifdef VISUALIZER
    tracker.display(I, cMo, cam_depth, vpColor::blue, 1);
    vpDisplay::flush(I);
#endif
#endif

#endif
}

vector<string> split(string s, string delimiter) {
    vector<string> list;
    size_t pos = 0;
    string token;
    while ((pos = s.find(delimiter)) != string::npos) {
        token = s.substr(0, pos);
        list.push_back(token);
        s.erase(0, pos + delimiter.length());
    }
    list.push_back(s);
    return list;
}

void copy_to_buffer(std::string &source_path, std::string &dest_path) {
    std::ifstream src(source_path, std::ios::binary);
    std::ofstream dst(dest_path, std::ios::binary);
    dst << src.rdbuf();
}

PointCloud<PointXYZ>::Ptr cloud_from_xyz_excess(string path, int length) {
    std::ifstream file(path);
    std::string line;
    int count = 0;
    int num_points = 0;

    vector<PointXYZ> vec_point;

    while (std::getline(file, line)) {
        if (count >= (CAO_HEADER_LENGTH + length - 1)) {
            vector<string> s = split(line, " ");
            PointXYZ p;
            p.x = stof(s[0]);
            p.y = stof(s[1]);
            p.z = stof(s[2]);
            vec_point.push_back(p);
            num_points++;
        }

        count++;
    }

    PointCloud<PointXYZ>::Ptr cloud(new PointCloud<PointXYZ>);
    cloud->width = num_points;
    cloud->height = 1;
    cloud->is_dense = false;

    if (vec_point.size() == num_points) {
        for (int i = 0; i < num_points; i++) {
            cloud->push_back(vec_point[i]);
        }
    } else {
        cout << "Mismatch between vec point list and cloud size in xyz file" << endl;
    }

    return cloud;
}

PointCloud<PointXYZ>::Ptr cloud_from_model(string path, int &length) {
    std::ifstream file(path);
    std::string line;
    int count = 0;
    int num_points = 0;

    vector<PointXYZ> vec_point;

    while (std::getline(file, line)) {
        if (count == 2)
            num_points = std::stoi(line);
        else if ((count > (CAO_HEADER_LENGTH - 1)) && (count < (CAO_HEADER_LENGTH + num_points))) {
            vector<string> s = split(line, " ");
            PointXYZ p;
            p.x = stof(s[0]);
            p.y = stof(s[1]);
            p.z = stof(s[2]);
            vec_point.push_back(p);
        }

        count++;
    }

    PointCloud<PointXYZ>::Ptr cloud(new PointCloud<PointXYZ>);
    cloud->width = num_points;
    cloud->height = 1;
    cloud->is_dense = false;

    if (vec_point.size() == num_points) {
        for (int i = 0; i < num_points; i++) {
            cloud->push_back(vec_point[i]);
        }
    } else {
        cout << "Mismatch between vec point list and cloud size" << endl;
    }

    length = num_points;
    return cloud;
}

int get_index(vector<Eigen::Vector3d> def_points, string line) {
    float thresh = 0.0001f;

    vector<string> s = split(line, " ");
    float x = stof(s[0]);

    int index = -1;

    for (int i = 0; i < def_points.size(); i++) {
        if ((fabs(def_points[i][0] - x) < thresh) && (fabs(def_points[i][1] - x) < thresh) && (fabs(def_points[i][2] - x) < thresh)) {
            if (index == -1)
                index = 1;
            else {
                cout << "multiple index for same point, quitting" << endl;
                exit(0);
            }
        }
    }

    return index;
}

void RigidTracking::reset_tracker(vpMbGenericTracker &tracker, pcl::PointCloud<pcl::PointXYZRGB>::Ptr frame, vpHomogeneousMatrix &cMo, bool should_reset) {
    if (should_reset) {
        vpImage<unsigned char> I, I_depth;
        vpImage<uint16_t> I_depth_raw;
        vpImage<vpRGBa> I_color;

        string new_model_file_path = "";  // replacement stub. calling this method will be lethal!

        std::vector<vpColVector> pointcloud;
        read_data_batch(frame, I, I_depth_raw, pointcloud, I_color);

        vpImageConvert::createDepthHistogram(I_depth_raw, I_depth);

        std::map<std::string, const vpImage<unsigned char> *> mapOfImages;
        mapOfImages["Camera1"] = &I;
        mapOfImages["Camera2"] = &I;

        std::map<std::string, std::string> mapOfModelFiles;
        mapOfModelFiles["Camera1"] = new_model_file_path;
        mapOfModelFiles["Camera2"] = new_model_file_path;

        std::map<std::string, vpHomogeneousMatrix> mapOfCameraPoses;
        mapOfCameraPoses["Camera1"] = cMo;
        mapOfCameraPoses["Camera2"] = cMo;

        tracker.reInitModel(mapOfImages, mapOfModelFiles, mapOfCameraPoses);

        tracker.loadModel(new_model_file_path, new_model_file_path);
        vpHomogeneousMatrix T;

        T[0][0] = 1.0; /**NO TRANSFORMATION BETWEEN TWO CAMERAS**/
        T[0][3] = 0.0;
        T[1][1] = 1;
        T[1][2] = 0;
        T[1][3] = 0.0;
        T[2][1] = 0;
        T[2][2] = 1;
        T[2][3] = 0.0;

        tracker.loadModel(new_model_file_path, false, T);
    }
}

void RigidTracking::initialize(std::string config_file_path, std::string cao_model_path, pcl::PointCloud<pcl::PointXYZRGB>::Ptr init_frame, vpHomogeneousMatrix &cMo, vpMbGenericTracker &tracker, bool is_sensor_active) {
    try {
        if (is_sensor_active && !is_camera_initialized) {
            cerr << "[RigidTracking::initialize]: ERROR: Either camera is not initialized or the sensor is inactive. Please check. Exiting for now." << endl;
            exit(0);
        }

        if (is_sensor_active) {
            _log_("Setting camera parameters for sensor") tracker.setCameraParameters(cam_color, cam_depth);
        } else {
            cout << "[RigidTracking::initialize]: Loading configuration file (" << config_file_path << ")" << endl;
            tracker.loadConfigFile(config_file_path, config_file_path);
        }

        bool useScanline = false;

        /// vpKltOpencv klt;
        /// tracker.setKltMaskBorder(5);
        /// klt.setMaxFeatures(10000);
        /// klt.setWindowSize(5);
        /// klt.setQuality(0.01);
        /// klt.setMinDistance(5);
        /// klt.setHarrisFreeParameter(0.02);
        /// klt.setBlockSize(3);
        /// klt.setPyramidLevels(3);
        ///
        /// tracker.setKltOpencv(klt);
        ///
        /// tracker.setDepthNormalFeatureEstimationMethod(vpMbtFaceDepthNormal::ROBUST_FEATURE_ESTIMATION);
        /// tracker.setDepthNormalPclPlaneEstimationMethod(2);
        /// tracker.setDepthNormalPclPlaneEstimationRansacMaxIter(200);
        /// tracker.setDepthNormalPclPlaneEstimationRansacThreshold(0.1);
        /// tracker.setDepthNormalSamplingStep(2, 2);

        tracker.setDepthDenseSamplingStep(1, 1);

        tracker.setAngleAppear(vpMath::rad(85.0));
        tracker.setAngleDisappear(vpMath::rad(89.0));
        tracker.setNearClippingDistance(0.001);
        tracker.setFarClippingDistance(200.0);
        tracker.setClipping(tracker.getClipping() | vpMbtPolygon::FOV_CLIPPING);

        cout << "[RigidTracking::initialize]: Loading model file (" << cao_model_path << ")" << endl;

        tracker.loadModel(cao_model_path, cao_model_path);

        vpHomogeneousMatrix T;

        if (!is_sensor_active) {
            tracker.getCameraParameters(cam_color, cam_depth);
        }

        tracker.setDisplayFeatures(true);
        tracker.setScanLineVisibilityTest(useScanline);

        std::map<int, std::pair<double, double> > map_thresh;

        _log_("loaded color files")

#ifdef VISP_HAVE_COIN3D
            map_thresh[vpMbGenericTracker::EDGE_TRACKER] = useScanline ? std::pair<double, double>(0.005, 3.9) : std::pair<double, double>(0.007, 2.9);
#if defined(VISP_HAVE_MODULE_KLT) && (defined(VISP_HAVE_OPENCV) && (VISP_HAVE_OPENCV_VERSION >= 0x020100))
        map_thresh[vpMbGenericTracker::KLT_TRACKER] = useScanline ? std::pair<double, double>(0.006, 1.9) : std::pair<double, double>(0.005, 1.3);
        map_thresh[vpMbGenericTracker::EDGE_TRACKER | vpMbGenericTracker::KLT_TRACKER] = useScanline ? std::pair<double, double>(0.005, 3.2) : std::pair<double, double>(0.006, 2.8);
#endif
        map_thresh[vpMbGenericTracker::EDGE_TRACKER | vpMbGenericTracker::DEPTH_DENSE_TRACKER] = useScanline ? std::pair<double, double>(0.003, 1.7) : std::pair<double, double>(0.002, 0.8);
#if defined(VISP_HAVE_MODULE_KLT) && (defined(VISP_HAVE_OPENCV) && (VISP_HAVE_OPENCV_VERSION >= 0x020100))
        map_thresh[vpMbGenericTracker::KLT_TRACKER | vpMbGenericTracker::DEPTH_DENSE_TRACKER] = std::pair<double, double>(0.002, 0.3);
        map_thresh[vpMbGenericTracker::EDGE_TRACKER | vpMbGenericTracker::KLT_TRACKER | vpMbGenericTracker::DEPTH_DENSE_TRACKER] = useScanline ? std::pair<double, double>(0.002, 1.8) : std::pair<double, double>(0.002, 0.7);
#endif
#else
            map_thresh[vpMbGenericTracker::EDGE_TRACKER] = useScanline ? std::pair<double, double>(0.007, 2.3) : std::pair<double, double>(0.007, 2.1);
#if defined(VISP_HAVE_MODULE_KLT) && (defined(VISP_HAVE_OPENCV) && (VISP_HAtrackerVE_OPENCV_VERSION >= 0x020100))
        map_thresh[vpMbGenericTracker::KLT_TRACKER] = useScanline ? std::pair<double, double>(0.006, 1.7) : std::pair<double, double>(0.005, 1.4);
        map_thresh[vpMbGenericTracker::EDGE_TRACKER | vpMbGenericTracker::KLT_TRACKER] = useScanline ? std::pair<double, double>(0.004, 1.2) : std::pair<double, double>(0.004, 1.0);
#endif
        map_thresh[vpMbGenericTracker::EDGE_TRACKER | vpMbGenericTracker::DEPTH_DENSE_TRACKER] = useScanline ? std::pair<double, double>(0.002, 0.7) : std::pair<double, double>(0.001, 0.4);
#if defined(VISP_HAVE_MODULE_KLT) && (defined(VISP_HAVE_OPENCV) && (VISP_HAVE_OPENCV_VERSION >= 0x020100))
        map_thresh[vpMbGenericTracker::KLT_TRACKER | vpMbGenericTracker::DEPTH_DENSE_TRACKER] = std::pair<double, double>(0.002, 0.3);
        map_thresh[vpMbGenericTracker::EDGE_TRACKER | vpMbGenericTracker::KLT_TRACKER | vpMbGenericTracker::DEPTH_DENSE_TRACKER] = useScanline ? std::pair<double, double>(0.001, 0.5) : std::pair<double, double>(0.001, 0.4);
#endif
#endif

        std::vector<vpColVector> pointcloud;

        read_data_batch(init_frame, I, I_depth_raw, pointcloud, I_color);

        display.init(I, 641, 0, "Model-based tracker");

        /// if(is_sensor_active)
        ///{
        ///     tracker.initClick(I, tracker_init_file, true);
        /// }
        /// else
        ///{
        ///     cout<<"[RigidTracking::initialize]: Initializing the tracker using the transformation matrix: "<<endl<<cMo<<endl;
        ///     tracker.initFromPose(I, cMo);
        ///     tracker.setPose(I,I_depth,cMo,cMo);
        /// }

        tracker.initClick(I, tracker_init_file, true);

        vpImageConvert::createDepthHistogram(I_depth_raw, I_depth);

        tracker.setCameraTransformationMatrix("Camera2", T);
        cMo = tracker.getPose();

        /// if(is_sensor_active)
        ///{
        ///     vpDisplay::flush(I);
        /// }

        _log_("done loading")

            vpDisplay::flush(I);

    } catch (const vpException &e) {
        std::cout << "Caught an exception: " << e << std::endl;
        std::cout << "Tracker initialization failed!" << std::endl;
    }
}
