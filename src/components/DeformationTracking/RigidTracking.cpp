
#include "RigidTracking.h"

#ifdef _WIN32
#include <intrin.h>
#elif defined(__i386__) || defined(__x86_64__)
#include <immintrin.h>
#elif defined(__ARM_FEATURE_SIMD32) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include "OcclusionCheck.h"


namespace {

    void parse_pointcloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, std::vector<vpColVector>& pointcloud, vpImage<unsigned char>& I_, vpImage<uint16_t>& I_depth_, vpImage<vpRGBa>& I_color) {
        float depth_scaling = 1.0f;//Tres Important: change value of this scaling term based on dataset!
        //for plank dataset, it is: p.z*100.0;   ...for all other dataset(?), it is: p.z*1000.0, for torus-simple Blender dataset, it is 1.0f (but torus does not work anyways!)
        float depth_visualization_scaling = 1000.0f; //this is just the scaling required for visualizing the depth data, not actually pusing back the pointcloud

        vpImage<vpRGBa> I_color_(480, 640);
        I_color_.resize(480, 640);
        I_.resize(480, 640);
        I_color.resize(480, 640);
        I_depth_.resize(480, 640);
        pointcloud.resize((480 * 640) + 1);
        for (int i = 0; i < 640; i++) {
            for (int j = 0; j < 480; j++) {
                pcl::PointXYZRGB p = cloud->at(i, j);
                I_color_[j][i].R = p.r;
                I_color_[j][i].G = p.g;
                I_color_[j][i].B = p.b;

                double x = 0.0f;
                double y = 0.0f;

                vpColVector pt3d(4, 1.0);
                pt3d[0] = p.x;
                pt3d[1] = p.y;
                pt3d[2] = p.z;
                pointcloud[(j * 640) + i] = pt3d;


                /**TEST CODE for PROJECTION VERIFICATION*/
                /*Uncomment this code to test if the depth_scaling is correct or not*/
                /*If correct -> depth image and color image would be the same. Would be dissimilar if depth_scaling incorrect*/
                //  int x_ = round(((p.x/(p.z*depth_scaling))*F_x) + C_x);
                //  int y_ = round(((p.y/(p.z*depth_scaling))*F_y) + C_y);
                //  if((y_ > 0) && (y_ < 480) && (x_ > 0) && (x_ < 640))
                //    {
                //      I_depth_[y_][x_] = p.z*depth_scaling;
                //      I_color_[y_][x_].R = p.r;
                //      I_color_[y_][x_].G = p.g;
                //      I_color_[y_][x_].B = p.b;
                //    }
                /*Always comment out the next line of code (i.e, I_depth_[j][i] = p.z*depth_scaling;) while doing this test*/
                /**TEST CODE for PROJECTION VERIFICATION*/

                I_depth_[j][i] = p.z * depth_scaling * depth_visualization_scaling; ///THIS iS the REAL CODE. Uncomment, except when testing the depth scale!

            }
        }

        vpImageConvert::convert(I_color_, I_);
        I_color = I_color_;

    }

    bool read_data(const std::string& input_directory, const int cpt, const vpCameraParameters& cam_depth,
        vpImage<unsigned char>& I_, vpImage<uint16_t>& I_depth_,
        std::vector<vpColVector>& pointcloud, vpHomogeneousMatrix& cMo, vpImage<vpRGBa>& I_color) {

        std::cout << "Reading data" << std::endl;
        char buffer[256];
        sprintf(buffer, std::string(input_directory + "/Images/Image_%04d.pgm").c_str(), cpt);
        std::string image_filename = buffer;

        sprintf(buffer, std::string(input_directory + "/pcd/%d.pcd").c_str(), cpt);
        std::string depth_filename = buffer;

        sprintf(buffer, std::string(input_directory + "/gt.txt").c_str(), cpt);
        std::string pose_filename = buffer;

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(depth_filename, *cloud) == -1) //* load the file
        {
            PCL_ERROR("Couldn't read pointcloud for display \n");
            return (-1);
        }

        parse_pointcloud(cloud, pointcloud, I_, I_depth_, I_color);

        std::ifstream file_pose(pose_filename.c_str());
        if (!file_pose.is_open()) {
            return false;
        }

        for (unsigned int i = 0; i < 4; i++) {
            for (unsigned int j = 0; j < 4; j++) {
                file_pose >> cMo[i][j];
            }
        }

        return true;
    }
}

bool read_data_batch(pcl::PointCloud<pcl::PointXYZRGB>::Ptr init_frame, vpImage<unsigned char>& I_, vpImage<uint16_t>& I_depth_, std::vector<vpColVector>& pointcloud, vpImage<vpRGBa>& I_color) {
    parse_pointcloud(init_frame, pointcloud, I_, I_depth_, I_color);
    //pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, std::vector<vpColVector> &pointcloud,
    //vpImage<unsigned char> &I, vpImage<uint16_t> &I_depth, const vpCameraParameters &cam_depth
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


double RigidTracking::track(pcl::PointCloud<pcl::PointXYZRGB>::Ptr frame, vpHomogeneousMatrix& cMo, vpMbGenericTracker& tracker, std::vector<vpColVector>& err_map, int count, bool should_track) {
    vpHomogeneousMatrix cMo_backup;
    if (!should_track) {
        cMo_backup = cMo;
    }

    std::vector<vpColVector> pointcloud;
    read_data_batch(frame, I, I_depth_raw, pointcloud, I_color);

    cout << "parsed batch" << endl;

    //tracker.setPose(I,I_depth,cMo,cMo);

    vpImageConvert::createDepthHistogram(I_depth_raw, I_depth);

    std::map<std::string, const vpImage<unsigned char>*> mapOfImages;
    mapOfImages["Camera1"] = &I;
    std::map<std::string, const std::vector<vpColVector>*> mapOfPointclouds;
    mapOfPointclouds["Camera2"] = &pointcloud;
    std::map<std::string, unsigned int> mapOfWidths, mapOfHeights;

    mapOfWidths["Camera2"] = I_depth.getWidth();
    mapOfHeights["Camera2"] = I_depth.getHeight();

    if (!should_track) {
        tracker.setPose(I, I_depth, cMo_backup, cMo_backup);
    }

    //tracker.track(mapOfImages, mapOfPointclouds, mapOfWidths, mapOfHeights, err_map);
    cMo = tracker.getPose();

    if (!should_track) {
        //cMo_backup[0][3] = cMo[0][3];  //nastiest hack ever!
        //cMo_backup[1][3] = cMo[1][3];
        //cMo_backup[2][3] = cMo[2][3];
        tracker.setPose(I, I_depth, cMo_backup, cMo_backup);
        cMo = cMo_backup;
    }

#ifdef DEBUG
    //display1.init(I, 0, 0, "Image");
    //display2.init(I_depth, I.getWidth(), 0, "Depth");



    float min_ = 1000.0f;
    float max_ = -10000.0f;

    for (int i = 0; i < err_map.size(); i++) {
        float error = err_map[i][3];
        if (error < min_) {
            min_ = error;
        }
        if (error > max_) {
            max_ = error;
        }
    }

    float range = max_ - min_;
    float mean = (max_ + min_) / 2;


    for (int i = 0; i < err_map.size(); i++) {
        float error = err_map[i][3];
        if (fabs(error) > CLUSTER_TOLERANCE_RIGID) {
            int x_ = round(((err_map[i][0] / (err_map[i][2])) * F_x) + C_x);
            int y_ = round(((err_map[i][1] / (err_map[i][2])) * F_y) + C_y);

            I_color[y_][x_].R = ((error - min_) / (max_ - min_)) * 255;
            //cout<<"assigning: "<<((error - min_)/(max_ - min_))*255<<", "<<min_<<","<<max_<<","<<error<<endl;
            I_color[y_][x_].G = 0.0f;
            I_color[y_][x_].B = 0.0f;
        }
    }

    vpDisplay::display(I_color);

    vpDisplay::display(I);
    vpDisplay::display(I_depth);

    vpCameraParameters cam_color, cam_depth;
    tracker.getCameraParameters(cam_color, cam_depth);
    vpHomogeneousMatrix depth_M_color;
    depth_M_color[0][3] = 0.0;

    tracker.display(I, I_depth, depth_M_color * cMo, depth_M_color * cMo, cam_color, cam_depth, vpColor::red, 1);
    vpDisplay::displayFrame(I, depth_M_color * cMo, cam_depth, 0.05, vpColor::none, 1);
    vpDisplay::displayFrame(I_depth, depth_M_color * cMo, cam_depth, 0.05, vpColor::none, 1);

    std::stringstream ss;
    ss << "Frame: " << count;
    vpDisplay::displayText(I_depth, 20, 20, ss.str(), vpColor::red);
    ss.str("");
    ss << "Nb features: " << tracker.getError().euclideanNorm() / ((float)tracker.getError().getRows());
    vpDisplay::displayText(I_depth, 40, 20, ss.str(), vpColor::red);

#endif


#ifdef DEBUG
    vpDisplay::flush(I);
    vpDisplay::flush(I_depth);
    vpDisplay::flush(I_color);
    debug_log_img(I, I_depth, count);
#endif

    vpColVector err = tracker.getError();

    //std::cout<<"error map size : "<<err_map.size()<<std::endl;

    //tracker.modifyModel();

    cout << "done rigid init" << endl;


    return (err.euclideanNorm() / (float)err.getRows());

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

void copy_to_buffer(std::string& source_path, std::string& dest_path) {

    std::ifstream  src(source_path, std::ios::binary);
    std::ofstream  dst(dest_path, std::ios::binary);
    dst << src.rdbuf();
}

PointCloud<PointXYZ>::Ptr cloud_from_xyz_excess(string path, int length) {
    std::ifstream file(path);
    std::string line;
    int count = 0;
    int num_points = 0;

    vector<PointXYZ> vec_point;

    //cout<<"begining excess @"<<path<<" with length: "<<length<<endl;

    while (std::getline(file, line)) {
        if (count >= (CAO_HEADER_LENGTH + length - 1)) {
            vector<string> s = split(line, " ");
            PointXYZ p;
            p.x = stof(s[0]);
            p.y = stof(s[1]);
            p.z = stof(s[2]);
            vec_point.push_back(p);
            cout << "points new: (" << p.x << "," << p.y << "," << p.z << ")" << endl;
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

PointCloud<PointXYZ>::Ptr cloud_from_model(string path, int& length) {
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
            //cout<<"line: "<<line<<endl;
            PointXYZ p;
            p.x = stof(s[0]);
            p.y = stof(s[1]);
            p.z = stof(s[2]);
            //cout<<"points org: ("<<p.x<<","<<p.y<<","<<p.z<<")"<<endl;
            vec_point.push_back(p);
        }

        count++;
    }

    PointCloud<PointXYZ>::Ptr cloud(new PointCloud<PointXYZ>);
    cloud->width = num_points;
    cloud->height = 1;
    cloud->is_dense = false;

    //cout<<"adding to pointcloud"<<endl;

    if (vec_point.size() == num_points) {
        for (int i = 0; i < num_points; i++) {
            cloud->push_back(vec_point[i]);
        }
    } else {
        cout << "Mismatch between vec point list and cloud size" << endl;
    }

    //cout<<"done with first step"<<endl;

    length = num_points;
    return cloud;
}

int get_index(vector<Eigen::Vector3d> def_points, string line) {
    float thresh = 0.0001f;

    vector<string> s = split(line, " ");
    //cout<<"trying to split line: "+line<<endl;
    float x = stof(s[0]);
    float y = stof(s[1]);
    float z = stof(s[2]);

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

////string update_model_file_buffer()
////{
////    std::string debug_dir(JACOBIAN_DIR);
////    std::string model_file_origin_path(debug_dir+"model.cao");
////    std::string model_file_dest_path(debug_dir+"model_1.cao");
////    string mod_mesh_file(MODIFIED_MESH_FILE);
////
////    int num_point;
////    PointCloud<PointXYZ>::Ptr model_org = cloud_from_model(model_file_origin_path,num_point);
////    PointCloud<PointXYZ>::Ptr new_points = cloud_from_xyz_excess(mod_mesh_file,num_point);
////
////    OcclusionCheck ocl;
////    vector<Eigen::Vector3d> def_points = ocl.nearest_neighbor_generalized(*model_org,new_points, 0.5f);
////
////
////    std::ifstream file(model_file_origin_path);
////    std::ifstream file_mesh(mod_mesh_file);
////    std::ofstream write_file(model_file_dest_path);
////    std::string line;
////    std::string line2;
////    int count = 0;
////    while (std::getline(file, line) && (write_file.is_open()))
////    {
////        string line_write = line;
////        if((count >= CAO_HEADER_LENGTH) && (count < (CAO_HEADER_LENGTH + num_point)))
////        {
////            int index = get_index(def_points, line);
////            if(index > -1)
////                line_write = to_string(new_points->points[index].x)+" "+to_string(new_points->points[index].y)+" "+to_string(new_points->points[index].z);
////        }
////
////        write_file << line_write << endl;
////        count++;
////
////    }
////    write_file.close();
////
////    std::remove(model_file_origin_path.c_str());
////    std::rename(model_file_dest_path.c_str(), model_file_origin_path.c_str());
////
////    return model_file_origin_path;
////}

void RigidTracking::reset_tracker(vpMbGenericTracker& tracker, pcl::PointCloud<pcl::PointXYZRGB>::Ptr frame, vpHomogeneousMatrix& cMo, bool should_reset) {
    if (should_reset) {
        //string new_model_file_path = update_model_file_buffer(); //this method is being removed on 24-june-2019, replacement not in place.
                                                                   //develop new code when required

        string new_model_file_path = "";                         //replacement stub. calling this method will be lethal!

        std::vector<vpColVector> pointcloud;
        read_data_batch(frame, I, I_depth_raw, pointcloud, I_color);

        //tracker.setPose(I,I_depth,cMo,cMo);

        vpImageConvert::createDepthHistogram(I_depth_raw, I_depth);

        std::map<std::string, const vpImage<unsigned char>*> mapOfImages;
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

        T[0][0] = 1.0;   /**NO TRANSFORMATION BETWEEN TWO CAMERAS**/
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

void RigidTracking::initialize(std::string config_file_path, std::string cao_model_path, pcl::PointCloud<pcl::PointXYZRGB>::Ptr init_frame, vpHomogeneousMatrix cMo_truth, vpMbGenericTracker& tracker) {
    try {
        cout << "started, trying to load: " << config_file_path << endl;
        tracker.loadConfigFile(config_file_path);
        cout << "loaded config file" << endl;


        //std::string debug_dir(JACOBIAN_DIR);
            //std::string model_file_copy_path(debug_dir+"model.cao");
            //copy_to_buffer(cao_model_path,model_file_copy_path);

        bool opt_click_allowed = true;
        bool opt_display = true;
        bool useScanline = false;

        // Edge
        vpMe me;
        me.setMaskSize(5);
        me.setMaskNumber(180);
        me.setRange(8);
        me.setThreshold(10000);
        me.setMu1(0.5);
        me.setMu2(0.5);
        me.setSampleStep(5);
        tracker.setMovingEdge(me);

        cout << "set me" << endl;

        vpKltOpencv klt;
        tracker.setKltMaskBorder(5);
        klt.setMaxFeatures(10000);
        klt.setWindowSize(5);
        klt.setQuality(0.01);
        klt.setMinDistance(5);
        klt.setHarrisFreeParameter(0.02);
        klt.setBlockSize(3);
        klt.setPyramidLevels(3);

        tracker.setKltOpencv(klt);

        tracker.setDepthNormalFeatureEstimationMethod(vpMbtFaceDepthNormal::ROBUST_FEATURE_ESTIMATION);
        tracker.setDepthNormalPclPlaneEstimationMethod(2);
        tracker.setDepthNormalPclPlaneEstimationRansacMaxIter(200);
        tracker.setDepthNormalPclPlaneEstimationRansacThreshold(0.1);
        tracker.setDepthNormalSamplingStep(2, 2);

        tracker.setDepthDenseSamplingStep(2, 2);

        tracker.setAngleAppear(vpMath::rad(85.0));
        tracker.setAngleDisappear(vpMath::rad(89.0));
        tracker.setNearClippingDistance(0.001);
        tracker.setFarClippingDistance(200.0);
        tracker.setClipping(tracker.getClipping() | vpMbtPolygon::FOV_CLIPPING);

        cout << "trying to load model" << endl;
        tracker.loadModel(cao_model_path, cao_model_path);
        cout << "done loading model" << endl;

        vpHomogeneousMatrix T;

        T[0][0] = 1.0;   /**NO TRANSFORMATION BETWEEN TWO CAMERAS**/
        T[0][3] = 0.0;
        T[1][1] = 1;
        T[1][2] = 0;
        T[1][3] = 0.0;
        T[2][1] = 0;
        T[2][2] = 1;
        T[2][3] = 0.0;

        tracker.loadModel(cao_model_path, false, T);
        vpCameraParameters cam_color, cam_depth;
        tracker.getCameraParameters(cam_color, cam_depth);
        tracker.setDisplayFeatures(true);
        tracker.setScanLineVisibilityTest(useScanline);

        std::map<int, std::pair<double, double> > map_thresh;

#ifdef VISP_HAVE_COIN3D
        map_thresh[vpMbGenericTracker::EDGE_TRACKER]
            = useScanline ? std::pair<double, double>(0.005, 3.9) : std::pair<double, double>(0.007, 2.9);
#if defined(VISP_HAVE_MODULE_KLT) && (defined(VISP_HAVE_OPENCV) && (VISP_HAVE_OPENCV_VERSION >= 0x020100))
        map_thresh[vpMbGenericTracker::KLT_TRACKER]
            = useScanline ? std::pair<double, double>(0.006, 1.9) : std::pair<double, double>(0.005, 1.3);
        map_thresh[vpMbGenericTracker::EDGE_TRACKER | vpMbGenericTracker::KLT_TRACKER]
            = useScanline ? std::pair<double, double>(0.005, 3.2) : std::pair<double, double>(0.006, 2.8);
#endif
        map_thresh[vpMbGenericTracker::EDGE_TRACKER | vpMbGenericTracker::DEPTH_DENSE_TRACKER]
            = useScanline ? std::pair<double, double>(0.003, 1.7) : std::pair<double, double>(0.002, 0.8);
#if defined(VISP_HAVE_MODULE_KLT) && (defined(VISP_HAVE_OPENCV) && (VISP_HAVE_OPENCV_VERSION >= 0x020100))
        map_thresh[vpMbGenericTracker::KLT_TRACKER | vpMbGenericTracker::DEPTH_DENSE_TRACKER]
            = std::pair<double, double>(0.002, 0.3);
        map_thresh[vpMbGenericTracker::EDGE_TRACKER | vpMbGenericTracker::KLT_TRACKER | vpMbGenericTracker::DEPTH_DENSE_TRACKER]
            = useScanline ? std::pair<double, double>(0.002, 1.8) : std::pair<double, double>(0.002, 0.7);
#endif
#else
        map_thresh[vpMbGenericTracker::EDGE_TRACKER]
            = useScanline ? std::pair<double, double>(0.007, 2.3) : std::pair<double, double>(0.007, 2.1);
#if defined(VISP_HAVE_MODULE_KLT) && (defined(VISP_HAVE_OPENCV) && (VISP_HAVE_OPENCV_VERSION >= 0x020100))
        map_thresh[vpMbGenericTracker::KLT_TRACKER]
            = useScanline ? std::pair<double, double>(0.006, 1.7) : std::pair<double, double>(0.005, 1.4);
        map_thresh[vpMbGenericTracker::EDGE_TRACKER | vpMbGenericTracker::KLT_TRACKER]
            = useScanline ? std::pair<double, double>(0.004, 1.2) : std::pair<double, double>(0.004, 1.0);
#endif
        map_thresh[vpMbGenericTracker::EDGE_TRACKER | vpMbGenericTracker::DEPTH_DENSE_TRACKER]
            = useScanline ? std::pair<double, double>(0.002, 0.7) : std::pair<double, double>(0.001, 0.4);
#if defined(VISP_HAVE_MODULE_KLT) && (defined(VISP_HAVE_OPENCV) && (VISP_HAVE_OPENCV_VERSION >= 0x020100))
        map_thresh[vpMbGenericTracker::KLT_TRACKER | vpMbGenericTracker::DEPTH_DENSE_TRACKER]
            = std::pair<double, double>(0.002, 0.3);
        map_thresh[vpMbGenericTracker::EDGE_TRACKER | vpMbGenericTracker::KLT_TRACKER | vpMbGenericTracker::DEPTH_DENSE_TRACKER]
            = useScanline ? std::pair<double, double>(0.001, 0.5) : std::pair<double, double>(0.001, 0.4);
#endif
#endif



        /*vpImage<unsigned char> I, I_depth;
        vpImage<uint16_t> I_depth_raw;*/

        std::vector<vpColVector> pointcloud;

        read_data_batch(init_frame, I, I_depth_raw, pointcloud, I_color);

        vpImageConvert::createDepthHistogram(I_depth_raw, I_depth);

        vpHomogeneousMatrix depth_M_color;
        depth_M_color[0][3] = 0.0;
        tracker.setCameraTransformationMatrix("Camera2", depth_M_color);
        tracker.initFromPose(I, cMo_truth);

        tracker.setPose(I, I_depth, cMo_truth, cMo_truth);

#ifdef DEBUG
        display1.init(I, 0, 0, "Image");
        display2.init(I_depth, I.getWidth(), 0, "Depth");
        display3.init(I_color, (I_color.getWidth() * 2), 0, "Error");
#endif


#ifdef PRINT_LOG
        std::cout << "Tracker initialized uneventfully!" << std::endl;
#endif

        /*vpDisplay::display(I);
        vpDisplay::display(I_depth);*/


    } catch (const vpException& e) {
        std::cout << "Caught an exception: " << e << std::endl;
        std::cout << "Tracker initialization failed!" << std::endl;
    }
}
