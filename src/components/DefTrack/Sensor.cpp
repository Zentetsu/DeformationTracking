#include "Sensor.h"


Sensor::Sensor():
#ifdef SENSOR_INPUT_ACTIVE
_align(nullptr),
_ppointCloud(new PointCloud<PointXYZRGB>),
#endif
_is_d435_initialized(false),
pointHomogeneousCoord(4,1.0f),
x_max(0.0f),
x_min(0.0f),
y_max(0.0f),
y_min(0.0f),
z_max(0.0f),
z_min(0.0f),
cloud_pass(new PCLPointCloud2),
filter_cloud(new PCLPointCloud2)
{

}

#ifdef SENSOR_INPUT_ACTIVE
int Sensor::initialize_markers(bool is_robot_marker)
{

//initialize marker model

    double dimensionTag = 0.025;
    double cornerPosition = dimensionTag/2.;
    std::vector<vpPoint> cornersRobotTag = {
        vpPoint(-cornerPosition, -cornerPosition, 0), // QCcode point 0 3D coordinates in plane Z=0
        vpPoint(cornerPosition, -cornerPosition, 0), // QCcode point 1 3D coordinates in plane Z=0
        vpPoint(cornerPosition, cornerPosition, 0), // QCcode point 2 3D coordinates in plane Z=0
        vpPoint(-cornerPosition, cornerPosition, 0) // QCcode point 3 3D coordinates in plane Z=0
    };
    _vMarkerRobotConfigModel = cornersRobotTag;

_qMeff[0][0] = 1.0f;    _qMeff[0][1] = 0.0f;    _qMeff[0][2] = 0.0f;    _qMeff[0][3] = 0.0f;
_qMeff[1][0] = 0.0f;    _qMeff[1][1] = 1.0f;    _qMeff[1][2] = 0.0f;    _qMeff[1][3] = -0.08f;
_qMeff[2][0] = 0.0f;    _qMeff[2][1] = 0.0f;    _qMeff[2][2] = 1.0f;    _qMeff[2][3] = -0.03f;
_qMeff[3][0] = 0.0f;    _qMeff[3][1] = 0.0f;    _qMeff[3][2] = 0.0f;    _qMeff[3][3] = 1.0f;



// Initialise the tracking
  vpMouseButton::vpMouseButtonType button = vpMouseButton::button1;
  vpImagePoint ip;
  int nbPoint = 0;

  // Used to have a unique function able to be used in both ROBOT_MARKERS case and OBJECT_MARKERS case
  std::vector<vpDot2*>* pvp_features = nullptr;
  std::vector<vpImagePoint>* pv_cog = nullptr;


    if(is_robot_marker)
     { pvp_features = &_vpRobotMarkers;
      pv_cog = &_vRobotCog;
      _isFirstRobotMarkerPoseEstimation = true; // The pose estimation has never been performed because we are initializing the visual features to track
      }
    else{
      pvp_features = &_vpVisualFeatures;
      pv_cog = &_vcog;
	} 

  // Ensuring that the vector are clean
  if(pvp_features->size() != 0){ // The function is called once again e.g during the tool calibration for capturing a new pose
    for(unsigned int i= 0; i < pvp_features->size(); i++){
      delete (*pvp_features)[i];
    }
    pvp_features->clear();
    pv_cog->clear();
  }

  do {
    _grabber.acquire((unsigned char*)_I.bitmap, NULL, NULL, _ppointCloud, NULL, _align);
    vpDisplay::display(_I);
    vpDisplay::flush(_I);
    if(is_robot_marker)
	{
        vpDisplay::displayText(_I, 10, 10, "Selection of the ROBOT markers with left clicks, right click to change step", vpColor::red);
       }
      else{
        vpDisplay::displayText(_I, 10, 10, "Selection of the OBJECT markers with left clicks, right click to change step", vpColor::orange);
        }
    

    vpDisplay::displayText(_I, 30,10, "Left click to select a point, right to start tracking", vpColor::red);

    if (vpDisplay::getClick(_I, ip, button, false)) {
      if (button == vpMouseButton::button1) {
        vpDot2* p_dot = new vpDot2();

        p_dot->setGraphics(true);
        p_dot->setGraphicsThickness(2);
        vpImageConvert::convert(_I,_Igrey);
        p_dot->initTracking(_Igrey, ip);

        pvp_features->push_back(p_dot);
        pv_cog->push_back(p_dot->getCog());

        vpDisplay::displayCross(_I, ip, 12, vpColor::green);
        std::list<vpImagePoint> edges;
        p_dot->getEdges(edges);
        std::list<vpImagePoint>::const_iterator it;
        for (it = edges.begin(); it != edges.end(); ++it) {
          vpDisplay::displayPoint(_I, *it, vpColor::blue);
        }
        nbPoint++;
      }
    }
    vpDisplay::flush(_I);
    vpTime::wait(20);
  } while (button != vpMouseButton::button3);

  return nbPoint;

}



bool Sensor::trackScene(vpColVector& robotObservation){
  bool isSuccess = false;
  vpHomogeneousMatrix cMq, local_cMeff;

  vpImageConvert::convert(_I,_Igrey);

  // // Looking for the 3D coordinates of the tip of the end-effector
  robotObservation.resize(3);
  isSuccess = keypointTracking(); // Tracking the markers making the tag
  if(isSuccess){(x_max > 0.0f) &&
    computePose(cMq); // Computing the pose of the tag from the markers positions
    local_cMeff = cMq * _qMeff; // We knew the QR code position in the camera frame and the transformation end-effector -> QR code frame so we can deduce the position of the end-effector in the camera frame
    vpTranslationVector coordinatesTipEffector = local_cMeff.getTranslationVector();
    robotObservation[0] = coordinatesTipEffector[0];
    robotObservation[1] = coordinatesTipEffector[1];
    robotObservation[2] = coordinatesTipEffector[2];

  }

  return true; // if isSuccess = isObjectTrackingSuccessful = SUCCESS then the sum is equal to FULL_SUCCESS
}

bool Sensor::keypointTracking(){
  
  //Begin the tracking
  for(unsigned int i=0; i<_vpRobotMarkers.size();i++){
    vpDot2* p_dot = _vpRobotMarkers[i];
    try{
      _vpRobotMarkers[i]->track(_Igrey, false);

      _vRobotCog[i] = _vpRobotMarkers[i]->getCog();
      
    }catch(...){
      //std::cout << "Failed on marker " << i << std::endl;
      return false;
   }

  }

  return true;
}
#endif



bool Sensor::computePose(vpHomogeneousMatrix &cMq)
{
  vpPose pose;
  double x = 0, y = 0;
  for (unsigned int i = 0; i < _vMarkerRobotConfigModel.size(); i++) {

      vpPixelMeterConversion::convertPoint(_param, _vRobotCog[i], x, y);
      _vMarkerRobotConfigModel[i].set_x(x);
      _vMarkerRobotConfigModel[i].set_y(y);
      pose.addPoint(_vMarkerRobotConfigModel[i]);
    
  }
  if (_isFirstRobotMarkerPoseEstimation) { // First estimation of the pose => tries 2 different estimation method and keeps the best result
    vpHomogeneousMatrix cMo_dem;
    vpHomogeneousMatrix cMo_lag;
    try
    {
      pose.computePose(vpPose::DEMENTHON, cMo_dem);
      pose.computePose(vpPose::LAGRANGE, cMo_lag);
      double residual_dem = pose.computeResidual(cMo_dem);
      double residual_lag = pose.computeResidual(cMo_lag);
      if (residual_dem < residual_lag)
        _cMlastRobot = cMo_dem;
      else
        _cMlastRobot = cMo_lag;
        _isFirstRobotMarkerPoseEstimation = false;
    }catch(...){
      return false;
    }
  }
  try{
    pose.computePose(vpPose::VIRTUAL_VS, _cMlastRobot); // Computes the pose from the last estimation which should be close enough to the current one
    cMq = _cMlastRobot;
  }catch(...){
    return false;
  }
  return true;
}

vpHomogeneousMatrix Sensor::get_extrinsics_depth2color()
{
  return extrinsics;
}

vpCameraParameters Sensor::get_intrinsics_color_d435()
{
  return _param;
}

vpCameraParameters Sensor::get_intrinsics_depth_d435()
{
  return _paramDepth;
}

#ifdef SENSOR_INPUT_ACTIVE
void Sensor::initialize_d435()
{
  try
  {
  unsigned int width = 640;
  unsigned int height = 480;
  unsigned int fps = 30;

  _config.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_RGBA8, fps);
  _config.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16, fps);

  _grabber.open(_config);
  _I.init((unsigned int)_grabber.getIntrinsics(RS2_STREAM_COLOR).height, (unsigned int)_grabber.getIntrinsics(RS2_STREAM_COLOR).width);


  _display.init(_I,0,0,"Deformation tracking");


  _param = _grabber.getCameraParameters(RS2_STREAM_COLOR);
  _paramDepth = _grabber.getCameraParameters(RS2_STREAM_DEPTH);

   cout<<"color: "<<_param<<endl;
   cout<<"depth: "<<_paramDepth<<endl;

  _Igrey.init((unsigned int)_grabber.getIntrinsics(RS2_STREAM_COLOR).height, (unsigned int)_grabber.getIntrinsics(RS2_STREAM_COLOR).width);

  extrinsics =  _grabber.getTransformation(RS2_STREAM_DEPTH, RS2_STREAM_COLOR); 

  if(initialize_markers(true)==4)
  {
	cout<<"robot marker intialized"<<endl;

	vpHomogeneousMatrix pose;
	computePose(pose);
	cout<<pose<<endl;
  }

  _is_d435_initialized = true;


	} catch (const vpException &e) {
    std::cerr << "Sensor::initialize_sr300 error " << e.what() << std::endl;
    } catch (const rs2::error &e) {
    std::cerr << "Sensor::initialize_sr300 error calling " << e.get_failed_function() << "(" << e.get_failed_args()
              << "): " << e.what() << std::endl;
    } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    }


}
#else
void Sensor::init_visualizer(vpImage<vpRGBa> &I)
{
    _display.init(I,0,0,"Deformation tracking");
}
#endif

void Sensor::set_extrinsics_depth2color(vpHomogeneousMatrix &extrinsicMat)
{
    extrinsics = extrinsicMat;
}

void Sensor::extract_boundingbox(PointCloud<PointXYZ>::Ptr &visible_vertices)
{
   PointXYZ min, max;
   getMinMax3D(*visible_vertices, min, max);

   x_max = max.x + BOUNDING_BOX_PADDING;
   x_min = min.x - BOUNDING_BOX_PADDING;
   y_max = max.y + BOUNDING_BOX_PADDING;
   y_min = min.y - BOUNDING_BOX_PADDING;
   z_max = max.z + BOUNDING_BOX_PADDING;
   z_min = min.z;

   //cout<<"[Sensor::extract_boundingbox] 3D RoI: ("<<setprecision(10)<<x_min<<","<<y_min<<","<<z_min<<") to ("<<x_max<<","<<y_max<<","<<z_max<<")"<<endl;
}

#ifdef SENSOR_INPUT_ACTIVE
void Sensor::acquire_d435(vpImage<vpRGBa> &color, PointCloud<PointXYZRGB>::Ptr &pointcloud)
{
    if(_is_d435_initialized)
    {
        _grabber.acquire((unsigned char*)_I.bitmap, NULL, NULL, _ppointCloud,NULL, _align);


        if((z_min > 0.0f) && (z_max > 0.0f))
        {
            //cout<<"[Sensor::acquire_d435] filtering data"<<endl;
            toPCLPointCloud2( *_ppointCloud, *filter_cloud );
            PassThrough<PCLPointCloud2> pass;
            pass.setKeepOrganized(true);
            pass.setUserFilterValue(0.0f);
            pass.setInputCloud(filter_cloud);

            pass.setFilterFieldName ("z");
            pass.setFilterLimits (z_min, z_max);
            pass.filter(*filter_cloud);

            fromPCLPointCloud2( *filter_cloud, *_ppointCloud);
        }


        pointcloud = _ppointCloud;
        color = _I;

#ifdef VISUALIZER
        vpDisplay::display(_I);
#endif

    }
    else
    {
        cerr<<"Initialize the D435 sensor first"<<endl;
    }

}

void Sensor::convert_to_image(float x, float y, float z, vpImagePoint &ip)
{
    pointHomogeneousCoord[0] = x;
    pointHomogeneousCoord[1] = y;
    pointHomogeneousCoord[2] = z;

    cMpoint = extrinsics * pointHomogeneousCoord;

    double _x = cMpoint[0];
    double _y = cMpoint[1];
    double _z = cMpoint[2];

    double xNormalized = _x/_z;
    double yNormalized = _y/_z;


    vpMeterPixelConversion::convertPoint (_param, xNormalized, yNormalized, ip);

}
#else
void Sensor::convert_to_image(float x, float y, float z, vpImagePoint &ip, vpCameraParameters &param)
{
    pointHomogeneousCoord[0] = x;
    pointHomogeneousCoord[1] = y;
    pointHomogeneousCoord[2] = z;

    cMpoint = extrinsics * pointHomogeneousCoord;

    double _x = cMpoint[0];
    double _y = cMpoint[1];
    double _z = cMpoint[2];

    double xNormalized = _x/_z;
    double yNormalized = _y/_z;


    vpMeterPixelConversion::convertPoint (param, xNormalized, yNormalized, ip);

}
#endif

#ifdef VISUALIZER

#ifdef SENSOR_INPUT_ACTIVE
void Sensor::updateScreen(mesh_map &visible_mesh, vector<vpColVector> &err_map, residual_statistics &res_stats,vector<PointXYZ> matched_points, bool display_error_map, bool display_centroids)
#else
void Sensor::updateScreen(mesh_map &visible_mesh, vector<vpColVector> &err_map, residual_statistics &res_stats,vector<PointXYZ> matched_points, vpCameraParameters &param, vpImage<vpRGBa> &I, bool display_error_map, bool display_centroids, bool debug_folder_active, string &debug_path, int debug_flag, int pcd_count)
#endif
{
#ifdef SENSOR_INPUT_ACTIVE
    vpDisplay::display(_I);
#else
    vpDisplay::display(I);
#endif
    unsigned int index(0);

    vpColor color;
    vpColor color_line;
    color_line.setColor(117, 173, 76);

    vpImagePoint ip;

    if(display_error_map)
    {
        for(int i = 0; i < err_map.size(); i++)
        {
#ifdef SENSOR_INPUT_ACTIVE
            convert_to_image(err_map[i][0], err_map[i][1], err_map[i][2], ip);
#else
            convert_to_image(err_map[i][0], err_map[i][1], err_map[i][2], ip, param);
#endif

        float color_val = ((err_map[i][3] - res_stats.minimum)/(res_stats.maximum-res_stats.minimum)*255.0f);
        color.setColor(color_val,0.0f,255.0f - color_val);

#ifdef SENSOR_INPUT_ACTIVE
    vpDisplay::displayCircle (_I, ip, 2, color, true, 1);
#else
    vpDisplay::displayCircle (I, ip, 2, color, true, 1);
#endif

        }
    }

    for(auto polygon = visible_mesh.mesh.polygons.begin(); polygon != visible_mesh.mesh.polygons.end(); ++polygon)
    {
      std::vector< vpImagePoint >  ips;
        for(int i = 0; i < 3; i++)
        {
          pcl::PointXYZ point = visible_mesh.mesh_points[index];

#ifdef SENSOR_INPUT_ACTIVE
            convert_to_image(point.x, point.y, point.z, ip);
            vpDisplay::displayPoint(_I, ip,vpColor::blue, 1);
#else
            convert_to_image(point.x, point.y, point.z, ip, param);
            vpDisplay::displayPoint(I, ip,vpColor::blue, 1);
#endif

          ips.push_back(ip);
          index++;
        }
#ifdef SENSOR_INPUT_ACTIVE
        vpDisplay::displayLine( _I,  ips, true, color_line, 1);
#else
        vpDisplay::displayLine(I,  ips, true, color_line, 1);
#endif

    }

	if(display_centroids)
	{
	    for(int i = 0; i < matched_points.size(); i++)
	    {
		PointXYZ p = matched_points[i];

#ifdef SENSOR_INPUT_ACTIVE
            convert_to_image(p.x, p.y, p.z,  ip);
            vpDisplay::displayCross(_I, ip, 9, vpColor::green, 2);
            vpDisplay::displayCircle (_I, ip, 11, vpColor::red, false, 2);
#else
            convert_to_image(p.x, p.y, p.z,  ip, param);
            vpDisplay::displayCross(I, ip, 9, vpColor::green, 2);
            vpDisplay::displayCircle(I, ip, 11, vpColor::red, false, 2);
#endif
	    }

	}

#ifdef SENSOR_INPUT_ACTIVE
    vpDisplay::flush(_I);
#else
    vpDisplay::flush(I);
#endif

    if((debug_flag >= 1) && debug_folder_active)
    {
        vpImage<vpRGBa> I_composite;
        vpDisplay::getImage(I, I_composite);
        std::string ofilename(debug_path+"/"+to_string(pcd_count)+".ppm");
        vpImageIo::write(I_composite, ofilename) ;

    }
}
#endif

