#include "DeformationTracking.h"

#include <sofa/core/ObjectFactory.h>

using namespace std;

// namespace sofa
//{

// namespace component
//{

// namespace controller
//{

std::vector<int> tracker_type = {vpMbGenericTracker::KLT_TRACKER, vpMbGenericTracker::DEPTH_DENSE_TRACKER};

vpMbGenericTracker tracker_;

DeformationTracking::DeformationTracking()
    : objectModel(initData(&objectModel, "objectModel", "Object Model")),
      mechModel(initData(&mechModel, "mechModel", "Mechanical Model")),
      fixedPoints(initData(&fixedPoints, "fixedPoints", "Fixed Points from the Simulation")),
      activePoints(initData(&activePoints, "activePoints", "Active nodes for which to track the object")),
      jacobianUp(initData(&jacobianUp, "jacobianUp", "Model when object pushed up for Jacobian")),
      jacobianDown(initData(&jacobianDown, "jacobianDown", "Model when object pushed down for Jacobian")),
      forcePoints(initData(&forcePoints, "forcePoints", "Points where the deforming forces are to be applied")),
      objFileName(initData(&objFileName, "objFileName", "Object file name")),
      mechFileName(initData(&mechFileName, "mechFileName", "Mechanical model file name")),
      update_tracker(initData(&update_tracker, "update_tracker", "Update coming from the tracker, array dimension = (3 x n), where n = no. of nodes being tracked")),
      transformFileName(initData(&transformFileName, "transformFileName", "Transformation Matrix file name")),
      dataFolder(initData(&dataFolder, "dataFolder", "Folder containing captured data")),
      outputDirectory(initData(&outputDirectory, "outputDirectory", "Directory where debug output is to be dumped")),
      configPath(initData(&configPath, "configPath", "Rigid tracking config path")),
      caoModelPath(initData(&caoModelPath, "caoModelPath", "Path to CAO model path")),
      calibrationFile(initData(&calibrationFile, "calibrationFile", "Path to file generated from calibration")),
      C_x(initData(&C_x, "C_x", "C_x")),
      C_y(initData(&C_y, "C_y", "C_y")),
      F_x(initData(&F_x, "F_x", "F_x")),
      F_y(initData(&F_y, "F_y", "F_y")),
      colorImageFolder(initData(&colorImageFolder, "colorImageFolder", "colorImageFolder")),
      depthDataFolder(initData(&depthDataFolder, "depthDataFolder", "depthDataFolder")),
      dataOffset(initData(&dataOffset, "dataOffset", "dataOffset")),
      frequency(initData(&frequency, "frequency", "frequency")),
      colorC_x(initData(&colorC_x, "colorC_x", "colorC_x")),
      colorC_y(initData(&colorC_y, "colorC_y", "colorC_y")),
      colorF_x(initData(&colorF_x, "colorF_x", "colorF_x")),
      colorF_y(initData(&colorF_y, "colorF_y", "colorF_y")),
      depthToColorExtrinsicFile(initData(&depthToColorExtrinsicFile, "depthToColorExtrinsicFile", "depthToColorExtrinsicFile")),
      trackerInitFile(initData(&trackerInitFile, "trackerInitFile", "Initial config for rigid tracking pose - from click")),
      iterations(initData(&iterations, "iterations", "Maximum iterations allowed for the minimization")),
      simulationMessage(initData(&simulationMessage, "simulationMessage", "Message from SOFA simulation (front-end) to tracker (back-end)")),
      trackerMessage(initData(&trackerMessage, "trackerMessage", "Message from SOFA tracker (back-end) to simulation (front-end)")),  //,
#ifdef COMMUNICATION
      _dataTransmitter(CommunicationUtilities::getInputFolderForEstimation(true), false),  // data_transmission
#endif
      initLambda(initData(&initLambda, "initLambda", "Initial value of lambda used by Gauss-Newton")),
      debugFlag(initData(&debugFlag, "debugFlag", "Decides whether to dump debug data or not. 0: debug off, 1: structured debug on, 2: jacobian and residual mesh dump turned on")),
      additionalDebugFolder(initData(&additionalDebugFolder, "additionalDebugFolder", "Folder containing jacobian and residual mesh + visible nodes")),
      jacobianForce(initData(&jacobianForce, "jacobianForce", "Force used for obtaining the Jacobian")),
      tracker(tracker_type),
      robot_marker_initialized(false),
      prev_time(vpTime::measureTimeMicros()),
      registration_required(false),
      reserved_counter_1(0),
      load_new_data(true),
      is_cloud_transformed(false),
      clustering_mode(0),
      is_first_initialized(false) {
    this->f_listening.setValue(true);
}

/**
Increment the pcd_count, so that the next call to the 'DeformationTracking' plugin loads a new PCD file

@returns void **/
void DeformationTracking::incrementPcdCount() {
    _log_("Incrementing counter") pcd_count++;  /// increase the PCD counter

    jFem.no_more_updates = false;

    load_new_data = true;           /// indicate that a new data needs to be loaded
    jFem.set_pcd_count(pcd_count);  /// set the current PCD count to the JacobianFEM class
    node_count = 0;
    jFem.set_prev_pointcloud(frame);                     /// Assign the current pointcloud (from the current data frame) as the previous pointcloud
    jFem.set_prev_colored_depth(colored_depth);          /// Assign the current depth image (from the current data frame) as the previous depth image
    prev_colored_depth = jFem.get_prev_colored_depth();  /// Making a copy of the same variable locally
    jFem.setCurr_image(image);                           /// Set the current  image
    jFem.set_is_photo_initialized(true);                 /// Indicate that photometry has been initialized
    jFem.is_first_node = true;
    jFem.is_first_iteration = true;
    jFem.setLmLambda(initLambda.getValue());  /// Resetting the lambda value for Levenberg Marquardt
    jFem.setPrev_residual(0.0f);
    is_cloud_transformed = false;
#ifdef SPARSE_MINIMIZATION
    jFem.clear_sparse_buffer();
#endif
    _warn_("Setting previous frame")
}

/**
Formats the shared_mesh variable with the updated vertex coordinates

@returns void **/
void DeformationTracking::prepSharedMesh() {
    vector<vector<sofa::type::Vec3>> unformatted_meshes;         /// Vector of unformatted meshes allows us to share multiple meshes between SOFA simulation/Python and our C++ code here
    formatList(mechModel.getValue(), unformatted_meshes, true);  /// Formatting the mesh coming in from SOFA
    shared_mesh.resize(unformatted_meshes[0].size() * 3, false);

    _log_("Shared mesh size: " << unformatted_meshes[0].size()) for (int i = 0; i < unformatted_meshes[0].size(); i++)  /// Assigning the values
    {
        sofa::type::Vec3 v = unformatted_meshes[0][i];
        shared_mesh[3 * i + 0] = (v[0]);
        shared_mesh[3 * i + 1] = (v[1]);
        shared_mesh[3 * i + 2] = (v[2]);
    }
}

/**
Updates the local copy of the mechanical mesh based on the SOFA simulation

@returns void **/
void DeformationTracking::updateMechanicalModel() {
    _log_("Previous mechanical cloud size: " << mechanical_cloud->size()) _log_("Previous mechanical cloud size: " << (shared_mesh.size() / 3)) if ((shared_mesh.size() / 3) == mechanical_cloud->size())  /// updates the local copy of the mechanical model
    {
        for (int i = 0; i < mechanical_cloud->size(); i++) {
            mechanical_cloud->at(i).x = shared_mesh[3 * i + 0];
            mechanical_cloud->at(i).y = shared_mesh[3 * i + 1];
            mechanical_cloud->at(i).z = shared_mesh[3 * i + 2];
        }
    }
    else {
        cerr << "ERROR: DeformationTracking::updateMechanicalModel: shared mechanical model and local mechanical model at DeformationTracker has different sizes. Failed to update!" << endl;
        _fatal_("Something wrong here")
    }
}

/**
Updates the local copy of the mechanical mesh based on the SOFA simulation - prepares the vertices that have been assigned the fixed constraint in the properties file

@returns void **/
void DeformationTracking::prepFixedConstraints() {
    fixed_constraints.clear();
    for (int i = 0; i < fixedPoints.getValue().size(); i++) {
        fixed_constraints.push_back((int)fixedPoints.getValue()[i]);  /// create a local copy of the fixed constraints
    }
}

/**
Clear buffers, reset counters and do house-keeping to prepare the code to accept a new data frame

@returns void **/
void DeformationTracking::seekNextData() {
    _log_("Seeking next data") prepSharedMesh();
    updateMechanicalModel();

    node_count++;                  /// increment the number of node
    jFem.node_count = node_count;  /// share the data with the JacobianFEM Object

    load_new_data = false;

    if (node_count == activePoints.getValue().size()) incrementPcdCount();  /// Increment the PCD count

    counter = 0;
    double t = vpTime::measureTimeMicros();
    _info_("Time elapsed since last frame: " << setprecision(10) << (t - prev_time) / 1000000.0f << " : frame #" << pcd_count << " (node:" << node_count << ")");

    prev_time = t;
    simulationMessage.setValue("ready");
    trackerMessage.setValue("ready");
}

/**
SOFA provides the deformed mesh in a strange format, this method tames the data into a more agreeable format

@param simulationDump : the raw data coming from SOFA
@param unformatted_meshes : output from the formatting operation
@param jacobian_mesh : indicates if the last deformation was a part of 'update' or 'jacobian computation'

@returns void **/
void DeformationTracking::formatList(const sofa::type::vector<sofa::type::Vec3> &simulationDump, vector<vector<sofa::type::Vec3>> &unformatted_meshes, bool jacobian_mesh) {
    int count = -1;

    if (jacobian_mesh) {
        x_.clear();
        _x.clear();
        y_.clear();
        _y.clear();
        z_.clear();
        _z.clear();
    } else {
        x_.clear();
    }

    _log_("Received from simulation: " << simulationDump.size())

        for (int i = 0; i < simulationDump.size(); i++) {
        sofa::type::Vec3 v = simulationDump[i];

        if (count >= 0) {
            if ((v[0] == -100) && (v[1] = -100) && (v[2] = -100))  /// we use '-100' as a seperator
            {
                count++;
            } else {
                sofa::type::Vec3 v_;
                v_[0] = v[0];
                v_[1] = v[1];
                v_[2] = v[2];

                if (jacobian_mesh) {
                    if (count == 0) {
                        x_.push_back(v_);
                    } else if (count == 1) {
                        _x.push_back(v_);
                    } else if (count == 2) {
                        y_.push_back(v_);
                    } else if (count == 3) {
                        _y.push_back(v_);
                    } else if (count == 4) {
                        z_.push_back(v_);
                    } else if (count == 5) {
                        _z.push_back(v_);
                    }
                } else {
                    x_.push_back(v_);
                }
            }
        } else if ((v[0] == -100) && (v[1] = -100) && (v[2] = -100)) {
            count++;
        }
    }

    if (jacobian_mesh) {
        unformatted_meshes.push_back(x_);
        unformatted_meshes.push_back(_x);
        unformatted_meshes.push_back(y_);
        unformatted_meshes.push_back(_y);
        unformatted_meshes.push_back(z_);
        unformatted_meshes.push_back(_z);
    } else {
        unformatted_meshes.push_back(x_);
    }
}

/**
This method handles clustering and rigid tracking

@returns void **/
void DeformationTracking::matching() {
    if (robot_marker_initialized) {
#ifdef SENSOR_INPUT_ACTIVE
        vpImage<vpRGBa> color;
        PointCloud<PointXYZRGB>::Ptr pointcloud(new PointCloud<PointXYZRGB>);
        sensor.acquire_d435(color, pointcloud);
        sensor.trackScene(robotObservation);
        frame.swap(pointcloud);
#endif
    }

    vector<vpColVector> err_map;
    residual_statistics res_stat;

    if (load_new_data) {
        _warn_("Loading new data") nrm.initialize(frame, transform, rTrack, tracker, configPath.getValue().c_str(), caoModelPath.getValue().c_str(), cMo, offline_data, image, depth_image, colored_depth, pcd_count, node_count, dataOffset.getValue(), rigid_tracking_initialized,
                                                  registration_required);  /// initialize the data

        vpHomogeneousMatrix cMo_prio = cMo;  /// set a local copy of the cMo matrix
        cv::Mat geometric_error;

        jFem.get_residual_simple(model, frame, cMo, pcd_count, err_map, true, res_stat, visible_mesh, correspond, residual, geometric_error);  /// does rigid tracking and also provides a map of residual error

        if ((pcd_count > 2) && (!is_cloud_transformed)) {
            Eigen::Matrix4f trns_inv;
            cMo_prio = cMo_prio * cMo.inverse();  /// updating the last estimate of 'cMo'
            ocl.visp_to_eigen_4x4(cMo_prio, trns_inv);

            cv::Mat img_depth = jFem.get_prev_colored_depth();

            PointCloud<PointXYZRGB>::Ptr colored_depth_trns = jFem.get_prev_pointcloud();
            colored_depth_trns = ocl.transformPointcloudXYZRGB(colored_depth_trns, img_depth, trns_inv, offline_data.depth_Fx, offline_data.depth_Fy, offline_data.depth_Cx, offline_data.depth_Cy);

            jFem.set_prev_pointcloud(colored_depth_trns);

            is_cloud_transformed = true;
        }

        ocl.visp_to_eigen_4x4(cMo, transform);

        jFem.augment_transformation(transform);  /// this sets the Kronecker product of the rotation matrix part of cMo with the "influence matrix"
        _log_("Influence matrix transformation done")

            if ((pcd_count > CLUSTERING_INITIALIZATION_POINT)) {
            clustering_mode = 1;  /// some different options for clustering, based ont the dataset. the heuristics are provided inside these methods

            /// Clustering depends on some heuristics.
            /// Note: this can be an important future work, to develop more robust method of clustering (deep learning etc.)
            /// For future use: it is important to have temporal consistency with clusters...
            /// ... if the cluster centroids keep changing while the external applied force do not change, it does not result in good accuracy
            /// Another possible future work: to include some error term / penalty cost for ensuring temporal smoothness (not directly related to clustering)
            if (clustering_mode == 0) {
                nrm.align_and_cluster_2(prev_colored_depth, colored_depth, mechanical_cloud, visible_mesh, fixed_constraints, transform, active_indices, matched_points, geometric_error, additionalDebugFolder.getValue(), debugFlag.getValue(), 3, pcd_count);
            } else if (clustering_mode == 1) {
                if (!is_first_initialized) {
                    first_colored_depth = prev_colored_depth.clone();
                    is_first_initialized = true;
                }
                nrm.align_and_cluster_3(first_colored_depth, colored_depth, mechanical_cloud, visible_mesh, fixed_constraints, transform, active_indices, matched_points, geometric_error, additionalDebugFolder.getValue(), debugFlag.getValue(), 3, pcd_count);
            }

            matched_points.clear();  /// clear the previously obtained clusters

            for (int i = 0; i < active_indices.size(); i++) {
                matched_points.push_back(mechanical_cloud->points[active_indices[i]]);  /// inserting the new matched clusters
            }

            _info_("Active point: " << active_indices[0])
        }
    }

#ifdef SENSOR_INPUT_ACTIVE
    sensor.extract_boundingbox(visible_mesh.visible_vertices);
#endif

#ifdef VISUALIZER
#ifdef SENSOR_INPUT_ACTIVE
    sensor.updateScreen(visible_mesh, err_map, res_stat, matched_points, true, false);
#else
    if (!rigid_tracking_initialized) {
        sensor.init_visualizer(rTrack.I_color);
    }
    bool showMarker = false;

    sensor.updateScreen(visible_mesh, err_map, res_stat, matched_points, ocl._colorCamParam, rTrack.I_color, false, showMarker, debug_folder_initialized, image_debug_path, debugFlag.getValue(), pcd_count);
#endif
#endif

    rigid_tracking_initialized = true;

    trackerMessage.setValue("matched");

    list_pts.clear();

    if ((pcd_count > CLUSTERING_INITIALIZATION_POINT) && (active_indices.size() > 0)) {
        for (int i = 0; i < active_indices.size(); i++) {
            list_pts.push_back(active_indices[i]);
            prev_control_points.push_back(active_indices[i]);
        }

    } else {
        /// Some hard-coded variables if clustering fails
        /// Normally, this code block should not be triggered
        list_pts.push_back(270);
        prev_control_points.push_back(270);

        list_pts.push_back(66);
        prev_control_points.push_back(66);

        list_pts.push_back(268);
        prev_control_points.push_back(268);
    }

    _info_("Tracking node " << activePoints.getValue()[node_count])
}

/**
Calls the method that computes residual and Jacobian

@returns void **/
void DeformationTracking::update() {
    _log_("update")

        vector<float>
            v;

    sofa::type::vector<sofa::type::Vec4> list_values;

    for (int i = 0; i < mechanical_cloud->size(); i++) {
        bool should_nullify = false;

        for (int j = 0; j < prev_control_points.size(); j++) {
            if (prev_control_points[j] == i) {
                should_nullify = true;
            }
        }

        for (int j = 0; j < list_pts.size(); j++) {
            if (list_pts[j] == i) {
                should_nullify = false;
            }
        }

        if (should_nullify) {
            sofa::type::Vec4 values;
            values[0] = i;
            values[1] = 0.0f;
            values[2] = 0.0f;
            values[3] = 0.0f;
            list_values.push_back(values);

            v.push_back(0.000f);
            v.push_back(0.000f);
            v.push_back(0.000f);
        }
    }

    for (int i = 0; i < list_pts.size(); i++) {
        sofa::type::Vec4 values;
        values[0] = list_pts[i];
        values[1] = 0.0f;
        values[2] = 0.0f;
        values[3] = 0.0f;
        list_values.push_back(values);
    }

    forcePoints.setValue(list_values);

    for (int i = 0; i < list_pts.size(); i++) {
        increment.resize(3, 1);
        jFem.active_point.clear();
        jFem.active_point.push_back(list_pts[i]);

        if (!jFem.no_more_updates) {
            /// this method triggers the computation of the error vector, Jacobian and update computation (with Levenberg-Marquardt)
            current_residual = jFem.compute_update(frame, model, transform, cMo, increment, colored_depth);  /// the update is stored in the "increment" variable

            for (int i = 0; i < increment.rows(); i++) {
                v.push_back(increment(i, 0));
            }
        } else {
            for (int i = 0; i < increment.rows(); i++) {
                v.push_back(0.0f);
            }
        }
    }

    update_tracker.setValue(v);  /// setting the increment as message from tracker
    trackerMessage.setValue("updated");
    counter++;
}

/**
Receives update from SOFA simulation and updates the local copy of the deformed meshes

@returns void **/
void DeformationTracking::assimilate() {
    _log_("assimilating mesh") vector<vector<sofa::type::Vec3>> unformatted_meshes;
    formatList(objectModel.getValue(), unformatted_meshes, false);
    nrm.update_polygon(model, unformatted_meshes);  /// updating the local copy of the polygonal model

    if (debugFlag.getValue()) {
        if (node_count == (activePoints.getValue().size() - 1)) {
            /// if debug is required, this method dumps certain files into the debug directory
            nrm.log_output(frame, model, pcd_count, cMo, jFem.getDebugCloudList(), jFem.getDebugCloudLabel(), depth_image, current_residual, dataFolder.getValue(), outputDirectory.getValue(), !debug_folder_initialized, image_debug_path);
            if (!debug_folder_initialized) debug_folder_initialized = true;
        }
    }

    _log_("assimilation done")

        if (counter > iterations.getValue()) {
        prepSharedMesh();

#ifdef COMMUNICATION
        vpColVector applicationPointCoordinates(3, 0.);  // data_transmission
        _dataTransmitter.transmitTrackingResults(shared_mesh, applicationPointCoordinates);
#endif
        seekNextData();
    }
    else {
        trackerMessage.setValue("matched");
    }
}

/**
Delegates the different steps for tracking to different modules based on the incoming messages from the SOFA Python simulation

@returns void **/
void DeformationTracking::track() {
    /********************************Status Messages:********************************/
    /* 'ready': Simulation has just started/loaded a new PCD. Can do matching now.  */
    /* 'applying_J': Simulation has received matched node list. Applying deformation*/
    /*              for Jacobian estimation. Need to wait till the process finishes.*/
    /* 'jacobian_ready': jacobian deformed meshes ready from FEM simulation         */
    /* 'update_ready':  new mesh ready after application of update-force            */
    /********************************************************************************/

    jFem.set_iteration(counter);
    if (simulationMessage.getValue() == STATUS_0) {
        _warn_("READY") matching();
    } else if (simulationMessage.getValue() == STATUS_1) {
        _warn_("JACOBIAN_READY") update();

    } else if (simulationMessage.getValue() == STATUS_2) {
        _warn_("UPDATE_READY") assimilate();
    } else if (!((simulationMessage.getValue() == STATUS_3) || (simulationMessage.getValue() == STATUS_4))) {
        _fatal_("Invalid simulation message received. Exiting!")
    }
}

DeformationTracking::~DeformationTracking() {}

/**
Callback method for SOFA Python plugin, initializes tracker

This is called when the scene is loaded and we utilize the initialize function to load data, initialize variables etc.
Follow 'initTracker' for the details

@returns void **/
void DeformationTracking::init() { initTracker(); }

/**
Callback method for SOFA Python plugin, does nothing

@returns void **/
void DeformationTracking::reinit() {}

/**
Callback method for SOFA Python plugin, does nothing

@returns void **/
void DeformationTracking::updatePosition(double dt) {}

/**
Callback method for SOFA Python plugin, does nothing

@returns void **/
void DeformationTracking::draw(const sofa::core::visual::VisualParams *vparams) {}

/**
Callback method for SOFA Python plugin, triggers tracking

This method is called at every SOFA simulation step and serves as the entry point for the C++ part of the tracking methodology

@returns void **/
void DeformationTracking::handleEvent(sofa::core::objectmodel::Event *event) {
    if (dynamic_cast<sofa::simulation::AnimateBeginEvent *>(event)) {
        track();
    }
}

/**
Initializes the tracker, and all associated variables. Description added with inline comments

@returns void **/
void DeformationTracking::initTracker() {
#ifdef SENSOR_INPUT_ACTIVE                                                           /// Activate this block if using data directly from sensor
    sensor.initialize_d435();                                                        /// At this point, code is tested to work with Intel RealSense D435. It is initialized here
    vpCameraParameters color_param = sensor.get_intrinsics_color_d435();             /// Get itrinsics for color camera
    vpCameraParameters depth_param = sensor.get_intrinsics_depth_d435();             /// Get intrinsics for depth camera. The two intrinsics are not the same
    rTrack.initialize_camera(depth_param, depth_param, trackerInitFile.getValue());  /// Setting camera parameters for the rigid tracking class
#else
    rTrack.tracker_init_file = trackerInitFile.getValue();  /// In offline mode, reading camera intrinsic parameters from a file
#endif

#ifdef COMMUNICATION  /// DO NOT enable this block. Legacy code for communicating with physical parameter estimation module (in the STEPE approach for ICRA 2020)
    CommunicationUtilities::ParametersForEstimation paramEstim;
    CommunicationUtilities::ParametersForSimulation paramSimu;
    _dataTransmitter.readConfigFile("config.toml", paramSimu, paramEstim);
    _dataTransmitter.prepareLocalContainers(paramSimu._x0, paramSimu._xf, paramSimu._3DpositionApplicationPointForce, paramSimu._vForceKeytimes, paramSimu._vForces);
    Eigen::VectorXd p_est(2);
    _dataTransmitter.receivePhysicalParameters(p_est);
#endif
    counter = pcd_count = current_residual = node_count = 0;          /// Setting all counters to zero
    pcl::io::loadPolygonFile(objFileName.getValue().c_str(), model);  /// Loading visual mesh from disk
    prepSharedMesh();                                                 /// Initializing the shared mesh variable that enables us to communicate with SOFA Python
#ifdef COMMUNICATION
    _dataTransmitter.transmitInitialStateOfTheObject(shared_mesh);
#endif
    _log_("Initial state transmitted") mechanical_cloud = ocl.load_vtk_mesh(mechFileName.getValue().c_str());  /// Loading the mechanical mesh from disk
    ocl.get_transform(transformFileName.getValue().c_str(), transform);                                        /// Get initial object frame to camera frame transformation cTo from a file
    _log_("Initial transform: " << transform) nrm.initialize_rigid_tracking(tracker);                          /// Initialize rigid tracking variable

    std::cout << calibrationFile.getValue() << std::endl;
    ocl.read_calibration_file(I, calibrationFile.getValue(), mechanical_cloud->size());  /// Read the output of the calibration file (we term it "influence matrix" inside the code, which is equivalen to the $\frac{d\xi}{d u_C}$ variable in the IROS 2020 paper)
    jFem.inflMat = I;                                                                    /// Assigning the influence matrix to the object of the JacobianFEM class
    jFem.num_vertices = mechanical_cloud->size();                                        /// Setting number of vertices

    if (dataFolder.getValue().length() >= 1) {
        nrm.data_folder = dataFolder.getValue().c_str();                                                                      /// Setting the path to the data folder (path where the color/depth files are located)
        ocl.set_color_camera_params(F_x.getValue(), F_y.getValue(), C_x.getValue(), C_y.getValue(), 0.0f, 0.0f, 0.0f, 0.0f);  /// Setting color camera parameters
    } else {
        _log_("Data in PCD format not found, trying to look for color and depth images seperately")

            offline_data = ocl.check_offline_data(colorImageFolder.getValue(), depthDataFolder.getValue(), dataOffset.getValue());  /// This method checks the sanity of the color and depth files provided and populates the structure termed "offline_data_map"...
                                                                                                                                    /// ...This method also tries to synchronize the number of files in color and depth folder (if the no. of color images and depth images are not equal)

        if ((offline_data.color_files.size() > 1) && (offline_data.depth_files.size() > 1)) {
            /// By now, we are sure that the offline data provided is not insane and can be processed
            registration_required = true;
            _log_("Reading " << offline_data.color_files.size() << " color images and " << offline_data.depth_files.size() << " depth images, starting from frame #" << dataOffset.getValue())

                /// Setting the intrinsics in the "offline_data" struct
                nrm.color_data_folder = colorImageFolder.getValue();
            nrm.depth_data_folder = depthDataFolder.getValue();
            offline_data.color_Cx = colorC_x.getValue();
            offline_data.color_Cy = colorC_y.getValue();
            offline_data.color_Fx = colorF_x.getValue();
            offline_data.color_Fy = colorF_y.getValue();

            /// Continuing with assignment of intrinsics
            offline_data.depth_Cx = C_x.getValue();
            offline_data.depth_Cy = C_y.getValue();
            offline_data.depth_Fx = F_x.getValue();
            offline_data.depth_Fy = F_y.getValue();
            ocl.get_transform(depthToColorExtrinsicFile.getValue(), offline_data.depth2color_extrinsics);  /// This reads and initializes the depth to color camera extrinsic transformation matrix

            /// Some more house-keeping operations
            vpHomogeneousMatrix extrinsicMat;
            ocl.eigen_to_visp_4x4(offline_data.depth2color_extrinsics, extrinsicMat);
            _log_("Loaded extrinsic param: " << extrinsicMat) sensor.set_extrinsics_depth2color(extrinsicMat);
            _log_("Done with init of registration") ocl.set_color_camera_params(colorF_x.getValue(), colorF_y.getValue(), colorC_x.getValue(), colorC_y.getValue(), 0.0f, 0.0f, 0.0f, 0.0f);
            jFem.setData_map(offline_data);

        } else {
            _fatal_("Neither was a valid PCD data folder path provided, nor was color and depth image file paths valid. Exiting now." << endl
                                                                                                                                      << "Color image path provided was: " << colorImageFolder.getValue() << endl
                                                                                                                                      << "Depth image path provided was: " << depthDataFolder.getValue() << ", with data offset: " << dataOffset.getValue() << endl
                                                                                                                                      << "PCD path provided was: " << dataFolder.getValue() << endl)
        }
    }

    nrm.prev_centroid.x = 0.0f;
    nrm.prev_centroid.y = 0.0f;

    nrm.set_valid_index();
    jFem.set_file_num(pcd_count);
    jFem.set_pcd_count(0);

#ifdef SENSOR_INPUT_ACTIVE
    jFem.set_camera_parameters(depth_param.get_px(), depth_param.get_py(), depth_param.get_u0(), depth_param.get_v0());
#else
    jFem.set_camera_parameters(F_x.getValue(), F_y.getValue(), C_x.getValue(), C_y.getValue());
#endif

    jFem.set_debug_flag(debugFlag.getValue());                           /// If set to true, enables some debug output from JacobianFEM class
    jFem.set_additional_debug_folder(additionalDebugFolder.getValue());  /// Path to write additional debug images and data. Make sure this path exists before using this option (can be set from properties file, set debug level to 2)
    jFem.setInitialLambda(initLambda.getValue());                        /// Initial value of lambda for the Levenberg-Marquardt
    jFem.occlusion_init();                                               /// House-keeping value assignments
    jFem.setJacobian_displacement(jacobianForce.getValue());             /// Sets the magnitude of the displacement that the mechanical mesh vertex undergoes along all 3-axes while calibrating, i.e., the $\Delta X$ displacement along all three axes
                                                                         /// The nomenclature of the incoming variable 'jacobianForce' is slightly misleading, but this is because this variable can also be used to transmit maghitude of force in case of force based tracking
    robot_marker_initialized = true;

    /// House-keeping variables indicating that the next iteration would indeed be the first iteration of the tracker
    jFem.is_first_node = true;
    jFem.is_first_iteration = true;
    jFem.undef_mesh_initialized = false;
    jFem.setPrev_residual(0.0f);

    jFem.no_more_updates = false;

    ocl.set_camera_params(F_x.getValue(), F_y.getValue(), C_x.getValue(), C_y.getValue(), 0.0f, 0.0f, 0.0f, 0.0f);

    /// Prepping up the OpenCV windows for display
    namedWindow(OPENCV_WINDOW_1, WINDOW_AUTOSIZE);
    namedWindow(OPENCV_WINDOW_2, WINDOW_AUTOSIZE);
    moveWindow(OPENCV_WINDOW_2, 100, 0);

    if (fixedPoints.getValue().size() > 0) prepFixedConstraints();
    debug_folder_initialized = false;
    rigid_tracking_initialized = false;
}

int DeformationTrackingClass = sofa::core::RegisterObject("Component to track non-rigid deformation in objects.").add<DeformationTracking>();

SOFA_DECL_CLASS(DeformationTracking)

//} // namespace deform_tracking

//} // namespace component

//} // namespace sofa
