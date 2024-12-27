#include "DeformationTracking.h"

#include <sofa/core/ObjectFactory.h>

using namespace std;

namespace sofa::component::controller {

DeformationTracking::DeformationTracking()
    : objectModel(initData(&objectModel, "objectModel", "Object Model")),
      mechModel(initData(&mechModel, "mechModel", "Mechanical Model")),
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
      C_x(initData(&C_x, "C_x", "C_x")),
      C_y(initData(&C_y, "C_y", "C_y")),
      F_x(initData(&F_x, "F_x", "F_x")),
      F_y(initData(&F_y, "F_y", "F_y")),
      iterations(initData(&iterations, "iterations", "Maximum iterations allowed for the minimization")),
      simulationMessage(initData(&simulationMessage, "simulationMessage", "Message from SOFA simulation (front-end) to tracker (back-end)")),
      trackerMessage(initData(&trackerMessage, "trackerMessage", "Message from SOFA tracker (back-end) to simulation (front-end)")) {
    cout << "C++: " << "DeformationTracking" << endl;
    this->f_listening.setValue(true);
}

void DeformationTracking::incrementPcdCount() {
    cout << "C++: " << "incrementPcdCount" << endl;
    pcd_count++;
    jFem.set_pcd_count(pcd_count);
    // cout<<"incremented pcd"<<endl;
}

void DeformationTracking::formatList(sofa::type::vector<Vec3> simulationDump, vector<vector<Vec3>>& unformatted_meshes, bool jacobian_mesh) {
    cout << "C++: " << "formatList" << endl;
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

    for (int i = 0; i < simulationDump.size(); i++) {
        Vec3 v = simulationDump[i];

        if (count >= 0) {
            if ((v[0] == -100) && (v[1] = -100) && (v[2] = -100)) {
                count++;
            } else {
                Vec3 v_;
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

void DeformationTracking::matching() {
    cout << "C++: " << "matching" << endl;
    nrm.initialize(frame, transform, rTrack, tracker, configPath.getValue().c_str(), caoModelPath.getValue().c_str(), cMo, pcd_count);
    nrm.align_and_cluster(frame, mechanical_cloud, transform, model, rTrack, tracker, matched_points, indices, 0, pcd_count);

    if (matched_points.size() > 0) {
        trackerMessage.setValue("matched");
        sofa::type::vector<Vec4> list_values;

        for (int i = 0; i < matched_points.size(); i++) {
            cout << "C++: " << matched_points[i].x << "," << matched_points[i].y << "," << matched_points[i].z << "," << indices[i] << endl;
            Vec4 values;
            values[0] = indices[i];
            values[1] = matched_points[i].x;
            values[2] = matched_points[i].y;
            values[3] = matched_points[i].z;
            list_values.push_back(values);
        }

        forcePoints.setValue(list_values);
    } else {
        incrementPcdCount();
    }
}

void DeformationTracking::matching_static() {
    cout << "C++: " << "matching_static" << endl;
    double t = vpTime::measureTimeMicros();
    cout << "C++: " << "before loading: ";
    cout.setf(ios::fixed);
    cout << "C++: " << setprecision(0) << t << endl;
    nrm.initialize(frame, transform, rTrack, tracker, configPath.getValue().c_str(), caoModelPath.getValue().c_str(), cMo, pcd_count);
    t = vpTime::measureTimeMicros();
    cout << "C++: " << "after loading: ";
    cout.setf(ios::fixed);
    cout << "C++: " << setprecision(0) << t << endl;

    sofa::type::vector<Vec4> list_values;
    Vec4 values;
    values[0] = 28;
    values[1] = -0.028288;
    values[2] = -0.006f;
    values[3] = 0.142213f;

    // /*values[0] = 252;
    // values[1] = 0.0f;
    // values[2] = 20.0f;
    // values[3] = 0.0f;*/

    list_values.push_back(values);
    forcePoints.setValue(list_values);
    trackerMessage.setValue("matched");
}

void DeformationTracking::update() {
    cout << "C++: " << "update" << endl;
    vector<PolygonMesh> formatted_meshes;
    vector<vector<Vec3>> unformatted_meshes;
    formatList(objectModel.getValue(), unformatted_meshes, true);
    cout << "C++: " << "debug 1" << endl;
    nrm.format_deformed_polygons(model, unformatted_meshes, formatted_meshes);
    cout << "C++: " << "debug 2" << endl;

    PointCloud<PointXYZ>::Ptr frame_(new pcl::PointCloud<pcl::PointXYZ>);
    copyPointCloud(*frame, *frame_);
    cout << "C++: " << "debug 3" << endl;

    current_residual = jFem.compute_update(frame_, model, formatted_meshes, transform, increment);

    cout << "C++: " << "debug 4" << endl;
    vector<float> v;
    for (int i = 0; i < increment.rows(); i++) {
        cout << "C++: " << "inc: " << increment(i, 0) << endl;
        v.push_back(increment(i, 0));
    }
    update_tracker.setValue(v);
    trackerMessage.setValue("updated");
    counter++;
    cout << "C++: " << "debug 5" << endl;
}

void DeformationTracking::assimilate() {
    cout << "C++: " << "assimilate" << endl;
    vector<vector<Vec3>> unformatted_meshes;
    formatList(objectModel.getValue(), unformatted_meshes, false);
    nrm.update_polygon(model, unformatted_meshes);

#ifdef DEBUG_DUMP
    nrm.log_output(model, pcd_count, cMo, current_residual, dataFolder.getValue(), outputDirectory.getValue());
#endif

    if (counter > iterations.getValue()) {
        incrementPcdCount();
        counter = 0;
        double t = vpTime::measureTimeMicros();
        cout << "C++: " << "***PCD UPDATE***: ";
        cout.setf(ios::fixed);
        cout << "C++: " << setprecision(0) << t << endl;
        simulationMessage.setValue("ready");
        trackerMessage.setValue("ready");
    } else {
        trackerMessage.setValue("matched");
    }
}

/********************************Status Messages:********************************/
/*                                                                              */
/* 'ready': Simulation has just started/loaded a new PCD. Can do matching now.  */
/* 'applying_J': Simulation has received matched node list. Applying deformation*/
/*              for Jacobian estimation. Need to wait till the process finishes.*/
/* 'jacobian_ready': jacobian deformed meshes ready from FEM simulation         */
/* 'update_ready':  new mesh ready after application of update-force            */
/*                                                                              */
/********************************************************************************/
void DeformationTracking::track() {
    cout << "C++: " << "track" << endl;
    cout << "C++: " << "M: " << simulationMessage.getValue() << "," << trackerMessage.getValue() << endl;
    jFem.set_iteration(counter);
    if (simulationMessage.getValue() == "ready") {
        matching_static();
    } else if (simulationMessage.getValue() == "jacobian_ready") {
        update();
    } else if (simulationMessage.getValue() == "update_ready") {
        assimilate();
    } else if (!((simulationMessage.getValue() == "applying_J") || (simulationMessage.getValue() == "applying_update"))) {
        cout << "C++: " << "Invalid simulation message received. Exiting!" << endl;
        exit(0);
    }
}

DeformationTracking::~DeformationTracking() { cout << "C++: " << "~" << endl; }

void DeformationTracking::init() {
    cout << "C++: " << "init" << endl;
    initTracker();
}

void DeformationTracking::reinit() { cout << "C++: " << "reinit" << endl; }

void DeformationTracking::updatePosition(double dt) { cout << "C++: " << "updatePosition" << endl; }

void DeformationTracking::draw(const sofa::core::visual::VisualParams* vparams) {
    // cout << "C++: " << "draw" << endl;
}

void DeformationTracking::handleEvent(core::objectmodel::Event* event) {
    // cout << "C++: " << "handleEvent" << endl;
    if (dynamic_cast<sofa::simulation::AnimateBeginEvent*>(event)) {
        track();
    }
}

void DeformationTracking::initTracker() {
    cout << "C++: " << "initTracker" << endl;
    counter = pcd_count = current_residual = 0;
    OcclusionCheck ocl;
    pcl::io::loadPolygonFile(objFileName.getValue().c_str(), model);
    mechanical_cloud = ocl.load_vtk_mesh(mechFileName.getValue().c_str());
    ocl.get_transform(transformFileName.getValue().c_str(), transform);
    tracker = nrm.initialize_rigid_tracking();
    nrm.data_folder = dataFolder.getValue().c_str();
    jFem.set_file_num(pcd_count);
    jFem.set_pcd_count(0);
    jFem.set_camera_parameters(F_x.getValue(), F_y.getValue(), C_x.getValue(), C_y.getValue());
}

int DeformationTrackingClass = sofa::core::RegisterObject("Component to track non-rigid deformation in objects.").add<DeformationTracking>();

// SOFA_DECL_CLASS(DeformationTracking)
}  // namespace sofa::component::controller
