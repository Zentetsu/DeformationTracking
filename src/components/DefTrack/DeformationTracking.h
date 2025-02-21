#ifndef DEFORMATIONTRACKING_DEFORMATIONTRACKING_H
#define DEFORMATIONTRACKING_DEFORMATIONTRACKING_H

#include <sofa/component/controller/Controller.h>
#include <sofa/core/BehaviorModel.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/behavior/BaseController.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/objectmodel/MouseEvent.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/SolidTypes.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/Simulation.h>
// #include <sofa/defaulttype/Vec.h>
// #include <SofaSimulationTree/GNode.h>
// #include <SofaUserInteraction/Controller.h>

// #include <sofa/defaulttype/Quat.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/io/obj_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>

// #include <pcl/io/io.h> // need to veryfy if this is needed
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/point_types.h>
#include <string.h>
#include <vtkActor.h>
#include <vtkCallbackCommand.h>
#include <vtkCellArray.h>
#include <vtkCommand.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkMath.h>
#include <vtkOBBTree.h>
#include <vtkPLYReader.h>
#include <vtkPointSource.h>
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
#include <vtkVersion.h>
#include <vtkWidgetEvent.h>
#include <vtkWidgetEventTranslator.h>

// #include <cstring>
#include <iostream>
#include <list>
#include <sstream>
#include <vector>

#ifdef COMMUNICATION
#include <DataTransmitter.h>
#include <common.h>
#endif

#include "JacobianFEM.h"
#include "NonRigidMatching.h"
#include "Sensor.h"
#include "config.h"

// namespace sofa
//{

// namespace component
//{

// namespace controller
//{

// using namespace sofa::defaulttype;
// using core::objectmodel::Data;

/**
 * This BehaviorModel does nothing but contain a custom data widget.
 */
class DeformationTracking : public sofa::component::controller::Controller {
   public:
    SOFA_CLASS(DeformationTracking, sofa::component::controller::Controller);

    virtual void init();
    virtual void reinit();
    void updatePosition(double dt);
    virtual void track();
    virtual void initTracker();

    /****Data Variables****/
    sofa::Data<sofa::type::vector<sofa::type::Vec3>> objectModel;
    sofa::Data<sofa::type::vector<sofa::type::Vec3>> mechModel;
    sofa::Data<sofa::type::vector<int>> fixedPoints;
    sofa::Data<sofa::type::vector<int>> activePoints;

    sofa::Data<sofa::type::vector<sofa::type::Vec3>> jacobianUp;
    sofa::Data<sofa::type::vector<sofa::type::Vec3>> jacobianDown;

    sofa::Data<sofa::type::vector<sofa::type::Vec4>> forcePoints;

    sofa::Data<sofa::type::vector<float>> update_tracker;

    sofa::Data<string> objFileName;
    sofa::Data<string> mechFileName;
    sofa::Data<string> transformFileName;
    sofa::Data<string> dataFolder;
    sofa::Data<string> outputDirectory;
    sofa::Data<string> configPath;
    sofa::Data<string> caoModelPath;
    sofa::Data<string> trackerInitFile;
    sofa::Data<string> calibrationFile;
    sofa::Data<float> C_x;
    sofa::Data<float> C_y;
    sofa::Data<float> F_x;
    sofa::Data<float> F_y;
    sofa::Data<float> jacobianForce;

    sofa::Data<float> initLambda;
    sofa::Data<int> debugFlag;
    sofa::Data<string> additionalDebugFolder;

    std::vector<int> list_pts;

    sofa::Data<float> iterations;

    sofa::Data<string> simulationMessage;
    sofa::Data<string> trackerMessage;

    sofa::Data<string> colorImageFolder;
    sofa::Data<string> depthDataFolder;

    sofa::Data<string> depthToColorExtrinsicFile;

    sofa::Data<int> dataOffset;
    sofa::Data<float> frequency;
    sofa::Data<float> colorC_x;
    sofa::Data<float> colorC_y;
    sofa::Data<float> colorF_x;
    sofa::Data<float> colorF_y;
    /****Data Variables****/

    std::vector<PolygonMesh> formatted_meshes;

    void draw(const sofa::core::visual::VisualParams *vparams);
    void handleEvent(sofa::core::objectmodel::Event *event);

    vpMbGenericTracker tracker;

   protected:
    EigenMatrix I;

    std::vector<std::vector<double>> correspond;
    EigenMatrix residual;
    std::vector<int> active_indices;

    int clustering_mode;

    bool is_cloud_transformed;

    PolygonMesh model;
    PointCloud<PointXYZ>::Ptr cloud;
    PointCloud<PointXYZ>::Ptr mechanical_cloud;
    PointCloud<PointXYZRGB>::Ptr frame;
    cv::Mat image;
    cv::Mat depth_image;
    cv::Mat colored_depth;
    cv::Mat prev_colored_depth;
    cv::Mat first_colored_depth;
    bool is_first_initialized;

    Eigen::Matrix4f transform;
    std::vector<PointXYZ> matched_points;
    std::vector<int> indices;
    EigenMatrix increment;
    vpColVector shared_mesh;
    vpColVector robotObservation;
    int reserved_counter_1;

    offline_data_map offline_data;

    OcclusionCheck ocl;

    RigidTracking rTrack;

    DeformationTracking();
    virtual ~DeformationTracking();

    int counter;
    int pcd_count;
    int node_count;
    NonRigidMatching nrm;
    vpHomogeneousMatrix cMo;
    JacobianFEM jFem;
    Sensor sensor;

    double current_residual;
    bool debug_folder_initialized;
    bool rigid_tracking_initialized;
    string image_debug_path;
    bool registration_required;
    bool load_new_data;

    bool robot_marker_initialized;
    vpHomogeneousMatrix cMr;

    std::vector<sofa::type::Vec3> x_;
    std::vector<sofa::type::Vec3> _x;
    std::vector<sofa::type::Vec3> y_;
    std::vector<sofa::type::Vec3> _y;
    std::vector<sofa::type::Vec3> z_;
    std::vector<sofa::type::Vec3> _z;

    std::vector<int> fixed_constraints;

    std::vector<int> prev_control_points;

    double prev_time;

#ifdef COMMUNICATION
    DataTransmitter _dataTransmitter;  // data_transmission
#endif

    mesh_map visible_mesh;

    void incrementPcdCount();
    void matching();
    void update();
    void assimilate();
    void prepSharedMesh();
    void formatList(const sofa::type::vector<sofa::type::Vec3> &simulationDump, std::vector<std::vector<sofa::type::Vec3>> &unformatted_meshes, bool jacobian_mesh);
    void prepFixedConstraints();
    void seekNextData();
    void updateMechanicalModel();
};

//} // namespace behaviormodel

//} // namespace component

//} // namespace sofa

#endif
