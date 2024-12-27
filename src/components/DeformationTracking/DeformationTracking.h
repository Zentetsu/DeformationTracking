
#include <sofa/component/controller/Controller.h>
#include <sofa/core/behavior/BaseController.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/SolidTypes.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/simulation/AnimateBeginEvent.h>
// // #include <sofa/simulation/Node.h>
// // #include <sofa/simulation/Simulation.h>
// #include <sofa/defaulttype/Vec.h>
// // #include <sofa/helper/Quater.h>
// // #include <sofa/simulation/Node.h>
// // #include <SofaSimulationTree/GNode.h>
// #include <sofa/core/objectmodel/MouseEvent.h>
// #include <SofaUserInteraction/Controller.h>
// // #include <sofa/core/BehaviorModel.h>
// // #include <SofaUserInteraction/Controller.h>
// #include <sofa/defaulttype/Quat.h>
// #include <sofa/core/DataEngine.h>
// #include <sofa/core/objectmodel/DataFileName.h>
// #include <sofa/defaulttype/RigidTypes.h>

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

#include <cstring>
#include <iostream>
#include <list>
#include <sstream>
#include <vector>

#include "JacobianFEM.h"
#include "NonRigidMatching.h"
#include "config.h"

namespace sofa::component::controller {

using namespace sofa::defaulttype;
using core::objectmodel::Data;
using sofa::component::controller::Controller;

/**
 * This BehaviorModel does nothing but contain a custom data widget.
 */
class DeformationTracking : public Controller {
   public:
    SOFA_CLASS(DeformationTracking, Controller);

    typedef sofa::type::Vec<3, double> Vec3;
    typedef sofa::type::Vec<4, double> Vec4;

    void init();
    virtual void reinit();
    void updatePosition(double dt);
    virtual void track();
    virtual void initTracker();

    /****Data Variables****/
    Data<sofa::type::vector<Vec3>> objectModel;
    Data<sofa::type::vector<Vec3>> mechModel;
    Data<sofa::type::vector<Vec3>> jacobianUp;
    Data<sofa::type::vector<Vec3>> jacobianDown;

    Data<sofa::type::vector<Vec4>> forcePoints;

    Data<sofa::type::vector<float>> update_tracker;

    Data<std::string> objFileName;
    Data<std::string> mechFileName;
    Data<std::string> transformFileName;
    Data<std::string> dataFolder;
    Data<std::string> outputDirectory;
    Data<std::string> configPath;
    Data<std::string> caoModelPath;
    Data<float> C_x;
    Data<float> C_y;
    Data<float> F_x;
    Data<float> F_y;

    Data<float> iterations;

    Data<std::string> simulationMessage;
    Data<std::string> trackerMessage;

    /****Data Variables****/
    void draw(const sofa::core::visual::VisualParams* vparams);
    void handleEvent(core::objectmodel::Event* event);

   protected:
    pcl::PolygonMesh model;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr mechanical_cloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr frame;
    Eigen::Matrix4f transform;
    std::vector<pcl::PointXYZ> matched_points;
    std::vector<int> indices;
    Eigen::MatrixXd increment;

    RigidTracking rTrack;
    vpMbGenericTracker tracker;

    DeformationTracking();
    virtual ~DeformationTracking();

    int counter;
    int pcd_count;
    NonRigidMatching nrm;
    vpHomogeneousMatrix cMo;
    JacobianFEM jFem;

    double current_residual;

    std::vector<Vec3> x_;
    std::vector<Vec3> _x;
    std::vector<Vec3> y_;
    std::vector<Vec3> _y;
    std::vector<Vec3> z_;
    std::vector<Vec3> _z;

    void incrementPcdCount();
    void matching();
    void matching_static();
    void update();
    void assimilate();
    void formatList(sofa::type::vector<Vec3> simulationDump, std::vector<std::vector<Vec3>>& unformatted_meshes, bool jacobian_mesh);
};

}  // namespace sofa::component::controller