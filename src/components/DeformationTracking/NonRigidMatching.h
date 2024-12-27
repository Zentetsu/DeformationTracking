#ifndef NONRIGIDMATCHING_H
#define NONRIGIDMATCHING_H

#include <sofa/component/controller/Controller.h>
#include <sofa/core/behavior/BaseController.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/SolidTypes.h>
#include <sofa/defaulttype/VecTypes.h>

#include <iostream>
#include <list>
#include <sstream>
#include <vector>

#include "OcclusionCheck.h"
#include "RigidTracking.h"

class NonRigidMatching {
   public:
    typedef sofa::type::Vec<3, double> Vec3;
    typedef sofa::type::Vec<4, double> Vec4;

    void align_and_cluster(PointCloud<PointXYZRGB>::Ptr& frame, PointCloud<PointXYZ>::Ptr& mechanical_mesh_points, Matrix4f& transform_init, PolygonMesh& model, RigidTracking& rTrack,
                           vpMbGenericTracker& tracker, vector<PointXYZ>& matched_points, vector<int>& indices, int status, int count);
    vpMbGenericTracker initialize_rigid_tracking();
    void initialize(PointCloud<PointXYZRGB>::Ptr& frame_, Matrix4f& transform_init, RigidTracking& rTrack, vpMbGenericTracker& tracker, string config_path, string cao_model_path,
                    vpHomogeneousMatrix& cMo, int pcd_count);
    string data_folder;
    void format_deformed_polygons(PolygonMesh model, vector<vector<Vec3>> deformed_meshes, vector<PolygonMesh>& formatted_meshes);
    void update_polygon(PolygonMesh& model, vector<vector<Vec3>> deformed_mesh);

    // #ifdef DEBUG_DUMP
    void log_output(PolygonMesh& model, int frame_num, vpHomogeneousMatrix& cMo, double residual, string data_path, string opfilepath);
    // #endif

   protected:
    void read_frame(PointCloud<PointXYZRGB>::Ptr& frame, int count);
    OcclusionCheck ocl;
};

#endif
