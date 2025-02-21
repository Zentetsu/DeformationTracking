#include "JacobianFEM.h"

vector<std::vector<double>> correspondence;
vector<std::vector<double>> correspondence_photo;
vector<PointXYZ> correspondence_barycentric_photo;
vector<float> correspondence_barycentric_transport;
vector<point_2d> correspondence_img_pt_photo;

vector<std::vector<std::vector<double>>> set_of_model_list;

int file_num_;

OcclusionCheck ocl;

float initLambda;

vpMatrix R;
vpMatrix J_;

vpColVector W_gnvr;
vpMatrix eyeJ;

vpMatrix AtWA;
vpMatrix lm;
vpMatrix v;

vpColVector W;
vpMatrix W_mat;

pcl::PointXYZ normal_cross_prod;

pcl::PointXYZ A_;
pcl::PointXYZ B_;

int iter_debug = 0;

pcl::PointXYZ out_vec;

#define NB_THREADS_RESIDUAL 1

const float inlier_threshold = 2.5f;  // Distance threshold to identify inliers with homography check
const float nn_match_ratio = 0.8f;    // Nearest neighbor matching ratio

bool are_points_same(const pcl::PointXYZ &a, const pcl::PointXYZ &b) {
    if ((a.x == b.x) && (a.y == b.y) && (a.z == b.z)) {
        return true;
    } else {
        return false;
    }
}

double norm_(PointXYZ &a) { return sqrt((a.x * a.x) + (a.y * a.y) + (a.z * a.z)); }

// Normal is being treated as a point
// WARNING: This method will not work if by any chance the centroid of the object lies outside the object. Careful with future experiments!

bool JacobianFEM::normal_estimation(const pcl::PointXYZ &a, const pcl::PointXYZ &b, const pcl::PointXYZ &c, const pcl::PointXYZ &centroid, pcl::PointXYZ &normal, bool should_debug) {
    if (are_points_same(a, b) || are_points_same(b, c) || are_points_same(a, c)) {
        if (should_debug) {
            cerr << "JacobianFEM::normal_estimation: Degenerate Triangle Detected. This is not expected, and is probably being caused by some issue in the FEM simulation. Please check." << std::endl;
        }
        pcl::PointXYZ normal_degenerate;
        normal_degenerate.x = 0.0f;
        normal_degenerate.y = 0.0f;
        normal_degenerate.z = 0.0f;

        return false;
    }

    A_.x = c.x - a.x;
    A_.y = c.y - a.y;
    A_.z = c.z - a.z;
    ////////////std::////cout<<c.x<<","<<a.x<<","<<A_.x<<","<<A_.y<<","<<A_.z<<std::endl;

    B_.x = b.x - a.x;
    B_.y = b.y - a.y;
    B_.z = b.z - a.z;

    out_vec.x = a.x - centroid.x;
    out_vec.y = a.y - centroid.y;
    out_vec.z = a.z - centroid.z;

    double norm_out = norm_(out_vec);

    out_vec.x /= norm_out;
    out_vec.y /= norm_out;
    out_vec.z /= norm_out;

    ocl.cross_product(A_, B_, normal);

    double sign = ocl.dot_product_raw(normal, out_vec);

    if (sign < 0) {
        normal.x = -normal.x;
        normal.y = -normal.y;
        normal.z = -normal.z;
    }

    return true;
}

inline double round(double val) {
    if (val < 0) return ceil(val - 0.5);
    return floor(val + 0.5);
}

float min_val(float a, float b, float c) { return a < b ? (a < c ? a : c) : (b < c ? b : c); }

float max_val(float a, float b, float c) { return a > b ? (a > c ? a : c) : (b > c ? b : c); }

void assign_weights(std::vector<double> &residual) {
    vpColVector error(residual.size());
    int i;
    int n = residual.size();

#pragma omp parallel for
    for (i = 0; i < n; i++) {
        error[i] = residual[i];
    }

    vpColVector weight(n);
    vpRobust robust(n);

    robust.setThreshold(0.00001f);
    robust.MEstimator(vpRobust::TUKEY, error, weight);  // it is also possible to use 'vpRobust::HUBER'
}

EigenMatrix assign_weights_to_matrix(EigenMatrix &residual) {
    EigenMatrix weights = EigenMatrix::Zero(residual.rows(), residual.rows());
    vpColVector error(residual.rows());

    int i;
    int n = error.size();

    memcpy(error.data, residual.data(), n * sizeof(double));

    vpColVector weight(error.size());
    vpRobust robust(error.size());

    robust.setThreshold(0.00001f);
    robust.MEstimator(vpRobust::TUKEY, error, weight);  // it is also possible to use 'vpRobust::HUBER'

    for (i = 0; i < n; i++) {
        weights(i, i) = weight[i];
    }

    return weights;
}

void assign_weights_to_matrix_visp(EigenMatrix &residual, vpColVector &W, float threshold) {
    vpColVector error(residual.rows());

    int n = error.size();

    memcpy(error.data, residual.data(), n * sizeof(double));

    W = 1;

    vpRobust robust(n);

    robust.setIteration(0);
    robust.setThreshold(threshold);
    robust.MEstimator(vpRobust::TUKEY, error, W);  // it is also possible to use 'vpRobust::HUBER'
}

inline void pcl_to_visp_3Dpoint(PointXYZ &pcl_P, vpPoint &visp_P) {
    visp_P.set_X(pcl_P.x);
    visp_P.set_Y(pcl_P.y);
    visp_P.set_Z(pcl_P.z);
}

float JacobianFEM::get_inter_triangular_distance(PointXYZ &A1, PointXYZ &A2, PointXYZ &A3, PointXYZ &B1, PointXYZ &B2, PointXYZ &B3) {
    return EUCLEDIAN_DIST(A1.x, A1.y, A1.z, B1.x, B1.y, B1.z) + EUCLEDIAN_DIST(A2.x, A2.y, A2.z, B2.x, B2.y, B2.z) + EUCLEDIAN_DIST(A3.x, A3.y, A3.z, B3.x, B3.y, B3.z);
}

void JacobianFEM::debug_residual(vector<Eigen::Vector4f> &residual_map, string &debug_path, string &file_name, int iteration) {
    float sum = 0.0f;
    float min = 99999999.0f;
    float max = -99999999.0f;

    for (int i = 0; i < residual_map.size(); i++) {
        sum += residual_map[i](3);
        residual_map[i](3) < min ? (min = residual_map[i](3)) : 0;
        residual_map[i](3) > max ? (max = residual_map[i](3)) : 0;
    }

    sum /= (float)residual_map.size();

    PointCloud<PointXYZRGB> cloud;
    cloud.width = residual_map.size();
    cloud.height = 1;
    cloud.is_dense = false;
    cloud.points.resize(cloud.width * cloud.height);

    for (size_t i = 0; i < residual_map.size(); ++i) {
        cloud.points[i].x = residual_map[i](0);
        cloud.points[i].y = residual_map[i](1);
        cloud.points[i].z = residual_map[i](2);
        cloud.points[i].r = 0;
        cloud.points[i].g = 0;
        cloud.points[i].b = 0;
        int color_val = (int)(((residual_map[i](3) - min) / (max - min)) * 255.0f);
        cloud.points[i].r = color_val;
        cloud.points[i].b = 255 - color_val;
    }

    debugCloudList.push_back(cloud);
    debugCloudLabel.push_back(file_name + to_string(iteration) + '_' + to_string(sum));
}

inline double get_point_to_plane_error(float pt_x, float pt_y, float pt_z, float model_X, float model_Y, float model_Z, float nX, float nY, float nZ) { return (nX * (pt_x - model_X)) + (nY * (pt_y - model_Y)) + (nZ * (pt_z - model_Z)); }

inline double get_point_to_plane_error_rigid(float pt_x, float pt_y, float pt_z, vpPlane &plane) { return ((plane.getA() * (pt_x)) + (plane.getB() * (pt_y)) + (plane.getC() * (pt_z)) + (plane.getD())); }

bool compare_projection_5d(point_5d lhs, point_5d rhs) {
    if ((lhs.u == rhs.u) && (lhs.v = rhs.v))
        return true;
    else
        return false;
}

void JacobianFEM::get_residual_simple(PolygonMesh poly_mesh, PointCloud<PointXYZRGB>::Ptr &pcl_cloud, vpHomogeneousMatrix &cMo, int iteration, vector<vpColVector> &err_map, bool create_error_map, residual_statistics &res_stats, mesh_map &visible_mesh, vector<vector<double>> &correspond,
                                      EigenMatrix &residual, cv::Mat &geometric_error_map) {
    int iteration_rigid = 0;

    Eigen::Matrix4f transform;
    ocl.visp_to_eigen_4x4(cMo, transform);
    ocl.transformPolygonMesh(poly_mesh, transform);  //  <----- the transformed model file is here
    pcl::PointCloud<pcl::PointXYZ>::Ptr vertices(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(poly_mesh.cloud, *vertices);  // vertices of the base model

    vector<vpPlane> correspond_model;

    visible_mesh = ocl.get_visibility_normal(poly_mesh);

    Eigen::Vector4f xyz_centroid;
    pcl::compute3DCentroid(*vertices, xyz_centroid);

    pcl::PointXYZ centroid;
    centroid.x = xyz_centroid[0];
    centroid.y = xyz_centroid[1];
    centroid.z = xyz_centroid[2];

    int K = 1;
    float dist = 0.25;

    std::vector<pcl::PointXYZ> visible_list;
    std::vector<pcl::PointXYZ> corresponding_model_list;

    std::vector<double> _residual;

    std::vector<pcl::Vertices, std::allocator<pcl::Vertices>>::iterator face;
    pcl::PointCloud<pcl::PointXYZ>::Ptr vertices_visible(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(visible_mesh.mesh.cloud, *vertices_visible);

    visible_mesh.visible_vertices = vertices_visible;

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(vertices);

    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(dist);

    Eigen::Vector3f vecNorm;

    for (face = visible_mesh.mesh.polygons.begin(); face != visible_mesh.mesh.polygons.end(); ++face) {
        for (int i = 0; i < 3; i++) {
            unsigned int v = face->vertices[i];
            pcl::PointXYZ p = vertices_visible->points.at(v);
            visible_mesh.mesh_points.push_back(p);
            visible_list.push_back(p);

            if (kdtree.nearestKSearch(p, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
                for (size_t j = 0; j < pointIdxNKNSearch.size(); ++j) {
                    pcl::PointXYZ pt;
                    pt.x = vertices->points[pointIdxNKNSearch[j]].x;
                    pt.y = vertices->points[pointIdxNKNSearch[j]].y;
                    pt.z = vertices->points[pointIdxNKNSearch[j]].z;

                    corresponding_model_list.push_back(pt);
                }
            }
        }
    }

    correspond.clear();
    if (create_error_map) err_map.clear();
    bool isDone = false;

    visible_mesh.roi.min_x = 10000.0f;
    visible_mesh.roi.min_y = 10000.0f;
    visible_mesh.roi.max_x = 0.0f;
    visible_mesh.roi.max_y = 0.0f;

    vector<point_5d> projected_points;

    cv::Mat geometric_error_mask;
    geometric_error_mask = cv::Mat::zeros(cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), CV_32FC1);
    PointXYZ geometric_buffer_point;
    point_2d geometric_buffer_image_pt;

    if (pcl_cloud->isOrganized()) {
        vpPoint A;
        vpColVector n_visp(3);
        point_2d a;
        point_2d b;
        point_2d c;
        point_2d target;
        for (int k = 0; (k < visible_list.size()) && (!isDone); k += 3) {
            double res = 0.0f;

            pcl::PointXYZ normal;
            vecNorm = visible_mesh.visible_normal[k / 3];

            normal.x = vecNorm[0];
            normal.y = vecNorm[1];
            normal.z = vecNorm[2];

            pcl_to_visp_3Dpoint(corresponding_model_list[k], A);

            n_visp[0] = normal.x;
            n_visp[1] = normal.y;
            n_visp[2] = normal.z;

            vpPlane model_plane(A, n_visp);

            a = ocl.projection(corresponding_model_list[k]);     /************************************************************************/
            b = ocl.projection(corresponding_model_list[k + 1]); /*These three points are supposed to represent a single triangular plane*/
            c = ocl.projection(corresponding_model_list[k + 2]); /************************************************************************/

            point_5d aA, bB, cC;

            /// d2c rotation needed
            aA.u = a.x;
            aA.v = a.y;
            aA.X = corresponding_model_list[k + 0].x;
            aA.Y = corresponding_model_list[k + 0].y;
            aA.Z = corresponding_model_list[k + 0].z;
            bB.u = b.x;
            bB.v = b.y;
            bB.X = corresponding_model_list[k + 1].x;
            bB.Y = corresponding_model_list[k + 1].y;
            bB.Z = corresponding_model_list[k + 1].z;
            cC.u = c.x;
            cC.v = c.y;
            cC.X = corresponding_model_list[k + 2].x;
            cC.Y = corresponding_model_list[k + 2].y;
            cC.Z = corresponding_model_list[k + 2].z;

            projected_points.push_back(aA);
            projected_points.push_back(bB);
            projected_points.push_back(cC);

            float min_x, min_y, max_x, max_y;

            min_x = min_val(a.x, b.x, c.x);
            max_x = max_val(a.x, b.x, c.x);
            min_y = min_val(a.y, b.y, c.y);
            max_y = max_val(a.y, b.y, c.y);

            (visible_mesh.roi.min_x >= min_x) ? visible_mesh.roi.min_x = min_x : 0;
            (visible_mesh.roi.min_y >= min_y) ? visible_mesh.roi.min_y = min_y : 0;
            (visible_mesh.roi.max_y < max_y) ? visible_mesh.roi.max_y = max_y : 0;
            (visible_mesh.roi.max_x < max_x) ? visible_mesh.roi.max_x = max_x : 0;

            float area = ocl.getTriangleArea(a, b, c);

            if ((min_x > 0) && (min_y > 0) && (max_x < IMAGE_WIDTH) && (max_y < IMAGE_HEIGHT) && (area > TRIANGLE_AREA_THRESHOLD)) {
                for (int i = min_x; i < max_x; i++) {
                    for (int j = min_y; j < max_y; j++) {
                        pcl::PointXYZRGB pt = pcl_cloud->at(i, j);

                        if (((pt.x != 0) || (pt.y != 0) || (pt.z != 0))) {
                            if ((i > min_x) && (i < max_x) && (j > min_y) && (j < max_y)) {
                                target.x = i;
                                target.y = j;

                                if (ocl.point_in_triangle(target, a, b, c)) {
                                    res = get_point_to_plane_error_rigid(pt.x, pt.y, pt.z, model_plane);
                                    if (fabs(res) < 0.05f) {
                                        geometric_buffer_point.x = pt.x;
                                        geometric_buffer_point.y = pt.y;
                                        geometric_buffer_point.z = pt.z;
                                        geometric_buffer_image_pt = ocl.projection_raw(geometric_buffer_point, data_map.color_Fx, data_map.color_Fy, data_map.color_Cx, data_map.color_Cy);
                                        geometric_error_mask.at<float>(geometric_buffer_image_pt.y, geometric_buffer_image_pt.x) = res;

                                        if ((create_error_map) && (iteration_rigid == 0)) {
                                            vpColVector v(4);
                                            v[0] = pt.x;
                                            v[1] = pt.y;
                                            v[2] = pt.z;
                                            v[3] = res;
                                            err_map.push_back(v);
                                        }

                                    } else {
                                        if ((create_error_map) && (iteration_rigid == 0)) {
                                            vpColVector v(4);
                                            v[0] = pt.x;
                                            v[1] = pt.y;
                                            v[2] = pt.z;
                                            v[3] = 0.0f;
                                            err_map.push_back(v);
                                        }
                                    }

                                    _residual.push_back(res);
                                    if (iteration_rigid == 0) {
                                        std::vector<double> corr;
                                        corr.push_back(k);
                                        corr.push_back(pt.x);
                                        corr.push_back(pt.y);
                                        corr.push_back(pt.z);
                                        correspond.push_back(corr);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    sort(projected_points.begin(), projected_points.end(), less_than_point_5d());
    projected_points.erase(unique(projected_points.begin(), projected_points.end(), compare_projection_5d), projected_points.end());
    visible_mesh.projected_points_5d = projected_points;

#ifdef CONTINIOUS_RIGID_TRACKING

    float prev_res = 0.0f;
    double mean_depth = 0.0f;

    if ((iteration >= VISP2CUSTOM_TRACKING_SWITCH)) {
        int nullifier = 0;  /// set to 1 -> to enable sparse pt.-to-pt. tracking, set 0 -> to disable

        EigenMatrix L_rigid(_residual.size() + (nullifier * sparse_correspondence.keypoints1.size()), 6);
        EigenMatrix error_rigid(_residual.size() + (nullifier * sparse_correspondence.keypoints1.size()), 1);
        vpColVector update;
        vpHomogeneousMatrix delta_cMo;
        vpHomogeneousMatrix prev_cMo;
        prev_cMo = cMo;
        EigenMatrix point_r(4, 1);
        EigenMatrix eigen_update(4, 4);
        float init_lambda_rigid = 0.5f;
        vpPlane model_plane;

        for (int i = 0; i < MAX_RIGID_TRACKING_ITERATION; i++) {
            double X;
            double nX;
            double Y;
            double nY;
            double Z;
            double nZ;
            if (i == 0) {
                for (int j = 0; j < _residual.size(); j++) {
                    error_rigid(j, 0) = _residual[j];
                    mean_depth += _residual[j];

                    vecNorm = visible_mesh.visible_normal[correspond[j][0] / 3];
                    nX = vecNorm[0];
                    nY = vecNorm[1];
                    nZ = vecNorm[2];

                    X = correspond[j][1];
                    Y = correspond[j][2];
                    Z = correspond[j][3];

                    L_rigid(j, 0) = nX;
                    L_rigid(j, 1) = nY;
                    L_rigid(j, 2) = nZ;
                    L_rigid(j, 3) = ((nZ * Y) - (nY * Z));
                    L_rigid(j, 4) = ((nX * Z) - (nZ * X));
                    L_rigid(j, 5) = ((nY * X) - (nX * Y));
                }
            }

            mean_depth /= (float)_residual.size();

            int offset;

            offset = 0;

            float alpha = 0.6f;
#ifdef WEIGHT_VISUALIZATION
            vpColVector W = gauss_newton_visp_rigid(error_rigid, L_rigid, update, init_lambda_rigid, alpha, offset);

            for (int m = 0; m < err_map.size(); m++) {
                err_map[m][3] = W[m];
            }
#else
            gauss_newton_visp_rigid(error_rigid, L_rigid, update, init_lambda_rigid, alpha, offset);
#endif
            if (prev_res < error_rigid.colwise().norm()[0]) {
                init_lambda_rigid /= 3.0f;
            } else {
                init_lambda_rigid *= 1.3f;
            }
            prev_res = error_rigid.colwise().norm()[0];
            delta_cMo = vpExponentialMap::direct(update);

            for (int j = 0; j < _residual.size(); j++) {
                vecNorm = visible_mesh.visible_normal[correspond[j][0] / 3];

                eigen_update(0, 0) = delta_cMo[0][0];
                eigen_update(0, 1) = delta_cMo[0][1];
                eigen_update(0, 2) = delta_cMo[0][2];
                eigen_update(0, 3) = delta_cMo[0][3];
                eigen_update(1, 0) = delta_cMo[1][0];
                eigen_update(1, 1) = delta_cMo[1][1];
                eigen_update(1, 2) = delta_cMo[1][2];
                eigen_update(1, 3) = delta_cMo[1][3];
                eigen_update(2, 0) = delta_cMo[2][0];
                eigen_update(2, 1) = delta_cMo[2][1];
                eigen_update(2, 2) = delta_cMo[2][2];
                eigen_update(2, 3) = delta_cMo[2][3];
                eigen_update(3, 0) = delta_cMo[3][0];
                eigen_update(3, 1) = delta_cMo[3][1];
                eigen_update(3, 2) = delta_cMo[3][2];
                eigen_update(3, 3) = delta_cMo[3][3];

                point_r(0, 0) = correspond[j][1];
                point_r(1, 0) = correspond[j][2];
                point_r(2, 0) = correspond[j][3];
                point_r(3, 0) = 1.0f;

                point_r = eigen_update * point_r;

                correspond[j][1] = point_r(0, 0);
                correspond[j][2] = point_r(1, 0);
                correspond[j][3] = point_r(2, 0);

                model_plane = correspond_model[j];

                _residual[j] = get_point_to_plane_error_rigid(correspond[j][1], correspond[j][2], correspond[j][3], model_plane);
            }
        }
    }
#endif
    EigenMatrix residual_matrix(_residual.size(), 1);

    float min = 99999.99f;
    float max = -99999.99f;
    float avg = 0.0f;
    float avg_unweighted = 0.0f;

    unsigned int size = _residual.size();

    int i;
    for (i = 0; i < size; i++) {
        residual_matrix(i, 0) = _residual[i];
    }

    for (i = 0; i < size; i++) {
        avg_unweighted += err_map[i][3];
        err_map[i][3] *= 1.0f;
        if (err_map[i][3] < min) {
            min = err_map[i][3];
        }
        if (err_map[i][3] > max) {
            max = err_map[i][3];
        }
        avg += err_map[i][3];
    }

    cv::normalize(geometric_error_mask, geometric_error_mask, 0, 1.0, cv::NORM_MINMAX);
    geometric_error_map = geometric_error_mask;

    avg /= ((float)size);
    avg_unweighted /= ((float)size);

    res_stats.minimum = min;
    res_stats.maximum = max;
    res_stats.average = avg;
    res_stats.unweighted_average = avg_unweighted;

    residual = residual_matrix;
}

EigenMatrix JacobianFEM::get_residual_depth(pcl::PolygonMesh &poly_mesh, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pcl_cloud, bool is_first, std::string file_name, mesh_map &mesh, int iteration) {
    pcl::PCLPointCloud2 blob = poly_mesh.cloud;

    _log_("Obtaining depth residual for: " << file_name);

    pcl::PointCloud<pcl::PointXYZ>::Ptr vertices(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(blob, *vertices);

    Eigen::Vector4f xyz_centroid;
    pcl::compute3DCentroid(*vertices, xyz_centroid);

    std::vector<pcl::PointXYZ> visible_list;
    std::vector<pcl::PointXYZ> corresponding_model_list;

    std::vector<double> residual;

    std::vector<pcl::Vertices, std::allocator<pcl::Vertices>>::iterator face;
    pcl::PointCloud<pcl::PointXYZ>::Ptr vertices_visible(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(mesh.mesh.cloud, *vertices_visible);

    int num_points = 0;
    int poly_count = 0;

    PointXYZ _p;
    PointXYZ _P;
    visible_full_map.clear();
    for (int i = 0; i < vertices_visible->size(); i++) {
        _p = vertices_visible->points.at(i);
        bool isFound = false;
        for (int j = 0; ((j < vertices->size()) && (!isFound)); j++) {
            _P = vertices->points.at(j);

            if ((sqrt(pow((_p.x - _P.x), 2) + pow((_p.y - _P.y), 2) + pow((_p.z - _P.z), 2))) < FLOAT_ZERO) {
                isFound = true;
                visible_full_map.push_back(j);
            }
        }
    }

    for (face = mesh.mesh.polygons.begin(); face != mesh.mesh.polygons.end(); ++face, poly_count++) {
        {
            for (int i = 0; i < 3; i++) {
                unsigned int v = face->vertices[i];
                pcl::PointXYZ p = vertices_visible->points.at(v);
                visible_list.push_back(p);
                corresponding_model_list.push_back(p);
            }
            if (is_first) {
                corresponding_model_list_depth_first = corresponding_model_list;
            }
        }
    }

    if (is_first) {
        if (!correspondence.empty()) {
            correspondence.clear();
        }
    }

    int corr_count = 0;

    bool isDone = false;

    Eigen::Vector3f vecNorm;

    if (pcl_cloud->isOrganized()) {
        for (int k = 0; (k < visible_list.size()) && (!isDone); k += 3)  // <-- here 'k+=3' is needed because 'vertices_visible_list' had three points added per triangle in <<compute_jacobian>>
        {
            double res = 0.0f;

            if (is_first) {
                pcl::PointXYZ normal;

                vecNorm = mesh.visible_normal[k / 3];
                normal.x = vecNorm[0];
                normal.y = vecNorm[1];
                normal.z = vecNorm[2];

                point_2d a = ocl.projection(corresponding_model_list[k]);     /************************************************************************/
                point_2d b = ocl.projection(corresponding_model_list[k + 1]); /*These three points are supposed to represent a single triangular plane*/
                point_2d c = ocl.projection(corresponding_model_list[k + 2]); /************************************************************************/

                float area = ocl.getTriangleArea(a, b, c);

                float min_x, min_y, max_x, max_y;

                min_x = min_val(a.x, b.x, c.x);
                max_x = max_val(a.x, b.x, c.x);
                min_y = min_val(a.y, b.y, c.y);
                max_y = max_val(a.y, b.y, c.y);

                if ((min_x > 0) && (min_y > 0) && (max_x < IMAGE_WIDTH) && (max_y < IMAGE_HEIGHT) && (area > TRIANGLE_AREA_THRESHOLD)) {
                    for (int i = min_x; i < max_x; i++) {
                        for (int j = min_y; j < max_y; j++) {
                            pcl::PointXYZRGB pt = pcl_cloud->at(i, j);

                            if (((pt.x != 0) || (pt.y != 0) || (pt.z != 0)))  // nearly 2 secs here
                            {
                                if ((i > min_x) && (i < max_x) && (j > min_y) && (j < max_y)) {
                                    point_2d target;
                                    target.x = i;
                                    target.y = j;

                                    if (ocl.point_in_triangle(target, a, b, c))  // 7 msec
                                    {
                                        res = get_point_to_plane_error(pt.x, pt.y, pt.z, corresponding_model_list[k].x, corresponding_model_list[k].y, corresponding_model_list[k].z, normal.x, normal.y, normal.z);
                                        residual.push_back(res);
                                        std::vector<double> corr;
                                        corr.push_back(k);
                                        corr.push_back(pt.x);
                                        corr.push_back(pt.y);
                                        corr.push_back(pt.z);
                                        correspondence.push_back(corr);

                                        if (debugFlag) {
                                            Eigen::Vector4f residual_pt(pt.x, pt.y, pt.z, res);
                                            residual_map.push_back(residual_pt);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

            } else {
                for (int p = 0; p < correspondence.size(); p++) {
                    std::vector<double> corr = correspondence[p];
                    int index = corr[0];

                    pcl::PointXYZ normal;
                    vecNorm = mesh.visible_normal[index / 3];
                    normal.x = vecNorm[0];
                    normal.y = vecNorm[1];
                    normal.z = vecNorm[2];
                    res = get_point_to_plane_error(corr[1], corr[2], corr[3], corresponding_model_list[index].x, corresponding_model_list[index].y, corresponding_model_list[index].z, normal.x, normal.y, normal.z);
                    residual.push_back(res);
                    num_points++;

                    if (debugFlag > 1) {
                        Eigen::Vector4f residual_pt(corr[1], corr[2], corr[3], res);
                        residual_map.push_back(residual_pt);
                    }
                }

                isDone = true;
            }
            corr_count++;
        }
    }

    EigenMatrix residual_matrix(residual.size(), 1);

    float mean_res = 0.0f;

    for (int i = 0; i < residual.size(); i++) {
        residual_matrix(i, 0) = residual[i];
        mean_res += residual[i];
    }

    mean_res /= ((float)residual.size());

    if ((debugFlag > 1) || ((debugFlag) && (is_first))) {
        debug_residual(residual_map, additionalDebugFolder, file_name, iteration);
    }

    if ((debugFlag > 1)) {
        if (is_first) {
            pcl::io::savePLYFileBinary(additionalDebugFolder + file_name + "_" + to_string(file_num_) + "_" + to_string(pcd_count) + "_" + to_string(iteration) + ".RES." + to_string(iteration) + "_R:" + to_string(mean_res) + "_.ply", *vertices);
            pcl::io::savePLYFileBinary(additionalDebugFolder + file_name + "_data_" + to_string(iteration) + ".ply", *pcl_cloud);
            pcl::io::savePLYFileBinary(additionalDebugFolder + file_name + "_mesh_" + to_string(iteration) + ".ply", mesh.mesh);
            pcl::io::savePLYFileBinary(additionalDebugFolder + file_name + "_base_" + to_string(iteration) + ".ply", poly_mesh);
        } else {
            pcl::io::saveOBJFile(additionalDebugFolder + file_name + "_" + to_string(iteration) + ".obj", poly_mesh);
            pcl::io::saveOBJFile(additionalDebugFolder + file_name + "_m_" + to_string(iteration) + ".obj", mesh.mesh);
        }
    }

    return residual_matrix;
}

bool JacobianFEM::get_photometric_residual_secondary(cv::Mat &colored_depth, vector<vector<double>> &_correspondence_photo, vector<PointXYZ> &_corresponding_model_list, vector<PointXYZ> &_correspondence_barycentric_photo, vector<double> &residual, int k) {
#ifdef FATALIZE_PHOTOMETRIC
    _fatal_("Photometric is un-defined. Yet, something triggered this code block!")
#endif
        point_2d low;
    point_2d top_right;
    point_2d bottom_left;
    point_2d roof;

    float pt_low;
    float pt_top_right;
    float pt_bottom_left;
    float pt_roof;

    float whole_x, fractional_x;
    float whole_y, fractional_y;

    float thresh = 0.001f;
    int prev_index = -1;
    vpPlane model_plane;
    vpPoint A;
    vpPoint B;
    vpPoint C;

    double res = 0.0f;
    double sum_res = 0.0f;

    PointXYZ barycentric_prio;

    for (int p = 0; p < _correspondence_photo.size(); p++) {
        std::vector<double> corr = _correspondence_photo[p];
        int index = corr[0];

        if (index != prev_index) {
            pcl_to_visp_3Dpoint(_corresponding_model_list[k + 0], A);
            pcl_to_visp_3Dpoint(_corresponding_model_list[k + 1], B);
            pcl_to_visp_3Dpoint(_corresponding_model_list[k + 2], C);

            vpPlane model_plane_(A, B, C);
            model_plane = model_plane_;

            prev_index = index;
        }

        barycentric_prio = _correspondence_barycentric_photo[p];

        PointXYZ barycentric_inverse_cartesian;
        ocl.compute_inverse_barycentric_coordinates(_corresponding_model_list[index], _corresponding_model_list[index + 1], _corresponding_model_list[index + 2], barycentric_prio, barycentric_inverse_cartesian);

        point_2d img_point;
        img_point.x = round(barycentric_inverse_cartesian.x);
        img_point.y = round(barycentric_inverse_cartesian.y);

        if (((img_point.x) > IMAGE_WIDTH) || ((img_point.y) > IMAGE_HEIGHT) || ((img_point.x) < 0.0f) || ((img_point.y) < 0.0f)) {
            res = 0.0f;
            residual.push_back(res);

        } else {
            fractional_x = fabs(modf(img_point.x, &whole_x));
            fractional_y = fabs(modf(img_point.y, &whole_y));

            float curr_val = 0.0f;

            if ((fractional_x > thresh) || (fractional_y > thresh)) {
                low.x = floor(img_point.x);
                low.y = floor(img_point.y);
                top_right.x = ceil(img_point.x);
                top_right.y = floor(img_point.y);
                bottom_left.x = floor(img_point.x);
                bottom_left.y = ceil(img_point.y);
                roof.x = ceil(img_point.x);
                roof.y = ceil(img_point.y);

                pt_low = (float)colored_depth.at<uchar>(floor(low.y), floor(low.x));  //(current_image.at<cv::Vec3b>(floor(low.y),floor(low.x))[2] + current_image.at<cv::Vec3b>(floor(low.y),floor(low.x))[1] + current_image.at<cv::Vec3b>(floor(low.y),floor(low.x))[0])/3.0f;
                pt_top_right = (float)colored_depth.at<uchar>(
                    floor(top_right.y), floor(top_right.x));  //(current_image.at<cv::Vec3b>(floor(top_right.y),floor(top_right.x))[2] + current_image.at<cv::Vec3b>(floor(top_right.y),floor(top_right.x))[1] + current_image.at<cv::Vec3b>(floor(top_right.y),floor(top_right.x))[0])/3.0f;
                pt_bottom_left = (float)colored_depth.at<uchar>(
                    floor(bottom_left.y),
                    floor(bottom_left.x));  //(current_image.at<cv::Vec3b>(floor(bottom_left.y),floor(bottom_left.x))[2] + current_image.at<cv::Vec3b>(floor(bottom_left.y),floor(bottom_left.x))[1] + current_image.at<cv::Vec3b>(floor(bottom_left.y),floor(bottom_left.x))[0])/3.0f;
                pt_roof = (float)colored_depth.at<uchar>(floor(roof.y), floor(roof.x));  //(current_image.at<cv::Vec3b>(floor(roof.y),floor(roof.x))[2] + current_image.at<cv::Vec3b>(floor(roof.y),floor(roof.x))[1] + current_image.at<cv::Vec3b>(floor(roof.y),floor(roof.x))[0])/3.0f;

                curr_val = ocl.bilinearInterpolationAtValues(img_point.x, img_point.y, pt_low, pt_top_right, pt_bottom_left, pt_roof);
            } else {
                curr_val = (float)colored_depth.at<uchar>(img_point.y, img_point.x);
            }

            float prev_intensity = (float)prev_colored_depth.at<uchar>(correspondence_img_pt_photo[p].y, correspondence_img_pt_photo[p].x);

            res = curr_val - prev_intensity;
            sum_res += res;

            residual.push_back(res);
        }
    }
    return true;
}

EigenMatrix JacobianFEM::get_residual_photometric(cv::Mat &colored_depth, pcl::PolygonMesh poly_mesh, pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud, Eigen::Matrix4f transform, bool is_first, std::string file_name, mesh_map mesh, vector<int> observability_indices, int iteration, int balancer) {
    pcl::PCLPointCloud2 blob = poly_mesh.cloud;

#ifndef BARYCENTRIC_IN_IMAGE_PLANE
    _fatal_("Enable BARYCENTRIC_IN_IMAGE_PLANE for photometric deformation tracking!")
#endif
#ifdef FATALIZE_PHOTOMETRIC
        _fatal_("Photometric is un-defined. Yet, something triggered this code block!")
#endif

            _warn_("Obtaining photometric residual for: " << file_name);

    EigenMatrix extrinsics = EigenMatrix::Zero(4, 4);
    extrinsics(0, 0) = data_map.depth2color_extrinsics(0, 0);
    extrinsics(0, 1) = data_map.depth2color_extrinsics(0, 1);
    extrinsics(0, 2) = data_map.depth2color_extrinsics(0, 2);
    extrinsics(0, 3) = data_map.depth2color_extrinsics(0, 3);
    extrinsics(1, 0) = data_map.depth2color_extrinsics(1, 0);
    extrinsics(1, 1) = data_map.depth2color_extrinsics(1, 1);
    extrinsics(1, 2) = data_map.depth2color_extrinsics(1, 2);
    extrinsics(1, 3) = data_map.depth2color_extrinsics(1, 3);
    extrinsics(2, 0) = data_map.depth2color_extrinsics(2, 0);
    extrinsics(2, 1) = data_map.depth2color_extrinsics(2, 1);
    extrinsics(2, 2) = data_map.depth2color_extrinsics(2, 2);
    extrinsics(2, 3) = data_map.depth2color_extrinsics(2, 3);
    extrinsics(3, 0) = data_map.depth2color_extrinsics(3, 0);
    extrinsics(3, 1) = data_map.depth2color_extrinsics(3, 1);
    extrinsics(3, 2) = data_map.depth2color_extrinsics(3, 2);
    extrinsics(3, 3) = data_map.depth2color_extrinsics(3, 3);

    pcl::PointCloud<pcl::PointXYZ>::Ptr vertices(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(blob, *vertices);

    if (debugFlag > 1) {
        if (is_first) {
            pcl::io::savePLYFileBinary(additionalDebugFolder + file_name + "_photo_resi_" + to_string(iteration) + ".ply", *vertices);
            pcl::io::savePLYFileBinary(additionalDebugFolder + file_name + "_photo_data_" + to_string(iteration) + ".ply", *_prev_pointcloud);
            pcl::io::savePLYFileBinary(additionalDebugFolder + file_name + "_photo_mesh_" + to_string(iteration) + ".ply", mesh.mesh);
            pcl::io::savePLYFileBinary(additionalDebugFolder + file_name + "_photo_base_" + to_string(iteration) + ".ply", poly_mesh);
        } else {
            pcl::io::saveOBJFile(additionalDebugFolder + file_name + "_photo_" + to_string(iteration) + ".obj", poly_mesh);
            pcl::io::saveOBJFile(additionalDebugFolder + file_name + "_m_photo_" + to_string(iteration) + ".obj", mesh.mesh);
            Eigen::Matrix4f invT;
            ocl.eigen_to_inv_eigen_4x4(transform, invT);
            PolygonMesh poly_mesh_object_c;
            poly_mesh_object_c = poly_mesh;
            ocl.transformPolygonMesh(poly_mesh_object_c, invT);
            pcl::io::saveOBJFile(additionalDebugFolder + file_name + "_photo_" + to_string(file_num_) + "_" + to_string(pcd_count) + "_" + to_string(iteration) + ".Oc." + to_string(iteration) + "_.obj", poly_mesh_object_c);
        }

        if (debugFlag > 2) {
            debug_img = cv::Mat::zeros(cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), CV_8UC3);
        }
    }

    Eigen::Vector4f xyz_centroid;
    pcl::compute3DCentroid(*vertices, xyz_centroid);

    std::vector<pcl::PointXYZ> visible_list;
    std::vector<pcl::PointXYZ> corresponding_model_list;

    std::vector<double> residual;

    std::vector<pcl::Vertices, std::allocator<pcl::Vertices>>::iterator face;
    pcl::PointCloud<pcl::PointXYZ>::Ptr vertices_visible(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(mesh.mesh.cloud, *vertices_visible);

    int poly_count = 0;

    for (face = mesh.mesh.polygons.begin(); face != mesh.mesh.polygons.end(); ++face, poly_count++) {
        {
            for (int i = 0; i < 3; i++) {
                unsigned int v = face->vertices[i];
                pcl::PointXYZ p = vertices_visible->points.at(v);
                visible_list.push_back(p);

                if ((fabs(p.x) < MAX_DIMENSION_CHECK) && (fabs(p.y) < MAX_DIMENSION_CHECK) && (fabs(p.z) < MAX_DIMENSION_CHECK) && (p.z > 0.0f)) {
                    corresponding_model_list.push_back(p);
                } else {
                    _error_("Received contorted mesh. Vertices dimension/size exceeds " << MAX_DIMENSION_CHECK << " units. Found point (" << p.x << "," << p.y << "," << p.z << ") at index [" << v << "] for *" << file_name << "*")
                        _warn_("This error is going to produce erratic behaviour of the tracker. Most likely cause is very high deforming forces in the SOFA simulation. Please check you SOFA scene before proceeding.") EigenMatrix residual_matrix(residual.size(), 1);

                    for (int i = 0; i < residual.size(); i++) {
                        residual_matrix(i, 0) = 0.0f;
                    }

                    return residual_matrix;
                }
            }
        }
    }

    if (is_first) {
        if (!correspondence_photo.empty()) {
            correspondence_photo.clear();
        }

        if (!correspondence_barycentric_photo.empty()) {
            correspondence_barycentric_photo.clear();
            correspondence_img_pt_photo.clear();
            correspondence_barycentric_transport.clear();
        }
        _log_("Primary photometric residual computation for PCD #" << pcd_count)
    } else {
        _log_("Secondary photometric residual computation for PCD #" << pcd_count)
    }

    int corr_count = 0;
    float sum_res = 0.0f;

    bool isDone = false;

    Eigen::Vector3f vecNorm;

    try {
        if (_prev_pointcloud->isOrganized()) {
            for (int k = 0; (k < visible_list.size()) && (!isDone); k += 3)  // <-- here 'k+=3' is needed because 'vertices_visible_list' had three points added per triangle in <<compute_jacobian>>
            {
                double res = 0.0f;

                if (is_first) {
                    if (is_first_node) {
                        first_mesh_size = corresponding_model_list.size();
                        pcl::PointXYZ normal;

                        vecNorm = mesh.visible_normal[k / 3];
                        normal.x = vecNorm[0];
                        normal.y = vecNorm[1];
                        normal.z = vecNorm[2];

                        vpPoint A;
                        pcl_to_visp_3Dpoint(corresponding_model_list[k], A);

                        vpColVector n_visp(3);
                        n_visp[0] = normal.x;
                        n_visp[1] = normal.y;
                        n_visp[2] = normal.z;

                        point_2d a = ocl.projection(corresponding_model_list[k]);     /************************************************************************/
                        point_2d b = ocl.projection(corresponding_model_list[k + 1]); /*These three points are supposed to represent a single triangular plane*/
                        point_2d c = ocl.projection(corresponding_model_list[k + 2]); /************************************************************************/

                        float area = ocl.getTriangleArea(a, b, c);

                        float min_x, min_y, max_x, max_y;

                        min_x = min_val(a.x, b.x, c.x);
                        max_x = max_val(a.x, b.x, c.x);
                        min_y = min_val(a.y, b.y, c.y);
                        max_y = max_val(a.y, b.y, c.y);

                        if ((min_x > 0) && (min_y > 0) && (max_x < IMAGE_WIDTH) && (max_y < IMAGE_HEIGHT) && (area > TRIANGLE_AREA_THRESHOLD)) {
                            for (int i = min_x; i < max_x; i++) {
                                for (int j = min_y; j < max_y; j++) {
                                    PointXYZRGB pt = _prev_pointcloud->at(i, j);
                                    PointXYZRGB pt_curr = pcl_cloud->at(i, j);

                                    float previous_image_intensity = (float)prev_colored_depth.at<uchar>(j, i);
                                    float current_image_intensity = (float)colored_depth.at<uchar>(j, i);

                                    PointXYZ pt_xyz;
                                    pt_xyz.x = pt.x;
                                    pt_xyz.y = pt.y;
                                    pt_xyz.z = pt.z;

                                    if ((pt.z > 0.0f) && (pt.z < MAX_DEPTH) && (pt_curr.z > 0.0f) && (pt_curr.z < MAX_DEPTH)) {
                                        if ((i > min_x) && (i < max_x) && (j > min_y) && (j < max_y)) {
                                            point_2d target;
                                            target.x = i;
                                            target.y = j;

                                            if (ocl.point_in_triangle(target, a, b, c)) {
                                                res = current_image_intensity - previous_image_intensity;

                                                if ((res < 0.001f) && (res > 0.001f)) {
                                                    _warn_("Extremely low photometric residual detected. Previous point: " << pt << " and current point: " << pt_curr << ". Residual = " << res)
                                                }

                                                PointXYZ barycentric;
                                                ocl.compute_barycentric_coordinates(corresponding_model_list[k], corresponding_model_list[k + 1], corresponding_model_list[k + 2], pt_xyz, barycentric);

                                                correspondence_barycentric_photo.push_back(barycentric);
                                                correspondence_img_pt_photo.push_back(target);

                                                residual.push_back(res);

                                                sum_res += res;

                                                /// debug_img.at<cv::Vec3b>(j,i)[0] = res;//ToBeDeletedLater
                                                /// debug_img.at<cv::Vec3b>(j,i)[1] = res;//ToBeDeletedLater
                                                /// debug_img.at<cv::Vec3b>(j,i)[2] = res;//ToBeDeletedLater

                                                std::vector<double> corr;
                                                corr.push_back(k);
                                                corr.push_back(pt.x);
                                                corr.push_back(pt.y);
                                                corr.push_back(pt.z);
                                                correspondence_photo.push_back(corr);

                                                if (debugFlag) {
                                                    Eigen::Vector4f residual_pt(pt.x, pt.y, pt.z, res);
                                                    residual_map.push_back(residual_pt);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                    } else {
                        if (first_mesh_size != corresponding_model_list.size()) {
                            mesh_deform_regular = false;
                        } else {
                            file_name = "SECOND_RES_";
                            isDone = get_photometric_residual_secondary(colored_depth, correspondence_photo_first, corresponding_model_list, correspondence_barycentric_photo_first, residual, k);

                            _warn_("Secondary photometric residual logged, residual size now: " << residual.size())
                        }
                    }
                } else {
                    isDone = get_photometric_residual_secondary(colored_depth, correspondence_photo, corresponding_model_list, correspondence_barycentric_photo, residual, k);
                }
                corr_count++;
            }
        }

    } catch (const std::out_of_range &oor) {
        _error_("Out of Range error: " << oor.what());
    } catch (vpException &e) {
        _error_("ViSP matrix exception " << e.what())
    }

    EigenMatrix residual_matrix(residual.size(), 1);
    double res = 0.0f;

    for (int i = 0; i < residual.size(); i++) {
        residual_matrix(i, 0) = balancer * residual[i];
        res += residual[i];
    }

    if (is_first && is_first_node) {
        std::ofstream outfile;
        outfile.open(additionalDebugFolder + "/test/R_" + to_string(pcd_count) + ".txt", std::ios_base::app);
        outfile << "Main," << to_string(res / (float)residual.size()) << endl;
        outfile.close();
    }

    if (is_first && is_first_node) {
        correspondence_barycentric_photo_first = correspondence_barycentric_photo;
        correspondence_photo_first = correspondence_photo;
        is_first_node = false;
    } else if (is_first) {
        correspondence_barycentric_photo = correspondence_barycentric_photo_first;
        correspondence_photo = correspondence_photo_first;
    }

    if ((debugFlag > 1) || ((debugFlag) && (is_first))) {
        debug_residual(residual_map, additionalDebugFolder, file_name, iteration);
    }

    _log_("Finished computing residual")

        return residual_matrix;
}

void JacobianFEM::setData_map(const offline_data_map &value) { data_map = value; }

void JacobianFEM::setCurr_image(const cv::Mat &curr_image) {
    if (_is_photo_initialized) {
        _prev_image = _curr_image.clone();
        _warn_("Assigning previous photo to prev")
    } else {
        _prev_image = curr_image.clone();
        _warn_("Assigning current photo to prev")
    }
    _curr_image = curr_image.clone();
}

void clean_residual(EigenMatrix &residual, EigenMatrix &J) {
    vector<float> res_filtered;
    vector<vector<float>> J_x;
    vector<vector<float>> J_y;
    vector<vector<float>> J_z;

    for (int i = 0; i < residual.rows(); i++) {
        if (!((fabs(J(i, 0)) < FLOAT_ZERO) && (fabs(J(i, 1)) < FLOAT_ZERO) && (fabs(J(i, 2)) < FLOAT_ZERO))) {
            res_filtered.push_back(residual(i, 0));

            vector<float> J_x_;
            vector<float> J_y_;
            vector<float> J_z_;

            for (int j = 0; j < (J.cols() / 3); j++) {
                int k = 3 * j;
                J_x_.push_back(J(i, k + 0));
                J_y_.push_back(J(i, k + 1));
                J_z_.push_back(J(i, k + 2));
            }

            J_x.push_back(J_x_);
            J_y.push_back(J_y_);
            J_z.push_back(J_z_);
        }
    }

    EigenMatrix R_(res_filtered.size(), 1);
    EigenMatrix J_(res_filtered.size(), J.cols());

    for (int i = 0; i < res_filtered.size(); i++) {
        R_(i, 0) = res_filtered[i];

        for (int j = 0; j < (J.cols() / 3); j++) {
            int k = 3 * j;
            J_(i, k + 0) = J_x[i][j];
            J_(i, k + 1) = J_y[i][j];
            J_(i, k + 2) = J_z[i][j];
        }
    }

    residual = R_;
    J = J_;
}

void JacobianFEM::gauss_newton_visp_rigid(EigenMatrix &residual, EigenMatrix &J, vpColVector &update, double rigid_lambda, double alpha, int cutpoint) {
    try {
        eyeJ.eye(J.cols(), J.cols());

        if (cutpoint == 0) {
            W_gnvr.resize(residual.rows(), false);
            assign_weights_to_matrix_visp(residual, W_gnvr, 0.08f);
        } else {
            vpColVector W1;
            vpColVector W2;

            int width = residual.cols();

            EigenMatrix res1(cutpoint, width);
            EigenMatrix res2((residual.rows() - cutpoint), width);

            for (int i = 0; i < cutpoint; i++) {
                for (int j = 0; j < width; j++) {
                    res1(i, j) = residual(i, j);
                }
            }

            int count = 0;
            for (int i = cutpoint; i < residual.rows(); i++, count++) {
                for (int j = 0; j < width; j++) {
                    res2(count, j) = residual(i, j);
                }
            }

            W1.resize(res1.rows(), false);
            W2.resize(res2.rows(), false);
            W_gnvr.resize(residual.rows(), false);

            assign_weights_to_matrix_visp(res1, W1, 0.00000f);
            assign_weights_to_matrix_visp(res2, W2, 0.00000f);

            for (int i = 0; i < W1.size(); i++) {
                W_gnvr[i] = W1[i];
            }

            count = W1.size();
            for (int i = 0; i < W2.size(); i++, count++) {
                W_gnvr[count] = W2[i];
            }
        }
        for (int i = 0; i < residual.rows(); i++) {
            residual(i, 0) *= W_gnvr[i];
        }

        ocl.eigen_to_visp(residual, R);
        ocl.eigen_to_visp_weighted(J, J_, W_gnvr);

        v = -((J_.transpose() * J_) + (rigid_lambda * eyeJ)).inverseByLU() * J_.transpose() * R;

        update.resize(v.getRows(), true);

        for (int i = 0; i < v.getRows(); i++) {
            update[i] = alpha * v[i][0];
        }

    } catch (vpException e) {
        vpERROR_TRACE("[JacobianFEM::gauss_newton_visp] Exception from gauss_newton_visp");
        vpCTRACE << e;
    }
}

void gauss_newton_visp(EigenMatrix &residual, EigenMatrix &J, EigenMatrix &update, int iteration, double lmLambda, double _initLambda) {
    clean_residual(residual, J);

    vpMatrix R;
    vpMatrix J_;

    vpMatrix v;

    vpColVector W;

    W.resize(residual.rows(), false);
    assign_weights_to_matrix_visp(residual, W, 0.001f);

    for (int i = 0; i < residual.rows(); i++) {
        residual(i, 0) *= W[i];
    }

    ocl.eigen_to_visp(residual, R);
    ocl.eigen_to_visp(J, J_);

    vpMatrix eyeJ;
    eyeJ.eye(J.cols(), J.cols());

    v = -((J_.transpose() * J_) + (lmLambda * eyeJ)).inverseByLU() * J_.transpose() * R;

    update.resize(v.getRows(), 1);

    for (int i = 0; i < v.getRows(); i++) {
        update(i, 0) = _initLambda * v[i][0];
    }
}

PointCloud<PointXYZ>::Ptr JacobianFEM::inverse_transform_mesh_single(mesh_map &mesh, Eigen::Matrix4f &transform) {
    Eigen::Matrix4f invT;
    ocl.eigen_to_inv_eigen_4x4(transform, invT);
    Eigen::Matrix3f invR;
    ocl.extract_rotation(invT, invR);
    PointCloud<PointXYZ>::Ptr visible_vertices = mesh.visible_vertices;
    transformPointCloud(*visible_vertices, *visible_vertices, invT);
    return visible_vertices;
}

void JacobianFEM::set_prev_colored_depth(const cv::Mat &value) { prev_colored_depth = value.clone(); }

cv::Mat JacobianFEM::get_prev_colored_depth() { return prev_colored_depth; }

void JacobianFEM::decrementLmLambda() { lmLambda -= (lmLambda / 3.0f); }

void JacobianFEM::setLmLambda(double value) { lmLambda = value; }

void JacobianFEM::setPrev_residual(double value) { prev_residual = value; }

Eigen::Matrix4f JacobianFEM::inverse_transform_mesh(vector<mesh_map> &meshes_full, Eigen::Matrix4f &transform) {
    Eigen::Matrix4f invT;
    ocl.eigen_to_inv_eigen_4x4(transform, invT);
    Eigen::Matrix3f invR;
    ocl.extract_rotation(invT, invR);

    for (int i = 1; i < meshes_full.size(); i++) {
        ocl.transformPolygonMesh(meshes_full[i].mesh, invT);
        transformPointCloud(*meshes_full[i].visible_vertices, *meshes_full[i].visible_vertices, invT);
    }

    return invT;
}

Eigen::Matrix4f JacobianFEM::inverse_transform_mesh_complete(vector<mesh_map> &meshes_full, Eigen::Matrix4f &transform) {
    Eigen::Matrix4f invT;
    ocl.eigen_to_inv_eigen_4x4(transform, invT);
    Eigen::Matrix3f invR;
    ocl.extract_rotation(invT, invR);

    for (int i = 0; i < meshes_full.size(); i++) {
        ocl.transformPolygonMesh(meshes_full[i].mesh, invT);
        transformPointCloud(*meshes_full[i].visible_vertices, *meshes_full[i].visible_vertices, invT);
    }

    return invT;
}

void JacobianFEM::get_influence_matrix(EigenMatrix &influence_mat, int index_0, int index_1, int index_2, int control_point_index, int option) {
    control_point_index = 0;

    influence_mat(0, (3 * control_point_index) + 0) = inflMat_a(index_0 + 0, active_point[control_point_index] + 0) / jacobian_displacement;
    influence_mat(1, (3 * control_point_index) + 0) = inflMat_a(index_0 + num_vertices, active_point[control_point_index] + 0) / jacobian_displacement;
    influence_mat(2, (3 * control_point_index) + 0) = inflMat_a(index_0 + (2 * num_vertices), active_point[control_point_index] + 0) / jacobian_displacement;
    influence_mat(3, (3 * control_point_index) + 0) = inflMat_a(index_1 + 0, active_point[control_point_index] + 0) / jacobian_displacement;
    influence_mat(4, (3 * control_point_index) + 0) = inflMat_a(index_1 + num_vertices, active_point[control_point_index] + 0) / jacobian_displacement;
    influence_mat(5, (3 * control_point_index) + 0) = inflMat_a(index_1 + (2 * num_vertices), active_point[control_point_index] + 0) / jacobian_displacement;
    influence_mat(6, (3 * control_point_index) + 0) = inflMat_a(index_2 + 0, active_point[control_point_index] + 0) / jacobian_displacement;
    influence_mat(7, (3 * control_point_index) + 0) = inflMat_a(index_2 + num_vertices, active_point[control_point_index] + 0) / jacobian_displacement;
    influence_mat(8, (3 * control_point_index) + 0) = inflMat_a(index_2 + (2 * num_vertices), active_point[control_point_index] + 0) / jacobian_displacement;

    influence_mat(0, (3 * control_point_index) + 1) = inflMat_a(index_0 + 0, active_point[control_point_index] + num_vertices) / jacobian_displacement;
    influence_mat(1, (3 * control_point_index) + 1) = inflMat_a(index_0 + num_vertices, active_point[control_point_index] + num_vertices) / jacobian_displacement;
    influence_mat(2, (3 * control_point_index) + 1) = inflMat_a(index_0 + (2 * num_vertices), active_point[control_point_index] + num_vertices) / jacobian_displacement;
    influence_mat(3, (3 * control_point_index) + 1) = inflMat_a(index_1 + 0, active_point[control_point_index] + num_vertices) / jacobian_displacement;
    influence_mat(4, (3 * control_point_index) + 1) = inflMat_a(index_1 + num_vertices, active_point[control_point_index] + num_vertices) / jacobian_displacement;
    influence_mat(5, (3 * control_point_index) + 1) = inflMat_a(index_1 + (2 * num_vertices), active_point[control_point_index] + num_vertices) / jacobian_displacement;
    influence_mat(6, (3 * control_point_index) + 1) = inflMat_a(index_2 + 0, active_point[control_point_index] + num_vertices) / jacobian_displacement;
    influence_mat(7, (3 * control_point_index) + 1) = inflMat_a(index_2 + num_vertices, active_point[control_point_index] + num_vertices) / jacobian_displacement;
    influence_mat(8, (3 * control_point_index) + 1) = inflMat_a(index_2 + (2 * num_vertices), active_point[control_point_index] + num_vertices) / jacobian_displacement;

    influence_mat(0, (3 * control_point_index) + 2) = inflMat_a(index_0 + 0, active_point[control_point_index] + (2 * num_vertices)) / jacobian_displacement;
    influence_mat(1, (3 * control_point_index) + 2) = inflMat_a(index_0 + num_vertices, active_point[control_point_index] + (2 * num_vertices)) / jacobian_displacement;
    influence_mat(2, (3 * control_point_index) + 2) = inflMat_a(index_0 + (2 * num_vertices), active_point[control_point_index] + (2 * num_vertices)) / jacobian_displacement;
    influence_mat(3, (3 * control_point_index) + 2) = inflMat_a(index_1 + 0, active_point[control_point_index] + (2 * num_vertices)) / jacobian_displacement;
    influence_mat(4, (3 * control_point_index) + 2) = inflMat_a(index_1 + num_vertices, active_point[control_point_index] + (2 * num_vertices)) / jacobian_displacement;
    influence_mat(5, (3 * control_point_index) + 2) = inflMat_a(index_1 + (2 * num_vertices), active_point[control_point_index] + (2 * num_vertices)) / jacobian_displacement;
    influence_mat(6, (3 * control_point_index) + 2) = inflMat_a(index_2 + 0, active_point[control_point_index] + (2 * num_vertices)) / jacobian_displacement;
    influence_mat(7, (3 * control_point_index) + 2) = inflMat_a(index_2 + num_vertices, active_point[control_point_index] + (2 * num_vertices)) / jacobian_displacement;
    influence_mat(8, (3 * control_point_index) + 2) = inflMat_a(index_2 + (2 * num_vertices), active_point[control_point_index] + (2 * num_vertices)) / jacobian_displacement;

    if ((influence_mat.colwise().norm()[2] < 0.00000f) /* && (option == _PHOTO)*/) {
        influence_mat(0, (3 * control_point_index) + 0) = 0.0f;
        influence_mat(1, (3 * control_point_index) + 0) = 0.0f;
        influence_mat(2, (3 * control_point_index) + 0) = 0.0f;
        influence_mat(3, (3 * control_point_index) + 0) = 0.0f;
        influence_mat(4, (3 * control_point_index) + 0) = 0.0f;
        influence_mat(5, (3 * control_point_index) + 0) = 0.0f;
        influence_mat(6, (3 * control_point_index) + 0) = 0.0f;
        influence_mat(7, (3 * control_point_index) + 0) = 0.0f;
        influence_mat(8, (3 * control_point_index) + 0) = 0.0f;
        influence_mat(0, (3 * control_point_index) + 1) = 0.0f;
        influence_mat(1, (3 * control_point_index) + 1) = 0.0f;
        influence_mat(2, (3 * control_point_index) + 1) = 0.0f;
        influence_mat(3, (3 * control_point_index) + 1) = 0.0f;
        influence_mat(4, (3 * control_point_index) + 1) = 0.0f;
        influence_mat(5, (3 * control_point_index) + 1) = 0.0f;
        influence_mat(6, (3 * control_point_index) + 1) = 0.0f;
        influence_mat(7, (3 * control_point_index) + 1) = 0.0f;
        influence_mat(8, (3 * control_point_index) + 1) = 0.0f;
        influence_mat(0, (3 * control_point_index) + 2) = 0.0f;
        influence_mat(1, (3 * control_point_index) + 2) = 0.0f;
        influence_mat(2, (3 * control_point_index) + 2) = 0.0f;
        influence_mat(3, (3 * control_point_index) + 2) = 0.0f;
        influence_mat(4, (3 * control_point_index) + 2) = 0.0f;
        influence_mat(5, (3 * control_point_index) + 2) = 0.0f;
        influence_mat(6, (3 * control_point_index) + 2) = 0.0f;
        influence_mat(7, (3 * control_point_index) + 2) = 0.0f;
        influence_mat(8, (3 * control_point_index) + 2) = 0.0f;
    }
#ifdef DEPTH_JACOBIAN_BINARIZED
    else if (option == _DEPTH) {
        influence_mat(0, (3 * control_point_index) + 0) = 1.0f;
        influence_mat(1, (3 * control_point_index) + 0) = 1.0f;
        influence_mat(2, (3 * control_point_index) + 0) = 1.0f;
        influence_mat(3, (3 * control_point_index) + 0) = 1.0f;
        influence_mat(4, (3 * control_point_index) + 0) = 1.0f;
        influence_mat(5, (3 * control_point_index) + 0) = 1.0f;
        influence_mat(6, (3 * control_point_index) + 0) = 1.0f;
        influence_mat(7, (3 * control_point_index) + 0) = 1.0f;
        influence_mat(8, (3 * control_point_index) + 0) = 1.0f;
        influence_mat(0, (3 * control_point_index) + 1) = 1.0f;
        influence_mat(1, (3 * control_point_index) + 1) = 1.0f;
        influence_mat(2, (3 * control_point_index) + 1) = 1.0f;
        influence_mat(3, (3 * control_point_index) + 1) = 1.0f;
        influence_mat(4, (3 * control_point_index) + 1) = 1.0f;
        influence_mat(5, (3 * control_point_index) + 1) = 1.0f;
        influence_mat(6, (3 * control_point_index) + 1) = 1.0f;
        influence_mat(7, (3 * control_point_index) + 1) = 1.0f;
        influence_mat(8, (3 * control_point_index) + 1) = 1.0f;
        influence_mat(0, (3 * control_point_index) + 2) = 1.0f;
        influence_mat(1, (3 * control_point_index) + 2) = 1.0f;
        influence_mat(2, (3 * control_point_index) + 2) = 1.0f;
        influence_mat(3, (3 * control_point_index) + 2) = 1.0f;
        influence_mat(4, (3 * control_point_index) + 2) = 1.0f;
        influence_mat(5, (3 * control_point_index) + 2) = 1.0f;
        influence_mat(6, (3 * control_point_index) + 2) = 1.0f;
        influence_mat(7, (3 * control_point_index) + 2) = 1.0f;
        influence_mat(8, (3 * control_point_index) + 2) = 1.0f;
    }
#endif
}

void JacobianFEM::set_is_photo_initialized(bool is_photo_initialized) { _is_photo_initialized = is_photo_initialized; }

PointCloud<PointXYZRGB>::Ptr JacobianFEM::get_prev_pointcloud() { return _prev_pointcloud; }

void JacobianFEM::set_prev_pointcloud(const PointCloud<PointXYZRGB>::Ptr &prev_pointcloud) { _prev_pointcloud = prev_pointcloud; }

EigenMatrix rotate_3x1_vector(Eigen::Matrix4f &transform, double _X, double _Y, double _Z) {
    _log_("Rotating force vector") EigenMatrix R(3, 3);
    EigenMatrix V(3, 1);

    R(0, 0) = transform(0, 0);
    R(0, 1) = transform(0, 1);
    R(0, 2) = transform(0, 2);
    R(1, 0) = transform(1, 0);
    R(1, 1) = transform(1, 1);
    R(1, 2) = transform(1, 2);
    R(2, 0) = transform(2, 0);
    R(2, 1) = transform(2, 1);
    R(2, 2) = transform(2, 2);

    V(0, 0) = _X;
    V(1, 0) = _Y;
    V(2, 0) = _Z;

    V = R * V;

    return V;
}

/** ************************************************** **/
/** EigenMatrix &V needs to be a n x 1 matrix, strictly**/
/**                 where n%3==0                       **/
/** ************************************************** **/
void rotate_3x1_vector_inverse(Eigen::Matrix4f transform, EigenMatrix &V) {
    _log_("Inverse rotating force vector, Jacobian has rows: " << V.rows()) if ((V.rows() < 3) || (V.cols() != 1)) {
        cerr << "[JacobianFEM::transform_jacobian_force_vector_inverse] ERROR: received a badly shaped vector for transformation. Please resize the matrix in the 2nd argument to a [3 x 1] matrix."
             << "If the actual matrix is larger, please take responsibility of splitting it into smaller chunks from outside this method." << endl;
        exit(0);
    }
    else {
        vpHomogeneousMatrix cMo;
        ocl.eigen_to_visp_4x4(transform, cMo);
        ocl.visp_to_eigen_4x4(cMo.inverse(), transform);
        for (int i = 0; i < (V.rows() / 3); i++) {
            int k = 3 * i;
            _log_("trying to rotate: " << V(k, 0) << "," << V(k + 1, 0) << "," << V(k + 2, 0)) EigenMatrix V_ = rotate_3x1_vector(transform, V(k, 0), V(k + 1, 0), V(k + 2, 0));

            V(k, 0) = V_(0, 0);
            V(k + 1, 0) = V_(1, 0);
            V(k + 2, 0) = V_(2, 0);
        }
    }
}

double get_mean_of_col(EigenMatrix M, int col_num) {
    double sum = 0.000f;
    int rows = M.rows();
    for (int i = 0; i < rows; i++) {
        sum += M(i, col_num);
    }

    return (sum / (double)rows);
}

void JacobianFEM::preformat_mesh(PolygonMesh &mesh, mesh_map &mesh_struct, pcl::PointCloud<pcl::PointXYZ>::Ptr &vertices_visible) { mesh_struct = ocl.get_visibility_normal(mesh); }

void JacobianFEM::setJacobian_displacement(float value) { jacobian_displacement = value; }

inline double L1_distance(PointXYZ &a, PointXYZ &b) { return (fabs(a.x - b.x) + fabs(a.y - b.y) + fabs(a.z - b.z)); }

bool JacobianFEM::preformat_all_models(PolygonMesh &base_model, Eigen::Matrix4f &transform, pcl::PointCloud<pcl::PointXYZ>::Ptr &vertices_visible, vector<mesh_map> &meshes) {
    mesh_map base_mesh;

    ocl.transformPolygonMesh(base_model, transform);
    preformat_mesh(base_model, base_mesh, vertices_visible);
    meshes.push_back(base_mesh);

    return true;
}

bool JacobianFEM::compute_analytic_part_jacobian(vector<mesh_map> meshes_full, EigenMatrix &J, Eigen::Matrix4f &transform, int num_forces) {
    int all_mesh_similar = true;

    _log_("Depth based Jacobian computation - analytic")

        /*********DEBUG STUFF*************/
        /// cv::Mat J_combined;
        /// J_combined = cv::Mat::zeros(cv::Size(IMAGE_WIDTH,IMAGE_HEIGHT), CV_8UC3);
        /*********DEBUG STUFF*************/

        vector<mesh_map>
            meshes;
    meshes.push_back(meshes_full[0]);
    meshes.push_back(meshes_full[0]);
    meshes.push_back(meshes_full[0]);

    for (int i = 0; i < (meshes.size() - 1); i++) {
        _log_(" mesh size: " << meshes[i].visible_indices.size() << " , " << meshes[i + 1].visible_indices.size()) if ((meshes[i].visible_indices.size() != meshes[i + 1].visible_indices.size()) || (meshes[i].visible_indices.size() == 0)) { all_mesh_similar = false; }
    }

    if (all_mesh_similar) {
        EigenMatrix P1(3, 1);
        EigenMatrix P2(3, 1);
        EigenMatrix P3(3, 1);

        float x1, y1, z1, x2, y2, z2, x3, y3, z3;

        EigenMatrix Ps(3, 1);

        EigenMatrix C(3, 1);

        EigenMatrix N_ps(3, 1);

        EigenMatrix J_elem_1 = EigenMatrix::Zero(1, 9);
        EigenMatrix J_elem_2 = EigenMatrix::Zero(9, num_forces * 3);
        EigenMatrix J_elem_combined = EigenMatrix::Zero(1, num_forces * 3);

        EigenMatrix eye = EigenMatrix::Identity(3, 3);
        EigenMatrix diff_normal = EigenMatrix::Identity(3, 3);

        EigenMatrix unit_normal(3, 1);
        EigenMatrix normal(3, 1);

        int prev_index = -1;

        pcl::PointCloud<pcl::PointXYZ>::Ptr mesh_base_C(new pcl::PointCloud<pcl::PointXYZ>);

        int prev_index_temp = -1;

        for (int p = 0; p < correspondence.size(); p++) {
            std::vector<double> corr = correspondence[p];
            int index = corr[0];

            if (prev_index_temp != index) {
                mesh_base_C->push_back(meshes_full[0].visible_vertices->points[index + 0]);
                mesh_base_C->push_back(meshes_full[0].visible_vertices->points[index + 1]);
                mesh_base_C->push_back(meshes_full[0].visible_vertices->points[index + 2]);

                prev_index_temp = index;
            }
        }

        if (!undef_mesh_initialized) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr mesh_temp(new pcl::PointCloud<pcl::PointXYZ>);
            for (int i = 0; i < mesh_base_C->size(); i++) {
                mesh_temp->push_back(mesh_base_C->points[i]);
            }
            undef_mesh_original = mesh_temp;
            undef_mesh_initialized = true;
        }

        for (int p = 0; p < correspondence.size(); p++) {
            std::vector<double> corr = correspondence[p];
            int index = corr[0];

            Ps(0, 0) = corr[1];
            Ps(1, 0) = corr[2];
            Ps(2, 0) = corr[3];

            if (prev_index != index) {
                P1(0, 0) = meshes_full[0].visible_vertices->points[index + 0].x;
                P1(1, 0) = meshes_full[0].visible_vertices->points[index + 0].y;
                P1(2, 0) = meshes_full[0].visible_vertices->points[index + 0].z;
                P2(0, 0) = meshes_full[0].visible_vertices->points[index + 1].x;
                P2(1, 0) = meshes_full[0].visible_vertices->points[index + 1].y;
                P2(2, 0) = meshes_full[0].visible_vertices->points[index + 1].z;
                P3(0, 0) = meshes_full[0].visible_vertices->points[index + 2].x;
                P3(1, 0) = meshes_full[0].visible_vertices->points[index + 2].y;
                P3(2, 0) = meshes_full[0].visible_vertices->points[index + 2].z;

                unit_normal(0, 0) = meshes_full[0].visible_normal[index / 3](0);
                unit_normal(1, 0) = meshes_full[0].visible_normal[index / 3](1);
                unit_normal(2, 0) = meshes_full[0].visible_normal[index / 3](2);
                normal(0, 0) = meshes_full[0].visible_normal_unnormalized[index / 3](0);
                normal(1, 0) = meshes_full[0].visible_normal_unnormalized[index / 3](1);
                normal(2, 0) = meshes_full[0].visible_normal_unnormalized[index / 3](2);

                diff_normal = (1 / normal.colwise().norm()(0)) * (eye - unit_normal * unit_normal.transpose());  // WARNING: diff_normal turns into an identity matrix when this line is commented out

                x1 = P1(0, 0);
                y1 = P1(1, 0);
                z1 = P1(2, 0);
                x2 = P2(0, 0);
                y2 = P2(1, 0);
                z2 = P2(2, 0);
                x3 = P3(0, 0);
                y3 = P3(1, 0);
                z3 = P3(2, 0);

                for (int j = 0; j < (num_forces * 3); j++) {
                    J_elem_2(0, j) = 0.0f;
                    J_elem_2(1, j) = 0.0f;
                    J_elem_2(2, j) = 0.0f;
                    J_elem_2(3, j) = 0.0f;
                    J_elem_2(4, j) = 0.0f;
                    J_elem_2(5, j) = 0.0f;
                    J_elem_2(6, j) = 0.0f;
                    J_elem_2(7, j) = 0.0f;
                    J_elem_2(8, j) = 0.0f;
                }

                int index_0 = visible_full_map[index + 0];
                int index_1 = visible_full_map[index + 1];
                int index_2 = visible_full_map[index + 2];

                for (int control_index = 0; control_index < num_forces; control_index++) {
                    get_influence_matrix(J_elem_2, index_0, index_1, index_2, control_index, _DEPTH);
                }
            }

            N_ps = Ps - P1;

            C(0, 0) = 0.0f;
            C(1, 0) = z3 - z2;
            C(2, 0) = y2 - y3;
            J_elem_1(0, 0) = -unit_normal(0, 0) + (N_ps.transpose() * (diff_normal * (-C)))(0, 0);

            C(0, 0) = z2 - z3;
            C(1, 0) = 0.0f;
            C(2, 0) = x3 - x2;
            J_elem_1(0, 1) = -unit_normal(1, 0) + (N_ps.transpose() * (diff_normal * (-C)))(0, 0);

            C(0, 0) = y3 - y2;
            C(1, 0) = x2 - x3;
            C(2, 0) = 0.0f;
            J_elem_1(0, 2) = -unit_normal(2, 0) + (N_ps.transpose() * (diff_normal * (-C)))(0, 0);

            C(0, 0) = 0.0f;
            C(1, 0) = z1 - z3;
            C(2, 0) = y3 - y1;
            J_elem_1(0, 3) = (N_ps.transpose() * (diff_normal * (-C)))(0, 0);

            C(0, 0) = z3 - z1;
            C(1, 0) = 0.0f;
            C(2, 0) = x1 - x3;
            J_elem_1(0, 4) = (N_ps.transpose() * (diff_normal * (-C)))(0, 0);

            C(0, 0) = y1 - y3;
            C(1, 0) = x3 - x1;
            C(2, 0) = 0.0f;
            J_elem_1(0, 5) = (N_ps.transpose() * (diff_normal * (-C)))(0, 0);

            C(0, 0) = 0.0f;
            C(1, 0) = z2 - z1;
            C(2, 0) = y1 - y2;
            J_elem_1(0, 6) = (N_ps.transpose() * (diff_normal * (-C)))(0, 0);

            C(0, 0) = z1 - z2;
            C(1, 0) = 0.0f;
            C(2, 0) = x2 - x1;
            J_elem_1(0, 7) = (N_ps.transpose() * (diff_normal * (-C)))(0, 0);

            C(0, 0) = y2 - y1;
            C(1, 0) = x1 - x2;
            C(2, 0) = 0.0f;
            J_elem_1(0, 8) = (N_ps.transpose() * (diff_normal * (-C)))(0, 0);

            J_elem_combined = (J_elem_1)*J_elem_2;

            J(p, 0) = J_elem_combined(0, 0);
            J(p, 1) = J_elem_combined(0, 1);
            J(p, 2) = J_elem_combined(0, 2);

            prev_index = index;

            /*****************The following block is just for DEBUG***********************/
            /// PointXYZ pt_of_intrst;
            /// pt_of_intrst.x = Ps(0,0);
            /// pt_of_intrst.y = Ps(1,0);
            /// pt_of_intrst.z = Ps(2,0);
            ///
            /// float multiplier = 50.0f;
            ///
            ///
            /// point_2d img_pt = ocl.projection(pt_of_intrst);
            //////
            /// J_combined.at<cv::Vec3b>(img_pt.y,img_pt.x)[0] = multiplier*fabs(J_elem_2(0,0) + J_elem_2(3,0) + J_elem_2(6,0));//J(p,0);
            /// J_combined.at<cv::Vec3b>(img_pt.y,img_pt.x)[1] = multiplier*fabs(J_elem_2(1,1) + J_elem_2(4,1) + J_elem_2(7,1));//J(p,1);
            /// J_combined.at<cv::Vec3b>(img_pt.y,img_pt.x)[2] = multiplier*fabs(J_elem_2(2,2) + J_elem_2(5,2) + J_elem_2(8,2));//J(p,2);
            /*****************The preceding block is just for DEBUG***********************/
        }

        /*****************The following block is just for DEBUG***********************/
        /// cv::normalize(J_combined, J_combined, 0, 255, cv::NORM_MINMAX);
        /// cv::imwrite( additionalDebugFolder+"/_depth_"+to_string(pcd_count)+"_"+to_string(iteration)+"_"+to_string(active_point[0])+"_J_elem_2.jpg", J_combined );
        /*****************The preceding block is just for DEBUG***********************/

        return true;
    } else {
        _error_("Jacobian deformation is producing different mesh for each deformation. Can't compute Jacobian")

            mesh_deform_regular = false;
        return false;
    }
}

void JacobianFEM::compute_analytic_part_jacobian_photometric(vector<mesh_map> meshes_full, EigenMatrix &J, Eigen::Matrix4f &transform, cv::Mat image, int num_forces) {
#ifdef FATALIZE_PHOTOMETRIC
    _fatal_("Photometric is un-defined. Yet, something triggered this code block!")
#endif

        _log_("Photometry based Jacobian computation - analytic")

            if (mesh_deform_regular) {
        EigenMatrix P1(3, 1);
        EigenMatrix P2(3, 1);
        EigenMatrix P3(3, 1);

        cv::Mat grad_x, grad_y;

        cv::Sobel(image, grad_x, CV_32FC1, 1, 0, 3, 1, 2, cv::BORDER_DEFAULT);
        cv::Sobel(image, grad_y, CV_32FC1, 0, 1, 3, 1, 2, cv::BORDER_DEFAULT);

        /*********DEBUG STUFF*************/
        /// cv::Mat Jx, Jy, Jz, J_combined;
        /// Jx = cv::Mat::zeros(cv::Size(IMAGE_WIDTH,IMAGE_HEIGHT), CV_8UC3);
        /// Jy = cv::Mat::zeros(cv::Size(IMAGE_WIDTH,IMAGE_HEIGHT), CV_8UC3);
        /// Jz = cv::Mat::zeros(cv::Size(IMAGE_WIDTH,IMAGE_HEIGHT), CV_8UC3);
        /// J_combined = cv::Mat::zeros(cv::Size(IMAGE_WIDTH,IMAGE_HEIGHT), CV_8UC3);
        /*********DEBUG STUFF*************/

        EigenMatrix S1(3, 9);
        EigenMatrix S2(3, 9);
        EigenMatrix S3(3, 9);

        EigenMatrix R1(2, 3);
        EigenMatrix R2(2, 3);
        EigenMatrix R3(2, 3);

        EigenMatrix J_elem_2 = EigenMatrix::Zero(9, num_forces * 3);

        EigenMatrix J_elem_combined(1, 3);
        EigenMatrix J_elem_1(1, 9);
        EigenMatrix J_elem_1_1(1, 2);
        EigenMatrix J_elem_1_2(2, 9);

        point_2d img_pt;
        PointXYZ B;

        pcl::PointCloud<pcl::PointXYZ>::Ptr mesh_base_C(new pcl::PointCloud<pcl::PointXYZ>);

        int prev_index_ = -1;

        for (int p = 0; p < correspondence_photo.size(); p++) {
            std::vector<double> corr = correspondence_photo[p];
            int index = corr[0];

            if (prev_index_ != index) {
                mesh_base_C->push_back(meshes_full[0].visible_vertices->points[index + 0]);
                mesh_base_C->push_back(meshes_full[0].visible_vertices->points[index + 1]);
                mesh_base_C->push_back(meshes_full[0].visible_vertices->points[index + 2]);
                prev_index_ = index;
            }
        }

        Eigen::Matrix4f invT;
        ocl.eigen_to_inv_eigen_4x4(transform, invT);

        float x1, y1, z1, x2, y2, z2, x3, y3, z3;
        int prev_index = -1;

        for (int p = 0; p < correspondence_photo.size(); p++) {
            std::vector<double> corr = correspondence_photo[p];
            int index = corr[0];

            if (prev_index != index) {
                P1(0, 0) = meshes_full[0].visible_vertices->points[index + 0].x;
                P1(1, 0) = meshes_full[0].visible_vertices->points[index + 0].y;
                P1(2, 0) = meshes_full[0].visible_vertices->points[index + 0].z;
                P2(0, 0) = meshes_full[0].visible_vertices->points[index + 1].x;
                P2(1, 0) = meshes_full[0].visible_vertices->points[index + 1].y;
                P2(2, 0) = meshes_full[0].visible_vertices->points[index + 1].z;
                P3(0, 0) = meshes_full[0].visible_vertices->points[index + 2].x;
                P3(1, 0) = meshes_full[0].visible_vertices->points[index + 2].y;
                P3(2, 0) = meshes_full[0].visible_vertices->points[index + 2].z;

                x1 = P1(0, 0);
                y1 = P1(1, 0);
                z1 = P1(2, 0);
                x2 = P2(0, 0);
                y2 = P2(1, 0);
                z2 = P2(2, 0);
                x3 = P3(0, 0);
                y3 = P3(1, 0);
                z3 = P3(2, 0);

                R1(0, 0) = (data_map.color_Fx / z1) / 1000.0f;
                R1(0, 1) = 0.0f;
                R1(0, 2) = (-(data_map.color_Fx * x1 / (z1 * z1))) / 1000.0f;
                R1(1, 0) = 0.0f;
                R1(1, 1) = (data_map.color_Fy / z1) / 1000.0f;
                R1(1, 2) = (-(data_map.color_Fy * y1 / (z1 * z1))) / 1000.0f;

                R2(0, 0) = (data_map.color_Fx / z2) / 1000.0f;
                R2(0, 1) = 0.0f;
                R2(0, 2) = (-(data_map.color_Fx * x2 / (z2 * z2))) / 1000.0f;
                R2(1, 0) = 0.0f;
                R2(1, 1) = (data_map.color_Fy / z2) / 1000.0f;
                R2(1, 2) = (-(data_map.color_Fy * y2 / (z2 * z2))) / 1000.0f;

                R3(0, 0) = (data_map.color_Fx / z3) / 1000.0f;
                R3(0, 1) = 0.0f;
                R3(0, 2) = (-(data_map.color_Fx * x3 / (z3 * z3))) / 1000.0f;
                R3(1, 0) = 0.0f;
                R3(1, 1) = (data_map.color_Fy / z3) / 1000.0f;
                R3(1, 2) = (-(data_map.color_Fy * y3 / (z3 * z3))) / 1000.0f;

                S1(0, 0) = 1.0f;
                S1(0, 1) = 0.0f;
                S1(0, 2) = 0.0f;
                S1(0, 3) = 0.0f;
                S1(0, 4) = 0.0f;
                S1(0, 5) = 0.0f;
                S1(0, 6) = 0.0f;
                S1(0, 7) = 0.0f;
                S1(0, 8) = 0.0f;
                S1(1, 0) = 0.0f;
                S1(1, 1) = 1.0f;
                S1(1, 2) = 0.0f;
                S1(1, 3) = 0.0f;
                S1(1, 4) = 0.0f;
                S1(1, 5) = 0.0f;
                S1(1, 6) = 0.0f;
                S1(1, 7) = 0.0f;
                S1(1, 8) = 0.0f;
                S1(2, 0) = 0.0f;
                S1(2, 1) = 0.0f;
                S1(2, 2) = 1.0f;
                S1(2, 3) = 0.0f;
                S1(2, 4) = 0.0f;
                S1(2, 5) = 0.0f;
                S1(2, 6) = 0.0f;
                S1(2, 7) = 0.0f;
                S1(2, 8) = 0.0f;

                S2(0, 0) = 0.0f;
                S2(0, 1) = 0.0f;
                S2(0, 2) = 0.0f;
                S2(0, 3) = 1.0f;
                S2(0, 4) = 0.0f;
                S2(0, 5) = 0.0f;
                S2(0, 6) = 0.0f;
                S2(0, 7) = 0.0f;
                S2(0, 8) = 0.0f;
                S2(1, 0) = 0.0f;
                S2(1, 1) = 0.0f;
                S2(1, 2) = 0.0f;
                S2(1, 3) = 0.0f;
                S2(1, 4) = 1.0f;
                S2(1, 5) = 0.0f;
                S2(1, 6) = 0.0f;
                S2(1, 7) = 0.0f;
                S2(1, 8) = 0.0f;
                S2(2, 0) = 0.0f;
                S2(2, 1) = 0.0f;
                S2(2, 2) = 0.0f;
                S2(2, 3) = 0.0f;
                S2(2, 4) = 0.0f;
                S2(2, 5) = 1.0f;
                S2(2, 6) = 0.0f;
                S2(2, 7) = 0.0f;
                S2(2, 8) = 0.0f;

                S3(0, 0) = 0.0f;
                S3(0, 1) = 0.0f;
                S3(0, 2) = 0.0f;
                S3(0, 3) = 0.0f;
                S3(0, 4) = 0.0f;
                S3(0, 5) = 0.0f;
                S3(0, 6) = 1.0f;
                S3(0, 7) = 0.0f;
                S3(0, 8) = 0.0f;
                S3(1, 0) = 0.0f;
                S3(1, 1) = 0.0f;
                S3(1, 2) = 0.0f;
                S3(1, 3) = 0.0f;
                S3(1, 4) = 0.0f;
                S3(1, 5) = 0.0f;
                S3(1, 6) = 0.0f;
                S3(1, 7) = 1.0f;
                S3(1, 8) = 0.0f;
                S3(2, 0) = 0.0f;
                S3(2, 1) = 0.0f;
                S3(2, 2) = 0.0f;
                S3(2, 3) = 0.0f;
                S3(2, 4) = 0.0f;
                S3(2, 5) = 0.0f;
                S3(2, 6) = 0.0f;
                S3(2, 7) = 0.0f;
                S3(2, 8) = 1.0f;

                for (int j = 0; j < (num_forces * 3); j++) {
                    J_elem_2(0, j) = 0.0f;
                    J_elem_2(1, j) = 0.0f;
                    J_elem_2(2, j) = 0.0f;
                    J_elem_2(3, j) = 0.0f;
                    J_elem_2(4, j) = 0.0f;
                    J_elem_2(5, j) = 0.0f;
                    J_elem_2(6, j) = 0.0f;
                    J_elem_2(7, j) = 0.0f;
                    J_elem_2(8, j) = 0.0f;
                }

                int index_0 = visible_full_map[index + 0];
                int index_1 = visible_full_map[index + 1];
                int index_2 = visible_full_map[index + 2];

                for (int control_index = 0; control_index < num_forces; control_index++) {
                    get_influence_matrix(J_elem_2, index_0, index_1, index_2, control_index, _PHOTO);
                }
            }

            img_pt = correspondence_img_pt_photo[p];
            B = correspondence_barycentric_photo[p];

            J_elem_1_2 = (B.x * R1 * S1) + (B.y * R2 * S2) + (B.z * R3 * S3);

            J_elem_1_1(0, 0) = grad_x.at<float>(img_pt.y, img_pt.x);
            J_elem_1_1(0, 1) = grad_y.at<float>(img_pt.y, img_pt.x);

            J_elem_1 = J_elem_1_1 * J_elem_1_2;

            J_elem_combined = (J_elem_1)*J_elem_2;

            for (int i = 0; i < num_forces; i++) {
                J(p, ((3 * i) + 0)) = J_elem_combined(((3 * i) + 0));
                J(p, ((3 * i) + 1)) = J_elem_combined(((3 * i) + 1));
                J(p, ((3 * i) + 2)) = J_elem_combined(((3 * i) + 2));
            }

            prev_index = index;

            /***********************************DEBUG STUFF***********************************************/
            /// float multiplier = 50.0f;
            /// Jx.at<cv::Vec3b>(img_pt.y,img_pt.x)[0] = multiplier*fabs(J_elem_2(0,0));//J_elem_1(0,0);//J(p,0);//ToBeDeletedLater
            /// Jx.at<cv::Vec3b>(img_pt.y,img_pt.x)[1] = multiplier*fabs(J_elem_2(0,0));//J_elem_1(0,1);//J(p,0);//ToBeDeletedLater
            /// Jx.at<cv::Vec3b>(img_pt.y,img_pt.x)[2] = multiplier*fabs(J_elem_2(0,0));//J_elem_1(0,2);//J(p,0);//ToBeDeletedLate
            ///
            /// Jy.at<cv::Vec3b>(img_pt.y,img_pt.x)[0] = multiplier*fabs(J_elem_2(1,1));//J_elem_1(0,3);//J(p,1);//ToBeDeletedLater
            /// Jy.at<cv::Vec3b>(img_pt.y,img_pt.x)[1] = multiplier*fabs(J_elem_2(1,1));//J_elem_1(0,4);//J(p,1);//ToBeDeletedLater
            /// Jy.at<cv::Vec3b>(img_pt.y,img_pt.x)[2] = multiplier*fabs(J_elem_2(1,1));//J_elem_1(0,5);//J(p,1);//ToBeDeletedLate
            ///
            /// Jz.at<cv::Vec3b>(img_pt.y,img_pt.x)[0] = multiplier*fabs(J_elem_2(2,2));//J_elem_1(0,6);//J(p,2);//ToBeDeletedLater
            /// Jz.at<cv::Vec3b>(img_pt.y,img_pt.x)[1] = multiplier*fabs(J_elem_2(2,2));//J_elem_1(0,7);//J(p,2);//ToBeDeletedLater
            /// Jz.at<cv::Vec3b>(img_pt.y,img_pt.x)[2] = multiplier*fabs(J_elem_2(2,2));//J_elem_1(0,8);//J(p,2);//ToBeDeletedLate
            ///
            /// J_combined.at<cv::Vec3b>(img_pt.y,img_pt.x)[0] = multiplier*fabs(J_elem_2(0,0) + J_elem_2(3,0) + J_elem_2(6,0));//J(p,0);
            /// J_combined.at<cv::Vec3b>(img_pt.y,img_pt.x)[1] = multiplier*fabs(J_elem_2(1,1) + J_elem_2(4,1) + J_elem_2(7,1));//J(p,1);
            /// J_combined.at<cv::Vec3b>(img_pt.y,img_pt.x)[2] = multiplier*fabs(J_elem_2(2,2) + J_elem_2(5,2) + J_elem_2(8,2));//J(p,2);
            /***********************************DEBUG STUFF***********************************************/
        }

        /***********************************DEBUG STUFF***********************************************/
        /// cv::imwrite( additionalDebugFolder+"/_photometricA_"+to_string(pcd_count)+"_"+to_string(iteration)+"_"+to_string(node_count)+"_Jx.jpg", Jx );
        /// cv::imwrite( additionalDebugFolder+"/_photometricA_"+to_string(pcd_count)+"_"+to_string(iteration)+"_"+to_string(node_count)+"_Jy.jpg", Jy );
        /// cv::imwrite( additionalDebugFolder+"/_photometricA_"+to_string(pcd_count)+"_"+to_string(iteration)+"_"+to_string(node_count)+"_Jz.jpg", Jz );
        /// cv::normalize(J_combined, J_combined, 0, 255, cv::NORM_MINMAX);
        /// cv::imwrite( additionalDebugFolder+"/_photometric_"+to_string(pcd_count)+"_"+to_string(iteration)+"_"+to_string(active_point[0])+"_J_elem_2.jpg", J_combined );
        ///_log_("written images")
        /***********************************DEBUG STUFF***********************************************/
    }
    else {
        _error_("Jacobian deformation is producing different mesh for each deformation. All gradients (photometric) set to zero")

            for (int p = 0; p < J.rows(); p++) {
            for (int i = 0; i < num_forces; i++) {
                J(p, ((3 * i) + 0)) = 0.0f;
                J(p, ((3 * i) + 1)) = 0.0f;
                J(p, ((3 * i) + 2)) = 0.0f;
            }
        }
        _log_("Copied all zeros to the photometric Jacobian matrix")
    }
}

void JacobianFEM::augment_transformation(const Eigen::Matrix4f &transform) {
    EigenMatrix R_augmented((3 * num_vertices), (3 * num_vertices));

    for (int i = 0; i < (3 * num_vertices); i++) {
        for (int j = 0; j < (3 * num_vertices); j++) {
            if ((i < num_vertices) && (j < num_vertices)) {
                if (i == j)
                    R_augmented(i, j) = transform(0, 0);
                else
                    R_augmented(i, j) = 0.0f;
            } else if ((i < num_vertices) && (j >= num_vertices) && (j < (2 * num_vertices))) {
                if (i == (j - num_vertices))
                    R_augmented(i, j) = transform(0, 1);
                else
                    R_augmented(i, j) = 0.0f;
            } else if ((i < num_vertices) && (j >= (2 * num_vertices))) {
                if (i == (j - (2 * num_vertices)))
                    R_augmented(i, j) = transform(0, 2);
                else
                    R_augmented(i, j) = 0.0f;
            } else if ((i >= num_vertices) && (i < (2 * num_vertices)) && (j < num_vertices)) {
                if ((i - num_vertices) == j)
                    R_augmented(i, j) = transform(1, 0);
                else
                    R_augmented(i, j) = 0.0f;
            } else if ((i >= num_vertices) && (i < (2 * num_vertices)) && (j >= num_vertices) && (j < (2 * num_vertices))) {
                if (i == j)
                    R_augmented(i, j) = transform(1, 1);
                else
                    R_augmented(i, j) = 0.0f;
            } else if ((i >= num_vertices) && (i < (2 * num_vertices)) && (j >= (2 * num_vertices))) {
                if ((i - num_vertices) == (j - (2 * num_vertices)))
                    R_augmented(i, j) = transform(1, 2);
                else
                    R_augmented(i, j) = 0.0f;
            } else if ((i >= (2 * num_vertices)) && (j < num_vertices)) {
                if ((i - (2 * num_vertices)) == j)
                    R_augmented(i, j) = transform(2, 0);
                else
                    R_augmented(i, j) = 0.0f;
            } else if ((i >= (2 * num_vertices)) && (j >= num_vertices) && (j < (2 * num_vertices))) {
                if ((i - (2 * num_vertices)) == (j - num_vertices))
                    R_augmented(i, j) = transform(2, 1);
                else
                    R_augmented(i, j) = 0.0f;
            } else if ((i >= (2 * num_vertices)) && (j >= (2 * num_vertices))) {
                if (i == j)
                    R_augmented(i, j) = transform(2, 2);
                else
                    R_augmented(i, j) = 0.0f;
            } else {
                _fatal_("Landed us in a strange soup while constructing augmented rotation matrix, bizzare indices (" << i << "," << j << "), while no. of vertices are #" << num_vertices)
            }
        }
    }

    inflMat_a = (R_augmented * inflMat);
}

double JacobianFEM::compute_jacobian(int num_control_points, pcl::PolygonMesh &visible_points, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pcl_cloud, Eigen::Matrix4f &transform, EigenMatrix &update, cv::Mat &colored_depth, int iteration, int pcd_count) {
    if (debugFlag) {
        residual_map.clear();
    }

#ifdef PHOTOMETRIC_MINIMIZATION
#ifdef SPARSE_MINIMIZATION
    _fatal_("Combined usage of sparse and photomeric minimization is not allowed at the moment, please comment out one of the macros in <OcclusionCheck.h>")
#endif
#endif

        std::string debugmsg = to_string(pcd_count) + "_" + to_string(node_count) + ".R_";

    bool is_node_optimization_possible = false;
    vector<int> observability_indices;

    bool fault_in_jacobian = false;
    bool fault_in_objective_function = false;

    vector<mesh_map> meshes;
    pcl::PointCloud<pcl::PointXYZ>::Ptr vertices(new pcl::PointCloud<pcl::PointXYZ>);

    is_node_optimization_possible = preformat_all_models(visible_points, transform, vertices, meshes);

    float photometric_additional_scaling = 0.8f;

    float depth_conditional_blocker = 1.0f;

    bool depth_conditional_blocker_flag = false;

    int mesh_count = 0;
    float scaling_factor = 1.0f;

    float optional_photometric_balancer = 1.0f;

    EigenMatrix res_depth = get_residual_depth(visible_points, pcl_cloud, true, debugmsg, meshes[mesh_count++], iteration);
    int num_points = 0;
#ifdef PHOTOMETRIC_MINIMIZATION
    EigenMatrix res_photo;
    if (_is_photo_initialized) {
        res_photo = get_residual_photometric(colored_depth, visible_points, pcl_cloud, transform, true, debugmsg, meshes[0], observability_indices, iteration, optional_photometric_balancer);
#ifdef PURELY_PHOTOMETRIC
        num_points = res_photo.rows();
#else
        num_points = res_depth.rows() + res_photo.rows();
#endif
    } else {
        num_points = res_depth.rows();
    }
#endif

#ifndef PHOTOMETRIC_MINIMIZATION
    num_points = res_depth.rows();
#endif

    EigenMatrix res_ = EigenMatrix::Zero(num_points, 1);
    EigenMatrix J = EigenMatrix::Zero(num_points, (3 * num_control_points));

#ifdef PHOTOMETRIC_MINIMIZATION

    if (_is_photo_initialized) {
#ifndef PURELY_PHOTOMETRIC
        scaling_factor = res_depth.colwise().norm()(0) / res_photo.colwise().norm()(0);

        for (int k = 0; k < res_depth.rows(); k++) {
            res_(k, 0) = depth_conditional_blocker * res_depth(k, 0);
        }
        _log_("Adding depth-based residual vector to stacked residual")
#endif

            _log_("Depth residual size: " << res_depth.rows() << " photo residual size: " << res_photo.rows() << " but combined num. of pt.s : " << num_points) _log_("Depth based correspondence size: " << correspondence.size() << " & photometric corrspondence size: " << correspondence_photo.size())
                _log_("Scaling factor: " << setprecision(10) << scaling_factor);

        if ((scaling_factor < -10000.0f) || (scaling_factor > 100000.0f)) {
            fault_in_objective_function = true;
            _warn_("Scaling factor is out of practical bounds, Gauus-Newton update will be zeroed in this iteration")
        }

        int p = 0;

#ifdef PURELY_PHOTOMETRIC
        int k_init = 0;
#else
        int k_init = res_depth.rows();
#endif

        for (int k = k_init; k < num_points; k++, p++) {
            res_(k, 0) = scaling_factor * photometric_additional_scaling * res_photo(p, 0);
        }
    } else {
        res_ = res_depth;
    }

#endif

#ifndef PHOTOMETRIC_MINIMIZATION
    res_ = res_depth;
    _warn_("depth residual added")
#endif

        double depth_jacobian_norm = 0.0f;

#ifndef PURELY_PHOTOMETRIC
    EigenMatrix J_depth_analytic = (EigenMatrix::Zero((res_.rows()), 3));
    if (_is_photo_initialized && (!fault_in_objective_function)) {
        compute_analytic_part_jacobian(meshes, J_depth_analytic, transform, 1);

        for (int i = 0; i < J_depth_analytic.rows(); i++) {
            J(i, 0) = depth_conditional_blocker * J_depth_analytic(i, 0);
            J(i, 1) = depth_conditional_blocker * J_depth_analytic(i, 1);
            J(i, 2) = depth_conditional_blocker * J_depth_analytic(i, 2);
        }
        depth_jacobian_norm = fabs(J_depth_analytic.colwise().norm()[0]) + fabs(J_depth_analytic.colwise().norm()[1]) + fabs(J_depth_analytic.colwise().norm()[2]);
    }
#endif

#ifdef PHOTOMETRIC_MINIMIZATION
    if (_is_photo_initialized && (!fault_in_objective_function)) {
        EigenMatrix J_photo_analytic = (EigenMatrix::Zero((res_.rows()), (num_control_points * 3)));
        compute_analytic_part_jacobian_photometric(meshes, J_photo_analytic, transform, colored_depth, num_control_points);
        for (int i = 0; i < num_control_points; i++) {
#ifdef PURELY_PHOTOMETRIC
            for (int k = 0; k < num_points; k++) {
                J(k, (i * 3) + 0) = J_photo_analytic(k, (i * 3) + 0);
                J(k, (i * 3) + 1) = J_photo_analytic(k, (i * 3) + 1);
                J(k, (i * 3) + 2) = J_photo_analytic(k, (i * 3) + 2);
            }
#else
            double photometric_jacobian_norm = fabs(J_photo_analytic.colwise().norm()[0]) + fabs(J_photo_analytic.colwise().norm()[1]) + fabs(J_photo_analytic.colwise().norm()[2]);

            double jacobian_scaling_factor;

            if (photometric_jacobian_norm > 0.000f)
                jacobian_scaling_factor = depth_jacobian_norm / photometric_jacobian_norm;
            else {
                jacobian_scaling_factor = 900000.0f;  // photometric residual's norm was ZERO, asking Gauss-Newton to be avoided for this iteration
                _warn_("Jacobian scaling went for a toss")
            }

            int l = 0;
            int k;
            if ((jacobian_scaling_factor > -100000.0f) && (jacobian_scaling_factor < 100000.0f)) {
                for (k = res_depth.rows(); k < num_points; k++, l++) {
                    J(k, (i * 3) + 0) = jacobian_scaling_factor * photometric_additional_scaling * J_photo_analytic(l, (i * 3) + 0);
                    J(k, (i * 3) + 1) = jacobian_scaling_factor * photometric_additional_scaling * J_photo_analytic(l, (i * 3) + 1);
                    J(k, (i * 3) + 2) = jacobian_scaling_factor * photometric_additional_scaling * J_photo_analytic(l, (i * 3) + 2);
                }
            } else {
                fault_in_jacobian = true;
                _warn_("Jacobian Scaling factor is out of practical bounds, Gauus-Newton update will be zeroed in this iteration. Jacobian scaling factor: " << setprecision(10) << jacobian_scaling_factor) no_more_updates = true;
            }
#endif
        }
    }

#endif

    _log_("Attempting to compute update next: " << _is_photo_initialized << "," << fault_in_jacobian << "," << fault_in_objective_function << "," << mesh_deform_regular)

        if ((_is_photo_initialized) && (!fault_in_jacobian) && (!fault_in_objective_function) && mesh_deform_regular) {
        try {
            if (depth_conditional_blocker_flag) {
                gauss_newton_visp(res_, J, update, iteration, lmLambda, initLambda / 100.0f);
            } else {
                gauss_newton_visp(res_, J, update, iteration, lmLambda, (initLambda));
            }
            _log_("update: " << update)

                if (res_.colwise().norm()[0] < prev_residual) {
                decrementLmLambda();
            }
            else {
                lmLambda *= 1.20f;
            }

            setPrev_residual(res_.colwise().norm()[0]);

        } catch (const std::out_of_range &oor) {
            _warn_("Out of Range error: " << oor.what());
            for (int ind = 0; ind < num_control_points * 3; ind++) update(ind) = 0.0f;
        } catch (vpException &e) {
            _warn_("ViSP matrix exception " << e.what()) for (int ind = 0; ind < num_control_points * 3; ind++) update(ind) = 0.0f;
        }
    }
    else {
        update.resize((num_control_points * 3), 1);

        for (int ind = 0; ind < num_control_points * 3; ind++) update(ind) = 0.0f;

        if (fault_in_jacobian || fault_in_objective_function) {
            _warn_("Residual/Jacobian scaling factor went out of bounds. Update not done for this iteration.") no_more_updates = true;
        } else {
            _error_("Not doing Gauss-Newton update for the first frame")
        }
    }

    rotate_3x1_vector_inverse(transform, update);
    _log_("Rotated Update: " << setprecision(10) << update.transpose()) _log_("Residual: " << setprecision(10) << res_.colwise().norm());

    if (is_first_iteration) {
        is_first_iteration = false;
    }

    return res_.colwise().norm()(0);
}

void JacobianFEM::set_iteration(int iter) { iteration = iter; }

void JacobianFEM::set_pcd_count(int count) { pcd_count = count; }

void JacobianFEM::set_camera_parameters(float Fx, float Fy, float Cx, float Cy) {
    F_x = Fx;
    F_y = Fy;
    C_x = Cx;
    C_y = Cy;
}

void JacobianFEM::set_debug_flag(int value) { debugFlag = value; }

void JacobianFEM::set_additional_debug_folder(const string &value) { additionalDebugFolder = value; }

vector<string> JacobianFEM::getDebugCloudLabel() { return debugCloudLabel; }

void JacobianFEM::resetDebugCloudLabel() { debugCloudLabel.clear(); }

void JacobianFEM::setInitialLambda(float value) {
    initial_lambda = value;
    global_counter = 0;
    lmLambda = value;
}

JacobianFEM::JacobianFEM()
    //  : v_residuals(NB_MP_THREADS) /ak
    : point_stack(BUFFER_SIZE, 3 * NORMAL_SIZE), normal_stack(3 * NORMAL_SIZE, 1), _prev_pointcloud(new pcl::PointCloud<pcl::PointXYZRGB>), _is_photo_initialized(false) {}

vector<PointCloud<PointXYZRGB>> JacobianFEM::getDebugCloudList() { return debugCloudList; }

void JacobianFEM::resetDebugCloudList() { debugCloudList.clear(); }

void JacobianFEM::set_file_num(int _file_num) { file_num_ = _file_num; }

void JacobianFEM::occlusion_init() {
    ocl.set_camera_params(F_x, F_y, C_x, C_y, 0.0f, 0.0f, 0.0f, 0.0f);
    initLambda = initial_lambda;
}

/**
Computes Jacobian and residual by delegating to the appropriate method

@param pointcloud : current frame
@param base_model : model registered to last frame
@param transform : cMo for last frame (in Eigen's Matrix4f format)
@param transform : cMo for last frame
@param update : output from the tracker
@param colored_depth : depth registered with color image

@returns norm of residual **/
double JacobianFEM::compute_update(PointCloud<PointXYZRGB>::Ptr &pointcloud, PolygonMesh base_model, Eigen::Matrix4f &transform, vpHomogeneousMatrix &cMo, EigenMatrix &update, cv::Mat &colored_depth) {
    if (debugFlag) {
        resetDebugCloudLabel();
        resetDebugCloudList();
        _log_("Incrementing counter")
    }

    if (pcd_count < 5) {
        init_cMo = cMo;
    }

    oMc = cMo.inverse();

    mesh_deform_regular = true;

    double residual = compute_jacobian(active_point.size(), base_model, pointcloud, transform, update, colored_depth, iteration, pcd_count);

    _log_("End JacobianFEM");

    global_counter++;

    return residual;
}
