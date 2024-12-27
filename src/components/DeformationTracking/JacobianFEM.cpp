
#include "JacobianFEM.h"

std::vector<std::vector<double>> correspondence;
std::vector<std::vector<std::vector<double>>> set_of_model_list;

int file_num_;

OcclusionCheck ocl;
vpMatrix R;
vpMatrix J_;

vpMatrix AtWA;
vpMatrix lm;
vpMatrix v;

vpColVector W;

void log(std::string text) {
    std::ofstream newFile("/home/agniv/Code/registration_iProcess/SOFA_dataset/cube_fem/output/log.txt", std::ios_base::app);
    newFile << text << std::endl;
    newFile.close();
}

pcl::PointXYZ cross_product_raw(pcl::PointXYZ a, pcl::PointXYZ b) {
    pcl::PointXYZ product;

    product.x = (a.y * b.z) - (a.z * b.y);
    product.y = (a.z * b.x) - (a.x * b.z);
    product.z = (a.x * b.y) - (a.y * b.x);

    return product;
}
double dot_product_raw(pcl::PointXYZ a, pcl::PointXYZ b) {
    pcl::PointXYZ product;

    product.x = a.x * b.x;
    product.y = a.y * b.y;
    product.z = a.z * b.z;

    return (product.x + product.y + product.z);
}

double norm_(pcl::PointXYZ a) { return sqrt((a.x * a.x) + (a.y * a.y) + (a.z * a.z)); }

pcl::PointXYZ cross_product(pcl::PointXYZ a, pcl::PointXYZ b) {
    pcl::PointXYZ product;

    ////////////std:://cout<<(a.y * b.z) - (a.z * b.y)<<std::endl;
    // Cross product formula
    product.x = (a.y * b.z) - (a.z * b.y);
    product.y = (a.z * b.x) - (a.x * b.z);
    product.z = (a.x * b.y) - (a.y * b.x);

    double norm = sqrt((product.x * product.x) + (product.y * product.y) + (product.z * product.z));

    ////////////std:://cout<<"before norm: "<<product.x<<","<<product.y<<","<<product.z<<std::endl;

    if ((norm > 0) || (norm < 0)) {
        product.x /= norm;
        product.y /= norm;
        product.z /= norm;
    }

    // log("norm: "+std::to_string(norm));

    ////////////std:://cout<<"after norm: "<<product.x<<","<<product.y<<","<<product.z<<std::endl;

    return product;
}

bool are_points_same(pcl::PointXYZ a, pcl::PointXYZ b) {
    if ((a.x == b.x) && (a.y == b.y) && (a.z == b.z)) {
        return true;
    } else {
        return false;
    }
}

// Normal is being treated as a point
// WARNING: This method will not work if by any chance the centroid of the object lies outside the object. Careful with future experiments!

pcl::PointXYZ normal_estimation(pcl::PointXYZ &a, pcl::PointXYZ &b, pcl::PointXYZ &c, pcl::PointXYZ &centroid) {
    pcl::PointXYZ A_;
    pcl::PointXYZ B_;
    pcl::PointXYZ out_vec;

    // log(std::to_string(a.x)+" "+std::to_string(a.y)+" "+std::to_string(a.z)+" "+std::to_string(b.x)+" "+std::to_string(b.y)+" "+std::to_string(b.z)+" "+std::to_string(c.x)+" "+std::to_string(c.y)+"
    // "+std::to_string(c.z));

    if (are_points_same(a, b) || are_points_same(b, c) || are_points_same(a, c)) {
        // cout<<"degenerate Triangle"<<std::endl;
        // log("degenerate Triangle");
        pcl::PointXYZ normal_degenerate;
        normal_degenerate.x = 0.0f;
        normal_degenerate.y = 0.0f;
        normal_degenerate.z = 0.0f;

        return normal_degenerate;
    }

    A_.x = c.x - a.x;
    A_.y = c.y - a.y;
    A_.z = c.z - a.z;
    ////////////std:://cout<<c.x<<","<<a.x<<","<<A_.x<<","<<A_.y<<","<<A_.z<<std::endl;

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

    pcl::PointXYZ normal = cross_product(A_, B_);

    double sign = dot_product_raw(normal, out_vec);
    // log("Sign: "+std::to_string(sign));

    if (sign < 0) {
        normal.x = -normal.x;
        normal.y = -normal.y;
        normal.z = -normal.z;
        // sign = dot_product_raw(normal,out_vec);
        // log("Sign again: ");
    }

    return normal;
}

inline double round(double val) {
    if (val < 0) return ceil(val - 0.5);
    return floor(val + 0.5);
}

float min_val(float a, float b, float c) { return a < b ? (a < c ? a : c) : (b < c ? b : c); }

float max_val(float a, float b, float c) { return a > b ? (a > c ? a : c) : (b > c ? b : c); }

void assign_weights(std::vector<double> &residual, int iteration) {
    vpColVector error(residual.size());

    for (int i = 0; i < residual.size(); i++) {
        error[i] = residual[i];
    }

    vpColVector weight;
    vpRobust robust;

    weight.resize(error.size(), false);

    robust.resize(error.size());
    robust.setThreshold(0.00001f);
    robust.MEstimator(vpRobust::TUKEY, error, weight);  // it is also possible to use 'vpRobust::HUBER'

    //    std::ofstream outfile;
    //    std::string debug_dir(DEBUG_DIR);
    //    outfile.open(debug_dir+"/weights"+to_string(iteration)+"_"+to_string(rand())+".csv", std::ios_base::app);
    //    outfile<<"Weight,Error"<<std::endl;
    //
    //    for (unsigned int i = 0; i < error.size(); i++)
    //    {
    //        residual[i] = weight[i] * error[i]; //final weighted error
    //        if(rand() % 100 > 98)
    //            outfile << to_string(weight[i])<<","<<to_string(error[i])<<std::endl;
    //    }
    //
    //    outfile<<"-10,-10"<<std::endl;
    //    outfile.close();
}

Eigen::MatrixXd assign_weights_to_matrix(Eigen::MatrixXd &residual) {
    Eigen::MatrixXd weights = Eigen::MatrixXd::Zero(residual.rows(), residual.rows());
    vpColVector error(residual.rows());

    for (int i = 0; i < residual.rows(); i++) {
        error[i] = residual(i, 0);
    }

    vpColVector weight;
    vpRobust robust;

    weight.resize(error.size(), false);

    robust.resize(error.size());
    robust.setThreshold(0.00001f);
    robust.MEstimator(vpRobust::TUKEY, error, weight);  // it is also possible to use 'vpRobust::HUBER'

    for (unsigned int i = 0; i < error.size(); i++) {
        weights(i, i) = weight[i];  // final weighted error
    }

    return weights;
}

void assign_weights_to_matrix_visp(Eigen::MatrixXd &residual, vpColVector &W) {
    std::cout << "C++: " << "assign_weights_to_matrix_visp" << std::endl;
    vpColVector error(residual.rows());
    // W.resize(residual.rows(),residual.rows(),true);

    std::cout << "C++: " << "debug 1" << std::endl;
    for (int i = 0; i < residual.rows(); i++) {
        error[i] = residual(i, 0);
    }

    std::cout << "C++: " << "debug 2" << std::endl;
    vpRobust robust;

    std::cout << "C++: " << "debug 3" << std::endl;
    robust.resize(error.size());
    std::cout << "C++: " << "debug 3.1" << std::endl;
    robust.setThreshold(0.1f);
    std::cout << "C++: " << "debug 3.2" << std::endl;
    std::cout << "C++: " << "ERROR: " << error.size() << std::endl;
    std::cout << "C++: " << "W: " << W.size() << std::endl;
    robust.MEstimator(vpRobust::TUKEY, error, W);  // it is also possible to use 'vpRobust::HUBER'
    std::cout << "C++: " << "debug 4" << std::endl;
}

Eigen::MatrixXd get_residual(pcl::PolygonMesh &poly_mesh, pcl::PointCloud<pcl::PointXYZ>::Ptr &pcl_cloud, Eigen::Matrix4f &transform, bool is_first, std::string file_name, mesh_map mesh, int iteration) {
    pcl::PCLPointCloud2 blob = poly_mesh.cloud;

    pcl::PointCloud<pcl::PointXYZ>::Ptr vertices(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(blob, *vertices);

    //    if(is_first)
    //    {
    //       pcl::io::savePLYFileBinary("/media/agniv/f9826023-e8c9-47ab-906c-2cbd7ccf196a/home/agniv/Documents/debug_non_rigid/"+to_string(iteration)+"_res"+file_name+".ply", *vertices);
    //       pcl::io::savePLYFileBinary("/media/agniv/f9826023-e8c9-47ab-906c-2cbd7ccf196a/home/agniv/Documents/debug_non_rigid/"+to_string(iteration)+"_data"+file_name+".ply", *pcl_cloud);
    //       pcl::io::savePLYFileBinary("/media/agniv/f9826023-e8c9-47ab-906c-2cbd7ccf196a/home/agniv/Documents/debug_non_rigid/"+to_string(iteration)+"_mesh"+file_name+".ply", mesh.mesh);
    //       pcl::io::savePLYFileBinary("/media/agniv/f9826023-e8c9-47ab-906c-2cbd7ccf196a/home/agniv/Documents/debug_non_rigid/"+to_string(iteration)+"_base"+file_name+".ply", poly_mesh);
    //    }
    //    else
    //    {
    //        pcl::io::savePLYFileBinary("/media/agniv/f9826023-e8c9-47ab-906c-2cbd7ccf196a/home/agniv/Documents/debug_non_rigid/"+to_string(iteration)+"_"+file_name+".ply", poly_mesh);
    //    }

    Eigen::Vector4f xyz_centroid;
    pcl::compute3DCentroid(*vertices, xyz_centroid);

    //////std:://cout<<"Centroid: "<<xyz_centroid<<std::endl;
    pcl::PointXYZ centroid;
    centroid.x = xyz_centroid[0];
    centroid.y = xyz_centroid[1];
    centroid.z = xyz_centroid[2];

    int K = 1;
    float dist = 0.25;

    std::vector<pcl::PointXYZ> visible_list;
    std::vector<pcl::PointXYZ> corresponding_model_list;

    std::vector<double> residual;

    std::vector<pcl::Vertices, std::allocator<pcl::Vertices>>::iterator face;
    pcl::PointCloud<pcl::PointXYZ>::Ptr vertices_visible(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(mesh.mesh.cloud, *vertices_visible);

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(vertices);

    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(dist);

    int num_points = 0;

    for (face = mesh.mesh.polygons.begin(); face != mesh.mesh.polygons.end(); ++face) {
        for (int i = 0; i < 3; i++) {
            unsigned int v = face->vertices[i];
            pcl::PointXYZ p = vertices_visible->points.at(v);
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

    if (is_first) {
        if (!correspondence.empty()) {
            correspondence.clear();
        }
    }

    int corr_count = 0;

    bool isDone = false;

    if (pcl_cloud->isOrganized()) {
        for (int k = 0; (k < visible_list.size()) && (!isDone); k += 3)  // <-- here 'k+=3' is needed because 'vertices_visible_list' had three points added per triangle in <<compute_jacobian>>
        {
            double res = 0.0f;

            if (is_first) {
                pcl::PointXYZ normal = normal_estimation(corresponding_model_list[k], corresponding_model_list[k + 1], corresponding_model_list[k + 2], centroid);

                point_2d a = ocl.projection(corresponding_model_list[k]);     /************************************************************************/
                point_2d b = ocl.projection(corresponding_model_list[k + 1]); /*These three points are supposed to represent a single triangular plane*/
                point_2d c = ocl.projection(corresponding_model_list[k + 2]); /************************************************************************/

                float min_x, min_y, max_x, max_y;

                min_x = min_val(a.x, b.x, c.x);
                max_x = max_val(a.x, b.x, c.x);
                min_y = min_val(a.y, b.y, c.y);
                max_y = max_val(a.y, b.y, c.y);

                if ((min_x > 0) && (min_y > 0) && (max_x < IMAGE_WIDTH) && (max_y < IMAGE_HEIGHT)) {
                    for (int i = min_x; i < max_x /*pcl_cloud->width*/; i++) {
                        for (int j = min_y; j < max_y /*pcl_cloud->height*/; j++) {
                            pcl::PointXYZ pt = pcl_cloud->at(i, j);

                            if (((pt.x != 0) || (pt.y != 0) || (pt.z != 0)) && (pt.z < MAX_DEPTH) /*pcl::isFinite (pt)*/)  // nearly 2 secs here
                            {
                                if ((i > min_x) && (i < max_x) && (j > min_y) && (j < max_y)) {
                                    point_2d target;
                                    target.x = i;
                                    target.y = j;

                                    if (ocl.point_in_triangle(target, a, b, c))  // 7 msec
                                    {
                                        res = (normal.x * (pt.x - corresponding_model_list[k].x)) + (normal.y * (pt.y - corresponding_model_list[k].y)) + (normal.z * (pt.z - corresponding_model_list[k].z));
                                        residual.push_back(res);
                                        std::vector<double> corr;
                                        corr.push_back(k);
                                        corr.push_back(pt.x);
                                        corr.push_back(pt.y);
                                        corr.push_back(pt.z);
                                        correspondence.push_back(corr);
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

                    pcl::PointXYZ normal = normal_estimation(corresponding_model_list[index], corresponding_model_list[index + 1], corresponding_model_list[index + 2], centroid);
                    res = (normal.x * (corr[1] - corresponding_model_list[index].x)) + (normal.y * (corr[2] - corresponding_model_list[index].y)) + (normal.z * (corr[3] - corresponding_model_list[index].z));

                    residual.push_back(res);
                    num_points++;
                }

                isDone = true;
            }
            corr_count++;
        }
    }

    Eigen::MatrixXd residual_matrix(residual.size(), 1);

    for (int i = 0; i < residual.size(); i++) {
        residual_matrix(i, 0) = residual[i];
    }

    return residual_matrix;
}

// void gauss_newton_eigen(Eigen::MatrixXd &residual, Eigen::MatrixXd &J, Eigen::MatrixXd &update, int iteration)
//{
//     Eigen::MatrixXd W = assign_weights_to_matrix(residual);
//
//     Eigen::MatrixXd AtWA = (J.transpose()*W*J);
//     update = -0.0005f*(AtWA.inverse())*J.transpose()*W*residual;
//
// }
//
// void gauss_newton_eigen_sparse(Eigen::MatrixXd &residual, Eigen::MatrixXd &J, Eigen::MatrixXd &update, int iteration)
//{
//     Eigen::MatrixXd W = assign_weights_to_matrix(residual);
//
//     Eigen::MatrixXd AtWA = (J.transpose()*W*J);
//
//     SparseMatrix<double> A_ = AtWA.sparseView();
//     SimplicialLLT<SparseMatrix<double> > solver;
//     solver.compute(A_);
//     auto A_inv = solver.solve(J.transpose()*W*residual);
//     Eigen::MatrixXd del = MatrixXd(A_inv);
//
//
//     update = -0.0005f*del;
//
// }
//
// void gauss_newton_eigen_sparse_unweighted(Eigen::MatrixXd &residual, Eigen::MatrixXd &J, Eigen::MatrixXd &update, int iteration)
//{
//     Eigen::MatrixXd AtWA = (J.transpose()*J);
//
//     SparseMatrix<double> A_ = AtWA.sparseView();
//     SimplicialLLT<SparseMatrix<double> > solver;
//     solver.compute(A_);
//     auto A_inv = solver.solve(J.transpose()*residual);
//     Eigen::MatrixXd del = MatrixXd(A_inv);
//
//     //update = -700.0f*del; //shar cube
//     update = -0.0002f*del;
//
// }
//
// void gauss_newton_eigen_unweighted(Eigen::MatrixXd &residual, Eigen::MatrixXd &J, Eigen::MatrixXd &update, int iteration)
//{
//     Eigen::MatrixXd AtWA = (J.transpose()*J);
//     update = -50.0f*(AtWA.inverse())*J.transpose()*residual;
// }

void gauss_newton_visp(Eigen::MatrixXd &residual, Eigen::MatrixXd &J, Eigen::MatrixXd &update, int iteration) {
    std::cout << "C++: " << "gauss_newton_visp" << std::endl;
    W.resize(residual.rows(), false);
    std::cout << "C++: " << "debug 0.1" << std::endl;
    assign_weights_to_matrix_visp(residual, W);
    std::cout << "C++: " << "residual rows: " << residual.rows() << std::endl;
    // std::cout << "C++: " << "W: " << W << std::endl;
    std::cout << "C++: " << "debug 1" << std::endl;

    for (int i = 0; i < residual.rows(); i++) {
        residual(i, 0) *= W[i];
    }

    std::cout << "C++: " << "debug 2" << std::endl;
    ocl.eigen_to_visp(residual, R);
    ocl.eigen_to_visp(J, J_);

    std::cout << "C++: " << "debug 3" << std::endl;
    AtWA = (J_.transpose() * J_);

    lm.eye(AtWA.getRows(), AtWA.getCols());

    std::cout << "C++: " << "debug 4" << std::endl;
    float lambda = 10.0f;

    if (iteration > 2) {
        lambda /= 100.0f;
    } else if (iteration > 5) {
        lambda /= 1000.0f;
    }

    AtWA = AtWA + (lambda * lm);

    v = -0.001 * (AtWA.pseudoInverse(AtWA.getRows() * std::numeric_limits<double>::epsilon())) * J_.transpose() * R;

    std::cout << "C++: " << "debug 5" << std::endl;
    update.resize(v.getRows(), 1);

    for (int i = 0; i < v.getRows(); i++) {
        update(i, 0) = v[i][0];
    }
    std::cout << "C++: " << "debug 6" << std::endl;
    std::cout << "C++: " << "END gauss_newton_visp" << std::endl;
}

void gauss_newton_visp_unweighted(Eigen::MatrixXd &residual, Eigen::MatrixXd &J, Eigen::MatrixXd &update, int iteration) {
    // vpMatrix W;
    // assign_weights_to_matrix_visp(residual, W);

    // cout<<residual.rows()<<","<<residual.cols()<<endl;
    // cout<<J.rows()<<","<<J.cols()<<endl;

    ocl.eigen_to_visp(residual, R);
    ocl.eigen_to_visp(J, J_);

    AtWA = (J_.transpose() /*W*/ * J_);
    //  J.AtA(AtWA) ;

    lm.eye(AtWA.getRows(), AtWA.getCols());

    float lambda = 10.0f;
    float mu = 0.00015f;

    if (iteration > 5) {
        lambda /= 2.0f;
        mu /= 1.1f;
    }

    // AtWA = AtWA + (lambda*lm);
    v = -mu * (AtWA.pseudoInverse(AtWA.getRows() * std::numeric_limits<double>::epsilon())) * J_.transpose() * R;

    update.resize(v.getRows(), 1);

    for (int i = 0; i < v.getRows(); i++) {
        update(i, 0) = v[i][0];
    }
}

void compute_analytic_part_jacobian() {
    // STUB: to compute partly analytic jacobian - coming up next
}

double get_mean_of_col(Eigen::MatrixXd M, int col_num) {
    double sum = 0.000f;
    int rows = M.rows();

    for (int i = 0; i < rows; i++) {
        sum += M(i, col_num);
    }

    return (sum / (double)rows);
}

double compute_jacobian(int num_forces, pcl::PolygonMesh &visible_points, pcl::PointCloud<pcl::PointXYZ>::Ptr &pcl_cloud, Eigen::Matrix4f &transform, mesh_map &mesh, vector<PolygonMesh> &deformed_models, Eigen::MatrixXd &update, int iteration, int pcd_count) {
    std::cout << "C++: " << "compute_jacobian" << std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr vertices_visible(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(visible_points.cloud, *vertices_visible);           // EXTRACTING VISIBLE VERTICES FROM VISIBLE MESH
    pcl::transformPointCloud(*vertices_visible, *vertices_visible, transform);  // transforming those vertices //NOTE::this is redundant transformation

    std::cout << "C++: " << "debug 1" << std::endl;
    // cout<<"visible size: "<<visible_points.cloud.width<<endl;

    int jacobian_count = 0;
    int force_count = 0;

    std::string debugmsg = to_string(pcd_count) + "R_";
    std::string debugmsg_J = "J";
    Eigen::MatrixXd res_ = get_residual(visible_points, pcl_cloud, transform, true, debugmsg, mesh, iteration);

    Eigen::MatrixXd J(res_.rows(), (num_forces));

    std::cout << "C++: " << "debug 2" << std::endl;
    J = Eigen::MatrixXd::Zero((res_.rows()), (num_forces) / 2);

    std::cout << "C++: " << "debug 3" << std::endl;
    for (int i = 0; i < (num_forces / 6); i++) {
        Eigen::MatrixXd j1, j2, j3, j4, j5, j6;
        pcl::PointCloud<pcl::PointXYZ>::Ptr vertices_visible_J(new pcl::PointCloud<pcl::PointXYZ>);

        int case_ = 0;

        for (int j = (6 * i); j < ((6 * i) + 6); j++) {
            Eigen::MatrixXd res;
            ocl.transformPolygonMesh(deformed_models[j], transform);

            switch (case_++) {
                case 0:
                    debugmsg_J = to_string(pcd_count) + "Jx+";
                    res = get_residual(deformed_models[j], pcl_cloud, transform, false, debugmsg_J, mesh, iteration);
                    j1 = res;
                    break;
                case 1:
                    debugmsg_J = to_string(pcd_count) + "Jx-";
                    j2 = get_residual(deformed_models[j], pcl_cloud, transform, false, debugmsg_J, mesh, iteration);
                    break;
                case 2:
                    debugmsg_J = to_string(pcd_count) + "Jy+";
                    j3 = get_residual(deformed_models[j], pcl_cloud, transform, false, debugmsg_J, mesh, iteration);
                    break;
                case 3:
                    debugmsg_J = to_string(pcd_count) + "Jy-";
                    j4 = get_residual(deformed_models[j], pcl_cloud, transform, false, debugmsg_J, mesh, iteration);
                    break;
                case 4:
                    debugmsg_J = to_string(pcd_count) + "Jz+";
                    j5 = get_residual(deformed_models[j], pcl_cloud, transform, false, debugmsg_J, mesh, iteration);
                    break;
                case 5:
                    debugmsg_J = to_string(pcd_count) + "Jz-";
                    j6 = get_residual(deformed_models[j], pcl_cloud, transform, false, debugmsg_J, mesh, iteration);
                    break;
                default:
                    std::  // cout<<"STRANGE SWITCH-CASE FOR JACOBIAN. EXITING"<<std::endl;
                        exit(0);
            }
        }

        Eigen::MatrixXd J_1 = (j1 - j2);
        Eigen::MatrixXd J_2 = (j3 - j4);
        Eigen::MatrixXd J_3 = (j5 - j6);

        for (int k = 0; k < J_1.size(); k++) {
            J(k, (i * 3) + 0) = J_1(k, 0);
            J(k, (i * 3) + 1) = J_2(k, 0);
            J(k, (i * 3) + 2) = J_3(k, 0);
        }
    }

    // gauss_newton_eigen(res_, J, update, iteration);

    // gauss_newton_eigen_unweighted(res_, J, update, iteration);

    // gauss_newton_eigen_sparse_unweighted(res_, J, update, iteration);

    // gauss_newton_eigen_sparse(res_, J, update, iteration);

    std::cout << "C++: " << "debug 4" << std::endl;
    gauss_newton_visp(res_, J, update, iteration);
    std::cout << "C++: " << "gauss_newton_visp" << std::endl;

    // gauss_newton_visp_unweighted(res_, J, update, iteration);

    std::cout << "C++: " << "debug 5" << std::endl;
#ifdef DEBUG_DUMP
    return get_mean_of_col(res_, 0);
#else
    return 0.0000f;
#endif
}

void JacobianFEM::set_iteration(int iter) { iteration = iter; }

void JacobianFEM::set_pcd_count(int count) { pcd_count = count; }

void JacobianFEM::set_camera_parameters(float Fx, float Fy, float Cx, float Cy) {
    F_x = Fx;
    F_y = Fy;
    C_x = Cx;
    C_y = Cy;
}

void JacobianFEM::set_file_num(int _file_num) { file_num_ = _file_num; }

/***************************************/
// PointCloud<PointXYZ>::Ptr pointcloud : current frame
// PolygonMesh &base_model : model registered to last frame
// vector<PolygonMesh> &deformed_models : set of deformed models generated from Jacobian deformation
// Eigen::Matrix4f &transform : cMo for last frame
/***************************************/
double JacobianFEM::compute_update(PointCloud<PointXYZ>::Ptr &pointcloud, PolygonMesh base_model, vector<PolygonMesh> &deformed_models, Eigen::Matrix4f &transform, Eigen::MatrixXd &update) {
    std::cout << "C++: " << "compute_update" << std::endl;
    std::cout << "C++: " << "debug 1" << std::endl;
    ocl.transformPolygonMesh(base_model, transform);  //  <----- the transformed model file is here
    std::cout << "C++: " << "debug 2" << std::endl;
    ocl.set_camera_params(F_x, F_y, C_x, C_y, 0.0f, 0.0f, 0.0f, 0.0f);
    std::cout << "C++: " << "debug 3" << std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr vertices(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(base_model.cloud, *vertices);  // vertices of the base model
    // cout<<"in here -> width: "<<pointcloud->width<<endl;
    std::cout << "C++: " << "debug 4" << std::endl;

    mesh_map mesh_ = ocl.get_visibility_vtk(base_model);
    std::cout << "C++: " << "debug 5" << std::endl;
    double residual = compute_jacobian(deformed_models.size(), base_model, pointcloud, transform, mesh_, deformed_models, update, iteration, pcd_count);
    // cout<<"computed J with "<<J.rows()<<" rows"<<endl;
    std::cout << "C++: " << "debug 6" << std::endl;
    return residual;
}
