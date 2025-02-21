/******************************************************************************
 *       SOFA, Simulation Open-Framework Architecture, development version     *
 *                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
 *                                                                             *
 * This program is free software; you can redistribute it and/or modify it     *
 * under the terms of the GNU Lesser General Public License as published by    *
 * the Free Software Foundation; either version 2.1 of the License, or (at     *
 * your option) any later version.                                             *
 *                                                                             *
 * This program is distributed in the hope that it will be useful, but WITHOUT *
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
 * for more details.                                                           *
 *                                                                             *
 * You should have received a copy of the GNU Lesser General Public License    *
 * along with this program. If not, see <http://www.gnu.org/licenses/>.        *
 *******************************************************************************
 * Authors: The SOFA Team and external contributors (see Authors.txt)          *
 *                                                                             *
 * Contact information: contact@sofa-framework.org                             *
 ******************************************************************************/
#ifndef SOFA_COMPONENT_FORCEFIELD_ORTHOTROPICTETRAHEDRALCOROTATIONALFEMFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_ORTHOTROPICTETRAHEDRALCOROTATIONALFEMFORCEFIELD_INL

// #include <sofa/component/topology/container/grid/GridTopology.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/decompose.h>
#include <sofa/simulation/Simulation.h>
// #include <SofaBaseTopology/GridTopology.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

// #include <SofaBaseTopology/TopologyData.inl>
#include <cassert>
#include <iostream>
#include <set>

#include "OrthotropicTetrahedralCorotationalFEMForceField.h"

int count;

namespace sofa {

namespace component {

namespace forcefield {

template <class DataTypes>
void OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::TetrahedronHandler::applyCreateFunction(unsigned int tetrahedronIndex, TetrahedronInformation&, const core::topology::BaseMeshTopology::Tetrahedron&, const sofa::type::vector<unsigned int>&, const sofa::type::vector<double>&) {
    if (ff) {
        const core::topology::BaseMeshTopology::Tetrahedron t = ff->_topology->getTetrahedron(tetrahedronIndex);
        Index a = t[0];
        Index b = t[1];
        Index c = t[2];
        Index d = t[3];

        switch (ff->method) {
            case SMALL:
                ff->computeMaterialStiffness(tetrahedronIndex, a, b, c, d);
                ff->initSmall(tetrahedronIndex, a, b, c, d);
                break;
            case LARGE:
                ff->computeMaterialStiffness(tetrahedronIndex, a, b, c, d);
                ff->initLarge(tetrahedronIndex, a, b, c, d);

                break;
            case POLAR:
                ff->computeMaterialStiffness(tetrahedronIndex, a, b, c, d);
                ff->initPolar(tetrahedronIndex, a, b, c, d);
                break;
        }
    }
}

template <class DataTypes>
OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::OrthotropicTetrahedralCorotationalFEMForceField()
    : tetrahedronInfo(initData(&tetrahedronInfo, "tetrahedronInfo", "Internal tetrahedron data")),
      f_method(initData(&f_method, std::string("large"), "method", "\"small\", \"large\" (by QR) or \"polar\" displacements")),
      _poissonRatio12(core::objectmodel::BaseObject::initData(&_poissonRatio12, (Real)0.45f, "poissonRatio12", "FEM Poisson Ratio12")),
      _poissonRatio21(core::objectmodel::BaseObject::initData(&_poissonRatio21, (Real)0.45f, "poissonRatio21", "FEM Poisson Ratio21")),
      _poissonRatio13(core::objectmodel::BaseObject::initData(&_poissonRatio13, (Real)0.45f, "poissonRatio13", "FEM Poisson Ratio13")),
      _poissonRatio31(core::objectmodel::BaseObject::initData(&_poissonRatio31, (Real)0.45f, "poissonRatio31", "FEM Poisson Ratio31")),
      _poissonRatio23(core::objectmodel::BaseObject::initData(&_poissonRatio23, (Real)0.45f, "poissonRatio23", "FEM Poisson Ratio23")),
      _poissonRatio32(core::objectmodel::BaseObject::initData(&_poissonRatio32, (Real)0.45f, "poissonRatio32", "FEM Poisson Ratio32")),
      _youngModulus1(core::objectmodel::BaseObject::initData(&_youngModulus1, (Real)5000, "youngModulus1", "FEM Young Modulus1")),
      _youngModulus2(core::objectmodel::BaseObject::initData(&_youngModulus2, (Real)5000, "youngModulus2", "FEM Young Modulus2")),
      _youngModulus3(core::objectmodel::BaseObject::initData(&_youngModulus3, (Real)5000, "youngModulus3", "FEM Young Modulus3")),
      _localStiffnessFactor(
          core::objectmodel::BaseObject::initData(&_localStiffnessFactor, "localStiffnessFactor", "Allow specification of different stiffness per element. If there are N element and M values are specified, the youngModulus factor for element i would be localStiffnessFactor[i*M/N]")),
      _updateStiffnessMatrix(core::objectmodel::BaseObject::initData(&_updateStiffnessMatrix, false, "updateStiffnessMatrix", "")),
      _assembling(core::objectmodel::BaseObject::initData(&_assembling, false, "computeGlobalMatrix", "")),
      f_drawing(initData(&f_drawing, true, "drawing", " draw the forcefield if true")),
      drawColor1(initData(&drawColor1, sofa::type::RGBAColor(0.0f, 0.0f, 1.0f, 1.0f), "drawColor1", " draw color for faces 1")),
      drawColor2(initData(&drawColor2, sofa::type::RGBAColor(0.0f, 0.5f, 1.0f, 1.0f), "drawColor2", " draw color for faces 2")),
      drawColor3(initData(&drawColor3, sofa::type::RGBAColor(0.0f, 1.0f, 1.0f, 1.0f), "drawColor3", " draw color for faces 3")),
      drawColor4(initData(&drawColor4, sofa::type::RGBAColor(0.5f, 1.0f, 1.0f, 1.0f), "drawColor4", " draw color for faces 4")),
      tetrahedronHandler(NULL) {
    this->addAlias(&_assembling, "assembling");
    _poissonRatio12.setWidget("poissonRatio12");
    _poissonRatio21.setWidget("poissonRatio21");
    _poissonRatio13.setWidget("poissonRatio13");
    _poissonRatio31.setWidget("poissonRatio31");
    _poissonRatio23.setWidget("poissonRatio23");
    _poissonRatio32.setWidget("poissonRatio32");
    tetrahedronHandler = new TetrahedronHandler(this, &tetrahedronInfo);

    _poissonRatio12.setRequired(true);
    _poissonRatio21.setRequired(true);
    _poissonRatio13.setRequired(true);
    _poissonRatio31.setRequired(true);
    _poissonRatio23.setRequired(true);
    _poissonRatio32.setRequired(true);
    _youngModulus1.setRequired(true);
    _youngModulus2.setRequired(true);
    _youngModulus3.setRequired(true);
    count = 0;
}

template <class DataTypes>
void OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::init() {
    this->core::behavior::ForceField<DataTypes>::init();

    _topology = this->getContext()->getMeshTopology();

    if (_topology == NULL) {
        msg_error() << "No topology component found. This FEM needs to rely on a Tetrahedral Topology.";
        this->d_componentState = sofa::core::objectmodel::ComponentState::Invalid;
        return;
    }

    reinit();  // compute per-element stiffness matrices and other precomputed values
}

template <class DataTypes>
void OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::reinit() {
    if (f_method.getValue() == "small")
        this->setMethod(SMALL);
    else if (f_method.getValue() == "polar")
        this->setMethod(POLAR);
    else
        this->setMethod(LARGE);

    // Need to initialize the _stiffnesses vector before using it
    size_t sizeMO = this->mstate->getSize();
    if (_assembling.getValue()) {
        _stiffnesses.resize(sizeMO * 3);
    }

    type::vector<typename OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::TetrahedronInformation>& tetrahedronInf = *(tetrahedronInfo.beginEdit());

    tetrahedronInf.resize(_topology->getNbTetrahedra());

    for (size_t i = 0; i < _topology->getNbTetrahedra(); ++i) {
        tetrahedronHandler->applyCreateFunction(i, tetrahedronInf[i], _topology->getTetrahedron(i), (const std::vector<unsigned int>)0, (const std::vector<double>)0);
    }

    // Need to check _topology needs to be updated regarding SOFA 20.06

    tetrahedronInfo.createTopologyHandler(_topology);
    // tetrahedronInfo.createTopologyHandler(_topology, tetrahedronHandler);
    // tetrahedronInfo.registerTopologicalData();

    tetrahedronInfo.endEdit();
}

long long current_timestamp() {
    struct timeval te;
    gettimeofday(&te, NULL);                                          // get current time
    long long milliseconds = te.tv_sec * 1000LL + te.tv_usec / 1000;  // calculate milliseconds
    // printf("milliseconds: %lld\n", milliseconds);
    return milliseconds;
}

template <class DataTypes>
void OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::addForce(const core::MechanicalParams* /* mparams */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& /* d_v */) {
    VecDeriv& f = *d_f.beginEdit();
    const VecCoord& p = d_x.getValue();

    FILE* fp;

    /*long long time = current_timestamp();
    //char buffer[30];
    //sprintf(buffer,"%lu", time);
    char dummy[255];
    //char path[]

    time_t timer;
    char buffer[26];
    struct tm* tm_info;

    time(&timer);
    tm_info = localtime(&timer);

    strftime(buffer, 26, "%Y-%m-%d %H:%M:%S", tm_info);*/

    /*char buffer[26];
    int millisec;
    struct tm* tm_info;
    struct timeval tv;

    gettimeofday(&tv, NULL);

    millisec = lrint(tv.tv_usec/1000.0); // Round to nearest millisec
    if (millisec>=1000) { // Allow for rounding up to nearest second
      millisec -=1000;
      tv.tv_sec++;
    }

    tm_info = localtime(&tv.tv_sec);

    strftime(buffer, 26, "%Y:%m:%d %H:%M:%S", tm_info);*/

    // char fn[100+1];

    // snprintf(fn, 100, "/media/agniv/f9826023-e8c9-47ab-906c-2cbd7ccf196a/home/agniv/Documents/data/real_data/pizza_new/staging_area/addnl/Force%d.csv", count);

    fp = fopen("/media/agniv/f9826023-e8c9-47ab-906c-2cbd7ccf196a/home/agniv/Documents/data/real_data/pizza_new/staging_area/addnl/Force.csv", "a+");

    switch (method) {
        case SMALL: {
            for (size_t i = 0; i < _topology->getNbTetrahedra(); ++i) {
                accumulateForceSmall(f, p, i);
            }
            break;
        }
        case LARGE: {
            float sum_x = 0.0f;
            float sum_y = 0.0f;
            float sum_z = 0.0f;
            for (size_t i = 0; i < _topology->getNbTetrahedra(); ++i) {
                accumulateForceLarge(f, p, i);
                /*if((i==90)||(i==306)||(i==748)||(i==927)||(i==1015)||(i==1061)||(i==828)||(i==113)/*||(i==2140)||(i==2143)||(i==2249)||(i==144)||
                       (i==145)||(i==157)||(i==158)||(i==162)||(i==173)||(i==175)||(i==230)||(i==266)||(i==280)||(i==281)||
                        (i==287)|comment till here|)*/
                /*if((i== 18 )|| (i== 21 )|| (i== 29 )|| (i== 31 )|| (i== 55 )|| (i== 55 )|| (i== 76 )|| (i== 90 )|| (i== 90 )|| (i== 110 )|| (i== 113 )|| (i== 113 )|| (i== 115 )|| (i== 115 )|| (i== 117 )|| (i== 117 )|| (i== 132 )|| (i== 140 )|| (i== 158 )|| (i== 189 )|| (i== 208 )|| (i== 252 )||
                 * (i== 306 )|| (i== 306 )|| (i== 342 )|| (i== 351 )|| (i== 351 )|| (i== 380 )|| (i== 381 )|| (i== 394 )|| (i== 400 )|| (i== 451 )|| (i== 527 )|| (i== 527 )|| (i== 548 )|| (i== 577 )|| (i== 578 )|| (i== 596 )|| (i== 601 )|| (i== 611 )|| (i== 611 )|| (i== 655 )|| (i== 686 )|| (i==
                 688
                 * )|| (i== 688 )|| (i== 692 )|| (i== 696 )|| (i== 696 )|| (i== 705 )|| (i== 726 )|| (i== 726 )|| (i== 736 )|| (i== 748 )|| (i== 748 )|| (i== 783 )|| (i== 796 )|| (i== 804 )|| (i== 804 )|| (i== 814 )|| (i== 814 )|| (i== 820 )|| (i== 821 )|| (i== 821 )|| (i== 822 )|| (i== 823 )||
                 (i==
                 * 828 )|| (i== 828 )|| (i== 841 )|| (i== 844 )|| (i== 847 )|| (i== 847 )|| (i== 853 )|| (i== 853 )|| (i== 854 )|| (i== 855 )|| (i== 855 )|| (i== 856 )|| (i== 857 )|| (i== 857 )|| (i== 861 )|| (i== 862 )|| (i== 869 )|| (i== 927 )|| (i== 927 )|| (i== 1002 )|| (i== 1002 )|| (i==
                 1015
                 * )|| (i== 1015 )|| (i== 1019 )|| (i== 1025 )|| (i== 1025 )|| (i== 1026 )|| (i== 1030 )|| (i== 1030 )|| (i== 1059 )|| (i== 1061 )|| (i== 1061 )|| (i== 1105 )|| (i== 1151 )|| (i== 1151 )|| (i== 1221 )|| (i== 1224 )|| (i== 1241 )|| (i== 1256 )|| (i== 1265 )|| (i== 1265 ))*/

                /*if((i==1033)||(i==2003)||(i==2702)||(i==3343)||(i==2884)||(i==3003)||(i==2281)||(i==2166)||(i==89)||(i==2703)||(i==2597)||(i==1381)||(i==2590)||(i==2578)||(i==1032)
                            ||(i==950)||(i==1068)||(i==1076)||(i==164)||(i==1067)||(i==1413)||(i==341)||(i==185)||(i==294)||(i==1235)||(i==2786)||(i==2989)||(i==268)||(i==2578)||(i==2821)||(i==4190)||(i==253)||(i==4037)||(i==950)||(i==1078)||(i==1037)||(i==164)||(i==341)||(i==1032)||(i==1748)||(i==2252)||(i==2703)||(i==863)||(i==1415)||(i==1233)||(i==166)||(i==683)||(i==1084)||(i==357)||(i==243)||(i==1092)||(i==294)||(i==1235)||(i==1498)||(i==2136))*/
                ///--->last one for sponge deform

                if ((i == 189) || (i == 341) || (i == 477) || (i == 513) || (i == 683) || (i == 742) || (i == 865) || (i == 898) || (i == 982) || (i == 133) || (i == 137) || (i == 144) || (i == 145) || (i == 149) || (i == 155) || (i == 161) || (i == 165) || (i == 1252) || (i == 126) ||
                    (i == 1267) || (i == 1335) || (i == 1413) || (i == 1645) || (i == 1764) || (i == 1899) || (i == 1964) || (i == 1988) || (i == 27) || (i == 238) || (i == 24) || (i == 253) || (i == 261) || (i == 2129) || (i == 2166) || (i == 222) || (i == 2251) || (i == 259) || (i == 272) ||
                    (i == 2884) || (i == 3185) || (i == 441) || (i == 456) || (i == 457) || (i == 4133) || (i == 23) || (i == 185) || (i == 259) || (i == 294) || (i == 363) || (i == 541) || (i == 768) || (i == 138) || (i == 14) || (i == 143) || (i == 158) || (i == 167) || (i == 1235) ||
                    (i == 1254) || (i == 1312) || (i == 1576) || (i == 164))
                /*||(i==1895)||(i==191)||(i==23)||(i==239)||(i==258)||(i==26)||(i==21)||(i==2133)||(i==2281)||(i==273)||(i==2786)||(i==2816)||(i==2821)||(i==2927)||(i==33)||(i==395)||(i==381)||(i==443)||(i==452)||(i==453)||(i==458)||(i==4138)||(i==12)||(i==89)||(i==166)||(i==268)||(i==28)||(i==322)||(i==912)||(i==938)||(i==95)||(i==132)||(i==168)||(i==1213)||(i==1233)||(i==1381)||(i==1415)||(i==1495)||(i==1496)||(i==1498)||(i==1748)||(i==1771)||(i==1798)||(i==183)||(i==1919)||(i==1992)||(i==288)||(i==2136)||(i==2167)||(i==2197)||(i==2252)||(i==2253)||(i==2364)||(i==2578)||(i==2597)||(i==2651)||(i==2989)||(i==394)||(i==3219)||(i==3343)||(i==44)||(i==442)||(i==4134)||(i==4139)||(i==414)||(i==4141)||(i==419)||(i==4323)||(i==4324)||(i==36)||(i==39)||(i==236)||(i==356)||(i==448)||(i==55)||(i==516)||(i==561)||(i==574)||(i==725)||(i==728))*/
                {
                    sum_x += f[0][i];
                    sum_y += f[1][i];
                    sum_z += f[2][i];
                }
            }

            fprintf(fp, "%d,%f,%f,%f\n", count, sum_x, sum_y, sum_z);
            break;
        }
        case POLAR: {
            for (size_t i = 0; i < _topology->getNbTetrahedra(); ++i) {
                accumulateForcePolar(f, p, i);
            }
            break;
        }
    }

    fclose(fp);
    count++;

    d_f.endEdit();
}

template <class DataTypes>
void OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::addDForce(const sofa::core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx) {
    VecDeriv& df = *d_df.beginEdit();
    const VecDeriv& dx = d_dx.getValue();

    Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

    switch (method) {
        case SMALL: {
            for (size_t i = 0; i < _topology->getNbTetrahedra(); ++i) {
                const core::topology::BaseMeshTopology::Tetrahedron t = _topology->getTetrahedron(i);
                Index a = t[0];
                Index b = t[1];
                Index c = t[2];
                Index d = t[3];

                applyStiffnessSmall(df, dx, i, a, b, c, d, kFactor);
            }
            break;
        }
        case LARGE: {
            for (size_t i = 0; i < _topology->getNbTetrahedra(); ++i) {
                const core::topology::BaseMeshTopology::Tetrahedron t = _topology->getTetrahedron(i);
                Index a = t[0];
                Index b = t[1];
                Index c = t[2];
                Index d = t[3];

                applyStiffnessLarge(df, dx, i, a, b, c, d, kFactor);
            }
            break;
        }
        case POLAR: {
            for (size_t i = 0; i < _topology->getNbTetrahedra(); ++i) {
                const core::topology::BaseMeshTopology::Tetrahedron t = _topology->getTetrahedron(i);
                Index a = t[0];
                Index b = t[1];
                Index c = t[2];
                Index d = t[3];

                applyStiffnessPolar(df, dx, i, a, b, c, d, kFactor);
            }
            break;
        }
    }

    d_df.endEdit();
}

template <class DataTypes>
void OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::computeStrainDisplacement(StrainDisplacementTransposed& J, Coord a, Coord b, Coord c, Coord d) {
    // shape functions matrix
    type::Mat<2, 3, Real> M;

    M[0][0] = b[1];
    M[0][1] = c[1];
    M[0][2] = d[1];
    M[1][0] = b[2];
    M[1][1] = c[2];
    M[1][2] = d[2];
    J[0][0] = J[1][3] = J[2][5] = -peudo_determinant_for_coef(M);
    M[0][0] = b[0];
    M[0][1] = c[0];
    M[0][2] = d[0];
    J[0][3] = J[1][1] = J[2][4] = peudo_determinant_for_coef(M);
    M[1][0] = b[1];
    M[1][1] = c[1];
    M[1][2] = d[1];
    J[0][5] = J[1][4] = J[2][2] = -peudo_determinant_for_coef(M);

    M[0][0] = c[1];
    M[0][1] = d[1];
    M[0][2] = a[1];
    M[1][0] = c[2];
    M[1][1] = d[2];
    M[1][2] = a[2];
    J[3][0] = J[4][3] = J[5][5] = peudo_determinant_for_coef(M);
    M[0][0] = c[0];
    M[0][1] = d[0];
    M[0][2] = a[0];
    J[3][3] = J[4][1] = J[5][4] = -peudo_determinant_for_coef(M);
    M[1][0] = c[1];
    M[1][1] = d[1];
    M[1][2] = a[1];
    J[3][5] = J[4][4] = J[5][2] = peudo_determinant_for_coef(M);

    M[0][0] = d[1];
    M[0][1] = a[1];
    M[0][2] = b[1];
    M[1][0] = d[2];
    M[1][1] = a[2];
    M[1][2] = b[2];
    J[6][0] = J[7][3] = J[8][5] = -peudo_determinant_for_coef(M);
    M[0][0] = d[0];
    M[0][1] = a[0];
    M[0][2] = b[0];
    J[6][3] = J[7][1] = J[8][4] = peudo_determinant_for_coef(M);
    M[1][0] = d[1];
    M[1][1] = a[1];
    M[1][2] = b[1];
    J[6][5] = J[7][4] = J[8][2] = -peudo_determinant_for_coef(M);

    M[0][0] = a[1];
    M[0][1] = b[1];
    M[0][2] = c[1];
    M[1][0] = a[2];
    M[1][1] = b[2];
    M[1][2] = c[2];
    J[9][0] = J[10][3] = J[11][5] = peudo_determinant_for_coef(M);
    M[0][0] = a[0];
    M[0][1] = b[0];
    M[0][2] = c[0];
    J[9][3] = J[10][1] = J[11][4] = -peudo_determinant_for_coef(M);
    M[1][0] = a[1];
    M[1][1] = b[1];
    M[1][2] = c[1];
    J[9][5] = J[10][4] = J[11][2] = peudo_determinant_for_coef(M);

    // 0
    J[0][1] = J[0][2] = J[0][4] = J[1][0] = J[1][2] = J[1][5] = J[2][0] = J[2][1] = J[2][3] = 0;
    J[3][1] = J[3][2] = J[3][4] = J[4][0] = J[4][2] = J[4][5] = J[5][0] = J[5][1] = J[5][3] = 0;
    J[6][1] = J[6][2] = J[6][4] = J[7][0] = J[7][2] = J[7][5] = J[8][0] = J[8][1] = J[8][3] = 0;
    J[9][1] = J[9][2] = J[9][4] = J[10][0] = J[10][2] = J[10][5] = J[11][0] = J[11][1] = J[11][3] = 0;

    // m_deq( J, 1.2 ); //hack for stability ??
}

template <class DataTypes>
typename OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::Real OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::peudo_determinant_for_coef(const type::Mat<2, 3, Real>& M) {
    return M[0][1] * M[1][2] - M[1][1] * M[0][2] - M[0][0] * M[1][2] + M[1][0] * M[0][2] + M[0][0] * M[1][1] - M[1][0] * M[0][1];
}

template <class DataTypes>
void OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::computeStiffnessMatrix(StiffnessMatrix& S, StiffnessMatrix& SR, const MaterialStiffness& K, const StrainDisplacementTransposed& J, const Transformation& Rot) {
    type::MatNoInit<6, 12, Real> Jt;
    Jt.transpose(J);

    type::MatNoInit<12, 12, Real> JKJt;
    JKJt = J * K * Jt;

    type::MatNoInit<12, 12, Real> RR, RRt;
    RR.clear();
    RRt.clear();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            RR[i][j] = RR[i + 3][j + 3] = RR[i + 6][j + 6] = RR[i + 9][j + 9] = Rot[i][j];
            RRt[i][j] = RRt[i + 3][j + 3] = RRt[i + 6][j + 6] = RRt[i + 9][j + 9] = Rot[j][i];
        }

    S = RR * JKJt;
    SR = S * RRt;
}

template <class DataTypes>
void OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::computeMaterialStiffness(int i, Index& a, Index& b, Index& c, Index& d) {
    const VecReal& localStiffnessFactor = _localStiffnessFactor.getValue();

    type::vector<typename OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::TetrahedronInformation>& tetrahedronInf = *(tetrahedronInfo.beginEdit());

    computeMaterialStiffness(tetrahedronInf[i].materialMatrix, a, b, c, d, (localStiffnessFactor.empty() ? 1.0f : localStiffnessFactor[i * localStiffnessFactor.size() / _topology->getNbTetrahedra()]));

    tetrahedronInfo.endEdit();
}

template <class DataTypes>
void OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::computeMaterialStiffness(MaterialStiffness& materialMatrix, Index& a, Index& b, Index& c, Index& d, SReal localStiffnessFactor) {
    const Real E1 = _youngModulus1.getValue() * (Real)localStiffnessFactor;
    const Real E2 = _youngModulus2.getValue() * (Real)localStiffnessFactor;
    const Real E3 = _youngModulus3.getValue() * (Real)localStiffnessFactor;
    const Real nu12 = _poissonRatio12.getValue();
    const Real nu21 = _poissonRatio21.getValue();
    const Real nu13 = _poissonRatio13.getValue();
    const Real nu31 = _poissonRatio31.getValue();
    const Real nu23 = _poissonRatio23.getValue();
    const Real nu32 = _poissonRatio32.getValue();

    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            materialMatrix[i][j] = 0;
        }
    }

    materialMatrix[0][0] = E1 * (1 - (nu23 * nu32));
    materialMatrix[1][1] = E2 * (1 - (nu13 * nu31));
    materialMatrix[2][2] = E3 * (1 - (nu12 * nu21));

    materialMatrix[0][1] = E2 * (nu12 + (nu32 * nu13));
    materialMatrix[0][2] = E3 * (nu13 + (nu12 * nu23));
    materialMatrix[1][0] = E1 * (nu21 + (nu31 * nu23));
    materialMatrix[1][2] = E3 * (nu23 + (nu21 * nu13));
    materialMatrix[2][0] = E1 * (nu31 + (nu21 * nu32));
    materialMatrix[2][1] = E2 * (nu32 + (nu12 * nu31));
    materialMatrix[4][4] = std::sqrt(E1 * E2) / (2 * (1 + std::sqrt(nu12 * nu21)));
    materialMatrix[5][5] = std::sqrt(E2 * E3) / (2 * (1 + std::sqrt(nu23 * nu32)));
    materialMatrix[6][6] = std::sqrt(E3 * E1) / (2 * (1 + std::sqrt(nu13 * nu31)));

    materialMatrix *= 1 / (1 - (nu12 * nu21) - (nu23 * nu32) - (nu31 * nu13) - (2 * nu21 * nu32 * nu13));

    // divide by 36 times volumes of the element
    const VecCoord X0 = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    Coord A = (X0)[b] - (X0)[a];
    Coord B = (X0)[c] - (X0)[a];
    Coord C = (X0)[d] - (X0)[a];
    Coord AB = cross(A, B);
    Real volumes6 = fabs(dot(AB, C));
    if (volumes6 < 0) {
        std::cerr << "ERROR: Negative volume for tetra " << a << ',' << b << ',' << c << ',' << d << "> = " << volumes6 / 6 << std::endl;
    }
    //	materialMatrix  /= (volumes6);//*6 christian
    // @TODO: in TetrahedronFEMForceField, the stiffness matrix is divided by 6 compared to the code in OrthotropicTetrahedralCorotationalFEMForceField. Check which is the correct one...
    // FF:  there is normally  a factor 1/6v in the strain-displacement matrix. Times transpose makes 1/36v². Integrating accross the volume multiplies by v, so the factor is 1/36v
    materialMatrix /= (volumes6 * 6);
}

template <class DataTypes>
inline void OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::computeForce(Displacement& F, const Displacement& Depl, const MaterialStiffness& K, const StrainDisplacementTransposed& J) {
    // Unit of K = unit of youngModulus / unit of volume = Pa / m^3 = kg m^-4 s^-2
    // Unit of J = m^2
    // Unit of JKJt =  kg s^-2
    // Unit of displacement = m
    // Unit of force = kg m s^-2

    /* We have these zeros
                                  K[0][3]   K[0][4]   K[0][5]
                                  K[1][3]   K[1][4]   K[1][5]
                                  K[2][3]   K[2][4]   K[2][5]
    K[3][0]   K[3][1]   K[3][2]             K[3][4]   K[3][5]
    K[4][0]   K[4][1]   K[4][2]   K[4][3]             K[4][5]
    K[5][0]   K[5][1]   K[5][2]   K[5][3]   K[5][4]

              J[0][1]   J[0][2]             J[0][4]
    J[1][0]             J[1][2]                       J[1][5]
    J[2][0]   J[2][1]             J[2][3]
              J[3][1]   J[3][2]             J[3][4]
    J[4][0]             J[4][2]                       J[4][5]
    J[5][0]   J[5][1]             J[5][3]
              J[6][1]   J[6][2]             J[6][4]
    J[7][0]             J[7][2]                       J[7][5]
    J[8][0]   J[8][1]             J[8][3]
              J[9][1]   J[9][2]             J[9][4]
    J[10][0]            J[10][2]                      J[10][5]
    J[11][0]  J[11][1]            J[11][3]
    */

    type::VecNoInit<6, Real> JtD;
    JtD[0] = J[0][0] * Depl[0] + /*J[ 1][0]*Depl[ 1]+  J[ 2][0]*Depl[ 2]+*/
             J[3][0] * Depl[3] + /*J[ 4][0]*Depl[ 4]+  J[ 5][0]*Depl[ 5]+*/
             J[6][0] * Depl[6] + /*J[ 7][0]*Depl[ 7]+  J[ 8][0]*Depl[ 8]+*/
             J[9][0] * Depl[9] /*J[10][0]*Depl[10]+  J[11][0]*Depl[11]*/;
    JtD[1] = /*J[ 0][1]*Depl[ 0]+*/ J[1][1] * Depl[1] + /*J[ 2][1]*Depl[ 2]+*/
             /*J[ 3][1]*Depl[ 3]+*/ J[4][1] * Depl[4] + /*J[ 5][1]*Depl[ 5]+*/
             /*J[ 6][1]*Depl[ 6]+*/ J[7][1] * Depl[7] + /*J[ 8][1]*Depl[ 8]+*/
             /*J[ 9][1]*Depl[ 9]+*/ J[10][1] * Depl[10] /*J[11][1]*Depl[11]*/;
    JtD[2] = /*J[ 0][2]*Depl[ 0]+  J[ 1][2]*Depl[ 1]+*/ J[2][2] * Depl[2] +
             /*J[ 3][2]*Depl[ 3]+  J[ 4][2]*Depl[ 4]+*/ J[5][2] * Depl[5] +
             /*J[ 6][2]*Depl[ 6]+  J[ 7][2]*Depl[ 7]+*/ J[8][2] * Depl[8] +
             /*J[ 9][2]*Depl[ 9]+  J[10][2]*Depl[10]+*/ J[11][2] * Depl[11];
    JtD[3] = J[0][3] * Depl[0] + J[1][3] * Depl[1] + /*J[ 2][3]*Depl[ 2]+*/
             J[3][3] * Depl[3] + J[4][3] * Depl[4] + /*J[ 5][3]*Depl[ 5]+*/
             J[6][3] * Depl[6] + J[7][3] * Depl[7] + /*J[ 8][3]*Depl[ 8]+*/
             J[9][3] * Depl[9] + J[10][3] * Depl[10] /*J[11][3]*Depl[11]*/;
    JtD[4] = /*J[ 0][4]*Depl[ 0]+*/ J[1][4] * Depl[1] + J[2][4] * Depl[2] +
             /*J[ 3][4]*Depl[ 3]+*/ J[4][4] * Depl[4] + J[5][4] * Depl[5] +
             /*J[ 6][4]*Depl[ 6]+*/ J[7][4] * Depl[7] + J[8][4] * Depl[8] +
             /*J[ 9][4]*Depl[ 9]+*/ J[10][4] * Depl[10] + J[11][4] * Depl[11];
    JtD[5] = J[0][5] * Depl[0] + /*J[ 1][5]*Depl[ 1]*/ J[2][5] * Depl[2] + J[3][5] * Depl[3] + /*J[ 4][5]*Depl[ 4]*/ J[5][5] * Depl[5] + J[6][5] * Depl[6] + /*J[ 7][5]*Depl[ 7]*/ J[8][5] * Depl[8] + J[9][5] * Depl[9] + /*J[10][5]*Depl[10]*/ J[11][5] * Depl[11];

    type::VecNoInit<6, Real> KJtD;
    KJtD[0] = K[0][0] * JtD[0] + K[0][1] * JtD[1] + K[0][2] * JtD[2]
        /*K[0][3]*JtD[3]+  K[0][4]*JtD[4]+  K[0][5]*JtD[5]*/;
    KJtD[1] = K[1][0] * JtD[0] + K[1][1] * JtD[1] + K[1][2] * JtD[2]
        /*K[1][3]*JtD[3]+  K[1][4]*JtD[4]+  K[1][5]*JtD[5]*/;
    KJtD[2] = K[2][0] * JtD[0] + K[2][1] * JtD[1] + K[2][2] * JtD[2]
        /*K[2][3]*JtD[3]+  K[2][4]*JtD[4]+  K[2][5]*JtD[5]*/;
    KJtD[3] = /*K[3][0]*JtD[0]+  K[3][1]*JtD[1]+  K[3][2]*JtD[2]+*/
        K[3][3] * JtD[3] /*K[3][4]*JtD[4]+  K[3][5]*JtD[5]*/;
    KJtD[4] = /*K[4][0]*JtD[0]+  K[4][1]*JtD[1]+  K[4][2]*JtD[2]+*/
        /*K[4][3]*JtD[3]+*/ K[4][4] * JtD[4] /*K[4][5]*JtD[5]*/;
    KJtD[5] = /*K[5][0]*JtD[0]+  K[5][1]*JtD[1]+  K[5][2]*JtD[2]+*/
        /*K[5][3]*JtD[3]+  K[5][4]*JtD[4]+*/ K[5][5] * JtD[5];

    F[0] = J[0][0] * KJtD[0] + /*J[ 0][1]*KJtD[1]+  J[ 0][2]*KJtD[2]+*/
           J[0][3] * KJtD[3] + /*J[ 0][4]*KJtD[4]+*/ J[0][5] * KJtD[5];
    F[1] = /*J[ 1][0]*KJtD[0]+*/ J[1][1] * KJtD[1] + /*J[ 1][2]*KJtD[2]+*/
           J[1][3] * KJtD[3] + J[1][4] * KJtD[4] /*J[ 1][5]*KJtD[5]*/;
    F[2] = /*J[ 2][0]*KJtD[0]+  J[ 2][1]*KJtD[1]+*/ J[2][2] * KJtD[2] +
           /*J[ 2][3]*KJtD[3]+*/ J[2][4] * KJtD[4] + J[2][5] * KJtD[5];
    F[3] = J[3][0] * KJtD[0] + /*J[ 3][1]*KJtD[1]+  J[ 3][2]*KJtD[2]+*/
           J[3][3] * KJtD[3] + /*J[ 3][4]*KJtD[4]+*/ J[3][5] * KJtD[5];
    F[4] = /*J[ 4][0]*KJtD[0]+*/ J[4][1] * KJtD[1] + /*J[ 4][2]*KJtD[2]+*/
           J[4][3] * KJtD[3] + J[4][4] * KJtD[4] /*J[ 4][5]*KJtD[5]*/;
    F[5] = /*J[ 5][0]*KJtD[0]+  J[ 5][1]*KJtD[1]+*/ J[5][2] * KJtD[2] +
           /*J[ 5][3]*KJtD[3]+*/ J[5][4] * KJtD[4] + J[5][5] * KJtD[5];
    F[6] = J[6][0] * KJtD[0] + /*J[ 6][1]*KJtD[1]+  J[ 6][2]*KJtD[2]+*/
           J[6][3] * KJtD[3] + /*J[ 6][4]*KJtD[4]+*/ J[6][5] * KJtD[5];
    F[7] = /*J[ 7][0]*KJtD[0]+*/ J[7][1] * KJtD[1] + /*J[ 7][2]*KJtD[2]+*/
           J[7][3] * KJtD[3] + J[7][4] * KJtD[4] /*J[ 7][5]*KJtD[5]*/;
    F[8] = /*J[ 8][0]*KJtD[0]+  J[ 8][1]*KJtD[1]+*/ J[8][2] * KJtD[2] +
           /*J[ 8][3]*KJtD[3]+*/ J[8][4] * KJtD[4] + J[8][5] * KJtD[5];
    F[9] = J[9][0] * KJtD[0] + /*J[ 9][1]*KJtD[1]+  J[ 9][2]*KJtD[2]+*/
           J[9][3] * KJtD[3] + /*J[ 9][4]*KJtD[4]+*/ J[9][5] * KJtD[5];
    F[10] = /*J[10][0]*KJtD[0]+*/ J[10][1] * KJtD[1] + /*J[10][2]*KJtD[2]+*/
            J[10][3] * KJtD[3] + J[10][4] * KJtD[4] /*J[10][5]*KJtD[5]*/;
    F[11] = /*J[11][0]*KJtD[0]+  J[11][1]*KJtD[1]+*/ J[11][2] * KJtD[2] +
            /*J[11][3]*KJtD[3]+*/ J[11][4] * KJtD[4] + J[11][5] * KJtD[5];
}

template <class DataTypes>
inline void OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::computeForce(Displacement& F, const Displacement& Depl, const MaterialStiffness& K, const StrainDisplacementTransposed& J, SReal fact) {
    // Unit of K = unit of youngModulus / unit of volume = Pa / m^3 = kg m^-4 s^-2
    // Unit of J = m^2
    // Unit of JKJt =  kg s^-2
    // Unit of displacement = m
    // Unit of force = kg m s^-2

    /* We have these zeros
                                  K[0][3]   K[0][4]   K[0][5]
                                  K[1][3]   K[1][4]   K[1][5]
                                  K[2][3]   K[2][4]   K[2][5]
    K[3][0]   K[3][1]   K[3][2]             K[3][4]   K[3][5]
    K[4][0]   K[4][1]   K[4][2]   K[4][3]             K[4][5]
    K[5][0]   K[5][1]   K[5][2]   K[5][3]   K[5][4]

              J[0][1]   J[0][2]             J[0][4]
    J[1][0]             J[1][2]                       J[1][5]
    J[2][0]   J[2][1]             J[2][3]
              J[3][1]   J[3][2]             J[3][4]
    J[4][0]             J[4][2]                       J[4][5]
    J[5][0]   J[5][1]             J[5][3]
              J[6][1]   J[6][2]             J[6][4]
    J[7][0]             J[7][2]                       J[7][5]
    J[8][0]   J[8][1]             J[8][3]
              J[9][1]   J[9][2]             J[9][4]
    J[10][0]            J[10][2]                      J[10][5]
    J[11][0]  J[11][1]            J[11][3]
    */

    type::VecNoInit<6, Real> JtD;
    JtD[0] = J[0][0] * Depl[0] + /*J[ 1][0]*Depl[ 1]+  J[ 2][0]*Depl[ 2]+*/
             J[3][0] * Depl[3] + /*J[ 4][0]*Depl[ 4]+  J[ 5][0]*Depl[ 5]+*/
             J[6][0] * Depl[6] + /*J[ 7][0]*Depl[ 7]+  J[ 8][0]*Depl[ 8]+*/
             J[9][0] * Depl[9] /*J[10][0]*Depl[10]+  J[11][0]*Depl[11]*/;
    JtD[1] = /*J[ 0][1]*Depl[ 0]+*/ J[1][1] * Depl[1] + /*J[ 2][1]*Depl[ 2]+*/
             /*J[ 3][1]*Depl[ 3]+*/ J[4][1] * Depl[4] + /*J[ 5][1]*Depl[ 5]+*/
             /*J[ 6][1]*Depl[ 6]+*/ J[7][1] * Depl[7] + /*J[ 8][1]*Depl[ 8]+*/
             /*J[ 9][1]*Depl[ 9]+*/ J[10][1] * Depl[10] /*J[11][1]*Depl[11]*/;
    JtD[2] = /*J[ 0][2]*Depl[ 0]+  J[ 1][2]*Depl[ 1]+*/ J[2][2] * Depl[2] +
             /*J[ 3][2]*Depl[ 3]+  J[ 4][2]*Depl[ 4]+*/ J[5][2] * Depl[5] +
             /*J[ 6][2]*Depl[ 6]+  J[ 7][2]*Depl[ 7]+*/ J[8][2] * Depl[8] +
             /*J[ 9][2]*Depl[ 9]+  J[10][2]*Depl[10]+*/ J[11][2] * Depl[11];
    JtD[3] = J[0][3] * Depl[0] + J[1][3] * Depl[1] + /*J[ 2][3]*Depl[ 2]+*/
             J[3][3] * Depl[3] + J[4][3] * Depl[4] + /*J[ 5][3]*Depl[ 5]+*/
             J[6][3] * Depl[6] + J[7][3] * Depl[7] + /*J[ 8][3]*Depl[ 8]+*/
             J[9][3] * Depl[9] + J[10][3] * Depl[10] /*J[11][3]*Depl[11]*/;
    JtD[4] = /*J[ 0][4]*Depl[ 0]+*/ J[1][4] * Depl[1] + J[2][4] * Depl[2] +
             /*J[ 3][4]*Depl[ 3]+*/ J[4][4] * Depl[4] + J[5][4] * Depl[5] +
             /*J[ 6][4]*Depl[ 6]+*/ J[7][4] * Depl[7] + J[8][4] * Depl[8] +
             /*J[ 9][4]*Depl[ 9]+*/ J[10][4] * Depl[10] + J[11][4] * Depl[11];
    JtD[5] = J[0][5] * Depl[0] + /*J[ 1][5]*Depl[ 1]*/ J[2][5] * Depl[2] + J[3][5] * Depl[3] + /*J[ 4][5]*Depl[ 4]*/ J[5][5] * Depl[5] + J[6][5] * Depl[6] + /*J[ 7][5]*Depl[ 7]*/ J[8][5] * Depl[8] + J[9][5] * Depl[9] + /*J[10][5]*Depl[10]*/ J[11][5] * Depl[11];
    //         serr<<"TetrahedronFEMForceField<DataTypes>::computeForce, D = "<<Depl<<sendl;
    //         serr<<"TetrahedronFEMForceField<DataTypes>::computeForce, JtD = "<<JtD<<sendl;

    type::VecNoInit<6, Real> KJtD;
    KJtD[0] = K[0][0] * JtD[0] + K[0][1] * JtD[1] + K[0][2] * JtD[2]
        /*K[0][3]*JtD[3]+  K[0][4]*JtD[4]+  K[0][5]*JtD[5]*/;
    KJtD[1] = K[1][0] * JtD[0] + K[1][1] * JtD[1] + K[1][2] * JtD[2]
        /*K[1][3]*JtD[3]+  K[1][4]*JtD[4]+  K[1][5]*JtD[5]*/;
    KJtD[2] = K[2][0] * JtD[0] + K[2][1] * JtD[1] + K[2][2] * JtD[2]
        /*K[2][3]*JtD[3]+  K[2][4]*JtD[4]+  K[2][5]*JtD[5]*/;
    KJtD[3] = /*K[3][0]*JtD[0]+  K[3][1]*JtD[1]+  K[3][2]*JtD[2]+*/
        K[3][3] * JtD[3] /*K[3][4]*JtD[4]+  K[3][5]*JtD[5]*/;
    KJtD[4] = /*K[4][0]*JtD[0]+  K[4][1]*JtD[1]+  K[4][2]*JtD[2]+*/
        /*K[4][3]*JtD[3]+*/ K[4][4] * JtD[4] /*K[4][5]*JtD[5]*/;
    KJtD[5] = /*K[5][0]*JtD[0]+  K[5][1]*JtD[1]+  K[5][2]*JtD[2]+*/
        /*K[5][3]*JtD[3]+  K[5][4]*JtD[4]+*/ K[5][5] * JtD[5];

    KJtD *= fact;

    F[0] = J[0][0] * KJtD[0] + /*J[ 0][1]*KJtD[1]+  J[ 0][2]*KJtD[2]+*/
           J[0][3] * KJtD[3] + /*J[ 0][4]*KJtD[4]+*/ J[0][5] * KJtD[5];
    F[1] = /*J[ 1][0]*KJtD[0]+*/ J[1][1] * KJtD[1] + /*J[ 1][2]*KJtD[2]+*/
           J[1][3] * KJtD[3] + J[1][4] * KJtD[4] /*J[ 1][5]*KJtD[5]*/;
    F[2] = /*J[ 2][0]*KJtD[0]+  J[ 2][1]*KJtD[1]+*/ J[2][2] * KJtD[2] +
           /*J[ 2][3]*KJtD[3]+*/ J[2][4] * KJtD[4] + J[2][5] * KJtD[5];
    F[3] = J[3][0] * KJtD[0] + /*J[ 3][1]*KJtD[1]+  J[ 3][2]*KJtD[2]+*/
           J[3][3] * KJtD[3] + /*J[ 3][4]*KJtD[4]+*/ J[3][5] * KJtD[5];
    F[4] = /*J[ 4][0]*KJtD[0]+*/ J[4][1] * KJtD[1] + /*J[ 4][2]*KJtD[2]+*/
           J[4][3] * KJtD[3] + J[4][4] * KJtD[4] /*J[ 4][5]*KJtD[5]*/;
    F[5] = /*J[ 5][0]*KJtD[0]+  J[ 5][1]*KJtD[1]+*/ J[5][2] * KJtD[2] +
           /*J[ 5][3]*KJtD[3]+*/ J[5][4] * KJtD[4] + J[5][5] * KJtD[5];
    F[6] = J[6][0] * KJtD[0] + /*J[ 6][1]*KJtD[1]+  J[ 6][2]*KJtD[2]+*/
           J[6][3] * KJtD[3] + /*J[ 6][4]*KJtD[4]+*/ J[6][5] * KJtD[5];
    F[7] = /*J[ 7][0]*KJtD[0]+*/ J[7][1] * KJtD[1] + /*J[ 7][2]*KJtD[2]+*/
           J[7][3] * KJtD[3] + J[7][4] * KJtD[4] /*J[ 7][5]*KJtD[5]*/;
    F[8] = /*J[ 8][0]*KJtD[0]+  J[ 8][1]*KJtD[1]+*/ J[8][2] * KJtD[2] +
           /*J[ 8][3]*KJtD[3]+*/ J[8][4] * KJtD[4] + J[8][5] * KJtD[5];
    F[9] = J[9][0] * KJtD[0] + /*J[ 9][1]*KJtD[1]+  J[ 9][2]*KJtD[2]+*/
           J[9][3] * KJtD[3] + /*J[ 9][4]*KJtD[4]+*/ J[9][5] * KJtD[5];
    F[10] = /*J[10][0]*KJtD[0]+*/ J[10][1] * KJtD[1] + /*J[10][2]*KJtD[2]+*/
            J[10][3] * KJtD[3] + J[10][4] * KJtD[4] /*J[10][5]*KJtD[5]*/;
    F[11] = /*J[11][0]*KJtD[0]+  J[11][1]*KJtD[1]+*/ J[11][2] * KJtD[2] +
            /*J[11][3]*KJtD[3]+*/ J[11][4] * KJtD[4] + J[11][5] * KJtD[5];
}

//////////////////////////////////////////////////////////////////////
////////////////////  small displacements method  ////////////////////
//////////////////////////////////////////////////////////////////////

template <class DataTypes>
void OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::initSmall(int i, Index& a, Index& b, Index& c, Index& d) {
    const VecCoord X0 = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    type::vector<typename OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::TetrahedronInformation>& tetrahedronInf = *(tetrahedronInfo.beginEdit());

    computeStrainDisplacement(tetrahedronInf[i].strainDisplacementTransposedMatrix, (X0)[a], (X0)[b], (X0)[c], (X0)[d]);

    tetrahedronInfo.endEdit();

    this->printStiffnessMatrix(i);  ////////////////////////////////////////////////////////////////
}

template <class DataTypes>
void OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::accumulateForceSmall(Vector& f, const Vector& p, Index elementIndex) {
    const core::topology::BaseMeshTopology::Tetrahedron t = _topology->getTetrahedron(elementIndex);
    const VecCoord& X0 = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    Index a = t[0];
    Index b = t[1];
    Index c = t[2];
    Index d = t[3];

    // displacements
    Displacement D;
    D[0] = 0;
    D[1] = 0;
    D[2] = 0;
    D[3] = (X0)[b][0] - (X0)[a][0] - p[b][0] + p[a][0];
    D[4] = (X0)[b][1] - (X0)[a][1] - p[b][1] + p[a][1];
    D[5] = (X0)[b][2] - (X0)[a][2] - p[b][2] + p[a][2];
    D[6] = (X0)[c][0] - (X0)[a][0] - p[c][0] + p[a][0];
    D[7] = (X0)[c][1] - (X0)[a][1] - p[c][1] + p[a][1];
    D[8] = (X0)[c][2] - (X0)[a][2] - p[c][2] + p[a][2];
    D[9] = (X0)[d][0] - (X0)[a][0] - p[d][0] + p[a][0];
    D[10] = (X0)[d][1] - (X0)[a][1] - p[d][1] + p[a][1];
    D[11] = (X0)[d][2] - (X0)[a][2] - p[d][2] + p[a][2];

    // compute force on element
    Displacement F;

    const type::vector<typename OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::TetrahedronInformation>& tetrahedronInf = tetrahedronInfo.getValue();

    if (!_assembling.getValue()) {
        computeForce(F, D, tetrahedronInf[elementIndex].materialMatrix, tetrahedronInf[elementIndex].strainDisplacementTransposedMatrix);
    } else {
        Transformation Rot;
        Rot[0][0] = Rot[1][1] = Rot[2][2] = 1;
        Rot[0][1] = Rot[0][2] = 0;
        Rot[1][0] = Rot[1][2] = 0;
        Rot[2][0] = Rot[2][1] = 0;

        StiffnessMatrix JKJt, tmp;
        computeStiffnessMatrix(JKJt, tmp, tetrahedronInf[elementIndex].materialMatrix, tetrahedronInf[elementIndex].strainDisplacementTransposedMatrix, Rot);

        // erase the stiffness matrix at each time step
        if (elementIndex == 0) {
            for (unsigned int i = 0; i < _stiffnesses.size(); ++i) {
                _stiffnesses[i].resize(0);
            }
        }

        for (int i = 0; i < 12; ++i) {
            int row = t[i / 3] * 3 + i % 3;

            for (int j = 0; j < 12; ++j) {
                if (JKJt[i][j] != 0) {
                    int col = t[j / 3] * 3 + j % 3;
                    // search if the vertex is already take into account by another element
                    typename CompressedValue::iterator result = _stiffnesses[row].end();
                    for (typename CompressedValue::iterator it = _stiffnesses[row].begin(); it != _stiffnesses[row].end() && result == _stiffnesses[row].end(); ++it) {
                        if ((*it).first == col) result = it;
                    }

                    if (result == _stiffnesses[row].end())
                        _stiffnesses[row].push_back(Col_Value(col, JKJt[i][j]));
                    else
                        (*result).second += JKJt[i][j];
                }
            }
        }

        F = JKJt * D;
    }

    f[a] += Deriv(F[0], F[1], F[2]);
    f[b] += Deriv(F[3], F[4], F[5]);
    f[c] += Deriv(F[6], F[7], F[8]);
    f[d] += Deriv(F[9], F[10], F[11]);
}

template <class DataTypes>
void OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::applyStiffnessSmall(Vector& f, const Vector& x, int i, Index a, Index b, Index c, Index d, SReal fact) {
    Displacement X;

    X[0] = x[a][0];
    X[1] = x[a][1];
    X[2] = x[a][2];

    X[3] = x[b][0];
    X[4] = x[b][1];
    X[5] = x[b][2];

    X[6] = x[c][0];
    X[7] = x[c][1];
    X[8] = x[c][2];

    X[9] = x[d][0];
    X[10] = x[d][1];
    X[11] = x[d][2];

    Displacement F;

    const type::vector<typename OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::TetrahedronInformation>& tetrahedronInf = tetrahedronInfo.getValue();

    computeForce(F, X, tetrahedronInf[i].materialMatrix, tetrahedronInf[i].strainDisplacementTransposedMatrix, fact);

    f[a] += Deriv(-F[0], -F[1], -F[2]);
    f[b] += Deriv(-F[3], -F[4], -F[5]);
    f[c] += Deriv(-F[6], -F[7], -F[8]);
    f[d] += Deriv(-F[9], -F[10], -F[11]);
}

//////////////////////////////////////////////////////////////////////
////////////////////  large displacements method  ////////////////////
//////////////////////////////////////////////////////////////////////

template <class DataTypes>
inline void OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::computeRotationLarge(Transformation& r, const Vector& p, const Index& a, const Index& b, const Index& c) {
    // first vector on first edge
    // second vector in the plane of the two first edges
    // third vector orthogonal to first and second

    Coord edgex = p[b] - p[a];
    edgex.normalize();

    Coord edgey = p[c] - p[a];
    edgey.normalize();

    Coord edgez = cross(edgex, edgey);
    edgez.normalize();

    edgey = cross(edgez, edgex);
    edgey.normalize();

    r[0][0] = edgex[0];
    r[0][1] = edgex[1];
    r[0][2] = edgex[2];
    r[1][0] = edgey[0];
    r[1][1] = edgey[1];
    r[1][2] = edgey[2];
    r[2][0] = edgez[0];
    r[2][1] = edgez[1];
    r[2][2] = edgez[2];
}

template <class DataTypes>
inline void OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::getElementRotation(Transformation& R, unsigned int elementIdx) {
    type::vector<TetrahedronInformation>& tetraInf = *(tetrahedronInfo.beginEdit());
    TetrahedronInformation* tinfo = &tetraInf[elementIdx];
    Transformation r01, r21;
    r01 = tinfo->initialTransformation;
    r21 = tinfo->rotation * r01;
    R = r21;
}

template <class DataTypes>
inline void OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::getRotation(Transformation& R, unsigned int nodeIdx) {
    const type::vector<typename OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::TetrahedronInformation>& tetraInf = tetrahedronInfo.getValue();
    int numNeiTetra = _topology->getTetrahedraAroundVertex(nodeIdx).size();
    Transformation r;
    r.clear();

    for (int i = 0; i < numNeiTetra; i++) {
        int tetraIdx = _topology->getTetrahedraAroundVertex(nodeIdx)[i];
        const TetrahedronInformation* tinfo = &tetraInf[tetraIdx];
        Transformation r01, r21;
        r01 = tinfo->initialTransformation;
        r21 = tinfo->rotation * r01;
        r += r21;
    }
    R = r * (1.0f / numNeiTetra);

    // orthogonalization
    Coord ex, ey, ez;
    for (int i = 0; i < 3; i++) {
        ex[i] = R[0][i];
        ey[i] = R[1][i];
    }
    ex.normalize();
    ey.normalize();

    ez = cross(ex, ey);
    ez.normalize();

    ey = cross(ez, ex);
    ey.normalize();

    for (int i = 0; i < 3; i++) {
        R[0][i] = ex[i];
        R[1][i] = ey[i];
        R[2][i] = ez[i];
    }
}

template <class DataTypes>
inline void OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::getElementStiffnessMatrix(Real* stiffness, unsigned int elementIndex) {
    const type::vector<typename OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::TetrahedronInformation>& tetraInf = tetrahedronInfo.getValue();
    Transformation Rot;
    StiffnessMatrix JKJt, tmp;
    Rot[0][0] = Rot[1][1] = Rot[2][2] = 1;
    Rot[0][1] = Rot[0][2] = 0;
    Rot[1][0] = Rot[1][2] = 0;
    Rot[2][0] = Rot[2][1] = 0;
    computeStiffnessMatrix(JKJt, tmp, tetraInf[elementIndex].materialMatrix, tetraInf[elementIndex].strainDisplacementTransposedMatrix, Rot);
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 12; j++) stiffness[i * 12 + j] = JKJt(i, j);
    }
}

template <class DataTypes>
inline void OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::getElementStiffnessMatrix(Real* stiffness, core::topology::BaseMeshTopology::Tetra& te) {
    const VecCoord X0 = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    Index a = te[0];
    Index b = te[1];
    Index c = te[2];
    Index d = te[3];

    Transformation R_0_1;
    computeRotationLarge(R_0_1, (X0), a, b, c);

    MaterialStiffness materialMatrix;
    StrainDisplacementTransposed strainMatrix;
    type::fixed_array<Coord, 4> rotatedInitialElements;

    rotatedInitialElements[0] = R_0_1 * (X0)[a];
    rotatedInitialElements[1] = R_0_1 * (X0)[b];
    rotatedInitialElements[2] = R_0_1 * (X0)[c];
    rotatedInitialElements[3] = R_0_1 * (X0)[d];

    rotatedInitialElements[1] -= rotatedInitialElements[0];
    rotatedInitialElements[2] -= rotatedInitialElements[0];
    rotatedInitialElements[3] -= rotatedInitialElements[0];
    rotatedInitialElements[0] = Coord(0, 0, 0);

    computeMaterialStiffness(materialMatrix, a, b, c, d);
    computeStrainDisplacement(strainMatrix, rotatedInitialElements[0], rotatedInitialElements[1], rotatedInitialElements[2], rotatedInitialElements[3]);

    Transformation Rot;
    StiffnessMatrix JKJt, tmp;
    Rot[0][0] = Rot[1][1] = Rot[2][2] = 1;
    Rot[0][1] = Rot[0][2] = 0;
    Rot[1][0] = Rot[1][2] = 0;
    Rot[2][0] = Rot[2][1] = 0;
    computeStiffnessMatrix(JKJt, tmp, materialMatrix, strainMatrix, Rot);
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 12; j++) stiffness[i * 12 + j] = JKJt(i, j);
    }
}

template <class DataTypes>
void OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::initLarge(int i, Index& a, Index& b, Index& c, Index& d) {
    // Rotation matrix (initial Tetrahedre/world)
    // first vector on first edge
    // second vector in the plane of the two first edges
    // third vector orthogonal to first and second
    const VecCoord& X0 = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    Transformation R_0_1;
    computeRotationLarge(R_0_1, (X0), a, b, c);

    type::vector<typename OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::TetrahedronInformation>& tetrahedronInf = *(tetrahedronInfo.beginEdit());

    tetrahedronInf[i].rotatedInitialElements[0] = R_0_1 * (X0)[a];
    tetrahedronInf[i].rotatedInitialElements[1] = R_0_1 * (X0)[b];
    tetrahedronInf[i].rotatedInitialElements[2] = R_0_1 * (X0)[c];
    tetrahedronInf[i].rotatedInitialElements[3] = R_0_1 * (X0)[d];

    tetrahedronInf[i].initialTransformation = R_0_1;

    tetrahedronInf[i].rotatedInitialElements[1] -= tetrahedronInf[i].rotatedInitialElements[0];
    tetrahedronInf[i].rotatedInitialElements[2] -= tetrahedronInf[i].rotatedInitialElements[0];
    tetrahedronInf[i].rotatedInitialElements[3] -= tetrahedronInf[i].rotatedInitialElements[0];
    tetrahedronInf[i].rotatedInitialElements[0] = Coord(0, 0, 0);

    computeStrainDisplacement(tetrahedronInf[i].strainDisplacementTransposedMatrix, tetrahedronInf[i].rotatedInitialElements[0], tetrahedronInf[i].rotatedInitialElements[1], tetrahedronInf[i].rotatedInitialElements[2], tetrahedronInf[i].rotatedInitialElements[3]);

    tetrahedronInfo.endEdit();
}

template <class DataTypes>
void OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::accumulateForceLarge(Vector& f, const Vector& p, Index elementIndex) {
    const core::topology::BaseMeshTopology::Tetrahedron t = _topology->getTetrahedron(elementIndex);

    type::vector<typename OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::TetrahedronInformation>& tetrahedronInf = *(tetrahedronInfo.beginEdit());

    // Rotation matrix (deformed and displaced Tetrahedron/world)
    Transformation R_0_2;
    computeRotationLarge(R_0_2, p, t[0], t[1], t[2]);
    tetrahedronInf[elementIndex].rotation.transpose(R_0_2);

    // positions of the deformed and displaced Tetrahedron in its frame
    type::fixed_array<Coord, 4> deforme;
    for (int i = 0; i < 4; ++i) deforme[i] = R_0_2 * p[t[i]];

    deforme[1][0] -= deforme[0][0];
    deforme[2][0] -= deforme[0][0];
    deforme[2][1] -= deforme[0][1];
    deforme[3] -= deforme[0];

    // displacement
    Displacement D;
    D[0] = 0;
    D[1] = 0;
    D[2] = 0;
    D[3] = tetrahedronInf[elementIndex].rotatedInitialElements[1][0] - deforme[1][0];
    D[4] = 0;
    D[5] = 0;
    D[6] = tetrahedronInf[elementIndex].rotatedInitialElements[2][0] - deforme[2][0];
    D[7] = tetrahedronInf[elementIndex].rotatedInitialElements[2][1] - deforme[2][1];
    D[8] = 0;
    D[9] = tetrahedronInf[elementIndex].rotatedInitialElements[3][0] - deforme[3][0];
    D[10] = tetrahedronInf[elementIndex].rotatedInitialElements[3][1] - deforme[3][1];
    D[11] = tetrahedronInf[elementIndex].rotatedInitialElements[3][2] - deforme[3][2];

    Displacement F;
    if (_updateStiffnessMatrix.getValue()) {
        StrainDisplacementTransposed& J = tetrahedronInf[elementIndex].strainDisplacementTransposedMatrix;
        J[0][0] = J[1][3] = J[2][5] = (-deforme[2][1] * deforme[3][2]);
        J[1][1] = J[0][3] = J[2][4] = (deforme[2][0] * deforme[3][2] - deforme[1][0] * deforme[3][2]);
        J[2][2] = J[0][5] = J[1][4] = (deforme[2][1] * deforme[3][0] - deforme[2][0] * deforme[3][1] + deforme[1][0] * deforme[3][1] - deforme[1][0] * deforme[2][1]);

        J[3][0] = J[4][3] = J[5][5] = (deforme[2][1] * deforme[3][2]);
        J[4][1] = J[3][3] = J[5][4] = (-deforme[2][0] * deforme[3][2]);
        J[5][2] = J[3][5] = J[4][4] = (-deforme[2][1] * deforme[3][0] + deforme[2][0] * deforme[3][1]);

        J[7][1] = J[6][3] = J[8][4] = (deforme[1][0] * deforme[3][2]);
        J[8][2] = J[6][5] = J[7][4] = (-deforme[1][0] * deforme[3][1]);

        J[11][2] = J[9][5] = J[10][4] = (deforme[1][0] * deforme[2][1]);
    }

    if (!_assembling.getValue()) {
        // compute force on element
        computeForce(F, D, tetrahedronInf[elementIndex].materialMatrix, tetrahedronInf[elementIndex].strainDisplacementTransposedMatrix);
        for (int i = 0; i < 12; i += 3) f[t[i / 3]] += tetrahedronInf[elementIndex].rotation * Deriv(F[i], F[i + 1], F[i + 2]);
    } else {
        tetrahedronInf[elementIndex].strainDisplacementTransposedMatrix[6][0] = 0;
        tetrahedronInf[elementIndex].strainDisplacementTransposedMatrix[9][0] = 0;
        tetrahedronInf[elementIndex].strainDisplacementTransposedMatrix[10][1] = 0;

        StiffnessMatrix RJKJt, RJKJtRt;
        computeStiffnessMatrix(RJKJt, RJKJtRt, tetrahedronInf[elementIndex].materialMatrix, tetrahedronInf[elementIndex].strainDisplacementTransposedMatrix, tetrahedronInf[elementIndex].rotation);

        // erase the stiffness matrix at each time step
        if (elementIndex == 0) {
            for (unsigned int i = 0; i < _stiffnesses.size(); ++i) {
                _stiffnesses[i].resize(0);
            }
        }

        for (int i = 0; i < 12; ++i) {
            int row = t[i / 3] * 3 + i % 3;

            for (int j = 0; j < 12; ++j) {
                int col = t[j / 3] * 3 + j % 3;

                // search if the vertex is already take into account by another element
                typename CompressedValue::iterator result = _stiffnesses[row].end();
                for (typename CompressedValue::iterator it = _stiffnesses[row].begin(); it != _stiffnesses[row].end() && result == _stiffnesses[row].end(); ++it) {
                    if ((*it).first == col) {
                        result = it;
                    }
                }

                if (result == _stiffnesses[row].end()) {
                    _stiffnesses[row].push_back(Col_Value(col, RJKJtRt[i][j]));
                } else {
                    (*result).second += RJKJtRt[i][j];
                }
            }
        }

        F = RJKJt * D;

        for (int i = 0; i < 12; i += 3) f[t[i / 3]] += Deriv(F[i], F[i + 1], F[i + 2]);
    }

    tetrahedronInfo.endEdit();
}

template <class DataTypes>
void OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::applyStiffnessLarge(Vector& f, const Vector& x, int i, Index a, Index b, Index c, Index d, SReal fact) {
    const type::vector<typename OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::TetrahedronInformation>& tetrahedronInf = tetrahedronInfo.getValue();

    Transformation R_0_2;
    R_0_2.transpose(tetrahedronInf[i].rotation);

    Displacement X;
    Coord x_2;

    x_2 = R_0_2 * x[a];
    X[0] = x_2[0];
    X[1] = x_2[1];
    X[2] = x_2[2];

    x_2 = R_0_2 * x[b];
    X[3] = x_2[0];
    X[4] = x_2[1];
    X[5] = x_2[2];

    x_2 = R_0_2 * x[c];
    X[6] = x_2[0];
    X[7] = x_2[1];
    X[8] = x_2[2];

    x_2 = R_0_2 * x[d];
    X[9] = x_2[0];
    X[10] = x_2[1];
    X[11] = x_2[2];

    Displacement F;

    computeForce(F, X, tetrahedronInf[i].materialMatrix, tetrahedronInf[i].strainDisplacementTransposedMatrix, fact);

    f[a] += tetrahedronInf[i].rotation * Deriv(-F[0], -F[1], -F[2]);
    f[b] += tetrahedronInf[i].rotation * Deriv(-F[3], -F[4], -F[5]);
    f[c] += tetrahedronInf[i].rotation * Deriv(-F[6], -F[7], -F[8]);
    f[d] += tetrahedronInf[i].rotation * Deriv(-F[9], -F[10], -F[11]);
}

//////////////////////////////////////////////////////////////////////
////////////////////  polar decomposition method  ////////////////////
//////////////////////////////////////////////////////////////////////

template <class DataTypes>
void OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::initPolar(int i, Index& a, Index& b, Index& c, Index& d) {
    const VecCoord X0 = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

    type::vector<typename OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::TetrahedronInformation>& tetrahedronInf = *(tetrahedronInfo.beginEdit());

    Transformation A;
    A[0] = (X0)[b] - (X0)[a];
    A[1] = (X0)[c] - (X0)[a];
    A[2] = (X0)[d] - (X0)[a];
    tetrahedronInf[i].initialTransformation = A;

    Transformation R_0_1;
    helper::Decompose<Real>::polarDecomposition(A, R_0_1);

    tetrahedronInf[i].rotatedInitialElements[0] = R_0_1 * (X0)[a];
    tetrahedronInf[i].rotatedInitialElements[1] = R_0_1 * (X0)[b];
    tetrahedronInf[i].rotatedInitialElements[2] = R_0_1 * (X0)[c];
    tetrahedronInf[i].rotatedInitialElements[3] = R_0_1 * (X0)[d];

    computeStrainDisplacement(tetrahedronInf[i].strainDisplacementTransposedMatrix, tetrahedronInf[i].rotatedInitialElements[0], tetrahedronInf[i].rotatedInitialElements[1], tetrahedronInf[i].rotatedInitialElements[2], tetrahedronInf[i].rotatedInitialElements[3]);

    tetrahedronInfo.endEdit();
}

template <class DataTypes>
void OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::accumulateForcePolar(Vector& f, const Vector& p, Index elementIndex) {
    const core::topology::BaseMeshTopology::Tetrahedron t = _topology->getTetrahedron(elementIndex);

    Transformation A;
    A[0] = p[t[1]] - p[t[0]];
    A[1] = p[t[2]] - p[t[0]];
    A[2] = p[t[3]] - p[t[0]];

    Transformation R_0_2;
    type::MatNoInit<3, 3, Real> S;
    helper::Decompose<Real>::polarDecomposition(A, R_0_2);

    type::vector<typename OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::TetrahedronInformation>& tetrahedronInf = *(tetrahedronInfo.beginEdit());

    tetrahedronInf[elementIndex].rotation.transpose(R_0_2);

    // positions of the deformed and displaced Tetrahedre in its frame
    type::fixed_array<Coord, 4> deforme;
    for (int i = 0; i < 4; ++i) deforme[i] = R_0_2 * p[t[i]];

    // displacement
    Displacement D;
    D[0] = tetrahedronInf[elementIndex].rotatedInitialElements[0][0] - deforme[0][0];
    D[1] = tetrahedronInf[elementIndex].rotatedInitialElements[0][1] - deforme[0][1];
    D[2] = tetrahedronInf[elementIndex].rotatedInitialElements[0][2] - deforme[0][2];
    D[3] = tetrahedronInf[elementIndex].rotatedInitialElements[1][0] - deforme[1][0];
    D[4] = tetrahedronInf[elementIndex].rotatedInitialElements[1][1] - deforme[1][1];
    D[5] = tetrahedronInf[elementIndex].rotatedInitialElements[1][2] - deforme[1][2];
    D[6] = tetrahedronInf[elementIndex].rotatedInitialElements[2][0] - deforme[2][0];
    D[7] = tetrahedronInf[elementIndex].rotatedInitialElements[2][1] - deforme[2][1];
    D[8] = tetrahedronInf[elementIndex].rotatedInitialElements[2][2] - deforme[2][2];
    D[9] = tetrahedronInf[elementIndex].rotatedInitialElements[3][0] - deforme[3][0];
    D[10] = tetrahedronInf[elementIndex].rotatedInitialElements[3][1] - deforme[3][1];
    D[11] = tetrahedronInf[elementIndex].rotatedInitialElements[3][2] - deforme[3][2];
    // serr<<"D : "<<D<<sendl;

    Displacement F;
    if (_updateStiffnessMatrix.getValue()) {
        // shape functions matrix
        computeStrainDisplacement(tetrahedronInf[elementIndex].strainDisplacementTransposedMatrix, deforme[0], deforme[1], deforme[2], deforme[3]);
    }

    if (!_assembling.getValue()) {
        computeForce(F, D, tetrahedronInf[elementIndex].materialMatrix, tetrahedronInf[elementIndex].strainDisplacementTransposedMatrix);
        for (int i = 0; i < 12; i += 3) f[t[i / 3]] += tetrahedronInf[elementIndex].rotation * Deriv(F[i], F[i + 1], F[i + 2]);
    } else {
        std::cerr << "TODO(OrthotropicTetrahedralCorotationalFEMForceField): support for assembling system matrix when using polar method." << std::endl;
    }

    tetrahedronInfo.endEdit();
}

template <class DataTypes>
void OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::applyStiffnessPolar(Vector& f, const Vector& x, int i, Index a, Index b, Index c, Index d, SReal fact) {
    type::vector<typename OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::TetrahedronInformation>& tetrahedronInf = *(tetrahedronInfo.beginEdit());

    Transformation R_0_2;
    R_0_2.transpose(tetrahedronInf[i].rotation);

    Displacement X;
    Coord x_2;

    x_2 = R_0_2 * x[a];
    X[0] = x_2[0];
    X[1] = x_2[1];
    X[2] = x_2[2];

    x_2 = R_0_2 * x[b];
    X[3] = x_2[0];
    X[4] = x_2[1];
    X[5] = x_2[2];

    x_2 = R_0_2 * x[c];
    X[6] = x_2[0];
    X[7] = x_2[1];
    X[8] = x_2[2];

    x_2 = R_0_2 * x[d];
    X[9] = x_2[0];
    X[10] = x_2[1];
    X[11] = x_2[2];

    Displacement F;

    computeForce(F, X, tetrahedronInf[i].materialMatrix, tetrahedronInf[i].strainDisplacementTransposedMatrix, fact);

    f[a] -= tetrahedronInf[i].rotation * Deriv(F[0], F[1], F[2]);
    f[b] -= tetrahedronInf[i].rotation * Deriv(F[3], F[4], F[5]);
    f[c] -= tetrahedronInf[i].rotation * Deriv(F[6], F[7], F[8]);
    f[d] -= tetrahedronInf[i].rotation * Deriv(F[9], F[10], F[11]);

    tetrahedronInfo.endEdit();
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

template <class DataTypes>
void OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::computeBBox(const core::ExecParams* params, bool onlyVisible) {
    if (!onlyVisible) return;

    helper::ReadAccessor<DataVecCoord> x = this->mstate->read(core::VecCoordId::position());

    static const Real max_real = std::numeric_limits<Real>::max();
    static const Real min_real = std::numeric_limits<Real>::lowest();
    Real maxBBox[3] = {min_real, min_real, min_real};
    Real minBBox[3] = {max_real, max_real, max_real};
    for (size_t i = 0; i < x.size(); i++) {
        for (int c = 0; c < 3; c++) {
            if (x[i][c] > maxBBox[c])
                maxBBox[c] = (Real)x[i][c];
            else if (x[i][c] < minBBox[c])
                minBBox[c] = (Real)x[i][c];
        }
    }

    this->f_bbox.setValue(/* params, */ sofa::type::TBoundingBox<Real>(minBBox, maxBBox));
}

template <class DataTypes>
void OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::draw(const core::visual::VisualParams* vparams) {
    if (!vparams->displayFlags().getShowForceFields()) return;
    if (!this->mstate) return;
    if (!f_drawing.getValue()) return;

    vparams->drawTool()->saveLastState();

    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    if (vparams->displayFlags().getShowWireFrame()) vparams->drawTool()->setPolygonMode(0, true);

    std::vector<type::Vec3> points[4];
    for (size_t i = 0; i < _topology->getNbTetrahedra(); ++i) {
        const core::topology::BaseMeshTopology::Tetrahedron t = _topology->getTetrahedron(i);

        Index a = t[0];
        Index b = t[1];
        Index c = t[2];
        Index d = t[3];
        Coord center = (x[a] + x[b] + x[c] + x[d]) * 0.125;
        Coord pa = (x[a] + center) * (Real)0.666667;
        Coord pb = (x[b] + center) * (Real)0.666667;
        Coord pc = (x[c] + center) * (Real)0.666667;
        Coord pd = (x[d] + center) * (Real)0.666667;

        points[0].push_back(pa);
        points[0].push_back(pb);
        points[0].push_back(pc);

        points[1].push_back(pb);
        points[1].push_back(pc);
        points[1].push_back(pd);

        points[2].push_back(pc);
        points[2].push_back(pd);
        points[2].push_back(pa);

        points[3].push_back(pd);
        points[3].push_back(pa);
        points[3].push_back(pb);
    }

    vparams->drawTool()->drawTriangles(points[0], drawColor1.getValue());
    vparams->drawTool()->drawTriangles(points[1], drawColor2.getValue());
    vparams->drawTool()->drawTriangles(points[2], drawColor3.getValue());
    vparams->drawTool()->drawTriangles(points[3], drawColor4.getValue());

    if (vparams->displayFlags().getShowWireFrame()) vparams->drawTool()->setPolygonMode(0, false);

    vparams->drawTool()->restoreLastState();
}

template <class DataTypes>
void OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::addKToMatrix(sofa::linearalgebra::BaseMatrix* mat, SReal k, unsigned int& offset) {
    // Build Matrix Block for this ForceField
    unsigned int i, j, n1, n2, row, column, ROW, COLUMN;

    Transformation Rot;
    StiffnessMatrix JKJt, tmp;

    const type::vector<typename OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::TetrahedronInformation>& tetrahedronInf = tetrahedronInfo.getValue();

    Index noeud1, noeud2;

    Rot[0][0] = Rot[1][1] = Rot[2][2] = 1;
    Rot[0][1] = Rot[0][2] = 0;
    Rot[1][0] = Rot[1][2] = 0;
    Rot[2][0] = Rot[2][1] = 0;
    const sofa::core::topology::BaseMeshTopology::SeqTetrahedra& tetras = _topology->getTetrahedra();
    for (int IT = 0; IT != (int)tetras.size(); ++IT) {
        if (method == SMALL)
            computeStiffnessMatrix(JKJt, tmp, tetrahedronInf[IT].materialMatrix, tetrahedronInf[IT].strainDisplacementTransposedMatrix, Rot);
        else
            computeStiffnessMatrix(JKJt, tmp, tetrahedronInf[IT].materialMatrix, tetrahedronInf[IT].strainDisplacementTransposedMatrix, tetrahedronInf[IT].rotation);
        const core::topology::BaseMeshTopology::Tetrahedron t = tetras[IT];

        // find index of node 1
        for (n1 = 0; n1 < 4; n1++) {
            noeud1 = t[n1];

            for (i = 0; i < 3; i++) {
                ROW = offset + 3 * noeud1 + i;
                row = 3 * n1 + i;
                // find index of node 2
                for (n2 = 0; n2 < 4; n2++) {
                    noeud2 = t[n2];

                    for (j = 0; j < 3; j++) {
                        COLUMN = offset + 3 * noeud2 + j;
                        column = 3 * n2 + j;
                        mat->add(ROW, COLUMN, -tmp[row][column] * k);
                    }
                }
            }
        }
    }
}

template <class DataTypes>
void OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::printStiffnessMatrix(int idTetra) {
    const type::vector<typename OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::TetrahedronInformation>& tetrahedronInf = tetrahedronInfo.getValue();

    Transformation Rot;
    StiffnessMatrix JKJt, tmp;

    Rot[0][0] = Rot[1][1] = Rot[2][2] = 1;
    Rot[0][1] = Rot[0][2] = 0;
    Rot[1][0] = Rot[1][2] = 0;
    Rot[2][0] = Rot[2][1] = 0;

    computeStiffnessMatrix(JKJt, tmp, tetrahedronInf[idTetra].materialMatrix, tetrahedronInf[idTetra].strainDisplacementTransposedMatrix, Rot);
}

}  // namespace forcefield

}  // namespace component

}  // namespace sofa

#endif  // SOFA_COMPONENT_FORCEFIELD_TETRAHEDRALCOROTATIONALFEMFORCEFIELD_INL
