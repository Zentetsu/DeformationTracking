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
#ifndef SOFA_COMPONENT_FORCEFIELD_ORTHOTROPICTETRAHEDRALCOROTATIONALFEMFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_ORTHOTROPICTETRAHEDRALCOROTATIONALFEMFORCEFIELD_H

// #include <sofa/component/solidmechanics/spring/SpringForceField.h>
#include <sofa/component/solidmechanics/fem/elastic/TetrahedralCorotationalFEMForceField.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/topology/BaseTopology.h>
#include <sofa/core/topology/BaseTopologyData.h>
#include <sofa/core/topology/TopologyData.h>
#include <sofa/core/topology/TopologyDataHandler.h>
#include <sofa/core/topology/TopologyHandler.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/map.h>
#include <sofa/linearalgebra/BaseMatrix.h>

// #include <SofaBaseTopology/TopologyData.h>
// #include <sofa/defaulttype/Mat.h>
// #include <sofa/defaulttype/Vec.h>
// #include <sofa/helper/vector.h>

#include "config.h"

// corotational tetrahedron from
// @InProceedings{NPF05,
//   author       = "Nesme, Matthieu and Payan, Yohan and Faure, Fran\c{c}ois",
//   title        = "Efficient, Physically Plausible Finite Elements",
//   booktitle    = "Eurographics (short papers)",
//   month        = "august",
//   year         = "2005",
//   editor       = "J. Dingliana and F. Ganovelli",
//   keywords     = "animation, physical model, elasticity, finite elements",
//   url          = "http://www-evasion.imag.fr/Publications/2005/NPF05"
// }

namespace sofa {

namespace component {

namespace forcefield {

/** Compute Finite Element forces based on tetrahedral elements.
 */
template <class DataTypes>
class OrthotropicTetrahedralCorotationalFEMForceField : public core::behavior::ForceField<DataTypes> {
   public:
    SOFA_CLASS(SOFA_TEMPLATE(OrthotropicTetrahedralCorotationalFEMForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::VecReal VecReal;
    typedef VecCoord Vector;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;

    enum {
        SMALL = 0,  ///< Symbol of small displacements tetrahedron solver
        LARGE = 1,  ///< Symbol of large displacements tetrahedron solver
        POLAR = 2   ///< Symbol of polar displacements tetrahedron solver
    };

   protected:
    /// @name Per element (tetrahedron) data
    /// @{

    /// Displacement vector (deformation of the 4 corners of a tetrahedron
    typedef sofa::type::VecNoInit<12, Real> Displacement;

    /// Material stiffness matrix of a tetrahedron
    typedef sofa::type::Mat<6, 6, Real> MaterialStiffness;

    /// Strain-displacement matrix
    typedef sofa::type::Mat<12, 6, Real> StrainDisplacementTransposed;

    /// Rigid transformation (rotation) matrix
    typedef sofa::type::MatNoInit<3, 3, Real> Transformation;

    /// Stiffness matrix ( = RJKJtRt  with K the Material stiffness matrix, J the strain-displacement matrix, and R the transformation matrix if any )
    typedef sofa::type::Mat<12, 12, Real> StiffnessMatrix;

    /// @}

    /// the information stored for each tetrahedron
    class TetrahedronInformation {
       public:
        /// material stiffness matrices of each tetrahedron
        MaterialStiffness materialMatrix;
        /// the strain-displacement matrices vector
        StrainDisplacementTransposed strainDisplacementTransposedMatrix;
        /// large displacement method
        type::fixed_array<Coord, 4> rotatedInitialElements;
        Transformation rotation;
        /// polar method
        Transformation initialTransformation;

        TetrahedronInformation() {}

        /// Output stream
        inline friend std::ostream& operator<<(std::ostream& os, const TetrahedronInformation& /*tri*/) { return os; }

        /// Input stream
        inline friend std::istream& operator>>(std::istream& in, TetrahedronInformation& /*tri*/) { return in; }
    };
    /// container that stotes all requires information for each tetrahedron
    sofa::core::topology::TetrahedronData<sofa::type::vector<TetrahedronInformation>> tetrahedronInfo;

    /// @name Full system matrix assembly support
    /// @{

    typedef std::pair<int, Real> Col_Value;
    typedef type::vector<Col_Value> CompressedValue;
    typedef type::vector<CompressedValue> CompressedMatrix;
    typedef unsigned int Index;

    CompressedMatrix _stiffnesses;
    /// @}

    SReal m_potentialEnergy;

    sofa::core::topology::BaseMeshTopology* _topology;

   public:
    using TetraCorotationalFEM = sofa::component::solidmechanics::fem::elastic::TetrahedralCorotationalFEMForceField<sofa::defaulttype::Vec3Types>;
    class TetrahedronHandler : public sofa::core::topology::TopologyDataHandler<sofa::core::topology::Topology::Tetrahedron, sofa::type::vector<TetraCorotationalFEM::TetrahedronInformation>> {
       public:
        typedef typename OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>::TetrahedronInformation TetrahedronInformation;
        TetrahedronHandler(OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>* ff, sofa::core::topology::TetrahedronData<sofa::type::vector<TetrahedronInformation>>* data)
            : sofa::core::topology::TopologyDataHandler<sofa::core::topology::Topology::Tetrahedron, sofa::type::vector<TetrahedronInformation>>(data), ff(ff) {}

        void applyCreateFunction(unsigned int, TetrahedronInformation& t, const sofa::core::topology::Topology::Tetrahedron&, const sofa::type::vector<unsigned int>&, const sofa::type::vector<double>&);

       protected:
        OrthotropicTetrahedralCorotationalFEMForceField<DataTypes>* ff;
    };

   public:
    int method;
    Data<std::string> f_method;           ///< the computation method of the displacements
    Data<Real> _poissonRatio12;           ///< FEM Poisson Ratio
    Data<Real> _poissonRatio21;           ///< FEM Poisson Ratio
    Data<Real> _poissonRatio13;           ///< FEM Poisson Ratio
    Data<Real> _poissonRatio31;           ///< FEM Poisson Ratio
    Data<Real> _poissonRatio23;           ///< FEM Poisson Ratio
    Data<Real> _poissonRatio32;           ///< FEM Poisson Ratio
    Data<Real> _youngModulus1;            ///< FEM Young Modulus
    Data<Real> _youngModulus2;            ///< FEM Young Modulus
    Data<Real> _youngModulus3;            ///< FEM Young Modulus
    Data<VecReal> _localStiffnessFactor;  ///< Allow specification of different stiffness per element. If there are N element and M values are specified, the youngModulus factor for element i would be localStiffnessFactor[i*M/N]
    Data<bool> _updateStiffnessMatrix;
    Data<bool> _assembling;
    Data<bool> f_drawing;  ///<  draw the forcefield if true
    Data<bool> _displayWholeVolume;
    Data<sofa::type::RGBAColor> drawColor1;  ///<  draw color for faces 1
    Data<sofa::type::RGBAColor> drawColor2;  ///<  draw color for faces 2
    Data<sofa::type::RGBAColor> drawColor3;  ///<  draw color for faces 3
    Data<sofa::type::RGBAColor> drawColor4;  ///<  draw color for faces 4
    Data<std::map<std::string, sofa::type::vector<double>>> _volumeGraph;

   protected:
    OrthotropicTetrahedralCorotationalFEMForceField();
    TetrahedronHandler* tetrahedronHandler;

   public:
    void setPoissonRatio12(Real val) { this->_poissonRatio12.setValue(val); }
    void setPoissonRatio21(Real val) { this->_poissonRatio21.setValue(val); }
    void setPoissonRatio13(Real val) { this->_poissonRatio13.setValue(val); }
    void setPoissonRatio31(Real val) { this->_poissonRatio31.setValue(val); }
    void setPoissonRatio23(Real val) { this->_poissonRatio23.setValue(val); }
    void setPoissonRatio32(Real val) { this->_poissonRatio32.setValue(val); }

    void setYoungModulus1(Real val) { this->_youngModulus1.setValue(val); }
    void setYoungModulus2(Real val) { this->_youngModulus2.setValue(val); }
    void setYoungModulus3(Real val) { this->_youngModulus3.setValue(val); }

    void setMethod(int val) { method = val; }

    void setUpdateStiffnessMatrix(bool val) { this->_updateStiffnessMatrix.setValue(val); }

    void setComputeGlobalMatrix(bool val) { this->_assembling.setValue(val); }

    void init() override;
    void reinit() override;

    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v) override;
    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx) override;
    SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord& /* x */) const override {
        std::cerr << "Get potentialEnergy not implemented" << std::endl;
        return 0.0;
    }

    void addKToMatrix(sofa::linearalgebra::BaseMatrix* m, SReal kFactor, unsigned int& offset) override;

    // Getting the rotation of the vertex by averaing the rotation of neighboring elements
    void getRotation(Transformation& R, unsigned int nodeIdx);
    void getRotations() {}
    void getElementRotation(Transformation& R, unsigned int elementIdx);

    // Getting the stiffness matrix of index i
    void getElementStiffnessMatrix(Real* stiffness, unsigned int nodeIdx);
    void getElementStiffnessMatrix(Real* stiffness, sofa::core::topology::BaseMeshTopology::Tetrahedron& te);

    void draw(const core::visual::VisualParams* vparams) override;

    void computeBBox(const core::ExecParams* params, bool onlyVisible) override;

   protected:
    void computeStrainDisplacement(StrainDisplacementTransposed& J, Coord a, Coord b, Coord c, Coord d);
    Real peudo_determinant_for_coef(const sofa::type::Mat<2, 3, Real>& M);

    void computeStiffnessMatrix(StiffnessMatrix& S, StiffnessMatrix& SR, const MaterialStiffness& K, const StrainDisplacementTransposed& J, const Transformation& Rot);

    void computeMaterialStiffness(int i, Index& a, Index& b, Index& c, Index& d);

    /// overloaded by classes with non-uniform stiffness
    virtual void computeMaterialStiffness(MaterialStiffness& materialMatrix, Index& a, Index& b, Index& c, Index& d, SReal localStiffnessFactor = 1);

    void computeForce(Displacement& F, const Displacement& Depl, const MaterialStiffness& K, const StrainDisplacementTransposed& J);
    void computeForce(Displacement& F, const Displacement& Depl, const MaterialStiffness& K, const StrainDisplacementTransposed& J, SReal fact);

    ////////////// small displacements method
    void initSmall(int i, Index& a, Index& b, Index& c, Index& d);
    void accumulateForceSmall(Vector& f, const Vector& p, Index elementIndex);
    void applyStiffnessSmall(Vector& f, const Vector& x, int i = 0, Index a = 0, Index b = 1, Index c = 2, Index d = 3, SReal fact = 1.0);

    ////////////// large displacements method
    void initLarge(int i, Index& a, Index& b, Index& c, Index& d);
    void computeRotationLarge(Transformation& r, const Vector& p, const Index& a, const Index& b, const Index& c);
    void accumulateForceLarge(Vector& f, const Vector& p, Index elementIndex);
    void applyStiffnessLarge(Vector& f, const Vector& x, int i = 0, Index a = 0, Index b = 1, Index c = 2, Index d = 3, SReal fact = 1.0);

    ////////////// polar decomposition method
    void initPolar(int i, Index& a, Index& b, Index& c, Index& d);
    void accumulateForcePolar(Vector& f, const Vector& p, Index elementIndex);
    void applyStiffnessPolar(Vector& f, const Vector& x, int i = 0, Index a = 0, Index b = 1, Index c = 2, Index d = 3, SReal fact = 1.0);

    void printStiffnessMatrix(int idTetra);
};

#if !defined(SOFA_COMPONENT_FORCEFIELD_ORTHOTROPICTETRAHEDRALCOROTATIONALFEMFORCEFIELD_CPP)

extern template class SOFA_EXPORT_DYNAMIC_LIBRARY OrthotropicTetrahedralCorotationalFEMForceField<sofa::type::Vec3Types>;

#endif  //  !defined(SOFA_COMPONENT_FORCEFIELD_TETRAHEDRALCOROTATIONALFEMFORCEFIELD_CPP)

}  // namespace forcefield

}  // namespace component

}  // namespace sofa

#endif  // SOFA_COMPONENT_FORCEFIELD_TETRAHEDRALCOROTATIONALFEMFORCEFIELD_H
