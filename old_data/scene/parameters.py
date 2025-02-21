"""
File: parameters.py
Created Date: Thursday, December 5th 2024, 7:17:00 pm

----

Last Modified: Sat Feb 01 2025

----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""  # noqa

import configparser as ConfigParser
import Sofa  # type: ignore


class Parameters:
    """A class to handle parameters for FEM and tracking properties, and to create a simulation scene."""

    def femProperties(self, config: ConfigParser.ConfigParser) -> tuple:
        """Extract FEM properties from the configuration.

        Args:
            config (ConfigParser): The configuration parser object.

        Returns:
            tuple: A tuple containing FEM properties.

        """
        fem_handle = "fem_simulation"
        mechanical_model = config.get(fem_handle, "mechanical_model")
        visual_model = config.get(fem_handle, "visual_model")
        rayleigh_mass = config.get(fem_handle, "rayleigh_mass")
        rayleigh_stiffness = config.get(fem_handle, "rayleigh_stiffness")
        iteration = config.get(fem_handle, "iteration")
        young_modulus = config.get(fem_handle, "young_modulus")
        poisson_ratio = config.get(fem_handle, "poisson_ratio")
        fixed_constraint = config.get(fem_handle, "fixed_constraint")
        max_force_vectors = config.get(fem_handle, "max_force_vectors")
        jacobian_force = config.get(fem_handle, "jacobian_force")

        return (
            mechanical_model,
            visual_model,
            rayleigh_mass,
            rayleigh_stiffness,
            iteration,
            young_modulus,
            poisson_ratio,
            fixed_constraint,
            max_force_vectors,
            jacobian_force,
        )

    def trackingProperties(self, config: ConfigParser.ConfigParser) -> tuple:
        """Extract tracking properties from the configuration.

        Args:
            config (ConfigParser): The configuration parser object.

        Returns:
            tuple: A tuple containing tracking properties.

        """
        tracker_handle = "deformation_tracking"
        visual_model = config.get(tracker_handle, "visual_model")
        init_transform = config.get(tracker_handle, "init_transform")
        mechanical_model = config.get(tracker_handle, "mechanical_model")
        data_folder = config.get(tracker_handle, "data_folder")
        output_folder = config.get(tracker_handle, "output_folder")
        c_x = config.get(tracker_handle, "c_x")
        c_y = config.get(tracker_handle, "c_y")
        f_x = config.get(tracker_handle, "f_x")
        f_y = config.get(tracker_handle, "f_y")
        config_path = config.get(tracker_handle, "config_path")
        cao_model_path = config.get(tracker_handle, "cao_model_path")
        iterations = config.get(tracker_handle, "iterations")

        return (
            visual_model,
            init_transform,
            mechanical_model,
            data_folder,
            output_folder,
            c_x,
            c_y,
            f_x,
            f_y,
            config_path,
            cao_model_path,
            iterations,
        )

    def createScene(
        self,
        node: Sofa.Core.Node,
        mechanical_model: str,
        visual_model: str,
        rayleigh_mass: float,
        rayleigh_stiffness: float,
        iteration: int,
        young_modulus: float,
        poisson_ratio: float,
        fixed_constraint: str,
        max_force_vectors: int,
    ) -> tuple:
        """Create a simulation scene.

        Args:
            node: The root node of the scene.
            mechanical_model: The mechanical model file path.
            visual_model: The visual model file path.
            rayleigh_mass: The Rayleigh mass damping coefficient.
            rayleigh_stiffness: The Rayleigh stiffness damping coefficient.
            iteration: The number of iterations for the linear solver.
            young_modulus: The Young's modulus of the material.
            poisson_ratio: The Poisson's ratio of the material.
            fixed_constraint: The indices of the fixed constraints.
            max_force_vectors: The maximum number of force vectors.

        Returns:
            tuple: The root node and the visual model object.

        """
        # print(mechanical_model)
        node.addObject(
            "MeshVTKLoader",  # MeshGmshLoader
            name="meshLoader",
            filename="../" + mechanical_model,
        )
        node.addObject(
            "EulerImplicitSolver",
            rayleighMass=rayleigh_mass,
            rayleighStiffness=rayleigh_stiffness,
        )
        # node.addObject(
        #     "StaticSolver",
        #     rayleighMass=rayleigh_mass,
        #     rayleighStiffness=rayleigh_stiffness,
        # )
        node.addObject("CGLinearSolver", iterations=iteration, tolerance=0.001, threshold=0.001)
        node.addObject("TetrahedronSetTopologyContainer", name="topo", src="@meshLoader")
        node.addObject(
            "MechanicalObject",
            name="myMech",
            scale="1",
            position="@meshLoader.position",
            tags="NoPicking",
        )
        node.addObject("TetrahedronSetGeometryAlgorithms", template="Vec3d", name="GeomAlgo")
        node.addObject("DiagonalMass", name="computed using mass density", massDensity="1")
        # node.addObject("SphereModel", radius="0.5", group="1")
        node.addObject("SphereCollisionModel", radius="0.5", group="1")
        node.addObject(
            "FixedProjectiveConstraint",
            name="FixedProjectiveConstraint",
            indices=fixed_constraint,
        )
        node.addObject(
            "FastTetrahedralCorotationalForceField",
            template="Vec3d",
            name="FEM",
            method="large",
            poissonRatio=poisson_ratio,
            youngModulus=young_modulus,
        )
        vizNode = node.addChild("Visu")
        vizNode.addObject(
            "OglModel",
            name="Visual",
            fileMesh="../" + visual_model,
        )
        vizNode.addObject("BarycentricMapping", input="@..", output="@Visual")

        for i in range(0, int(max_force_vectors)):
            node.addObject("LinearForceField", name=str(i), points="0", forces="0 0 0", times="0")

        return node, vizNode.getObject("Visual")
