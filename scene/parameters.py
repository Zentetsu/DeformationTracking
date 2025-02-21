import configparser as ConfigParser
import Sofa


class Parameters:
    def femProperties(self, config):
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
        jacobian_type = int(config.get(fem_handle, "jacobian_type"))
        update_wait_steps = int(config.get(fem_handle, "update_wait_steps"))

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
            jacobian_type,
            update_wait_steps,
        )

    def trackingProperties(self, config):
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
        debug_flag = config.get(tracker_handle, "debug_flag")
        init_lambda = config.get(tracker_handle, "init_lambda")
        additional_debug_folder = config.get(tracker_handle, "additional_debug_folder")
        tracker_init_file = config.get(tracker_handle, "tracker_init_file")
        active_points = config.get(tracker_handle, "active_points")
        color_image_folder = config.get(tracker_handle, "color_image_folder")
        depth_data_folder = config.get(tracker_handle, "depth_data_folder")
        data_offset = config.get(tracker_handle, "data_offset")
        frequency = config.get(tracker_handle, "frequency")
        color_cx = config.get(tracker_handle, "color_cx")
        color_cy = config.get(tracker_handle, "color_cy")
        color_fx = config.get(tracker_handle, "color_fx")
        color_fy = config.get(tracker_handle, "color_fy")
        depth_to_color_extrinsic_file = config.get(
            tracker_handle, "depth_to_color_extrinsic_file"
        )
        calibration_file = config.get(tracker_handle, "calibration_file")

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
            debug_flag,
            init_lambda,
            additional_debug_folder,
            tracker_init_file,
            color_image_folder,
            depth_data_folder,
            data_offset,
            frequency,
            color_cx,
            color_cy,
            color_fx,
            color_fy,
            depth_to_color_extrinsic_file,
            active_points,
            calibration_file,
        )

    def createScene(
        self,
        node,
        mechanical_model,
        visual_model,
        rayleigh_mass,
        rayleigh_stiffness,
        iteration,
        young_modulus,
        poisson_ratio,
        fixed_constraint,
        max_force_vectors,
        jacobian_type,
    ):
        node.addObject("MeshVTKLoader", name="meshLoader", filename=mechanical_model)
        node.addObject(
            "EulerImplicitSolver",
            rayleighMass=rayleigh_mass,
            rayleighStiffness=rayleigh_stiffness,
        )
        node.addObject(
            "CGLinearSolver",
            iterations=iteration,
            tolerance="0.000001",
            threshold=0.000001,
        )
        node.addObject(
            "TetrahedronSetTopologyContainer", name="topo", src="@meshLoader"
        )
        node.addObject(
            "MechanicalObject",
            name="myMech",
            scale="1",
            position="@meshLoader.position",
            tags="NoPicking",
        )
        node.addObject(
            "TetrahedronSetGeometryAlgorithms", template="Vec3d", name="GeomAlgo"
        )
        print(fixed_constraint)
        node.addObject(
            "FixedConstraint", name="FixedConstraint", indices=fixed_constraint
        )

        node.addObject(
            "TetrahedralCorotationalFEMForceField",
            template="Vec3d",
            name="FEM",
            method="large",
            poissonRatio=poisson_ratio,
            youngModulus=young_modulus,
            computeGlobalMatrix="0",
        )

        vizNode = node.addChild("Visu")
        vizNode.addObject("OglModel", name="Visual", fileMesh=visual_model)
        vizNode.addObject("BarycentricMapping", input="@..", output="@Visual")

        if jacobian_type == 0:
            for i in range(0, int(max_force_vectors)):
                node.addObject(
                    "LinearForceField",
                    name=str(i),
                    points="2",
                    forces="0 0 0",
                    times="0",
                )
        else:
            for i in range(0, int(max_force_vectors)):
                node.addObject(
                    "LinearMovementConstraint",
                    name=str(i),
                    indices="2",
                    template="Vec3d",
                    relativeMovements="true",
                )  # to change indices such that it takes a vlue dynamically

        return node, vizNode.getObject("Visual")
