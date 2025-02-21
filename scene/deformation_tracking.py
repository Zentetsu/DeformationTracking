import Sofa
import sys
import numpy as np
import configparser as ConfigParser

import random
import os
from shutil import copyfile

path_param = (
    os.getcwd()
)  #'/home/agniv/Code/sofa/applications/plugins/DeformationTracking/scene/'
sys.path.append(path_param)

import parameters
import force_manager
import displacement_manager


class SofaDeform(Sofa.Core.Controller):
    count = 0
    properties_file = path_param + "/parameter_ball.properties"
    config = ConfigParser.RawConfigParser()
    config.read(properties_file)
    param = parameters.Parameters()
    force = force_manager.ForceManager()
    myMechanicalObjectPointer = 0
    tracker = 0
    current_node = 0
    force_nodes = []
    node_count = 0
    jacobian_phase = 0
    jacobian_toggle = 0
    jacobian_force = 0
    rest_velocity = 0
    obj_position = 0
    obj_normal = 0
    mesh_position = 0
    prev_force = 0
    update_stage = 0
    fixed_points = 0
    force_map = dict()
    jacobian_type = 0
    output_folder_ = ""
    dt = 0
    calibration_file = ""
    done_jacobian_once = 0
    update_wait_steps = 2
    max_force_vectors = 0

    def get_previous_force(self, node, prev_force):
        if node in self.force_map:
            return self.force_map[node]
        else:
            return prev_force

    def set_previous_force(self, node, force):
        self.force_map[node] = force

    def getFemProperties(self):
        return self.param.femProperties(self.config)

    def getTrackingProperties(self):
        return self.param.trackingProperties(self.config)

    def getTrackerMessage(self):
        return self.tracker.trackerMessage

    def spawnScene(self):
        node = self.rootNode.addChild("Fem_Simulation")
        (
            mechanical_model,
            visual_model,
            rayleigh_mass,
            rayleigh_stiffness,
            iteration,
            young_modulus,
            poisson_ratio,
            fixed_constraint,
            self.max_force_vectors,
            self.jacobian_force,
            self.jacobian_type,
            self.update_wait_steps,
        ) = self.getFemProperties()
        node, self.myMechanicalObjectPointer = self.param.createScene(
            node,
            mechanical_model,
            visual_model,
            rayleigh_mass,
            rayleigh_stiffness,
            iteration,
            young_modulus,
            poisson_ratio,
            fixed_constraint,
            self.max_force_vectors,
            self.jacobian_type,
        )
        self.fixed_points = fixed_constraint
        if self.jacobian_type == 1:
            self.force = displacement_manager.DisplacementManager()

        return node

    def onLoaded(self, node):
        self.rootNode = node

    def createGraph(self, node):
        node = self.spawnScene()
        node.init()
        self.prepTracker(node)
        copyfile(
            self.properties_file,
            self.tracker.outputDirectory + "/parameter_ball.properties",
        )
        self.rootNode = node

        return 0

    def prepTracker(self, node):
        (
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
            self.calibration_file,
        ) = self.getTrackingProperties()
        self.output_folder_ = output_folder
        self.tracker = node.addObject(
            "DeformationTracking",
            name="deform_tracker",
            objFileName=visual_model,
            mechFileName=mechanical_model,
            mechModel="",
            transformFileName=init_transform,
            dataFolder=data_folder,
            outputDirectory=output_folder,
            C_x=c_x,
            C_y=c_y,
            F_x=f_x,
            F_y=f_y,
            configPath=config_path,
            caoModelPath=cao_model_path,
            objectModel="",
            iterations=iterations,
            debugFlag=debug_flag,
            initLambda=init_lambda,
            additionalDebugFolder=additional_debug_folder,
            trackerInitFile=tracker_init_file,
            colorImageFolder=color_image_folder,
            depthDataFolder=depth_data_folder,
            dataOffset=data_offset,
            frequency=frequency,
            colorC_x=color_cx,
            colorC_y=color_cy,
            colorF_x=color_fx,
            colorF_y=color_fy,
            depthToColorExtrinsicFile=depth_to_color_extrinsic_file,
            activePoints=active_points,
            calibrationFile=self.calibration_file,
            jacobianForce=self.jacobian_force,
            fixedPoints=self.fixed_points,
        )
        self.tracker.simulationMessage = "ready"
        # print(self.jacobian_force)
        # self.tracker.jacobianForce = self.jacobian_force
        # self.tracker.fixedPoints = self.fixed_points

    def extract_previous_node_position(self, point):
        obj_force = self.rootNode.getObject("myMech").position

        return obj_force[int(point)]

    def preformat_current_node(self, current_node_local, point):
        previous_vertex = int(
            self.rootNode.getObject(str(int(current_node_local))).indices[0][0]
        )
        current_vertex = int(point)
        if previous_vertex == current_vertex:
            return current_node_local
        else:
            for i in range(0, int(self.max_force_vectors)):
                previous_vertex = int(
                    self.rootNode.getObject(str(int(i))).indices[0][0]
                )
                if previous_vertex == current_vertex:
                    return int(i)

            return int(current_node_local + 1)

    def update(self):
        num_of_vertices = int(len(self.tracker.forcePoints))

        for i in range(0, num_of_vertices):
            tracker_point = self.tracker.forcePoints[(i)][0]
            tracker_node = self.tracker.forcePoints[(i)]
            tracker_Fx = self.tracker.update_tracker[(3 * i) + 0][0]
            tracker_Fy = self.tracker.update_tracker[(3 * i) + 1][0]
            tracker_Fz = self.tracker.update_tracker[(3 * i) + 2][0]
            if self.update_stage == 0:
                self.tracker.simulationMessage = "applying_update"
                node_details = tracker_node
                self.node_count = len(self.tracker.forcePoints)
                self.current_node = self.preformat_current_node(
                    self.current_node, int(node_details[0])
                )
                self.rootNode.getObject(str(int(self.current_node))).indices[0][0] = (
                    int(node_details[0])
                )
                self.prev_force = self.extract_previous_node_position(node_details[0])
                self.current_node = self.current_node
                self.rootNode = self.force.apply_force(
                    self.current_node,
                    tracker_point,
                    self.rootNode,
                    self.tracker,
                    self.prev_force,
                    (self.count * self.dt),
                    self.dt,
                    tracker_Fx,
                    tracker_Fy,
                    tracker_Fz,
                )
            elif self.update_stage == self.update_wait_steps:
                (
                    self.rest_velocity,
                    self.obj_position,
                    self.obj_normal,
                    self.mesh_position,
                    self.prev_force,
                ) = self.force.pack_model(
                    self.rootNode,
                    self.tracker,
                    self.current_node,
                    self.rest_velocity,
                    self.obj_position,
                    self.obj_normal,
                    self.mesh_position,
                    self.prev_force,
                )
                self.tracker = self.force.pack_model_mechanical(
                    self.rootNode, self.tracker
                )
                self.set_previous_force(self.tracker.forcePoints[0][0], self.prev_force)
                self.update_stage = -1
                self.tracker.simulationMessage = "update_ready"

        self.update_stage = self.update_stage + 1

    def onBeginAnimationStep(self, dt):
        self.dt = dt
        self.count = self.count + 1
        if type(self.tracker) is int:
            if self.count > 10:
                exit(0)
        else:
            if self.getTrackerMessage() == "matched":
                self.tracker.simulationMessage = "jacobian_ready"
            elif self.getTrackerMessage() == "updated":
                self.update()


def createScene(rootNode: Sofa.Core.Node) -> None:
    """Create the scene with the given root node.

    Args:
        rootNode (Sofa.Core.Node): The root node of the scene.

    """
    print("DT createScene")
    rootNode.gravity = [0, 0, 0]
    rootNode.dt = 0.1
    rootNode.addObject("RequiredPlugin", name="MultiThreading")
    rootNode.addObject("RequiredPlugin", name="MyPlugin", pluginName="MyPlugin")
    rootNode.addObject(
        "RequiredPlugin", name="Sofa.Component.Collision.Detection.Algorithm"
    )
    rootNode.addObject(
        "RequiredPlugin", name="Sofa.Component.Collision.Detection.Intersection"
    )
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Collision.Geometry")
    rootNode.addObject(
        "RequiredPlugin", name="Sofa.Component.Collision.Response.Contact"
    )
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Constraint.Projective")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.IO.Mesh")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.LinearSolver.Iterative")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Mapping.Linear")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Mass")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.MechanicalLoad")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.ODESolver.Backward")
    rootNode.addObject(
        "RequiredPlugin", name="Sofa.Component.SolidMechanics.FEM.Elastic"
    )
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.StateContainer")
    rootNode.addObject(
        "RequiredPlugin", name="Sofa.Component.Topology.Container.Dynamic"
    )
    rootNode.addObject("RequiredPlugin", name="Sofa.GL.Component.Rendering3D")
    rootNode.addObject("RequiredPlugin", name="SofaMiscCollision")
    rootNode.addObject("RequiredPlugin", name="SofaPython", pluginName="SofaPython3")

    rootNode.addObject("CollisionPipeline", depth=15, verbose=0, draw=0)
    # rootNode.addObject('BruteForceDetection', name='N2')
    rootNode.addObject("ParallelBruteForceBroadPhase", name="BruteForceBroadPhase")
    rootNode.addObject("ParallelBVHNarrowPhase", name="BVHNarrowPhase")
    rootNode.addObject(
        "MinProximityIntersection",
        name="Proximity",
        alarmDistance=1.5,
        contactDistance=1,
    )
    rootNode.addObject(
        "CollisionResponse", name="Response", response="PenalityContactForceField"
    )
    rootNode.addObject("CollisionGroup", name="Group")

    print("#" * 50)
    deform = SofaDeform()
    rootNode.addObject(deform)
    deform.onLoaded(rootNode)
    deform.spawnScene()
    deform.prepTracker(rootNode)
    print(rootNode.Fem_Simulation.name.value)
    print("#" * 50)

    # print(deform.getFemProperties(), deform.properties_file)
    # deform.onLoaded(rootNode)
    # deform.spawnScene()
    # deform.prepTracker(rootNode)
    # print(deform.getFemProperties())
    # deform.createGraph(rootNode)

    # rootNode.addObject("DeformationTracking")

    # rootNode.addObject('OBJExporter', name='objExporter', listening=True, filename='/media/agniv/f9826023-e8c9-47ab-906c-2cbd7ccf196a/home/agniv/Documents/debug_non_rigid/', edges=1, triangles=1, quads=1, tetras=1, hexas=1, exportEveryNumberOfSteps=1, pointsDataFields='dofs.velocity dofs.rest_position dofs.acceleration dofs.force')
