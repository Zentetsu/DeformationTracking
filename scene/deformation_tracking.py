"""
File: deformation_tracking.py
Created Date: Thursday, December 5th 2024, 7:17:00 pm

----

Last Modified: Wed Dec 18 2024

----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""  # noqa

import Sofa  # type: ignore
import sys
import numpy as np
import configparser as ConfigParser

import random
import os

path_param = os.getcwd()  #'/home/agniv/Code/sofa/applications/plugins/DeformationTracking/scene/'
sys.path.append(path_param)

import parameters
import force_manager


class SofaDeform(Sofa.Core.Controller):
    """Controller class for deformation tracking in Sofa."""

    count = 0
    properties_file = path_param + "/parameter_cube5.properties"
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

    def getFemProperties(self) -> tuple:
        """Retrieve FEM properties from the configuration file.

        Returns:
            tuple: A tuple containing FEM properties.

        """
        print("DT getFemProperties")
        return self.param.femProperties(self.config)

    def getTrackingProperties(self) -> tuple:
        """Retrieve tracking properties from the configuration file.

        Returns:
            tuple: A tuple containing tracking properties.

        """
        print("DT getTrackingProperties")
        return self.param.trackingProperties(self.config)

    def getTrackerMessage(self) -> str:
        """Retrieve the tracker message.

        Returns:
            str: The tracker message.

        """
        print("DT getTrackerMessage")
        return self.tracker.trackerMessage

    def spawnScene(self) -> Sofa.Core.Node:
        """Create and return the FEM simulation scene.

        Returns:
            Sofa.Core.Node: The root node of the FEM simulation scene.

        """
        print("DT spawnScene")
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
            max_force_vectors,
            self.jacobian_force,
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
            max_force_vectors,
        )

        return node

    def onLoaded(self, node: Sofa.Core.Node) -> None:
        """Handle the event when the scene is loaded.

        Args:
            node (Sofa.Core.Node): The root node of the scene.

        """
        print("DT onLoaded")
        # print("onLoaded")
        self.rootNode = node

    def createGraph(self, node: Sofa.Core.Node) -> int:
        """Create the graph for the scene.

        Args:
            node (Sofa.Core.Node): The root node of the scene.

        Returns:
            int: Status code indicating success or failure.

        """
        print("DT createGraph")
        # print 'createGraph'
        node = self.spawnScene()
        node.init()
        self.prepTracker(node.Response.name.value)
        self.rootNode = node
        # print 'created graph'
        return 0

    def prepTracker(self, node: Sofa.Core.Node) -> None:
        """Prepare the tracker with the given node.

        Args:
            node (Sofa.Core.Node): The node to prepare the tracker with.

        """
        print("DT prepTracker")
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
        ) = self.getTrackingProperties()
        self.tracker = node.addObject(
            "DeformationTracking",
            name="deform_tracker",
            objFileName=visual_model,
            mechFileName=mechanical_model,
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
        )
        self.tracker.simulationMessage = "ready"

    def jacobian_deform(self) -> None:
        """Handle the Jacobian deformation process."""
        print("DT jacobian_deform")
        print(self.tracker.simulationMessage.value)
        if self.tracker.simulationMessage.value != "applying_J":
            print("preparing applying force")
            for i in range(0, len(self.tracker.forcePoints)):
                node_details = self.tracker.forcePoints[i]
                print("Node forcepoints", self.tracker.forcePoints[i])
                self.rootNode = self.force.append_nodes(self.rootNode, node_details[0], self.current_node)
            self.force_nodes.extend([self.current_node])
            self.current_node = self.current_node  # + 1
            self.tracker.simulationMessage.value = "applying_J"
            self.node_count = len(self.tracker.forcePoints)
            # print("node count: " + str(self.node_count))
        elif self.tracker.simulationMessage.value == "applying_J":
            print("reading force nodes")
            (
                self.jacobian_toggle,
                self.rest_velocity,
                self.obj_position,
                self.obj_normal,
                self.mesh_position,
                self.prev_force,
                self.rootNode,
                self.tracker,
            ) = self.force.apply_force_J(
                self.force_nodes[self.node_count - 1],
                self.rootNode,
                self.jacobian_phase,
                self.jacobian_toggle,
                self.rest_velocity,
                self.obj_position,
                self.obj_normal,
                self.mesh_position,
                self.prev_force,
                self.jacobian_force,
                self.tracker,
            )
            self.jacobian_phase, self.node_count, self.jacobian_toggle = self.force.update_force_J_counters(self.jacobian_phase, self.node_count, self.jacobian_toggle)
            if self.node_count == 0:
                self.tracker.simulationMessage.value = "jacobian_ready"

    def update(self) -> None:
        """Update the deformation tracking process."""
        print("DT update")
        if self.update_stage == 0:
            self.tracker.simulationMessage.value = "applying_update"
            # print 'F^'
            # print 'from Python: need to update force now'
            # print self.tracker.update_tracker
            self.node_count = len(self.tracker.forcePoints)
            # print 'node count: '+str(self.node_count)
            self.force.apply_force(
                self.force_nodes[self.node_count - 1],
                self.rootNode,
                self.tracker,
                self.prev_force,
            )
            self.update_stage = 1
        elif self.update_stage == 1:
            self.update_stage = 2
            # print 'Fv'
        elif self.update_stage == 2:
            (
                self.rest_velocity,
                self.obj_position,
                self.obj_normal,
                self.mesh_position,
                self.prev_force,
            ) = self.force.pack_model(
                self.rootNode,
                self.tracker,
                self.force_nodes[0],
                self.rest_velocity,
                self.obj_position,
                self.obj_normal,
                self.mesh_position,
                self.prev_force,
            )
            self.update_stage = 0
            self.tracker.simulationMessage.value = "update_ready"
            # print 'Fv'

    def onAnimateBeginEvent(self, event: any) -> None:
        """Handle the event at the beginning of the animation step.

        Args:
            event: The event object.

        """
        print("DT onAnimateBeginEvent")
        # print("start")
        # print("Animation step..." + str(self.count))
        self.count = self.count + 1
        # position = self.myMechanicalObjectPointer.findData('position').value
        # print("tracker message: " + self.getTrackerMessage().value)
        if self.getTrackerMessage().value == "matched":
            self.jacobian_deform()
        elif self.getTrackerMessage().value == "updated":
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
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Collision.Detection.Algorithm")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Collision.Detection.Intersection")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Collision.Geometry")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Collision.Response.Contact")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Constraint.Projective")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.IO.Mesh")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.LinearSolver.Iterative")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Mapping.Linear")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Mass")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.MechanicalLoad")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.ODESolver.Backward")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.SolidMechanics.FEM.Elastic")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.StateContainer")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Topology.Container.Dynamic")
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
    rootNode.addObject("CollisionResponse", name="Response", response="PenalityContactForceField")
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
