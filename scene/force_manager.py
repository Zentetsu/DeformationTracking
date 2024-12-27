"""
File: force_manager.py
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

from shutil import copyfile
import copy


class ForceManager:
    """Class to manage forces in the Fem_Simulation object."""

    def append_nodes(self, node: Sofa.Core.Node, point: int, count: int) -> None:
        """Append nodes to the Fem_Simulation object.

        Args:
            node: The node to which the points will be appended.
            point: The point to be appended.
            count: The count of the points.

        Returns:
            The updated node.

        """
        print("FM append_nodes")
        # print("#" * 50)
        # print("appending node: " + str(count))
        # print(node.Fem_Simulation.getObject(str(count)).points.value[0])
        # print(int(point))
        # print("#" * 50)
        node.Fem_Simulation.getObject(str(count)).points.value = [int(point)]
        return node

    def extract_previous_force(self, node: Sofa.Core.Node, point: int) -> list:
        """Extract the previous force applied to a point in the Fem_Simulation object.

        Args:
            node: The node containing the Fem_Simulation object.
            point: The point from which the force will be extracted.

        Returns:
            The previous force applied to the point.

        """
        print("FM extract_previous_force")
        obj_force = node.Fem_Simulation.getObject(str(point))
        return obj_force.forces

    def nullify_force(self, node: Sofa.Core.Node, point: int, prev_force: list) -> Sofa.Core.Node:
        """Nullify the force applied to a point in the Fem_Simulation object.

        Args:
            node: The node containing the Fem_Simulation object.
            point: The point to which the force will be nullified.
            prev_force: The previous force to be applied.

        Returns:
            The updated node.

        """
        print("FM nullify_force")
        node.Fem_Simulation.getObject(str(point)).forces = prev_force
        return node

    def extract_reset_parameters(self, node: Sofa.Core.Node, point: int) -> tuple:
        """Extract reset parameters from the Fem_Simulation object.

        Args:
            node: The node containing the Fem_Simulation object.
            point: The point from which the parameters will be extracted.

        Returns:
            A tuple containing rest_velocity, obj_position, obj_normal, mesh_position, and prev_force.

        """
        print("FM extract_reset_parameters")
        myMechanicalObjectPointer = node.Fem_Simulation.getChild("Visu").getObject("Visual")
        mesh_pos = myMechanicalObjectPointer.position
        mesh_norm = myMechanicalObjectPointer.normal
        mech_buff = node.Fem_Simulation.getObject("meshLoader").position

        rest_velocity = node.Fem_Simulation.getObject("myMech").findData("velocity").value
        obj_position = copy.deepcopy(mesh_pos.value)
        obj_normal = copy.deepcopy(mesh_norm.value)
        mesh_position = copy.deepcopy(mech_buff.value)

        prev_force = self.extract_previous_force(node, point)

        return rest_velocity, obj_position, obj_normal, mesh_position, prev_force

    def reset(
        self,
        node: Sofa.Core.Node,
        point: int,
        rest_velocity: list,
        obj_position: list,
        obj_normal: list,
        mesh_position: list,
        prev_force: list,
    ) -> Sofa.Core.Node:
        """Reset the Fem_Simulation object to its previous state.

        Args:
            node: The node containing the Fem_Simulation object.
            point: The point to be reset.
            rest_velocity: The rest velocity to be applied.
            obj_position: The object position to be reset.
            obj_normal: The object normal to be reset.
            mesh_position: The mesh position to be reset.
            prev_force: The previous force to be applied.

        Returns:
            The updated node.

        """
        print("FM reset")
        node.Fem_Simulation.getChild("Visu").getObject("Visual").reset()
        node.Fem_Simulation.getObject("myMech").reset()
        node.Fem_Simulation.getChild("Visu").getObject("Visual").position = obj_position
        node.Fem_Simulation.getChild("Visu").getObject("Visual").normal = obj_normal
        node.Fem_Simulation.getObject("myMech").position = mesh_position
        node.Fem_Simulation.getObject("myMech").findData("velocity").value = rest_velocity
        # print 'reset to:'
        # print prev_force

        node = self.nullify_force(node, point, prev_force)

        return node

    def phase_splitter(self, phase: int, jacobian_force: float) -> list:
        """Split the jacobian force into its components based on the phase.

        Args:
            phase: The phase of the force.
            jacobian_force: The magnitude of the jacobian force.

        Returns:
            A list containing the resolved force components.

        """
        print("FM phase_splitter")
        force_resolved = [0, 0, 0]
        if phase == 0:
            force_resolved = [float(jacobian_force), 0, 0]
        elif phase == 1:
            force_resolved = [-float(jacobian_force), 0, 0]
        elif phase == 2:
            force_resolved = [0, float(jacobian_force), 0]
        elif phase == 3:
            force_resolved = [0, -float(jacobian_force), 0]
        elif phase == 4:
            force_resolved = [0, 0, float(jacobian_force)]
        elif phase == 5:
            force_resolved = [0, 0, -float(jacobian_force)]
        else:
            print("ERROR: Wrong option for jacobian force. Catastrophic!")

        # print 'J:'
        # print force_resolved
        # print 'again:'
        # print force_resolved[0],force_resolved[1],force_resolved[2]
        return force_resolved

    def jacobian_deform(self, phase: int, node: Sofa.Core.Node, point: int, jacobian_force: float, prev_force: list) -> Sofa.Core.Node:
        """Apply the jacobian deformation to a point in the Fem_Simulation object.

        Args:
            phase: The phase of the force.
            node: The node containing the Fem_Simulation object.
            point: The point to which the force will be applied.
            jacobian_force: The magnitude of the jacobian force.
            prev_force: The previous force to be applied.

        Returns:
            The updated node.

        """
        print("FM jacobian_deform")
        force_J = self.phase_splitter(phase, jacobian_force)
        # print prev_force
        # print(node.Fem_Simulation.getObject(str(point)).forces.value)

        node.Fem_Simulation.getObject(str(point)).forces = [
            [
                force_J[0] + prev_force[0][0],
                force_J[1] + prev_force[0][1],
                force_J[2] + prev_force[0][2],
            ]
        ]
        # node.getObject(str(point)).forces = (
        #     str(force_J[0] + prev_force[0][0])
        #     + " "
        #     + str(force_J[1] + prev_force[0][1])
        #     + " "
        #     + str(force_J[2] + prev_force[0][2])
        # )
        # node.getObject(str(point)).times = node.findData('time').value
        # print 'applying J-force: '+str(force_J[0]+prev_force[0][0])+' '+str(force_J[1]+prev_force[0][1])+' '+str(force_J[2]+prev_force[0][2])
        return node

    def apply_force_J(
        self,
        point: int,
        node: Sofa.Core.Node,
        phase: int,
        toggle: int,
        rest_velocity: list,
        obj_position: list,
        obj_normal: list,
        mesh_position: list,
        prev_force: list,
        jacobian_force: float,
        tracker: object,
    ) -> tuple:
        """Apply the Jacobian force to a point in the Fem_Simulation object.

        Args:
            point: The point to which the force will be applied.
            node: The node containing the Fem_Simulation object.
            phase: The phase of the force.
            toggle: The toggle state.
            rest_velocity: The rest velocity to be applied.
            obj_position: The object position to be reset.
            obj_normal: The object normal to be reset.
            mesh_position: The mesh position to be reset.
            prev_force: The previous force to be applied.
            jacobian_force: The magnitude of the Jacobian force.
            tracker: The tracker object.

        Returns:
            A tuple containing the updated toggle, rest_velocity, obj_position, obj_normal, mesh_position, prev_force, node, and tracker.

        """
        print("FM apply_force_J")
        print("toggle: " + str(toggle))
        if toggle == 0:
            # print("Jacobian for: " + str(point) + " at phase: " + str(phase))
            rest_velocity, obj_position, obj_normal, mesh_position, prev_force = self.extract_reset_parameters(node, point)
            node = self.jacobian_deform(phase, node, point, jacobian_force, prev_force)
            # print 'J^'
            if phase == 0:
                tracker.objectModel = ""
                toggle = 1
        elif toggle == 1:
            # print 'Jv'
            # print 'Jacobian for: '+str(point)+' at phase: '+str(phase)+' :: RESET'
            # print node.getChild('Visu').getObject('Visual').position
            # node = self.reset(node, point, rest_velocity, obj_position, obj_normal, mesh_position, prev_force)
            toggle = 2
        elif toggle == 2:
            # print 'Jv'
            # print 'Jacobian for: '+str(point)+' at phase: '+str(phase)+' :: RESET'
            tracker.objectModel = np.vstack((tracker.objectModel.value, [[-100, -100, -100]], node.Fem_Simulation.getChild("Visu").getObject("Visual").position.value))
            # print(np.shape(tracker.objectModel.value))
            # tracker.objectModel = t
            node = self.reset(
                node,
                point,
                rest_velocity,
                obj_position,
                obj_normal,
                mesh_position,
                prev_force,
            )
            toggle = 3

        return (
            toggle,
            rest_velocity,
            obj_position,
            obj_normal,
            mesh_position,
            prev_force,
            node,
            tracker,
        )

    def apply_force(self, point: int, node: Sofa.Core.Node, tracker: object, prev_force: list) -> None:
        """Apply force to a point in the Fem_Simulation object.

        Args:
            point: The point to which the force will be applied.
            node: The node containing the Fem_Simulation object.
            tracker: The tracker object.
            prev_force: The previous force to be applied.

        """
        print("FM apply_force")
        # print 'prev force:'
        print(node.Fem_Simulation.getObject(str(point)).forces.value)
        print(tracker.update_tracker.value)

        node.Fem_Simulation.getObject(str(point)).forces = [
            [
                tracker.update_tracker[0][0] + prev_force[0][0],
                tracker.update_tracker[1][0] + prev_force[0][1],
                tracker.update_tracker[2][0] + prev_force[0][2],
            ]
        ]
        # node.Fem_Simulation.getObject(str(point)).forces = str(tracker.update_tracker[0][0] + prev_force[0][0]) + " " + str(tracker.update_tracker[1][0] + prev_force[0][1]) + " " + str(tracker.update_tracker[2][0] + prev_force[0][2])
        # print 'applying: '+str(tracker.update_tracker[0][0]+prev_force[0][0])+' '+str(tracker.update_tracker[1][0]+prev_force[0][1])+' '+str(tracker.update_tracker[2][0]+prev_force[0][2])

    def pack_model(
        self,
        node: Sofa.Core.Node,
        tracker: object,
        point: int,
        rest_velocity: list,
        obj_position: list,
        obj_normal: list,
        mesh_position: list,
        prev_force: list,
    ) -> tuple:
        """Pack the model with the current state.

        Args:
            node: The node containing the Fem_Simulation object.
            tracker: The tracker object.
            point: The point to be packed.
            rest_velocity: The rest velocity to be packed.
            obj_position: The object position to be packed.
            obj_normal: The object normal to be packed.
            mesh_position: The mesh position to be packed.
            prev_force: The previous force to be packed.

        Returns:
            A tuple containing rest_velocity, obj_position, obj_normal, mesh_position, and prev_force.

        """
        print("FM pack_model")
        tracker.objectModel = ""
        tracker.objectModel = tracker.objectModel + [[-100, -100, -100]] + node.Fem_Simulation.getChild("Visu").getObject("Visual").position
        rest_velocity, obj_position, obj_normal, mesh_position, prev_force = self.extract_reset_parameters(node, point)
        # print 'prev force while packing:'
        # print prev_force
        return rest_velocity, obj_position, obj_normal, mesh_position, prev_force

    def update_force_J_counters(self, jacobian_phase: int, node_count: int, toggle: int) -> tuple:
        """Update the Jacobian force counters.

        Args:
            jacobian_phase: The current phase of the Jacobian force.
            node_count: The current node count.
            toggle: The current toggle state.

        Returns:
            A tuple containing the updated jacobian_phase, node_count, and toggle.

        """
        print("FM update_force_J_counters")
        # print("Jacobian phase: " + str(jacobian_phase))
        if jacobian_phase == 5:
            if toggle == 3:
                jacobian_phase = 0
                node_count = node_count - 1
            toggle = 0
        elif toggle == 3:
            jacobian_phase = jacobian_phase + 1
            toggle = 0

        return jacobian_phase, node_count, toggle
