import Sofa
import sys
import numpy as np
import configparser as ConfigParser

import random
import os

from shutil import copyfile
import copy


class ForceManager:
    def append_nodes(self, node, point, count):
        print("PYTHON:: appending node: " + str(int(point)))
        node.getObject(str(count)).points = int(point)

        return node

    def extract_previous_force(self, node, point):
        obj_force = node.getObject(str(point))
        return obj_force.forces

    def nullify_force(self, node, point, prev_force):
        node.getObject(str(point)).forces = prev_force

        return node

    def extract_reset_parameters(self, node, point):
        myMechanicalObjectPointer = node.getChild("Visu").getObject("Visual")
        mesh_pos = myMechanicalObjectPointer.position
        mesh_norm = myMechanicalObjectPointer.normal
        mech_buff = node.getObject("meshLoader").position

        rest_velocity = node.getObject("myMech").findData("velocity").value
        obj_position = copy.deepcopy(mesh_pos)
        obj_normal = copy.deepcopy(mesh_norm)
        mesh_position = copy.deepcopy(mech_buff)

        prev_force = self.extract_previous_force(node, point)

        return rest_velocity, obj_position, obj_normal, mesh_position, prev_force

    def reset(
        self,
        node,
        point,
        rest_velocity,
        obj_position,
        obj_normal,
        mesh_position,
        prev_force,
    ):
        node.getChild("Visu").getObject("Visual").reset()
        node.getObject("myMech").reset()
        node.getChild("Visu").getObject("Visual").position = obj_position
        node.getChild("Visu").getObject("Visual").normal = obj_normal
        node.getObject("myMech").position = mesh_position
        node.getObject("myMech").findData("velocity").value = rest_velocity

        node = self.nullify_force(node, point, prev_force)

        return node

    def phase_splitter(self, phase, jacobian_force):
        force_resolved = [0, 0, 0]
        if phase == 0:
            force_resolved = [float(jacobian_force), 0, 0]
            # force_resolved = [0,0,0]
        elif phase == 1:
            force_resolved = [0, 0, 0]
            # force_resolved = [float(jacobian_force),0,0]
        elif phase == 2:
            force_resolved = [0, float(jacobian_force), 0]
            # force_resolved = [0,0,0]
        elif phase == 3:
            force_resolved = [0, 0, 0]
            # force_resolved = [0, float(jacobian_force), 0]
        elif phase == 4:
            force_resolved = [0, 0, float(jacobian_force)]
            # force_resolved = [0,0,0]
        elif phase == 5:
            force_resolved = [0, 0, 0]
            # force_resolved = [0, 0, float(jacobian_force)]
        else:
            print("ERROR: Wrong option for jacobian force. Catastrophic!")

        return force_resolved

    def jacobian_deform(self, phase, node, point, jacobian_force, prev_force):
        force_J = self.phase_splitter(phase, jacobian_force)
        try:
            print("***PYTHON:: Fetching point: " + str(point) + "***")
            print(force_J[0])
            print(prev_force[0])
            print(node.getObject(str(point)).points)
            print("***PYTHON***")

            node.getObject(str(point)).forces = (
                str(force_J[0] + prev_force[0][0])
                + " "
                + str(force_J[1] + prev_force[0][1])
                + " "
                + str(force_J[2] + prev_force[0][2])
            )
        except IndexError as e:
            print(e)
            print(sys.exc_type)
            exit(0)

        return node

    def apply_force_J(
        self,
        point,
        node,
        phase,
        toggle,
        rest_velocity,
        obj_position,
        obj_normal,
        mesh_position,
        prev_force,
        jacobian_force,
        tracker,
    ):
        if toggle == 0:
            rest_velocity, obj_position, obj_normal, mesh_position, prev_force = (
                self.extract_reset_parameters(node, point)
            )
            node = self.jacobian_deform(phase, node, point, jacobian_force, prev_force)
            if phase == 0:
                tracker.objectModel = ""
            toggle = 1
        elif toggle == 1:
            toggle = 2
        elif toggle == 2:
            tracker.objectModel = (
                tracker.objectModel
                + [[-100, -100, -100]]
                + node.getChild("Visu").getObject("Visual").position
            )
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

    def apply_force(self, point, node, tracker, prev_force):
        node.getObject(str(point)).forces = (
            str(tracker.update_tracker[0][0] + prev_force[0][0])
            + " "
            + str(tracker.update_tracker[1][0] + prev_force[0][1])
            + " "
            + str(tracker.update_tracker[2][0] + prev_force[0][2])
        )

    def pack_model(
        self,
        node,
        tracker,
        point,
        rest_velocity,
        obj_position,
        obj_normal,
        mesh_position,
        prev_force,
    ):
        tracker.objectModel = ""
        tracker.objectModel = (
            tracker.objectModel
            + [[-100, -100, -100]]
            + node.getChild("Visu").getObject("Visual").position
        )
        rest_velocity, obj_position, obj_normal, mesh_position, prev_force = (
            self.extract_reset_parameters(node, point)
        )

        return rest_velocity, obj_position, obj_normal, mesh_position, prev_force

    def pack_model_mechanical(self, node, tracker):
        tracker.mechModel = ""
        tracker.mechModel = (
            tracker.mechModel + [[-100, -100, -100]] + node.getObject("myMech").position
        )

        return tracker

    def update_force_J_counters(self, jacobian_phase, node_count, toggle):
        if jacobian_phase == 5:
            if toggle == 3:
                jacobian_phase = 0
                node_count = node_count - 1
                toggle = 0
        elif toggle == 3:
            jacobian_phase = jacobian_phase + 1
            toggle = 0

        return jacobian_phase, node_count, toggle
