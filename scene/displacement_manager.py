import Sofa
import sys
import numpy as np
import configparser as ConfigParser

import random
import os

from shutil import copyfile
import copy


class DisplacementManager:
    x_time = 10000
    y_time = 10000
    z_time = 10000

    def append_nodes(self, node, point, count):
        node.getObject(str(int(count))).indices = int(point)
        obj_force = node.getObject("myMech").position  # just for debugging
        return node

    def extract_previous_force(self, node, point):
        obj_force = node.Fem_Simulation.getObject("myMech").position
        return obj_force[node.Fem_Simulation.getObject(str(int(point))).indices[0]]

    def nullify_force(self, node, point, prev_force):
        node.getObject("myMech").position[
            node.getObject(str(int(point))).indices[0][0]
        ] = prev_force
        return node

    def extract_reset_parameters(self, node, point):
        myMechanicalObjectPointer = node.Fem_Simulation.getChild("Visu").getObject(
            "Visual"
        )
        mesh_pos = myMechanicalObjectPointer.position
        mesh_norm = myMechanicalObjectPointer.normal
        mech_buff = node.Fem_Simulation.getObject("myMech").position

        rest_velocity = (
            node.Fem_Simulation.getObject("myMech").findData("velocity").value
        )
        obj_position = copy.deepcopy(mesh_pos.value)
        obj_normal = copy.deepcopy(mesh_norm.value)
        mesh_position = copy.deepcopy(mech_buff.value)

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
        node.getChild("Visu").getObject("Visual").position = obj_position
        node.getChild("Visu").getObject("Visual").normal = obj_normal
        node.getObject("myMech").position = mesh_position
        node.getObject("myMech").findData("velocity").value = rest_velocity

        node = self.nullify_force(node, point, prev_force)

        return node

    def jacobian_deform(self, phase, node, point, jacobian_force, prev_force, time, dt):
        if phase == 0:
            node_obj = node.getObject(str(point))
            node_obj.relativeMovements = "true"
            node_obj.movements = (
                str(prev_force[0])
                + " "
                + str(prev_force[1])
                + " "
                + str(prev_force[2])
                + " "
                + str(prev_force[0] + float(jacobian_force))
                + " "
                + str(prev_force[1])
                + " "
                + str(prev_force[2])
                + " "
                + str(prev_force[0] + float(jacobian_force))
                + " "
                + str(prev_force[1])
                + " "
                + str(prev_force[2])
                + " "
                + str(prev_force[0])
                + " "
                + str(prev_force[1] + float(jacobian_force))
                + " "
                + str(prev_force[2])
                + " "
                + str(prev_force[0])
                + " "
                + str(prev_force[1] + float(jacobian_force))
                + " "
                + str(prev_force[2])
                + " "
                + str(prev_force[0])
                + " "
                + str(prev_force[1])
                + " "
                + str(prev_force[2] + float(jacobian_force))
                + " "
                + str(prev_force[0])
                + " "
                + str(prev_force[1])
                + " "
                + str(prev_force[2] + float(jacobian_force))
                + " "
                + str(prev_force[0])
                + " "
                + str(prev_force[1])
                + " "
                + str(prev_force[2])
                + " "
                + str(prev_force[0])
                + " "
                + str(prev_force[1])
                + " "
                + str(prev_force[2])
            )
            node_obj.keyTimes = (
                str(time)
                + " "
                + str(time + (3 * dt))
                + " "
                + str(time + (5 * dt))
                + " "
                + str(time + (8 * dt))
                + " "
                + str(time + (10 * dt))
                + " "
                + str(time + (13 * dt))
                + " "
                + str(time + (15 * dt))
                + " "
                + str(time + (16 * dt))
                + " "
                + str(time + (25 * dt))
            )
            self.x_time = time + (5 * dt)
            self.y_time = time + (10 * dt)
            self.z_time = time + (15 * dt)

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
        time,
        dt,
    ):
        if toggle == 0:
            rest_velocity, obj_position, obj_normal, mesh_position, prev_force = (
                self.extract_reset_parameters(node, point)
            )
            node = self.jacobian_deform(
                phase, node, point, jacobian_force, prev_force, time, dt
            )
            if phase == 0:
                tracker.objectModel = ""
            toggle = 2
        elif toggle == 1:
            toggle = 2
        elif toggle == 2:
            if (
                ("{0:.2f}".format(time) == "{0:.2f}".format(self.x_time))
                or ("{0:.2f}".format(time) == "{0:.2f}".format(self.y_time))
                or ("{0:.2f}".format(time) == "{0:.2f}".format(self.z_time))
            ):
                tracker.objectModel = (
                    tracker.objectModel
                    + [[-100, -100, -100]]
                    + node.getChild("Visu").getObject("Visual").position
                )
                if "{0:.2f}".format(time) == "{0:.2f}".format(self.z_time):
                    toggle = 3.5
                else:
                    toggle = 3
                node = self.reset(
                    node,
                    point,
                    rest_velocity,
                    obj_position,
                    obj_normal,
                    mesh_position,
                    prev_force,
                )

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

    def apply_force(
        self, point, node_index, node, tracker, prev_force, curr_time, dt, F_x, F_y, F_z
    ):
        # node.Fem_Simulation.getObject(str(int(point))).indices = str(int(node_index))
        node.Fem_Simulation.getObject(str(int(point))).indices.value = [
            int(node_index),
            *node.Fem_Simulation.getObject(str(int(point))).indices.value[1:],
        ]
        node.Fem_Simulation.getObject(str(int(point))).relativeMovements = True
        node.Fem_Simulation.getObject(str(int(point))).movements.value = np.array(
            [
                [prev_force[0], prev_force[1], prev_force[2]],
                [F_x + prev_force[0], F_y + prev_force[1], F_z + prev_force[2]],
                [F_x + prev_force[0], F_y + prev_force[1], F_z + prev_force[2]],
            ]
        )
        node.Fem_Simulation.getObject(str(int(point))).keyTimes.value = np.array(
            [curr_time, curr_time + dt["dt"], 99999999999 * dt["dt"]]
        )

        return node

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
        tracker.objectModel = np.vstack(
            (
                tracker.objectModel.value,
                [[-100, -100, -100]],
                node.Fem_Simulation.getChild("Visu").getObject("Visual").position.value,
            )
        )
        rest_velocity, obj_position, obj_normal, mesh_position, prev_force = (
            self.extract_reset_parameters(node, point)
        )

        return rest_velocity, obj_position, obj_normal, mesh_position, prev_force

    def pack_model_mechanical(self, node, tracker):
        tracker.mechModel = ""
        tracker.mechModel = np.vstack(
            (
                tracker.mechModel.value,
                [[-100, -100, -100]],
                node.Fem_Simulation.getObject("myMech").position.value,
            )
        )

        return tracker

    def update_force_J_counters(self, jacobian_phase, node_count, toggle):
        if (jacobian_phase == 3) and (toggle == 3):
            jacobian_phase = 0
            node_count = node_count - 1
            toggle = 0
        elif toggle == 3.5:
            toggle = 3
        elif toggle == 3:
            jacobian_phase = jacobian_phase + 1
            toggle = 2
        elif jacobian_phase == 2:
            jacobian_phase = jacobian_phase + 1

        return jacobian_phase, node_count, toggle
