from random import random, randint, uniform

import numpy as np
from pydrake.planning import (RobotDiagramBuilder,
                              SceneGraphCollisionChecker)

def collision_checker_and_joint_limits(models_path, com, rot, dims):
    # Get bounding box of object
    rpy = rot.ToRollPitchYaw().vector()
    bounding_box_urdf = """<?xml version="1.0"?>
<robot name="bounding_box">
<link name="bounding_box">
<collision name="bounding_box">
    <origin rpy="%f %f %f" xyz="%f %f %f"/>
<geometry>
    <box size="%f %f %f"/>
</geometry>
</collision>
</link>
<joint name="fixed_link_weld" type="fixed">
<parent link="world"/>
<child link="bounding_box"/>
</joint>
</robot>
    """ % (rpy[0], rpy[1], rpy[2], com[0], com[1], com[2], dims[0], dims[1], dims[2])
    
    params = dict(edge_step_size=0.125)
    builder = RobotDiagramBuilder()
    builder.parser().AddModels(models_path)
    builder.parser().AddModelsFromString(bounding_box_urdf, "urdf")
    iiwa_model_instance_index = builder.plant().GetModelInstanceByName("iiwa")
    wsg_model_instance_index = builder.plant().GetModelInstanceByName("wsg")
    plant = builder.plant()
    plant.Finalize()
    params["robot_model_instances"] = [iiwa_model_instance_index, wsg_model_instance_index]
    params["model"] = builder.Build()
    checker = SceneGraphCollisionChecker(**params)

    # Construct configuration space for IIWA.
    nq = 7
    joint_limits = np.zeros((nq, 2))
    for i in range(nq):
        joint = plant.GetJointByName("iiwa_joint_%i" % (i + 1))
        joint_limits[i, 0] = joint.position_lower_limits()
        joint_limits[i, 1] = joint.position_upper_limits()

    return checker, joint_limits

class TreeNode:
    def __init__(self, value, cost_to, parent=None):
        self.value = value  # tuple of floats representing a configuration
        self.cost_to = cost_to
        self.parent = parent  # another TreeNode
        self.children = []  # list of TreeNodes


class RRT:
    """
    RRT Tree.
    """

    def __init__(self, root: TreeNode, joint_limits: np.ndarray):
        self.root = root  # root TreeNode
        self.joint_limits = joint_limits  # robot.ConfigurationSpace
        self.size = 1  # int length of path
        self.max_recursion = 1000  # int length of longest possible path

    def add_configuration(self, parent_node, child_value):
        child_cost = parent_node.cost_to + np.linalg.norm(parent_node.value - child_value)
        child_node = TreeNode(child_value, child_cost, parent_node)
        parent_node.children.append(child_node)
        self.size += 1
        return child_node

    def valid_configuration(self, configuration):
        return (configuration >= self.joint_limits[:, 0]).all() and (configuration <= self.joint_limits[:, 1]).all()

    # Brute force nearest, handles general distance functions
    def nearest(self, configuration):
        """
        Finds the nearest node by distance to configuration in the
             configuration space.

        Args:
            configuration: tuple of floats representing a configuration of a
                robot

        Returns:
            closest: TreeNode. the closest node in the configuration space
                to configuration
            distance: float. distance from configuration to closest
        """
        assert self.valid_configuration(configuration)

        def recur(node, depth=0):
            closest, distance = node, np.linalg.norm(node.value - configuration)
            if depth < self.max_recursion:
                for child in node.children:
                    (child_closest, child_distance) = recur(child, depth + 1)
                    if child_distance < distance:
                        closest = child_closest
                        child_distance = child_distance
            return closest, distance

        return recur(self.root)[0]
    
class RRT_tools:
    def __init__(self, q_start, q_goal, joint_limits, collision_checker):
        # rrt is a tree
        self.rrt_tree = RRT(TreeNode(q_start, 0), joint_limits)
        print(joint_limits)
        print(q_start)
        print(q_goal)
        assert self.rrt_tree.valid_configuration(q_start)
        assert self.rrt_tree.valid_configuration(q_goal)

        self.joint_limits = joint_limits
        self.collision_checker = collision_checker
        self.q_goal = q_goal

    def find_nearest_node_in_RRT_graph(self, q_sample):
        nearest_node = self.rrt_tree.nearest(q_sample)
        return nearest_node

    def sample_node_in_configuration_space(self):
        q_sample = np.random.uniform(self.joint_limits[:, 0], self.joint_limits[:, 1])
        return q_sample

    def interpolated_path(self, one: np.ndarray, two: np.ndarray):
        # Assert that both configurations are valid
        assert self.rrt_tree.valid_configuration(one) and self.rrt_tree.valid_configuration(two)
        
        diffs = two - one
        max_diffs = np.array(7 * [np.pi / 180 * 2])  # three degrees
        
        samples = np.max(np.ceil(np.abs(diffs) / max_diffs).astype(int))
        samples = max(2, samples)  # Ensure at least 2 samples
        
        linear_interpolation = diffs / (samples - 1.0)
        
        path = np.array([one + i * linear_interpolation for i in range(samples-1)] + [two])

        return path


    def calc_intermediate_qs_wo_collision(self, q_start, q_end):
        """create more samples by linear interpolation from q_start
        to q_end. Return all samples that are not in collision

        Example interpolated path:
        q_start, qa, qb, (Obstacle), qc , q_end
        returns >>> q_start, qa, qb
        """
        path = self.interpolated_path(q_start, q_end)
        safe_path = []
        for configuration in path:
            if not self.collision_checker.CheckConfigCollisionFree(configuration):
                return safe_path
            safe_path.append(configuration)
        return safe_path

    def grow_rrt_tree(self, parent_node, q_sample):
        """
        add q_sample to the rrt tree as a child of the parent node
        returns the rrt tree node generated from q_sample
        """
        child_node = self.rrt_tree.add_configuration(parent_node, q_sample)
        return child_node

    def node_reaches_goal(self, node):
        return (node.value == self.q_goal).all()

    def backup_path_from_node(self, node):
        path = [node.value]
        while node.parent is not None:
            node = node.parent
            path.append(node.value)
        path.reverse()
        return path
    
    def shortcut_path(self, path):
        smoothed_path = path
        for attempt in range(100):
            if len(smoothed_path) <= 2:
                return smoothed_path
            i = randint(0, len(smoothed_path) - 1)
            j = randint(0, len(smoothed_path) - 1)
            if i == j or abs(i - j) == 1:
                continue
            one, two = i, j
            if j < i:
                one, two = j, i
            
            path = self.interpolated_path(smoothed_path[one], smoothed_path[two])
            path_is_safe = True
            for configuration in path:
                if not self.collision_checker.CheckConfigCollisionFree(configuration):
                    path_is_safe = False
                    break
        
            if path_is_safe:
                smoothed_path = smoothed_path[:one + 1] + smoothed_path[two:]
        return smoothed_path
    
def rrt_planning(q_start, q_goal, joint_limits, collision_checker, max_iterations=2000, prob_sample_q_goal=0.05):
    """
    Input:
        problem (IiwaProblem): instance of a utility class
        max_iterations: the maximum number of samples to be collected
        prob_sample_q_goal: the probability of sampling q_goal

    Output:
        path (list): [q_start, ...., q_goal].
                    Note q's are configurations, not RRT nodes
    """
    rrt_tools = RRT_tools(q_start, q_goal, joint_limits, collision_checker)
    
    path = None
    for i in range(max_iterations):
        q_sample = rrt_tools.sample_node_in_configuration_space()
        r = uniform(0, 1)
        if r < prob_sample_q_goal:
            q_sample = q_goal
        n_near = rrt_tools.find_nearest_node_in_RRT_graph(q_sample)

        intermediate_qs = list(rrt_tools.calc_intermediate_qs_wo_collision(n_near.value, q_sample))
        last_node = n_near
        for q in intermediate_qs:
            last_node = rrt_tools.grow_rrt_tree(last_node, q)
        
        if rrt_tools.node_reaches_goal(last_node):
            path = rrt_tools.backup_path_from_node(last_node)
            break
    
    if path is None:
        print("no path found")
        return None
    
    return rrt_tools.shortcut_path(path)
    