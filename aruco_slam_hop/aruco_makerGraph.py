#!/usr/bin/env python3

import numpy as np
from collections import deque
import rospy

class MarkerGraph:
    def __init__(self):
        self.nodes = {}  # marker_id -> PoseStamped (global pose if known)
        self.edges = {}  # (from_id, to_id) -> relative transform (4x4 numpy array)

    def add_node(self, marker_id, pose):
        self.nodes[marker_id] = pose

    def add_edge(self, from_id, to_id, relative_transform):
        self.edges[(from_id, to_id)] = relative_transform

    def has_edge(self, id1, id2):
        return (id1, id2) in self.edges or (id2, id1) in self.edges

    def get_pose(self, marker_id):
        return self.nodes.get(marker_id, None)

    def get_global_pose(self, target_id, known_poses_dict, pose_to_matrix_func):
        """
        Try to find the global pose of target_id using known marker poses and graph edges.

        Args:
            target_id (int): marker ID to localize
            known_poses_dict (dict): marker_id -> PoseStamped
            pose_to_matrix_func (function): converts PoseStamped to 4x4 numpy matrix

        Returns:
            (bool, np.ndarray): success flag, 4x4 global transform matrix for the target marker
        """
        visited = set()
        queue = deque([(target_id, np.eye(4))])

        while queue:
            current_id, T_total = queue.popleft()
            visited.add(current_id)

            for (from_id, to_id), T_rel in self.edges.items():
                # Case 1: current_id → known marker
                if from_id == current_id and to_id in known_poses_dict:
                    T_known = pose_to_matrix_func(known_poses_dict[to_id])
                    T_target = T_known @ np.linalg.inv(T_rel)
                    return True, T_target

                # Case 2: known marker → current_id
                elif to_id == current_id and from_id in known_poses_dict:
                    T_known = pose_to_matrix_func(known_poses_dict[from_id])
                    T_target = T_known @ T_rel
                    return True, T_target

                # Case 3: keep traversing the graph
                elif from_id == current_id and to_id not in visited:
                    queue.append((to_id, T_total @ T_rel))
                elif to_id == current_id and from_id not in visited:
                    queue.append((from_id, T_total @ np.linalg.inv(T_rel)))

        return False, None
    
    def log_graph_structure(self):
        rospy.loginfo("===== Marker Graph Structure =====")
        rospy.loginfo(f"Total nodes: {len(self.nodes)}")

        for mid, pose in self.nodes.items():
            pos = pose.pose.position
            rospy.loginfo(f"  Node ID {mid}: ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})")
        
        rospy.loginfo(f"Total edges: {len(self.edges)}")
        for (id1, id2), T in self.edges.items():
            rospy.loginfo(f"  Edge: {id1} <--> {id2}")
        rospy.loginfo("====================================")
