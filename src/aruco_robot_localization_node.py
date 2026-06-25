#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, TransformStamped
from aruco_slam_HOP.msg import MarkerWithPose
import tf2_ros
import tf2_geometry_msgs
import tf.transformations as tf_trans
from aruco_slam_hop.aruco_makerGraph import MarkerGraph 
from scipy.spatial.transform import Rotation as R


class ArucoRobotLocalizationNode:
    def __init__(self):
        rospy.init_node('aruco_robot_localization')

        self.frame_id = "world_ned"
        self.robot_pose_pub = rospy.Publisher("/robot/aruco_localized_pose", PoseStamped, queue_size=10)

        # Subscribe to global marker poses (published by arucoDetectorNode)
        rospy.Subscriber("/aruco/marker_pose", MarkerWithPose, self.marker_pose_callback)

        # TF buffer to get marker pose relative to camera
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        self.graph = MarkerGraph()
        self.known_markers = {}  # marker_id -> PoseStamped

        self.num_last_used_markers = 0

        rospy.logwarn("Aruco Robot Localization Node Initialized.")


    def marker_pose_callback(self, marker_msg):
        self.known_markers[marker_msg.id] = marker_msg.pose
        
        visible_markers = list(self.known_markers.items())
        robot_transforms = []

        source_camera_frame = "turtlebot/kobuki/realsense_color"

        for marker_id, marker_pose in visible_markers:
            try:
                tf_marker_in_camera = self.tf_buffer.lookup_transform(
                    target_frame = source_camera_frame,  # where is marker relative to camera?
                    source_frame = f"aruco_marker_{marker_id}",
                    time = rospy.Time(0),
                    timeout = rospy.Duration(0.3)
                )

                T_marker_world = self.pose_to_matrix(marker_pose.pose)
                T_marker_cam   = self.tf_to_matrix(tf_marker_in_camera.transform)
                T_cam_world    = T_marker_world @ np.linalg.inv(T_marker_cam)

                robot_transforms.append(T_cam_world)

            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
                continue

        if not robot_transforms:
            rospy.logwarn_throttle(3, "No valid marker transforms found for fusion.")
            return

        T_avg = self.average_transforms(robot_transforms)

        self.last_pose_matrix = T_avg

        T_base_in_camera = self.get_base_in_camera_transform()
        T_base_world = T_avg @ T_base_in_camera

        # Convert to pose and publish
        pose_world = self.matrix_to_pose(T_base_world)
        pose_world_stamped = PoseStamped()
        pose_world_stamped.header.stamp = rospy.Time.now()
        pose_world_stamped.header.frame_id = self.frame_id
        pose_world_stamped.pose = pose_world

        self.robot_pose_pub.publish(pose_world_stamped)

        # Only log if number of used markers increased
        if len(robot_transforms) > self.num_last_used_markers:
            self.num_last_used_markers = len(robot_transforms)

            # Log robot pose on terminal
            x = pose_world.position.x
            y = pose_world.position.y
            z = pose_world.position.z

            qx = pose_world.orientation.x
            qy = pose_world.orientation.y
            qz = pose_world.orientation.z
            qw = pose_world.orientation.w

            yaw = tf_trans.euler_from_quaternion([qx, qy, qz, qw])[2]

            rospy.sleep(0.03)
            rospy.loginfo(f"New marker(s) used! Total contributing markers: {len(robot_transforms)}")
            rospy.loginfo(f"""
                            ================ ArUco-Based Robot Pose =================
                            Pose ({self.frame_id}):
                            Position:  x = {x:.2f},  y = {y:.2f},  z = {z:.2f}
                            Yaw (deg): {np.degrees(yaw):.2f}
                            =========================================================
                            
                            """)

    def pose_to_matrix(self, pose):
        T = tf_trans.quaternion_matrix([
                                        pose.orientation.x,
                                        pose.orientation.y,
                                        pose.orientation.z,
                                        pose.orientation.w])
        T[0, 3] = pose.position.x
        T[1, 3] = pose.position.y
        T[2, 3] = pose.position.z
        return T


    def tf_to_matrix(self, tf):
        T = tf_trans.quaternion_matrix([
                                        tf.rotation.x,
                                        tf.rotation.y,
                                        tf.rotation.z,
                                        tf.rotation.w])
        T[0, 3] = tf.translation.x
        T[1, 3] = tf.translation.y
        T[2, 3] = tf.translation.z
        return T


    def matrix_to_pose(self, T):
        trans = tf_trans.translation_from_matrix(T)
        quat = tf_trans.quaternion_from_matrix(T)
        pose = PoseStamped().pose
        pose.position.x, pose.position.y, pose.position.z = trans
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quat
        return pose
    

    def average_transforms(self, transforms):
        """
            Averages a list of 4x4 transformation matrices.
            Returns a 4x4 averaged transformation.
        """
        if not transforms:
            return None

        translations = np.array([T[:3, 3] for T in transforms])
        avg_translation = np.mean(translations, axis=0)

        quaternions = [R.from_matrix(T[:3, :3]).as_quat() for T in transforms]
        A = np.zeros((4, 4))
        for q in quaternions:
            q = q.reshape(4, 1)
            A += q @ q.T
        eigvals, eigvecs = np.linalg.eigh(A)
        avg_quat = eigvecs[:, np.argmax(eigvals)]
        avg_rot = R.from_quat(avg_quat).as_matrix()

        T_avg = np.eye(4)
        T_avg[:3, :3] = avg_rot
        T_avg[:3, 3] = avg_translation
        return T_avg
    

    def get_base_in_camera_transform(self):
        try:
            tf_base_in_camera = self.tf_buffer.lookup_transform(
                target_frame="turtlebot/kobuki/realsense_color",       # camera frame
                source_frame="turtlebot/kobuki/base_link",             # base frame
                time=rospy.Time(0),
                timeout=rospy.Duration(0.3)
            )
            return self.tf_to_matrix(tf_base_in_camera.transform)
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
            rospy.logwarn("Could not get base_link -> camera transform")
            return np.eye(4)    


if __name__ == "__main__":
    try:
        node = ArucoRobotLocalizationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
