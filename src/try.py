#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from tf.transformations import quaternion_from_matrix
import tf2_ros
import tf2_geometry_msgs
import cv2.aruco as aruco

from aruco_slam_hop.aruco_markerHandler import MarkerHandler

class arucoDetectorNode:
    def __init__(self):
        rospy.init_node('aruco_detector')
        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber('/turtlebot/kobuki/realsense/color/image_color', Image, self.image_callback)

        self.pose_pub = rospy.Publisher('/turtlebot/kobuki/aruco_position_perception', PoseStamped, queue_size=10)
        self.aruco_rviz_marker_pub = rospy.Publisher('/aruco/visualization_aruco_perception', MarkerArray, queue_size=10)
        self.marker_lines_pub = rospy.Publisher('/aruco/marker_lines', MarkerArray, queue_size=10)
        self.marker_distances_pub = rospy.Publisher('/aruco/marker_distances', MarkerArray, queue_size=10)

        self.marker_handler = MarkerHandler()
        self.marker_length = 0.1875
        self.detected_order = []

        self.aruco_dict_type = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        self.camera_matrix = np.array([[1396.81, 0.0, 960.0], [0.0, 1396.81, 540.0], [0.0, 0.0, 1.0]])
        self.dist_coeffs = np.zeros(5)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.world_origin_marker_id = None
        self.initialized_world = False


    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        parameters = aruco.DetectorParameters()
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict_type, parameters=parameters)

        if ids is None:
            rospy.logerr_throttle(3, "No aruco marker detected or wrong dictionary selected")
            return

        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, self.marker_length, self.camera_matrix, self.dist_coeffs)

        for i in range(len(ids)):
            marker_id = ids[i][0]
            rvec = rvecs[i][0]
            tvec = tvecs[i][0]

            # Pose in camera frame
            pose_cam = PoseStamped()
            pose_cam.header.stamp = rospy.Time.now()
            pose_cam.header.frame_id = "turtlebot/kobuki/realsense_color"
            pose_cam.pose.position.x = tvec[0]
            pose_cam.pose.position.y = tvec[1]
            pose_cam.pose.position.z = tvec[2]

            rot_mat, _ = cv2.Rodrigues(rvec)
            T = np.eye(4)
            T[:3, :3] = rot_mat
            quat = quaternion_from_matrix(T)
            pose_cam.pose.orientation.x, pose_cam.pose.orientation.y = quat[0], quat[1]
            pose_cam.pose.orientation.z, pose_cam.pose.orientation.w = quat[2], quat[3]

            try:
                transform = self.tf_buffer.lookup_transform("world_ned", pose_cam.header.frame_id, rospy.Time(0), rospy.Duration(1.0))
                pose_world = tf2_geometry_msgs.do_transform_pose(pose_cam, transform)

                self.pose_pub.publish(pose_world)
                self.publish_marker(pose_world, marker_id)

                if marker_id not in self.marker_handler.markers:
                    # First detected marker becomes the world origin [0, 0, 0]
                    if not self.initialized_world:
                        self.marker_handler.add_marker(marker_id, pose_world)
                        self.detected_order.append(marker_id)
                        self.initialized_world = True
                        rospy.logwarn(f"First marker {marker_id} set as world origin based on detected pose.")
                        self.publish_marker(pose_world, marker_id)
                        continue

                    self.marker_handler.add_marker(marker_id, pose_world)

                    # Log relative position with respect to the first 

                    self.detected_order.append(marker_id)
                    
                    rospy.loginfo(f"Marker {marker_id} stored and added to detection order.")
                    rospy.logwarn(f"Detected Aruco markers ids: {self.detected_order}")

            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
                rospy.logwarn(f"TF transform failed for Marker {marker_id}: {e}")

        self.publish_marker_connections()

    
    def publish_marker(self, pose_world, marker_id):
        marker = Marker()
        marker.header.frame_id = pose_world.header.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "aruco_cube"
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position.x = pose_world.pose.position.x
        marker.pose.position.y = pose_world.pose.position.y
        marker.pose.position.z = -0.1
        marker.pose.orientation = pose_world.pose.orientation

        marker.scale.x = marker.scale.y = marker.scale.z = 0.2
        marker.color.r = marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        text_marker = Marker()
        text_marker.header.frame_id = pose_world.header.frame_id
        text_marker.header.stamp = rospy.Time.now()
        text_marker.ns = "aruco_label"
        text_marker.id = marker_id + 1000
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        
        text_marker.pose.position.x = pose_world.pose.position.x
        text_marker.pose.position.y = pose_world.pose.position.y
        text_marker.pose.position.z = -0.5
        text_marker.pose.orientation = pose_world.pose.orientation

        text_marker.scale.z = 0.2
        text_marker.color.r = text_marker.color.g = text_marker.color.b = 1.0
        text_marker.color.a = 1.0
        text_marker.text = f"ArUco {marker_id}"

        self.aruco_rviz_marker_pub.publish(MarkerArray(markers=[marker, text_marker]))


    def publish_marker_connections(self):
        connections = MarkerArray()
        distances = MarkerArray()
        idx = 0

        for i in range(len(self.detected_order) - 1):
            id1 = self.detected_order[i]
            id2 = self.detected_order[i + 1]
            pose1 = self.marker_handler.get_marker_position(id1).pose.position
            pose2 = self.marker_handler.get_marker_position(id2).pose.position

            line = Marker()
            line.header.frame_id = "world_ned"
            line.header.stamp = rospy.Time.now()
            line.ns = "marker_edges"
            line.id = idx
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.scale.x = 0.02
            line.color.r = 1.0
            line.color.g = 0.0
            line.color.b = 1.0
            line.color.a = 1.0
            line.points = [Point(x = pose1.x, y = pose1.y, z = -0.05), Point(x = pose2.x, y = pose2.y, z = -0.05)]
            connections.markers.append(line)

            dist = np.linalg.norm([pose2.x - pose1.x, pose2.y - pose1.y])
            text = Marker()
            text.header.frame_id = "world_ned"
            text.header.stamp = rospy.Time.now()
            text.ns = "marker_distances"
            text.id = 1000 + idx
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = (pose1.x + pose2.x) / 2
            text.pose.position.y = (pose1.y + pose2.y) / 2
            text.pose.position.z = -0.4
            text.scale.z = 0.2
            text.color.r = text.color.g = 1.0
            text.color.b = 0.0
            text.color.a = 1.0
            text.text = f"{dist:.2f}m"
            distances.markers.append(text)
            idx += 1

        self.marker_lines_pub.publish(connections)
        self.marker_distances_pub.publish(distances)

if __name__ == "__main__":
    try:
        node = arucoDetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
