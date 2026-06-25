#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Point, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from tf.transformations import quaternion_from_matrix, quaternion_matrix
import tf2_ros
import cv2.aruco as aruco

from aruco_slam_HOP.msg import MarkerWithPose
from aruco_slam_hop.aruco_markerHandler import MarkerHandler
from aruco_slam_hop.aruco_makerGraph import MarkerGraph 


class arucoDetectorNode:
    def __init__(self):
        rospy.init_node('aruco_detector')
        self.bridge = CvBridge()

        # =========================== Subscribers ===========================
        self.image_sub = rospy.Subscriber('/turtlebot/kobuki/realsense/color/image_color', Image, self.image_callback)
        # self.image_sub = rospy.Subscriber('/turtlebot/kobuki/realsense/color/image_raw', Image, self.image_callback)
        self.camera_info_sub = rospy.Subscriber('/turtlebot/kobuki/realsense/color/camera_info', CameraInfo, self.camera_info_callback)

        # =========================== Publishers ===========================
        self.marker_pose_pub = rospy.Publisher('/aruco/marker_pose', MarkerWithPose, queue_size=10)
        self.pose_pub = rospy.Publisher('/turtlebot/kobuki/aruco_position_perception', PoseStamped, queue_size=10)
        self.aruco_rviz_marker_pub = rospy.Publisher('/aruco/visualization_aruco_perception', MarkerArray, queue_size=10)
        self.marker_lines_pub = rospy.Publisher('/aruco/marker_lines', MarkerArray, queue_size=10)
        self.marker_distances_pub = rospy.Publisher('/aruco/marker_distances', MarkerArray, queue_size=10)

        self.marker_handler = MarkerHandler()
        self.graph = MarkerGraph()

        self.marker_length = 0.1875
        self.detected_order = []
        self.aruco_dict_type = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_frame = None
        self.frame_id = "world_ned"

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        self.initialized_world = False

        # 3D corner model for solvePnP
        h = self.marker_length / 2.0
        self.marker_obj_pts = np.array([
                                        [-h, h, 0],
                                        [ h, h, 0],
                                        [ h,-h, 0],
                                        [-h,-h, 0]], dtype=np.float32)


    def camera_info_callback(self, msg):
        self.camera_matrix = np.array(msg.K).reshape(3,3)
        self.dist_coeffs = np.array(msg.D)
        self.camera_frame = msg.header.frame_id
        self.camera_info_sub.unregister()
        rospy.loginfo(f"Camera calibration received. Using frame: {self.camera_frame}")


    def image_callback(self, msg):
        if self.camera_matrix is None:
            rospy.logwarn_throttle(5.0, "Waiting for camera intrinsics...")
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = aruco.ArucoDetector(self.aruco_dict_type, aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is None:
            return

        for i, marker_id in enumerate(ids.flatten()):
            image_points = corners[i][0].astype(np.float32)
            success, rvec, tvec = cv2.solvePnP(self.marker_obj_pts, image_points,
                                               self.camera_matrix, self.dist_coeffs,
                                               flags=cv2.SOLVEPNP_IPPE_SQUARE)

            if not success:
                rospy.logwarn(f"solvePnP failed for marker {marker_id}")
                continue

            # Broadcast transform for this marker
            t = TransformStamped()
            t.header.stamp = msg.header.stamp
            t.header.frame_id = self.camera_frame
            t.child_frame_id = f"aruco_marker_{marker_id}"
            t.transform.translation.x = tvec[0][0]
            t.transform.translation.y = tvec[1][0]
            t.transform.translation.z = tvec[2][0]

            rot_mat = cv2.Rodrigues(rvec)[0]
            T = np.eye(4)
            T[:3, :3] = rot_mat
            q = quaternion_from_matrix(T)
            t.transform.rotation.x, t.transform.rotation.y = q[0], q[1]
            t.transform.rotation.z, t.transform.rotation.w = q[2], q[3]
            self.tf_broadcaster.sendTransform(t)

            # Now lookup from world_ned to aruco_marker_X
            try:
                marker_frame = t.child_frame_id
                transform = self.tf_buffer.lookup_transform(self.frame_id, marker_frame, rospy.Time(0), rospy.Duration(0.5))

                pose_world = PoseStamped()
                pose_world.header.stamp = transform.header.stamp
                pose_world.header.frame_id = self.frame_id
                pose_world.pose.position.x = transform.transform.translation.x
                pose_world.pose.position.y = transform.transform.translation.y
                pose_world.pose.position.z = transform.transform.translation.z
                pose_world.pose.orientation = transform.transform.rotation

                self.markerID_pose_msg(marker_id, pose_world)
                self.pose_pub.publish(pose_world)
                self.publish_marker(pose_world, marker_id)

                self.graph.add_node(marker_id, pose_world)

                if marker_id not in self.marker_handler.markers:
                    if not self.initialized_world:
                        self.marker_handler.add_marker(marker_id, pose_world)
                        self.initialized_world = True
                        self.detected_order.append(marker_id)
                        rospy.logwarn(f"First marker {marker_id} set as world origin.")
                    else:
                        self.marker_handler.add_marker(marker_id, pose_world)
                        self.detected_order.append(marker_id)
                        rospy.logwarn(f"Marker {marker_id} stored.")
                        self.graph.log_graph_structure()

                # Add edges
                for m in range(len(ids)):
                    for n in range(m + 1, len(ids)):
                        id1, id2 = ids[m][0], ids[n][0]
                        if not self.graph.has_edge(id1, id2):
                            pose1 = self.marker_handler.get_marker_position(id1)
                            pose2 = self.marker_handler.get_marker_position(id2)
                            if pose1 is None or pose2 is None:
                                continue
                            T1 = self.pose_to_matrix(pose1.pose)
                            T2 = self.pose_to_matrix(pose2.pose)
                            T_rel = np.linalg.inv(T1) @ T2
                            self.graph.add_edge(id1, id2, T_rel)

            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
                rospy.logwarn(f"TF lookup failed for marker {marker_id}: {e}")

        self.publish_marker_connections()


    def markerID_pose_msg(self, marker_id, pose_world):
        msg = MarkerWithPose()
        msg.id = marker_id
        msg.pose = pose_world
        msg.pose.header.stamp = rospy.Time.now()
        msg.pose.header.frame_id = self.frame_id
        self.marker_pose_pub.publish(msg)


    def pose_to_matrix(self, pose):
        T = quaternion_matrix([
                                pose.orientation.x,
                                pose.orientation.y,
                                pose.orientation.z,
                                pose.orientation.w])
        T[0, 3] = pose.position.x
        T[1, 3] = pose.position.y
        T[2, 3] = pose.position.z
        return T
    
    
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
        marker.lifetime = rospy.Duration(0)

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
        text_marker.lifetime = rospy.Duration(0)

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
            line.header.frame_id = self.frame_id
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
            line.lifetime = rospy.Duration(0)
            connections.markers.append(line)

            dist = np.linalg.norm([pose2.x - pose1.x, pose2.y - pose1.y])
            text = Marker()
            text.header.frame_id = self.frame_id
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
            text.lifetime = rospy.Duration(0)
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
