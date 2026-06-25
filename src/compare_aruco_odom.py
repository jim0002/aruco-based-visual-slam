#!/usr/bin/env python3

import rospy
import tf.transformations as tf_trans
import tf2_ros
import tf2_geometry_msgs
import matplotlib.pyplot as plt
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

class PoseComparerPlotter:
    def __init__(self):
        rospy.init_node("pose_comparer_plotter")

        rospy.Subscriber("/robot/aruco_localized_pose", PoseStamped, self.aruco_callback)
        rospy.Subscriber("/turtlebot/kobuki/odom", Odometry, self.odom_callback)

        self.aruco_pose = None
        self.odom_pose_world = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.timestamps = []
        self.aruco_data = {'x': [], 'y': [], 'yaw': []}
        self.odom_data = {'x': [], 'y': [], 'yaw': []}
        self.error_data = {'x': [], 'y': [], 'yaw': []}

        rospy.on_shutdown(self.on_shutdown)

    def aruco_callback(self, msg):
        self.aruco_pose = msg
        self.collect_and_store()

    def odom_callback(self, msg):
        try:
            odom_pose = PoseStamped()
            odom_pose.header = msg.header
            odom_pose.pose = msg.pose.pose
            tf = self.tf_buffer.lookup_transform("world_ned", msg.header.frame_id, rospy.Time(0), rospy.Duration(1.0))
            self.odom_pose_world = tf2_geometry_msgs.do_transform_pose(odom_pose, tf)
            self.collect_and_store()
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
            pass

    def get_yaw(self, orientation):
        quat = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = tf_trans.euler_from_quaternion(quat)
        return yaw

    def collect_and_store(self):
        if self.aruco_pose and self.odom_pose_world:
            t = rospy.Time.now().to_sec()
            self.timestamps.append(t)

            ax = self.aruco_pose.pose.position.x
            ay = self.aruco_pose.pose.position.y
            ayaw = self.get_yaw(self.aruco_pose.pose.orientation)

            ox = self.odom_pose_world.pose.position.x
            oy = self.odom_pose_world.pose.position.y
            oyaw = self.get_yaw(self.odom_pose_world.pose.orientation)

            self.aruco_data['x'].append(ax)
            self.aruco_data['y'].append(ay)
            self.aruco_data['yaw'].append(ayaw)

            self.odom_data['x'].append(ox)
            self.odom_data['y'].append(oy)
            self.odom_data['yaw'].append(oyaw)

            self.error_data['x'].append(ax - ox)
            self.error_data['y'].append(ay - oy)
            self.error_data['yaw'].append(ayaw - oyaw)

    def on_shutdown(self):
        plt.figure(figsize=(15, 8))
        labels = ['Aruco', 'Odom', 'Ground Truth']

        # X Position
        plt.subplot(2, 3, 1)
        plt.plot(self.timestamps, self.aruco_data['x'], label='Aruco X')
        plt.plot(self.timestamps, self.odom_data['x'], label='Odom X')
        plt.title('X Position Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('X Position (m)')
        plt.legend()
        plt.grid(True)

        # Y Position
        plt.subplot(2, 3, 2)
        plt.plot(self.timestamps, self.aruco_data['y'], label='Aruco Y')
        plt.plot(self.timestamps, self.odom_data['y'], label='Odom Y')
        plt.title('Y Position Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Y Position (m)')
        plt.legend()
        plt.grid(True)

        # Yaw
        plt.subplot(2, 3, 3)
        plt.plot(self.timestamps, self.aruco_data['yaw'], label='Aruco Yaw')
        plt.plot(self.timestamps, self.odom_data['yaw'], label='Odom Yaw')
        plt.title('Yaw Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Yaw (rad)')
        plt.legend()
        plt.grid(True)

        # X Error
        plt.subplot(2, 3, 4)
        plt.plot(self.timestamps, self.error_data['x'], label='X Error')
        plt.title('X Error (Aruco - Odom)')
        plt.xlabel('Time (s)')
        plt.ylabel('X Error (m)')
        plt.legend()
        plt.grid(True)

        # Y Error
        plt.subplot(2, 3, 5)
        plt.plot(self.timestamps, self.error_data['y'], label='Y Error')
        plt.title('Y Error (Aruco - Odom)')
        plt.xlabel('Time (s)')
        plt.ylabel('Y Error (m)')
        plt.legend()
        plt.grid(True)

        # Yaw Error
        plt.subplot(2, 3, 6)
        plt.plot(self.timestamps, self.error_data['yaw'], label='Yaw Error')
        plt.title('Yaw Error (Aruco - Odom)')
        plt.xlabel('Time (s)')
        plt.ylabel('Yaw Error (rad)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('/home/priyam/aruco_ros/src/aruco_slam_HOP/pose_comparison_plot.png')
        plt.close()

PoseComparerPlotter()
rospy.spin()