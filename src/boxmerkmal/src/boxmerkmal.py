#!/usr/bin/env python

import rospy
import numpy as np
import math

# import matplotlib.pyplot as plt
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32


class AngleHistogramFromLaserScan:
    def __init__(self):
        rospy.init_node("angle_histogram_from_laserscan")

        laserscan_topic = rospy.get_param("~laserscan_topic", "/sensors/scanFront")
        angle_topic = rospy.get_param("~angle_topic", "/average_angle")

        self.k = rospy.get_param("~k_value", 5)  # Set the 'k' value for point pairs

        self.sub = rospy.Subscriber(laserscan_topic, LaserScan, self.callback)
        self.pub = rospy.Publisher(angle_topic, Float32, queue_size=10)

        # plt.ion()  # Interactive mode for live updating plots

    def callback(self, data):
        rospy.loginfo("Entered callback")
        x, y = self.convert_and_filter_scan_data(data)
        if x is None or y is None:  # Early exit if not enough points
            return

        angles = self.calculate_angles(x, y)
        self.process_and_publish_angles(angles)

    def convert_and_filter_scan_data(self, data):
        angles = np.linspace(data.angle_min, data.angle_max, len(data.ranges))
        distances = np.array(data.ranges)

        valid_indices = np.isfinite(distances)
        if not np.any(valid_indices):
            rospy.logwarn("No valid distances in laser scan data.")
            return None, None

        distances = distances[valid_indices]
        angles = angles[valid_indices]

        x = distances * np.cos(angles)
        y = distances * np.sin(angles)

        if len(x) <= self.k:
            rospy.logwarn("Not enough points in laser scan to calculate angles.")
            return None, None

        return x, y

    def calculate_angles(self, x, y):
        dx = x[self.k :] - x[: -self.k]
        dy = y[self.k :] - y[: -self.k]
        dx[dx == 0] = 1e-10
        angle_rad = np.arctan(dy / dx)  # Use np.arctan2 for element-wise arctangent
        angles = np.degrees(angle_rad)
        return angles[np.isfinite(angles)]  # Filter out NaN values

    def process_and_publish_angles(self, angles):
        if len(angles) == 0:
            rospy.logwarn("No valid angles left after filtering.")
            return

        bins = np.arange(-180, 190, 10)  # Define bins
        count, bins = np.histogram(angles, bins=bins)
        max_bin_index = np.argmax(count)
        avg_angle = np.mean(
            [
                angles[j]
                for j in range(len(angles))
                if bins[max_bin_index] <= angles[j] < bins[max_bin_index + 1]
            ]
        )

        self.pub.publish(avg_angle)

    def run(self):
        rospy.loginfo("Node is running...")
        while not rospy.is_shutdown():
            rospy.spin()


if __name__ == "__main__":
    node = AngleHistogramFromLaserScan()
    node.run()
