#!/usr/bin/env python

import rospy
import numpy as np
import matplotlib.pyplot as plt
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32


class AngleHistogramFromLaserScan:
    def __init__(self):
        rospy.init_node("angle_histogram_from_laserscan")

        self.laserscan_topic = rospy.get_param("~laserscan_topic", "/sensors/scanFront")
        self.angle_topic = rospy.get_param("~angle_topic", "/average_angle")

        self.k = rospy.get_param("~k_value", 5)  # Set the 'k' value for point pairs

        self.sub = rospy.Subscriber(self.laserscan_topic, LaserScan, self.callback)
        self.pub = rospy.Publisher(self.angle_topic, Float32, queue_size=10)

        self.angles_to_plot = None  # Initialize storage for angle data

        # Set up the plotting
        plt.ion()
        self.figure, self.ax = plt.subplots()

    def callback(self, data):
        rospy.loginfo("Entered callback")
        x, y = self.convert_and_filter_scan_data(data)
        if x is None or y is None:  # Early exit if not enough points
            return

        angles = self.calculate_angles(x, y)
        self.angles_to_plot = angles  # Store angles for plotting
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

        angles = np.degrees(np.arctan2(dy, dx))
        return angles[np.isfinite(angles)]

    def process_and_publish_angles(self, angles):
        if len(angles) == 0:
            rospy.logwarn("No valid angles left after filtering.")
            return

        bins = np.arange(-90, 100, 10)
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

    def visualize_histogram(self):
        if self.angles_to_plot is not None:
            self.ax.clear()
            bins = np.arange(-90, 100, 10)
            self.ax.hist(self.angles_to_plot, bins=bins, alpha=0.75)
            self.ax.set_title("Histogram of Angles")
            self.ax.set_xlabel("Angle (degrees)")
            self.ax.set_ylabel("Frequency")
            plt.draw()
            plt.pause(0.1)

    def run(self):
        rate = rospy.Rate(1)  # Update the plot at 1 Hz
        while not rospy.is_shutdown():
            self.visualize_histogram()
            rate.sleep()


if __name__ == "__main__":
    node = AngleHistogramFromLaserScan()
    node.run()
