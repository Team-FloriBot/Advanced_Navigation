#!/usr/bin/env python

import rospy
import numpy as np
import math
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from std_msgs.msg import Float32, Header, ColorRGBA
from geometry_msgs.msg import Point32, Pose, Vector3, Quaternion
from visualization_msgs.msg import Marker


class ClosestPairAngleCalculator:
    def __init__(self):
        rospy.init_node("closest_pair_angle_node")
        self.min_distance = 0.4
        self.threshold = 0.16
        self.closest_pair_angle_pub = rospy.Publisher(
            rospy.get_param("~angle_topic", "/closest_pair_angle_hist"),
            Float32,
            queue_size=10,
        )
        self.line_marker_pub = rospy.Publisher(
            "/line_visualization", Marker, queue_size=10
        )
        self.point_marker_pub = rospy.Publisher(
            "/point_visualization", Marker, queue_size=10
        )

        rospy.Subscriber("/merged_point_cloud", PointCloud2, self.point_cloud_callback)

    def point_distance(self, point1, point2):
        return math.hypot(point1.x - point2.x, point1.y - point2.y)

    def point_cloud_callback(self, msg):
        points = [
            Point32(p[0], p[1], p[2])
            for p in point_cloud2.read_points(
                msg, field_names=("x", "y", "z"), skip_nans=True
            )
        ]

        closest_points = sorted(points, key=lambda p: p.x**2 + p.y**2)[:30]

        line_list_marker, point_list_marker = self.prepare_markers(msg.header.frame_id)

        valid_angles = []
        for i, point1 in enumerate(closest_points[:-1]):
            for point2 in closest_points[i + 1 :]:
                dist = self.point_distance(point1, point2)
                if self.threshold < dist < self.min_distance:
                    dx = point2.x - point1.x
                    dy = point2.y - point1.y
                    angle_rad = (
                        math.atan(dy / dx)
                        if dx != 0
                        else (math.pi / 2 if dy > 0 else -math.pi / 2)
                    )
                    valid_angles.append(angle_rad)
                    self.update_markers(
                        line_list_marker, point_list_marker, point1, point2
                    )

        self.process_valid_pairs(valid_angles, line_list_marker, point_list_marker)

    def prepare_markers(self, frame_id):
        header = Header(frame_id=frame_id)
        line_list_marker = Marker(
            type=Marker.LINE_LIST,
            id=0,
            lifetime=rospy.Duration(0),
            pose=Pose(Point32(0, 0, 0), Quaternion(0, 0, 0, 1)),
            scale=Vector3(0.01, 0.0, 0.0),
            header=header,
            color=ColorRGBA(1.0, 0.0, 0.0, 1.0),
            points=[],
        )
        point_list_marker = Marker(
            type=Marker.POINTS,
            id=1,
            lifetime=rospy.Duration(0),
            pose=Pose(Point32(0, 0, 0), Quaternion(0, 0, 0, 1)),
            scale=Vector3(0.05, 0.05, 0.05),
            header=header,
            color=ColorRGBA(0.0, 1.0, 0.0, 1.0),
            points=[],
        )
        return line_list_marker, point_list_marker

    def update_markers(self, line_list_marker, point_list_marker, point1, point2):
        line_list_marker.points.append(point1)
        line_list_marker.points.append(point2)
        point_list_marker.points.append(point1)
        point_list_marker.points.append(point2)

    def process_valid_pairs(self, valid_angles, line_list_marker, point_list_marker):
        if valid_angles:
            angle_counts, angle_bins = np.histogram(valid_angles, bins="auto")
            max_bin_index = np.argmax(angle_counts)
            max_bin_start, max_bin_end = (
                angle_bins[max_bin_index],
                angle_bins[max_bin_index + 1],
            )
            angles_in_max_bin = [
                angle for angle in valid_angles if max_bin_start <= angle < max_bin_end
            ]

            if angles_in_max_bin:
                average_angle_rad = sum(angles_in_max_bin) / len(angles_in_max_bin)
                average_angle_deg = math.degrees(average_angle_rad)
                rospy.loginfo(
                    f"Average angle of max bin (degrees): {average_angle_deg}"
                )
                self.closest_pair_angle_pub.publish(average_angle_deg)
                self.line_marker_pub.publish(line_list_marker)
                self.point_marker_pub.publish(point_list_marker)
            else:
                rospy.logwarn("No angles found in the most common bin.")


if __name__ == "__main__":
    calculator = ClosestPairAngleCalculator()
    rospy.spin()
