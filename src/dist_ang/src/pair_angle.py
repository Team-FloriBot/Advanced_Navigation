#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import math
from std_msgs.msg import Float32, Header, ColorRGBA
from geometry_msgs.msg import Point32, Pose, Vector3, Quaternion
from visualization_msgs.msg import Marker


class ClosestPairAngleCalculator:
    def __init__(self):
        rospy.init_node("closest_pair_angle_node")
        self.min_distance = 0.4
        self.threshold = 0.16
        self.closest_pair_angle_pub = rospy.Publisher(
            rospy.get_param("~angle_topic", "/closest_pair_angle"),
            Float32,
            queue_size=10,
        )
        self.line_marker_pub = rospy.Publisher(
            "/line_visualization", Marker, queue_size=10
        )
        self.point_marker_pub = rospy.Publisher(
            "/point_visualization", Marker, queue_size=10
        )

        # Subscribe to the merged point cloud
        rospy.Subscriber("/merged_point_cloud", PointCloud2, self.point_cloud_callback)

    def point_distance(self, point1, point2):
        # Calculate Euclidean distance in the XY plane
        return math.hypot(point1.x - point2.x, point1.y - point2.y)

    def point_cloud_callback(self, msg):
        # Extract the points from the PointCloud2 message and convert to Point32
        points = [
            Point32(p[0], p[1], p[2])
            for p in point_cloud2.read_points(
                msg, field_names=("x", "y", "z"), skip_nans=True
            )
        ]

        # Select the closest 30 points to the origin (in XY plane)
        closest_points = sorted(points, key=lambda p: p.x**2 + p.y**2)[:30]

        # Prepare markers for visualization
        line_list_marker, point_list_marker = self.prepare_markers(msg.header.frame_id)

        valid_angles = []  # List to store valid angles
        # Iterate over all unique pairs of closest points
        for i, point1 in enumerate(closest_points[:-1]):
            for point2 in closest_points[i + 1 :]:
                dist = self.point_distance(point1, point2)
                if self.threshold < dist < self.min_distance:
                    # Avoid division by zero and calculate angle in radians
                    dx = point2.x - point1.x
                    dy = point2.y - point1.y
                    angle_rad = (
                        math.atan(dy / dx)
                        if dx != 0
                        else (math.pi / 2 if dy > 0 else -math.pi / 2)
                    )
                    valid_angles.append(angle_rad)
                    # Update visualization markers
                    self.update_markers(
                        line_list_marker, point_list_marker, point1, point2
                    )

        # Process valid pairs
        self.process_valid_pairs(valid_angles, line_list_marker, point_list_marker)

    def prepare_markers(self, frame_id):
        header = Header(frame_id=frame_id)
        marker_common_properties = {
            "header": header,
            "pose": Pose(orientation=Quaternion(w=1.0)),
            "lifetime": rospy.Duration(),
        }
        # Prepare line list marker
        line_list_marker = Marker(**marker_common_properties)
        line_list_marker.type = Marker.LINE_LIST
        line_list_marker.scale = Vector3(x=0.01)
        line_list_marker.color = ColorRGBA(r=1.0, g=1.0, a=1.0)

        # Prepare point list marker
        point_list_marker = Marker(**marker_common_properties)
        point_list_marker.type = Marker.POINTS
        point_list_marker.scale = Vector3(x=0.05, y=0.05)
        point_list_marker.color = ColorRGBA(g=1.0, a=1.0)

        return line_list_marker, point_list_marker

    def update_markers(self, line_list_marker, point_list_marker, point1, point2):
        # Add points to markers for visualization
        line_list_marker.points.extend([point1, point2])
        point_list_marker.points.extend([point1, point2])

    def process_valid_pairs(self, valid_angles, line_list_marker, point_list_marker):
        # If there are valid angles, calculate the average angle, publish it and the markers
        if valid_angles:
            average_angle_rad = sum(valid_angles) / len(valid_angles)
            average_angle_deg = math.degrees(average_angle_rad)
            rospy.loginfo(f"Average pair angle (degrees): {average_angle_deg}")
            self.closest_pair_angle_pub.publish(average_angle_deg)
            self.line_marker_pub.publish(line_list_marker)
            self.point_marker_pub.publish(point_list_marker)


if __name__ == "__main__":
    calculator = ClosestPairAngleCalculator()
    rospy.spin()
