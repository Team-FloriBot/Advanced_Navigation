import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import math
from std_msgs.msg import Float32
from geometry_msgs.msg import Point32, Point
import heapq


class ClosestPairAngleCalculator:
    def __init__(self):
        rospy.init_node("closest_pair_angle_node")
        self.min_distance = 0.5
        self.threshold = 0.1
        self.closest_pair_angle_pub = rospy.Publisher(
            "/closest_pair_angle", Float32, queue_size=10
        )

        # Subscribe to the merged point cloud
        rospy.Subscriber("/merged_point_cloud", PointCloud2, self.point_cloud_callback)

    def point_distance(self, point1, point2):
        return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

    def point_cloud_callback(self, msg):
        # Extract the points from the PointCloud2 message
        points = []
        for p in point_cloud2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True
        ):
            points.append(Point32(p[0], p[1], p[2]))

        # Maintain a min-heap for the closest 10 points
        closest_10_points = []
        min_heap = []

        for point in points:
            dist = self.point_distance(
                Point32(0, 0, 0), point
            )  # Robot's position assumed at (0, 0, 0)

            if len(closest_10_points) < 10:
                heapq.heappush(min_heap, (-dist, point))
            elif dist < -min_heap[0][0]:
                heapq.heappop(min_heap)
                heapq.heappush(min_heap, (-dist, point))

        closest_10_points = [point for _, point in min_heap]

        # Find the closest pair of points within the threshold
        closest_pair = None
        min_pair_distance = self.min_distance
        min_pair_threshold = self.threshold

        for i in range(len(closest_10_points)):
            for j in range(i + 1, len(closest_10_points)):
                dist = self.point_distance(closest_10_points[i], closest_10_points[j])
                if dist < min_pair_distance and dist > min_pair_threshold:
                    min_pair_distance = dist
                    closest_pair = (closest_10_points[i], closest_10_points[j])

        if closest_pair:
            angle = math.atan2(
                closest_pair[1].y - closest_pair[0].y,
                closest_pair[1].x - closest_pair[0].x,
            )
            self.closest_pair_angle_pub.publish(angle)


if __name__ == "__main__":
    calculator = ClosestPairAngleCalculator()
    rospy.spin()
