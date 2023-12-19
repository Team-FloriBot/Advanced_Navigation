#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField, Joy
from sensor_msgs import point_cloud2
from geometry_msgs.msg import Twist, Point32
from std_msgs.msg import Bool, Float32
from visualization_msgs.msg import Marker


class PID:
    def __init__(self, kp, ki, kd, integral_limit):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.previous_error = 0.0
        self.integral = 0.0

    def compute(self, error, dt):
        self.integral += error * dt
        self.integral = max(
            min(self.integral, self.integral_limit), -self.integral_limit
        )
        derivative = (error - self.previous_error) / dt
        self.previous_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


class FieldRobotNavigator:
    def __init__(self):
        rospy.init_node("field_robot_navigator")

        # Set up subscribers and publishers
        rospy.Subscriber("/merged_point_cloud", PointCloud2, self.point_cloud_callback)
        rospy.Subscriber("/teleop/automatic_mode", Bool, self.automatic_mode_callback)
        rospy.Subscriber(
            "/obstacle/obstacle_automatic_mode",
            Bool,
            self.obstacle_automatic_mode_callback,
        )
        rospy.Subscriber("/teleop/cmd_vel", Twist, self.teleop_cmd_vel_callback)
        rospy.Subscriber("/teleop/movement_sequence", Joy, self.pattern_callback)
        rospy.Subscriber("/closest_pair_angle", Float32, self.angle_callback)
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.points_pub = rospy.Publisher("/field_points", PointCloud2, queue_size=1)
        self.marker_pub = rospy.Publisher("/polynomial_markers", Marker, queue_size=10)
        self.angle_pub = rospy.Publisher("angle_to_row", Float32, queue_size=10)
        self.center_dist = rospy.Publisher("/center_dist", Float32, queue_size=10)

        # Initialize member variables
        self.robot_pose = None
        self.points = None
        self.drive_points = None

        self.box = rospy.set_param("box", "drive")
        self.both_sides = rospy.set_param("both_sides", "both")

        self.last_linear_speed = 0
        self.pid_controller = PID(kp=2.25, ki=0.01, kd=0.75, integral_limit=1.0)
        # Polyfit parameters
        self.poly_min_dist_req = 0.3
        self.polydist_to_robot = 0.05
        self.needed_points_for_fit = 6

        self.last_cycle_time = rospy.Time.now()

        # Initialize parameters
        self.x_min = self.x_min_drive_in_row = rospy.get_param("x_min_drive_in_row")
        self.x_max = self.x_max_drive_in_row = rospy.get_param("x_max_drive_in_row")
        self.y_min = self.y_min_drive_in_row = rospy.get_param("y_min_drive_in_row")
        self.y_max = self.y_max_drive_in_row = rospy.get_param("y_max_drive_in_row")

        self.x_min_turn_and_exit = rospy.get_param("x_min_turn_and_exit")
        self.x_max_turn_and_exit = rospy.get_param("x_max_turn_and_exit")
        self.y_min_turn_and_exit = rospy.get_param("y_min_turn_and_exit")
        self.y_max_turn_and_exit = rospy.get_param("y_max_turn_and_exit")

        self.x_min_counting_rows = rospy.get_param("x_min_counting_rows")
        self.x_max_counting_rows = rospy.get_param("x_max_counting_rows")
        self.y_min_counting_rows = rospy.get_param("y_min_counting_rows")
        self.y_max_counting_rows = rospy.get_param("y_max_counting_rows")

        self.x_min_turn_to_row = rospy.get_param("x_min_turn_to_row")
        self.x_max_turn_to_row = rospy.get_param("x_max_turn_to_row")
        self.y_min_turn_to_row = rospy.get_param("y_min_turn_to_row")
        self.y_max_turn_to_row = rospy.get_param("y_max_turn_to_row")

        self.x_min_turn_to_row_critic = rospy.get_param("x_min_turn_to_row_critic")
        self.x_max_turn_to_row_critic = rospy.get_param("x_max_turn_to_row_critic")
        self.y_min_turn_to_row_critic = rospy.get_param("y_min_turn_to_row_critic")
        self.y_max_turn_to_row_critic = rospy.get_param("y_max_turn_to_row_critic")

        self.row_width = rospy.get_param("row_width")
        self.drive_out_dist = rospy.get_param("drive_out_dist")
        self.max_dist_in_row = rospy.get_param("max_dist_in_row")
        self.critic_row = rospy.get_param("critic_row")

        self.vel_linear_drive = rospy.get_param("vel_linear_drive")
        self.vel_linear_count = rospy.get_param("vel_linear_count")
        self.vel_linear_turn = rospy.get_param("vel_linear_turn")

        self.last_state = "drive_in_row"

        # Initialize state variables
        self.current_state = "drive_in_row"  #'manual_mode'
        self.pattern = rospy.get_param("pattern")
        self.automatic_mode = True
        self.obstacle_automatic_mode = True
        self.teleop_cmd_vel = Twist()
        self.teleop_cmd_vel.linear.x = 0
        self.teleop_cmd_vel.angular.z = 0
        self.driven_row = 0

    def set_drive_params(self):
        self.x_min = self.x_min_drive_in_row
        self.x_max = self.x_max_drive_in_row
        self.y_min = self.y_min_drive_in_row
        self.y_max = self.y_max_drive_in_row

    def set_exit_params(self):
        self.x_min = self.x_min_turn_and_exit
        self.x_max = self.x_max_turn_and_exit
        self.y_min = self.y_min_turn_and_exit
        self.y_max = self.y_max_turn_and_exit

    def set_counting_params(self):
        self.x_min = self.x_min_counting_rows
        self.x_max = self.x_max_counting_rows
        self.y_min = self.y_min_counting_rows
        self.y_max = self.y_max_counting_rows

    def set_turn_to_row_params(self, flag):
        if flag != True:
            self.x_min = self.x_min_turn_to_row
            self.x_max = self.x_max_turn_to_row
            self.y_min = self.y_min_turn_to_row
            self.y_max = self.y_max_turn_to_row
        else:
            self.x_min = self.x_min_turn_to_row_critic
            self.x_max = self.x_max_turn_to_row_critic
            self.y_min = self.y_min_turn_to_row_critic
            self.y_max = self.y_max_turn_to_row_critic

    def publish_poly_marker(self, coeffs, ns, x_min, x_max):
        marker = Marker()
        marker.header.frame_id = "laserFront"
        marker.header.stamp = rospy.Time.now()
        marker.ns = ns
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05  # Line width
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.lifetime = rospy.Duration()
        x_vals = np.linspace(x_min, x_max, 100)
        y_vals = np.polyval(coeffs, x_vals)
        marker.points = [Point32(x, y, 0) for x, y in zip(x_vals, y_vals)]
        self.marker_pub.publish(marker)

    def point_cloud_callback(self, msg):
        points = []
        drive_points = []
        both_sides = rospy.get_param("both_sides")
        min_distance = float("inf")  # initialize minimum distance with infinity

        for p in point_cloud2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True
        ):
            point = Point32(*p)
            distance = np.sqrt(
                point.x**2 + point.y**2
            )  # calculate Euclidean distance

            if distance < min_distance:
                min_distance = distance
            if self.current_state == "drive_in_row":
                if (
                    self.y_min_drive_in_row < np.abs(point.y) < self.y_max_drive_in_row
                    and self.x_min_drive_in_row < point.x < self.x_max_drive_in_row
                ):
                    drive_points.append(point)
                    points = drive_points
            if both_sides == "both" and self.current_state != "drive_in_row":
                if (
                    self.y_min < np.abs(point.y) < self.y_max
                    and self.x_min < point.x < self.x_max
                ):
                    points.append(point)
            elif both_sides == "L":
                if (
                    -self.y_max < point.y < -self.y_min
                    and self.x_min < point.x < self.x_max
                ):
                    points.append(point)
            elif both_sides == "R":
                if (
                    self.y_min < point.y < self.y_max
                    and self.x_min < point.x < self.x_max
                ):
                    points.append(point)
        self.min_dist = min_distance
        self.points = points
        self.drive_points = drive_points

        # Publish self.points
        header = msg.header
        header.frame_id = "laserFront"
        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
        ]
        points = [(p.x, p.y, p.z) for p in self.points]
        cloud = point_cloud2.create_cloud(header, fields, points)
        self.points_pub.publish(cloud)

    def angle_callback(self, msg):
        # msg is a Float32 message
        # msg.data contains the float value
        self.angle = msg.data

    def navigate(self):
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            if self.points is None:
                rate.sleep()
                continue
            else:
                if self.current_state == "drive_in_row":
                    self.drive_in_row()
                elif self.current_state == "turn_and_exit":
                    self.turn_and_exit()
                elif self.current_state == "counting_rows":
                    self.counting_rows()
                elif self.current_state == "turn_to_row":
                    self.turn_to_row()
                elif self.current_state == "manual_mode":
                    self.manual_mode()

                rate.sleep()

    def pattern_callback(self, msg):
        # Callback function to process the received message
        buttons = msg.buttons
        pattern = []

        # Iterate over the buttons list with a step size of 2
        for i in range(0, len(buttons), 2):
            button_count = buttons[i]
            button_value = buttons[i + 1]
            if button_value == 1:
                button_value = "L"
            else:
                button_value = "R"
            pattern.append([button_count, button_value])
            self.pattern = pattern

    def automatic_mode_callback(self, msg):
        # Callback function to process the received message
        self.automatic_mode = msg.data

    def obstacle_automatic_mode_callback(self, msg):
        self.obstacle_automatic_mode = msg.data

    def teleop_cmd_vel_callback(self, msg):
        # Callback function to process the received message
        self.teleop_cmd_vel.linear.x = msg.linear.x
        self.teleop_cmd_vel.angular.z = msg.angular.z

    def manual_mode(self):
        rospy.loginfo("Manual mode...")
        cmd_vel = Twist()
        cmd_vel = self.teleop_cmd_vel
        self.cmd_vel_pub.publish(cmd_vel)
        if self.automatic_mode == True and self.obstacle_automatic_mode == True:
            self.current_state = "drive_in_row"  # self.last_state

    def drive_in_row(self):
        rospy.loginfo("Driving in row...")
        # Calculate the average distance to the robot on both sides within x and y limits
        rospy.set_param("box", "drive")
        rospy.set_param("both_sides", "both")

        # Separating the points to the left and right
        left_points = [p for p in self.drive_points if p.y < 0]
        right_points = [p for p in self.drive_points if p.y >= 0]
        # Calculate cycle time
        current_time = rospy.Time.now()
        cycle_time = (current_time - self.last_cycle_time).to_sec()
        self.last_cycle_time = current_time
        # vel_linear_drive = self.vel_linear_drive# Update last_cycle_time for the next cycle

        # Calculate validation_x using the last published linear speed
        validation_x = 5 * cycle_time * self.last_linear_speed
        self.last_linear_speed = 0

        if len(left_points) < 3 and len(right_points) < 3:
            # Not enough data to calculate center
            rospy.loginfo("At least one side has no maize")
            rospy.loginfo("Reached the end of a row.")
            cmd_vel = Twist()
            cmd_vel.linear.x = self.vel_linear_drive
            time = self.drive_out_dist / self.vel_linear_drive
            start_time = rospy.Time.now().to_sec()
            while (rospy.Time.now().to_sec() - start_time) < time:
                self.cmd_vel_pub.publish(cmd_vel)
                rospy.loginfo("Leaving the row...")
                rospy.sleep(0.1)

            self.set_exit_params()
            rospy.set_param("box", "exit")
            rospy.set_param("both_sides", self.pattern[self.driven_row][1])
            self.driven_row += 1
            self.current_state = "turn_and_exit"
            pass
        else:
            # Preparing data for polynomial fitting
            left_x = np.array([p.x for p in left_points])
            left_y = np.array([p.y for p in left_points])
            right_x = np.array([p.x for p in right_points])
            right_y = np.array([p.y for p in right_points])

            # Calculate min and max values of both sides individually
            min_left_x = (
                np.min(left_x) if len(left_x) > self.needed_points_for_fit else None
            )
            max_left_x = (
                np.max(left_x) if len(left_x) > self.needed_points_for_fit else None
            )

            min_right_x = (
                np.min(right_x) if len(right_x) > self.needed_points_for_fit else None
            )
            max_right_x = (
                np.max(right_x) if len(right_x) > self.needed_points_for_fit else None
            )

            # Calculate min_both and max_both using min/max with list filtering
            min_both = min(filter(None, [min_left_x, min_right_x]), default=None)
            max_both = max(filter(None, [max_left_x, max_right_x]), default=None)

            # Conditional logic
            # Use AVG Control if both sides have no too few points, too short polynomials, or too far away polynomials
            if (
                min_both is None
                or self.polydist_to_robot < min_both
                or (max_both is not None and self.poly_min_dist_req > max_both)
            ):
                rospy.loginfo("AVG Control!")
                left_dist = (
                    np.mean(left_y) if len(left_y) >= 2 else np.inf
                )  # left is negative usually
                right_dist = np.mean(right_y) if len(right_y) >= 2 else np.inf
                # One line mode
                if np.isinf(left_dist):
                    left_dist = right_dist - self.row_width
                if np.isinf(right_dist):
                    right_dist = left_dist + self.row_width
            else:
                # Fitting a polynomial of degree 3 to the points, if there are enough points
                left_poly_coeffs = (
                    np.polyfit(left_x, left_y, 3)
                    if len(left_points) > self.needed_points_for_fit
                    else None
                )
                right_poly_coeffs = (
                    np.polyfit(right_x, right_y, 3)
                    if len(right_points) > self.needed_points_for_fit
                    else None
                )
                # Conditions to check if Polyquality is enough
                if (
                    left_poly_coeffs is not None
                    and self.polydist_to_robot > min_left_x
                    and self.poly_min_dist_req < max_left_x
                ):
                    # Calculating the distance to the left at x=0 using the polynomial equation
                    left_dist = np.polyval(left_poly_coeffs, validation_x)
                    self.publish_poly_marker(left_poly_coeffs, "left_poly", -0.3, 1.5)
                    # left_derivative_poly = np.polyder(left_poly_coeffs)
                    # left_derivative_value = np.polyval(left_derivative_poly, validation_x)
                else:
                    # One Line Mode
                    # If not enough points on the left, use a default or previously calculated value
                    left_dist = (
                        np.polyval(right_poly_coeffs, validation_x) - self.row_width
                        if right_poly_coeffs is not None
                        else np.inf
                    )
                # Conditions to check if Polyquality is enough
                if (
                    right_poly_coeffs is not None
                    and self.polydist_to_robot > min_right_x
                    and self.poly_min_dist_req < max_right_x
                ):
                    # Calculating the distance to the right at x=0 using the polynomial equation
                    right_dist = np.polyval(right_poly_coeffs, validation_x)
                    self.publish_poly_marker(right_poly_coeffs, "right_poly", -0.3, 1.5)
                    # right_derivative_poly = np.polyder(right_poly_coeffs)
                    # right_derivative_value = np.polyval(right_derivative_poly, validation_x)
                else:
                    # One Line Mode
                    # If not enough points on the right, use a default or previously calculated value
                    right_dist = (
                        self.row_width + np.polyval(left_poly_coeffs, validation_x)
                        if left_poly_coeffs is not None
                        else np.inf
                    )

                # Calculate the average derivative value
                """if left_poly_coeffs is not None and right_poly_coeffs is not None:
                    avg_derivative_value = (left_derivative_value + right_derivative_value) / 2.0
                    # Calculate the angle to the row using atan
                    angle_to_row = np.arctan(avg_derivative_value)
                    angle_msg = Float32()
                    angle_msg.data = np.degrees(angle_to_row)
                    # Publish the angle message
                    self.angle_pub.publish(angle_msg)"""

            center_dist = right_dist + left_dist
            # Call of PID-Control
            angular_correction = self.pid_controller.compute(center_dist, cycle_time)

            rospy.loginfo("Distance to center: %f", center_dist)
            self.center_dist.publish(center_dist)
            # Adjust the angular velocity to center the robot between the rows
            cmd_vel = Twist()
            # Negative, because laser is mounted overhead
            cmd_vel.angular.z = -angular_correction
            # Limit the speed according to distance to row
            if np.abs(center_dist) > self.max_dist_in_row - 0.05:
                cmd_vel.linear.x = 0.1
                if np.abs(center_dist) > self.max_dist_in_row:
                    cmd_vel.linear.x = 0
                    rospy.logwarn("Too close to row!!!")
            else:
                # Norming the speed according to distance to row
                cmd_vel.linear.x = (
                    self.vel_linear_drive
                    * (self.max_dist_in_row - np.abs(center_dist))
                    / self.max_dist_in_row
                )
            rospy.loginfo("Publishing to cmd_vel: %s", cmd_vel)
        self.last_linear_speed = cmd_vel.linear.x
        self.cmd_vel_pub.publish(cmd_vel)
        if self.automatic_mode == False or self.obstacle_automatic_mode == False:
            self.last_state = self.current_state
            self.current_state = "manual_mode"

    def turn_and_exit(self):
        rospy.loginfo("Turn and exit...")
        # Calculate the average distance to the robot on both sides within x and y limits
        points_x = [p.x for p in self.points]
        x_mean = np.mean(points_x) if len(points_x) > 0 else np.inf
        rospy.loginfo("xmean: %f", x_mean)
        if -0.08 < x_mean < 0.08:
            # if np.abs(self.angle) > 75:
            cmd_vel = Twist()
            rospy.loginfo("Aligned to the rows...")
            self.set_counting_params()
            if self.pattern[self.driven_row - 1][0] == 1:
                if self.driven_row in self.critic_row:
                    self.set_turn_to_row_params(True)  # True means critic params
                    rospy.set_param("box", "turn_crit")
                else:
                    self.set_turn_to_row_params(False)
                    rospy.set_param("box", "turn")

                rospy.set_param("both_sides", "both")
                self.current_state = "turn_to_row"
            else:
                rospy.set_param("box", "count")
                self.row_counter = 1
                self.previous_row = 1
                self.actual_row = 1
                self.actual_dist = self.min_dist
                self.current_state = "counting_rows"

        else:
            if 0 <= self.driven_row < len(self.pattern):
                if self.pattern[self.driven_row - 1][1] == "L":
                    # Calculate the actual distance to the center of both sides
                    rospy.loginfo("Turning left until parallel...")
                    # Adjust the angular velocity to center the robot between the rows
                    cmd_vel = Twist()
                    cmd_vel.linear.x = self.vel_linear_turn
                    radius = self.row_width / 2
                    cmd_vel.angular.z = self.vel_linear_turn / radius
                elif self.pattern[self.driven_row - 1][1] == "R":
                    rospy.loginfo("Turning right until parallel...")
                    cmd_vel = Twist()
                    cmd_vel.linear.x = self.vel_linear_turn
                    radius = -self.row_width / 2
                    cmd_vel.angular.z = self.vel_linear_turn / radius
                else:
                    rospy.logwarn("Invalid direction at driven_row index")
                    cmd_vel = Twist()
            else:
                rospy.loginfo("Pattern is now finished")
                cmd_vel = Twist()

            rospy.loginfo("Publishing to cmd_vel: %s", cmd_vel)

        self.cmd_vel_pub.publish(cmd_vel)
        if self.automatic_mode == False or self.obstacle_automatic_mode == False:
            self.last_state = self.current_state
            self.current_state = "manual_mode"

    def counting_rows(self):
        rospy.loginfo("counting rows...")
        # Calculate the average distance to the robot on both sides within x and y limits
        if self.pattern[self.driven_row - 1][0] == self.row_counter:
            rospy.loginfo("start turning to row...")
            if self.driven_row in self.critic_row:
                self.set_turn_to_row_params(True)
                rospy.set_param("box", "turn_crit")
            else:
                self.set_turn_to_row_params(False)
                rospy.set_param("box", "turn")
            rospy.set_param("both_sides", "both")

            self.current_state = "turn_to_row"
        else:
            if self.pattern[self.driven_row - 1][1] == "L":
                gain = 2.5
            elif self.pattern[self.driven_row - 1][1] == "R":
                gain = -2.5

            # Calculate the actual distance to the center of both sides
            rospy.loginfo("Holding Distance, driving parallel")
            # Adjust the angular velocity to center the robot between the rows
            cmd_vel = Twist()
            cmd_vel.linear.x = self.vel_linear_count
            rospy.loginfo("No. of points in Box %i", len(self.points))
            if len(self.points) > 1:
                # Compute the shortest y distance
                cmd_vel.angular.z = gain * (self.min_dist - self.actual_dist)
                rospy.loginfo(
                    "Gap to desired distance:%f", self.min_dist - self.actual_dist
                )
                self.actual_row = 1
            else:
                cmd_vel.angular.z = 0
                self.actual_row = 0
            if self.actual_row > self.previous_row:
                self.row_counter += 1
                rospy.loginfo("Increment row_counter to %i", self.row_counter)

            rospy.loginfo(
                "Passing row %i of %i",
                self.row_counter,
                self.pattern[self.driven_row - 1][0],
            )
            self.previous_row = self.actual_row
            rospy.loginfo("Publishing to cmd_vel: %s", cmd_vel)
            self.cmd_vel_pub.publish(cmd_vel)
            if self.automatic_mode == False or self.obstacle_automatic_mode == False:
                self.last_state = self.current_state
                self.current_state = "manual_mode"

    def turn_to_row(self):
        rospy.loginfo("Turn to row...")
        # Calculate the average distance to the robot on both sides within x and y limits
        # points_y = [p.y for p in self.points]
        # y_mean = np.mean((points_y)) if len(points_y) > 0 else np.inf #left is negative usually
        # rospy.loginfo("ymean: %f", y_mean)
        if np.abs(self.angle) < 5:
            cmd_vel = Twist()
            rospy.loginfo("Start driving in row...")
            self.set_drive_params()
            self.current_state = "drive_in_row"
        else:
            if 0 <= self.driven_row < len(self.pattern):
                if self.pattern[self.driven_row - 1][1] == "L":
                    gain = 1
                elif self.pattern[self.driven_row - 1][1] == "R":
                    gain = -1
                else:
                    rospy.logwarn("Invalid direction at driven_row index")
                    cmd_vel = Twist()

                rospy.loginfo("Turning right until parallel...")
                cmd_vel = Twist()
                cmd_vel.linear.x = self.vel_linear_turn
                radius = gain * self.row_width / 2
                cmd_vel.angular.z = self.vel_linear_turn / radius
            else:
                rospy.logwarn("Driven_row index out of range")
                cmd_vel = Twist()

            rospy.loginfo("Publishing to cmd_vel: %s", cmd_vel)

        self.cmd_vel_pub.publish(cmd_vel)
        if self.automatic_mode == False or self.obstacle_automatic_mode == False:
            self.last_state = self.current_state
            self.current_state = "manual_mode"


if __name__ == "__main__":
    navigator = FieldRobotNavigator()
    navigator.navigate()
