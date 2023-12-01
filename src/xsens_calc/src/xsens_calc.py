#!/usr/bin/env python
import rospy
from geometry_msgs.msg import QuaternionStamped
from std_msgs.msg import Float64
import tf.transformations

def quaternion_to_euler(quaternion):
    """
    Converts a quaternion to Euler angles (roll, pitch, yaw).
    """
    return tf.transformations.euler_from_quaternion([
        quaternion.x,
        quaternion.y,
        quaternion.z,
        quaternion.w
    ])

def callback(data):
    """
    Callback function for the subscriber.
    This function is called whenever a new message is received on the 'filter/quaternion' topic.
    It converts the quaternion to Euler angles and publishes the z-axis orientation.
    """
    # Convert the quaternion to Euler angles
    euler = quaternion_to_euler(data.quaternion)

    # Extracting specific orientations:
    # euler[0] - Roll angle: Rotation around the x-axis
    # euler[1] - Pitch angle: Rotation around the y-axis
    # euler[2] - Yaw angle: Rotation around the z-axis
    z_orientation = euler[2]  # Here, we are specifically interested in the yaw angle

    # Publish the z-axis orientation (yaw angle)
    pub.publish(z_orientation)
    rospy.loginfo("Published z-axis orientation (yaw angle): %f", z_orientation)

if __name__ == '__main__':
    rospy.init_node('quaternion_to_euler_node', anonymous=True)

    # Subscriber to the 'filter/quaternion' topic
    rospy.Subscriber("filter/quaternion", QuaternionStamped, callback)

    # Publisher for the z-axis orientation (yaw angle)
    pub = rospy.Publisher("z_orientation", Float64, queue_size=10)

    rospy.spin()
