#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

import math
import numpy as np
from scipy.spatial import KDTree
from threading import Lock

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
UPDATER_FREQUENCY = 50 # Hertz
KDTREE_SEARCH_COUNT = 1
KDTREE_SEARCH_RESULT_IDX = 1 # The KDTree query returns (position, index) tuple. We only need the index value here..hence '1'.

class WaypointUpdater(object):
    def __init__(self):

        # This is so that we can make this node threadsafe, against race conditions
        # Always acquire the lock in the methods that follow
        self.lock = Lock()
        with self.lock:

            rospy.init_node('waypoint_updater')

            rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
            rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

            # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
            rospy.Subscriber('/traffic_waypoint', Lane, self.traffic_cb)
            rospy.Subscriber('/obstacle_waypoint', Lane, self.obstacle_cb)

            self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

            # TODO: Add other member variables you need below
            self.base_waypoints = None
            self.pose = None
            self.waypoints_2d = None
            self.waypoints_tree = None

            self.loop()

    def loop(self):
        '''
        This method loops at a defined rate, and each time it wakes up, it updates
        the subsequent waypoints topic with the next waypoints (since that keeps changing).
        This method serves as an automatic update of the waypoints topic, based on
        changes in the target velocity, pose etc of the car. The frequency should be high
        enough here that this topic is kept updated *at least* as frequently as the topics
        that this class subscribes to. For example, these updates are picked up by Autoware
        at 30Hz, so the waypoint publishing frequency here should not drop less than 30 Hz.
        :return: None
        '''
        rate = rospy.Rate(UPDATER_FREQUENCY)
        while not rospy.is_shutdown():
            # Get closest waypoint
            with self.lock:
                if self.pose and self.base_waypoints:
                    closest_waypoint_idx = self.get_closest_waypoint_idx()
                    self.publish_waypoints(closest_waypoint_idx)
            rate.sleep()

    def get_closest_waypoint_idx(self):
        '''
        Find the closest waypoint that is ahead of the vehicle
        :return: id of the closest waypoint in the base_waypoints list
        '''

        with self.lock:
            x = self.pose.pose.position.x
            y = self.pose.pose.position.y

            # Look for the closest in the KD Tree...
            closest_idx = self.waypoints_tree.query([x, y], KDTREE_SEARCH_COUNT)[KDTREE_SEARCH_RESULT_IDX]

            # Check if this waypoint is ahead or behind the vehicle, based on the motion vector of the vehicle
            # We use coordinates here that are relative to the absolute origin (just as all waypoints are).
            # We then utilize their coordinates as vectors, and perform the desired calculation
            pose_x_y_vector = np.array([x,y])
            closest_x_y_vector = np.array(self.waypoints_2d[closest_idx])
            prev_x_y_vector = np.array(self.waypoints_2d[closest_idx-1]) # This is guaranteed to be behind the vehicle. If it wasn't, it would have been returned as the closest waypoint.

            # With hyperplane perpendicular to the closest vector position, we define the following:
            previous_to_hyperplane = closest_x_y_vector - prev_x_y_vector # Vector pointing from prev --> hyperplane
            hyperplane_to_car = pose_x_y_vector - closest_x_y_vector # Vector pointing from hyperplane --> car

            # If the two vectors above are pointing in teh same direction, it means the car is ahead of the closest index
            # and the dot product will be positive (as below), in which case we pick the next waypoint (which is
            # guaranteed then to be AHEAD)
            product = np.dot(previous_to_hyperplane, hyperplane_to_car)
            if product > 0:
                closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
            else: # when product <= 0, the closest waypoint is ahead of the car (or at the same position)
                pass # So we can return the closest idx as it is
            return closest_idx

    def publish_waypoints(self, closest_waypoint_idx):
        with self.lock:
            lane = Lane()
            lane.header = self.base_waypoints.header

            # Only slice out the waypoints needed (If the final index spills over, python adjusts to the end index)
            lane.waypoints = self.base_waypoints.waypoints[closest_waypoint_idx: closest_waypoint_idx + LOOKAHEAD_WPS]
            self.final_waypoints_pub.publish(lane)

    def pose_cb(self, msg):
        # TODO: Implement
        with self.lock:
            self.pose = msg

    def waypoints_cb(self, waypoints):
        '''
        This will be called only once, because the /base_waypoints subscriber is a "latched"
        subscriber.
        :param waypoints:
        :return: None
        '''
        # TODO: Implement
        with self.lock:
            self.base_waypoints = waypoints
            if self.waypoints_2d is None:
                self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
                self.waypoints_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement

        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later

        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    @staticmethod
    def distance(waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
