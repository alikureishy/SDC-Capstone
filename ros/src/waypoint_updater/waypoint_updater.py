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

from waypoints_wrapper import WaypointsWrapper

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
UPDATER_FREQUENCY = 50 # Hertz

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
            self.base_waypoints_wrapper = None
            self.pose = None
            self.pose_xy = None

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
                if self.pose_xy and self.base_waypoints:
                    closest_waypoint_idx, _ = self.base_waypoints_wrapper.get_closest_waypoint_to(self.pose_xy, strictly_ahead=True)
                    self.publish_updated_waypoints(closest_waypoint_idx)
            rate.sleep()

    def publish_updated_waypoints(self, closest_waypoint_idx):
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
            x = msg.pose.position.x
            y = msg.pose.position.y
            self.pose_xy = [x,y]

    def waypoints_cb(self, msg):
        '''
        This will be called only once, because the /base_waypoints subscriber is a "latched"
        subscriber.
        :param waypoints:
        :return: None
        '''
        # TODO: Implement
        with self.lock:
            self.base_waypoints = msg
            self.base_waypoints = WaypointsWrapper(msg.waypoints)

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
