#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

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
MAX_DECEL = 0.5 # Deceleration dampener

class WaypointUpdater(object):
    def __init__(self):

        # This is so that we can make this node threadsafe, against race conditions
        # Always acquire the lock in the methods that follow
        self.lock = Lock()
        with self.lock:

            rospy.init_node('waypoint_updater')

            rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
            rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
            rospy.Subscriber('/traffic_waypoint', Lane, self.traffic_cb)

            # For future enhancements:
            rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)

            self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

            # TODO: Add other member variables you need below
            self.skeletal_waypoints = None
            self.skeletal_waypoints_wrapper = None
            self.pose = None
            self.pose_xy = None

            # Traffic-related:
            self.upcoming_stopline_wp_idx = -1

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
                if self.pose_xy and self.skeletal_waypoints:
                    closest_waypoint_idx, _ = self.skeletal_waypoints_wrapper.get_closest_waypoint_to(self.pose_xy, strictly_ahead=True)
                    self.publish_updated_waypoints(closest_waypoint_idx)
            rate.sleep()

    def publish_updated_waypoints(self, closest_waypoint_idx):
        with self.lock:
            final_lane = self.generate_lane()
            self.final_waypoints_pub.publish(final_lane)

    def generate_lane(self):
        with self.lock:
            lane = Lane()
            lane.header = self.skeletal_waypoints.header

            closest_wp_idx, _ = self.skeletal_waypoints_wrapper.get_closest_waypoint_to(self.pose_xy)
            farthest_wp_idx = closest_wp_idx + LOOKAHEAD_WPS

            # Only slice out the waypoints needed (If the final index spills over, python adjusts to the end index)
            skeletal_lane_waypoints = self.skeletal_waypoints[closest_wp_idx:farthest_wp_idx]

            if self.upcoming_stopline_wp_idx == -1 or self.upcoming_stopline_wp_idx >= farthest_wp_idx:
                # ~There's no traffic light coming up for the next LOOKAHEAD_WPS waypoints, so continue as-is
                lane.waypoints = skeletal_lane_waypoints
            else:
                # ~There's a traffic light coming up soon. We need to start decelerating appropriately
                lane.waypoints = WaypointUpdater.decelerate_waypoints(closest_wp_idx, skeletal_lane_waypoints, self.upcoming_stopline_wp_idx)

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
            self.skeletal_waypoints = msg
            self.skeletal_waypoints_wrapper = WaypointsWrapper(msg.waypoints)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        with self.lock:
            self.upcoming_stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    @staticmethod
    def decelerate_waypoints(closest_wp_idx, remaining_skeletal_waypoints, upcoming_stopline_wp_idx):
        """
        This function generates a list of waypoints from the skeletal waypoints that are ahead of us (i.e, the
        subsequent_waypoints parameter). Each waypoint from that list gets copied over to the new list, after
        setting/adjusting to the requisite velocity at that waypoint (to achieve the required deceleration)
        :param closest_wp_idx:
        :param remaining_skeletal_waypoints:
        :return:
        """
        decelerated_waypoints = []
        # Stop 2 waypoints before the stopline so that the front of the car stops at the stopline:
        required_stop_idx = max((upcoming_stopline_wp_idx - closest_wp_idx - 2), 0)

        for i, skeletal_waypoint in enumerate(remaining_skeletal_waypoints):
            updated_waypoint = Waypoint()
            updated_waypoint.pose = skeletal_waypoint.pose    # Deceleration isn't going to change the pose (which includes the orientation)

            # Check the physical distance between i-th waypoint and stop_idx
            dist = WaypointUpdater.distance(remaining_skeletal_waypoints, i, required_stop_idx)

            # This exponentially declines the vel as we get nearer to the stop_idx
            requisite_velocity = math.sqrt(2 * MAX_DECEL * dist)    # <---- We can use any appropriate function here

            # Since we might never really get to a distance of 0, the requisite_velocity will always likely be some
            # non zero real number. However, we do need to come to a commplete stop at the end, so this check here
            # ensures that the car doesn't keep inching forward when it has almost already reached the stopline
            if requisite_velocity < 1.:
                requisite_velocity = 0.

            # We assume that all base waypoints will be set at the speed limit (and we adjust that downwards, only as needed)
            # Here, if the velocity calculated above ends up being larger than the speed limit (i.e, the previous linear
            # velocity -- wp.twist.twist.linear.x -- we just retain our velocity at that that previous linear velocity
            # As the calculated vel above declines (as the i-th waypoint gets closer to the stop-idx, the velocity will
            # drop below the speed limit, which is what will make the car start slowing down. In other words, we assume
            # that the car will always travel at the speed limit, unless there's a reason to reduce the speed (as with
            # an approaching traffic light, in this case). [Another example would be when approaching an obstacle]
            updated_waypoint.twist.twist.linear.x = min(requisite_velocity, skeletal_waypoint.twist.twist.linear.x)
            decelerated_waypoints.append(updated_waypoint)

        return decelerated_waypoints

    @staticmethod
    def get_waypoint_velocity(waypoint):
        return waypoint.twist.twist.linear.x

    @staticmethod
    def set_waypoint_velocity(waypoints, waypoint, velocity):
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
