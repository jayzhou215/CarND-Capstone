#!/usr/bin/env python

import math

import numpy as np

import rospy
from geometry_msgs.msg import PoseStamped
from scipy.spatial import KDTree
from std_msgs.msg import Int32
from styx_msgs.msg import Lane, Waypoint

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

LOOKAHEAD_WPS = 200  # Number of waypoints we will publish. You can change this number

MAX_DECEL = 1.


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.stop_wp_idx = -1

        self.loop()

    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.pose and self.waypoint_tree:
                self.publish_waypoints()
            rate.sleep()

    def closest_waypoint(self, position, waypoints):
        closestLen = float("inf")
        closestWaypoint = 0
        dist = 0.0
        for idx in range(0, len(waypoints)):
            x = position.x
            y = position.y
            map_x = waypoints[idx].pose.pose.position.x
            map_y = waypoints[idx].pose.pose.position.y
            dist = self.distance_any(x, y, map_x, map_y)
            if (dist < closestLen):
                closestLen = dist
                closestWaypoint = idx
        return closestWaypoint

    def distance_any(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx = self.waypoint_tree.query([x, y], 10)[1]
        # closest_idx = self.closest_waypoint(self.pose.pose.position, self.base_waypoints.waypoints)
        # check if closest is ahead or behind vehicle
        closest_coord = self.waypoints_2d[closest_idx]
        pred_coord = self.waypoints_2d[closest_idx - 1]
        cl_vect = np.array(closest_coord)
        pre_vect = np.array(pred_coord)
        pos_vect = np.array([x, y])
        val = np.dot(cl_vect - pre_vect, pos_vect - cl_vect)
        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        # rospy.loginfo('pose=[%f, %f] closet=[%f, %f], pre=[%f, %f] closest_idx=%d %s', x,
        #               y, cl_vect[0], cl_vect[1], pre_vect[0], pre_vect[1], closest_idx,
        #               self.waypoints_2d[closest_idx:closest_idx + 5])
        return closest_idx

    def publish_waypoints(self):
        lane = self.generate_lane()
        self.final_waypoints_pub.publish(lane)

    def generate_lane(self):
        lane = Lane()
        lane.header = self.base_waypoints.header

        closest_idx = self.get_closest_waypoint_idx()
        fatherest_idx = closest_idx + LOOKAHEAD_WPS
        waypoints = self.base_waypoints.waypoints[closest_idx: fatherest_idx]
        if self.stop_wp_idx == -1 or (self.stop_wp_idx >= fatherest_idx):
            lane.waypoints = waypoints
            rospy.loginfo('normal closest_idx=%d', closest_idx)
        else:
            lane.waypoints = self.decelerate_waypoints(waypoints, closest_idx)
            rospy.loginfo('use decelerate closest_idx=%d', closest_idx)
        return lane

    def decelerate_waypoints(self, waypoints, closest_idx):
        temp = []
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            stop_idx = max(self.stop_wp_idx - closest_idx - 2, 0)
            dist = self.distance(waypoints, i, stop_idx)
            vel = math.sqrt(2 * dist * MAX_DECEL)
            if vel < 1.:
                vel = 0.
            p.pose = wp.pose
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)
        return temp

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y]
                                 for waypoint in self.base_waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.stop_wp_idx = msg.data
        rospy.loginfo('traffic_cb idx:%d ', self.stop_wp_idx)

    def obstacle_cb(self, msg):
        self.stop_wp_idx = msg.data

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        rospy.loginfo('set_waypoint_velocity idx:%d vel:%f', waypoint, velocity)
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0

        def dl(a, b): return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)
        for i in range(wp1, wp2 + 1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
