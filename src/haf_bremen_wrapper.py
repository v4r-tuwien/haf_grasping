#! /usr/bin/env python3
import sys
import open3d as o3d
from math import pi
import rospy
import actionlib
import numpy as np
import ros_numpy
import tf
from open3d_ros_helper import open3d_ros_helper as orh
from tf.transformations import (quaternion_about_axis, quaternion_from_matrix,
                                quaternion_multiply, unit_vector)
from geometry_msgs.msg import PoseStamped
from robokudo_msgs.msg import GenericImgProcAnnotatorAction
from haf_grasping.msg import CalcGraspPointsServerAction, CalcGraspPointsServerActionGoal
from visualization_msgs.msg import MarkerArray, Marker
from sensor_msgs.msg import CameraInfo


class HAF_Wrapper():

    def __init__(self):
        self.server = actionlib.SimpleActionServer('/pose_estimator/find_grasppose_haf', GenericImgProcAnnotatorAction, self.find_grasppose,
                                                   auto_start=False)

        rospy.loginfo('Connecting to Detector')
        self.detector = actionlib.SimpleActionClient('/pose_estimator/find_grasppose', GenericImgProcAnnotatorAction)
        res = self.detector.wait_for_server(rospy.Duration(10.0))
        if res is False:
            rospy.logerr('Timeout when trying to connect to actionserver find_grasppose')
            sys.exit(-1)

        rospy.loginfo('Connecting to HAF server')
        self.haf_client = actionlib.SimpleActionClient(
            '/calc_grasppoints_svm_action_server', CalcGraspPointsServerAction)
        res_haf = self.haf_client.wait_for_server(rospy.Duration(10.0))
        if res_haf is False:
            rospy.logerr('Timeout when trying to connect to actionserver calc_grasppoints_svm_action_server ')
            sys.exit(-1)

        self.marker_pub = rospy.Publisher('/pose_estimator/haf_grasp_markers', MarkerArray, queue_size=10, latch=True)
        self.Transformer = tf.TransformListener(True, rospy.Duration(10))

        self.server.start()
        rospy.loginfo('Server started')
    

    def find_grasppose(self, req):
        self.detector.send_goal(req)
        self.detector.wait_for_result()
        detector_result = self.detector.get_result()
        if len(detector_result.pose_results) == 0:
            rospy.logerr('No poses found by detector')
            self.server.set_aborted()
        rospy.loginfo('Got estimations from initial Detector')

        frame_id = req.depth.header.frame_id

        pc = self.convert_depth_img_to_pcd(req.depth)
        rospy.loginfo('Converted depth image to pcd')

        pose_list = []
        rospy.loginfo('Calling HAF')
        base_frame = rospy.get_param('/haf_wrapper/base_frame')
        for pose in detector_result.pose_results:
            # Haf preprocessing needs poses in a frame with the z-axis pointing upwards (relative to floor)
            self.Transformer.waitForTransform(base_frame, frame_id, rospy.Time(), rospy.Duration(4.0))
            pose_stamped = PoseStamped()
            pose_stamped.pose = pose
            pose_stamped.header.stamp = rospy.Time()
            pose_stamped.header.frame_id = frame_id
            pose_stamped_tr = self.Transformer.transformPose(base_frame, pose_stamped)

            haf_result = self.call_haf(pc, pose_stamped_tr)
            if haf_result.graspOutput.eval <= 0:
                rospy.logerr(
                    'HAF grasping did not deliver successful result. Eval below 0\n' +
                    'Eval: ' + str(haf_result.graspOutput.eval))

                # return pose_stamped from original detector in failure case
                pose_stamped = pose_stamped
            else:
                # return pose_stamped from haf in succesful case
                pose_stamped = self.convert_haf_result_to_moveit_convention(haf_result, frame_id, base_frame)
            pose_list.append(pose_stamped.pose)

        rospy.loginfo('Finished calling HAF')
        self.add_markers(pose_list, frame_id)
        rospy.loginfo('Published MarkerArray')
        detector_result.pose_results = pose_list

        self.server.set_succeeded(detector_result)
        rospy.loginfo('Done')



    def call_haf(self, pc, search_center, search_center_z_offset=0.1, grasp_area_length_x=30, grasp_area_length_y=30):
        # approach vector for top grasps
        approach_vector_x = 0.0
        approach_vector_y = 0.0
        approach_vector_z = 1.0

        grasp_goal = CalcGraspPointsServerActionGoal()
        grasp_goal.goal.graspinput.goal_frame_id = search_center.header.frame_id
        grasp_goal.goal.graspinput.grasp_area_center.x = search_center.pose.position.x
        grasp_goal.goal.graspinput.grasp_area_center.y = search_center.pose.position.y
        grasp_goal.goal.graspinput.grasp_area_center.z = search_center.pose.position.z + \
            search_center_z_offset
        grasp_goal.goal.graspinput.grasp_area_length_x = grasp_area_length_x
        grasp_goal.goal.graspinput.grasp_area_length_y = grasp_area_length_y

        grasp_goal.goal.graspinput.approach_vector.x = approach_vector_x
        grasp_goal.goal.graspinput.approach_vector.y = approach_vector_y
        grasp_goal.goal.graspinput.approach_vector.z = approach_vector_z

        grasp_goal.goal.graspinput.input_pc = pc
        grasp_goal.goal.graspinput.max_calculation_time = rospy.Duration(25)
        grasp_goal.goal.graspinput.gripper_opening_width = 1
        self.haf_client.wait_for_server()
        self.haf_client.send_goal(grasp_goal.goal)
        self.haf_client.wait_for_result()

        grasp_result = self.haf_client.get_result()
        return grasp_result
        

    def convert_depth_img_to_pcd(self, depth_img):
        cam_info_topic = rospy.get_param('/haf_wrapper/camera_info_topic')
        depth_scale = rospy.get_param('/haf_wrapper/depth_scale')
        cam_info = rospy.wait_for_message(cam_info_topic, CameraInfo, 10.0)
        width = cam_info.width
        height = cam_info.height
        intrinsics = np.array(cam_info.K).reshape(3, 3)
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        cam_intr = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

        depth_img_np = ros_numpy.numpify(depth_img)
        depth_img_o3d = o3d.geometry.Image(depth_img_np.astype(np.uint16))

        o3d_pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_img_o3d, cam_intr, depth_scale=depth_scale)
        ros_pcd = orh.o3dpc_to_rospc(o3d_pcd, frame_id = depth_img.header.frame_id, stamp=rospy.Time())

        return ros_pcd
    
    def convert_haf_result_to_moveit_convention(self, grasp_result_haf, target_frame_id, base_frame_id):
        '''
        Transforms pose and approachVector from haf into single pose.
        '''
        av = unit_vector([-grasp_result_haf.graspOutput.approachVector.x,
                          -grasp_result_haf.graspOutput.approachVector.y,
                          -grasp_result_haf.graspOutput.approachVector.z])

        gp1 = np.array([grasp_result_haf.graspOutput.graspPoint1.x,
                        grasp_result_haf.graspOutput.graspPoint1.y,
                        grasp_result_haf.graspOutput.graspPoint1.z])

        gp2 = np.array([grasp_result_haf.graspOutput.graspPoint2.x,
                        grasp_result_haf.graspOutput.graspPoint2.y,
                        grasp_result_haf.graspOutput.graspPoint2.z])

        gc = unit_vector(gp2 - gp1)
        c = np.cross(gc, av)
        rot_mat = np.array([[c[0], gc[0], av[0], 0], [c[1], gc[1], av[1], 0], [
                           c[2], gc[2], av[2], 0], [0, 0, 0, 1]])
        q = quaternion_from_matrix(rot_mat)

        self.Transformer.waitForTransform(
            target_frame_id, base_frame_id, rospy.Time(), rospy.Duration(4.0))

        grasp_pose_bl = PoseStamped()

        grasp_pose_bl.pose.orientation.x = q[0]
        grasp_pose_bl.pose.orientation.y = q[1]
        grasp_pose_bl.pose.orientation.z = q[2]
        grasp_pose_bl.pose.orientation.w = q[3]
        grasp_pose_bl.pose.position.x = grasp_result_haf.graspOutput.averagedGraspPoint.x - \
            rospy.get_param("/haf_wrapper/grasppoint_offset", default=0.04) * av[0]
        grasp_pose_bl.pose.position.y = grasp_result_haf.graspOutput.averagedGraspPoint.y - \
            rospy.get_param("/haf_wrapper/grasppoint_offset", default=0.04) * av[1]
        grasp_pose_bl.pose.position.z = grasp_result_haf.graspOutput.averagedGraspPoint.z - \
            rospy.get_param("/haf_wrapper/grasppoint_offset", default=0.04) * av[2]
        grasp_pose_bl.header.frame_id = base_frame_id

        grasp_pose = self.Transformer.transformPose(target_frame_id, grasp_pose_bl)
        return grasp_pose


    def create_marker(self, id, pose_goal, frame_id):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time()
        marker.ns = 'grasp_marker'
        marker.id = id
        marker.type = 0
        marker.action = 0

        q2 = [pose_goal.orientation.w, pose_goal.orientation.x,
              pose_goal.orientation.y, pose_goal.orientation.z]
        q = quaternion_about_axis(pi / 2, (0, 1, 0))
        q = quaternion_multiply(q, q2)

        marker.pose.orientation.w = q[0]
        marker.pose.orientation.x = q[1]
        marker.pose.orientation.y = q[2]
        marker.pose.orientation.z = q[3]
        marker.pose.position.x = pose_goal.position.x
        marker.pose.position.y = pose_goal.position.y
        marker.pose.position.z = pose_goal.position.z

        marker.scale.x = 0.1
        marker.scale.y = 0.05
        marker.scale.z = 0.01

        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0
        marker.color.b = 0
        return marker

    def add_markers(self, pose_goals, frame_id):
        marker_arr = MarkerArray()
        delete_marker = Marker()
        delete_marker.ns = 'grasp_marker'
        delete_marker.action = delete_marker.DELETEALL
        marker_arr.markers.append(delete_marker)
        for id, pose_goal in enumerate(pose_goals):
            marker = self.create_marker(id, pose_goal, frame_id)
            marker_arr.markers.append(marker)
        self.marker_pub.publish(marker_arr)

if __name__ == '__main__':
    rospy.init_node('Haf_grasping_bremen_wrapper')
    server = HAF_Wrapper()
    rospy.spin()
    
