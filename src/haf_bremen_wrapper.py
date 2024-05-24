#! /usr/bin/env python3
import os
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
from geometry_msgs.msg import PoseStamped, Pose
from robokudo_msgs.msg import GenericImgProcAnnotatorAction, GenericImgProcAnnotatorResult
from haf_grasping.msg import CalcGraspPointsServerAction, CalcGraspPointsServerActionGoal
from visualization_msgs.msg import MarkerArray, Marker
from sensor_msgs.msg import CameraInfo


class HAF_Wrapper():

    def __init__(self):
        self.server = actionlib.SimpleActionServer('/pose_estimator/find_grasppose_haf', GenericImgProcAnnotatorAction, self.find_grasppose,
                                                   auto_start=False)

        rospy.loginfo('Connecting to HAF server')
        self.haf_client = actionlib.SimpleActionClient(
            '/calc_grasppoints_svm_action_server', CalcGraspPointsServerAction)
        while True:
            res_haf = self.haf_client.wait_for_server(rospy.Duration(10.0))
            if res_haf is False:
                rospy.logerr("Timeout when trying to connect to actionserver calc_grasppoints_svm_action_server. Trying again ...")
            else:
                break

        self.marker_pub = rospy.Publisher('/pose_estimator/haf_grasp_markers', MarkerArray, queue_size=10, latch=True)
        self.Transformer = tf.TransformListener(True, rospy.Duration(10))

        self.server.start()
        rospy.loginfo('Server started')
        self.depth_scale = None
        self.cam_info = None
    

    def find_grasppose(self, req):
        self.get_cam_info()
        frame_id = req.depth.header.frame_id
        depth_img_np = ros_numpy.numpify(req.depth)
        ros_pcd, _ = self.convert_depth_img_to_pcd(depth_img_np, frame_id)
        rospy.loginfo('Converted depth image to pcd')

        if len(req.bb_detections) > 0:
            center_poses = self.get_center_poses_from_bbs_2D(req.bb_detections, depth_img_np)
        elif len(req.mask_detections) > 0:
            #TODO how to handle? Any value != 0 as 'True'?
            raise NotImplementedError
        else:
            rospy.logerr("No bounding boxes or masks where passed! Aborting ...")
            self.server.set_aborted("No bb or masks passed to HAF!")
        rospy.loginfo('Calling HAF')
        pose_list, _, scores = self.get_grasp_poses(center_poses, frame_id, ros_pcd)
        rospy.loginfo(f'Finished calling HAF. Detected grasp_poses for {len(pose_list)} object!')

        self.add_markers(pose_list, frame_id)
        rospy.loginfo('Published MarkerArray')

        result = GenericImgProcAnnotatorResult()
        result.success = True
        result.pose_results = pose_list
        result.class_names = ['Unknown Object'] * len(pose_list)
        result.class_confidences = scores
        self.server.set_succeeded(result)
        rospy.loginfo('Done')


    def get_cam_info(self):
        cam_info_topic = rospy.get_param('/haf_wrapper/camera_info_topic')
        self.depth_scale = rospy.get_param('/haf_wrapper/depth_scale')
        self.cam_info = rospy.wait_for_message(cam_info_topic, CameraInfo, 10.0)


    def get_center_poses_from_bbs_2D(self, bbs_2D, depth_img):
        center_poses = []
        o3d_pcd_scene = self.convert_depth_img_to_o3d_pcd(depth_img)
        for bb_2D in bbs_2D:
            depth_img_obj = np.full_like(depth_img, np.nan)
            y, x = bb_2D.y_offset, bb_2D.x_offset
            h, w = bb_2D.height, bb_2D.width
            depth_img_obj[y:y+h, x:x+w] = depth_img[y:y+h, x:x+w]
            o3d_pcd = self.convert_depth_img_to_o3d_pcd(depth_img_obj)
            center = o3d_pcd.get_center()
            center_pose = Pose()
            center_pose.position.x = center[0]
            center_pose.position.y = center[1]
            center_pose.position.z = center[2]
            center_poses.append(center_pose)
        return center_poses

    def get_grasp_poses(self, center_poses, frame_id, ros_pcd):
        pose_list = []
        unsuccesful_calls_idx = []
        eval_scores = []
        base_frame = rospy.get_param('/haf_wrapper/base_frame')
        for i, pose in enumerate(center_poses):
            # Haf preprocessing needs poses in a frame with the z-axis pointing upwards (relative to floor)
            self.Transformer.waitForTransform(base_frame, frame_id, rospy.Time(), rospy.Duration(4.0))
            pose_stamped = PoseStamped()
            pose_stamped.pose = pose
            pose_stamped.header.stamp = rospy.Time()
            pose_stamped.header.frame_id = frame_id
            pose_stamped_tr = self.Transformer.transformPose(base_frame, pose_stamped)

            haf_result = self.call_haf(ros_pcd, pose_stamped_tr)
            if haf_result.graspOutput.eval <= 0:
                rospy.logerr(
                    'HAF grasping did not deliver successful result. Eval below 0\n' +
                    'Eval: ' + str(haf_result.graspOutput.eval) + '\n' + 
                    'Skipping grasp point estimation for object with index i=' + str(i))
                unsuccesful_calls_idx.append(i)
            else:
                # return pose_stamped from haf in succesful case
                pose_stamped = self.convert_haf_result_to_moveit_convention(haf_result, frame_id, base_frame)
                pose_list.append(pose_stamped.pose)
                eval_scores.append(haf_result.graspOutput.eval)
        
        return pose_list, unsuccesful_calls_idx, eval_scores

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
        

    def convert_depth_img_to_o3d_pcd(self, depth_img):
        width = self.cam_info.width
        height = self.cam_info.height
        intrinsics = np.array(self.cam_info.K).reshape(3, 3)
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        cam_intr = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

        depth_img_o3d = o3d.geometry.Image(depth_img.astype(np.uint16))
        #o3d.io.write_image('/root/HSR/catkin_ws/src/haf_grasping/'+str(rospy.get_rostime())+'.png', depth_img_o3d)

        o3d_pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_img_o3d, cam_intr, depth_scale=self.depth_scale)
        return o3d_pcd

    def convert_depth_img_to_pcd(self, depth_img, frame_id):
        o3d_pcd = self.convert_depth_img_to_o3d_pcd(depth_img)
        ros_pcd = orh.o3dpc_to_rospc(o3d_pcd, frame_id = frame_id, stamp=rospy.Time())
        return ros_pcd, o3d_pcd
    

    def convert_haf_result_to_moveit_convention(self, grasp_result_haf, target_frame_id, base_frame_id):
        '''
        Transforms pose and approachVector from haf into single pose following moveit-convention for orientation.
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
        marker.color.r = 0.0
        marker.color.g = 0
        marker.color.b = 1.0
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
    
