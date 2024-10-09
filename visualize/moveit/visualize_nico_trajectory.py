#  Copyright (c) 2024. Jan-Gerrit Habekost. GNU General Public License. https://www.gnu.org/licenses/gpl-3.0.html.
import numpy as np
import rospy
import std_msgs.msg
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Quaternion, Pose, Point, Vector3
from std_msgs.msg import Header, ColorRGBA
from cycleik_pytorch import CycleIK


rospy.init_node('nico_vis_node', anonymous=True)
marker_publisher = rospy.Publisher("/visualization_marker", Marker, queue_size=100000)
marker_publisher_id_counter = 0

cycleik = CycleIK(robot='nico', chain='right_arm', verbose=True, cuda_device=1)


#(0.157, 0.0, 1.57, 1.57, 1.39, 0.0)
#(4, -10, 90, 93, 80, 0.0)
def create_marker(target_point):
    return Marker(
        type=Marker.SPHERE,
        action=Marker.ADD,
        id=marker_publisher_id_counter,
        lifetime=rospy.Duration(),
        pose=Pose(Point(target_point[0], target_point[1], target_point[2]), Quaternion(0, 0, 0, 1)),
        scale=Vector3(0.033, 0.033, 0.033),
        header=Header(frame_id='world'),
        color=ColorRGBA(0.0, 1.0, 0.0, 0.8),
        # lifetime=0,
        frame_locked=False)

def main():
    start_pose = np.array([0.2, -0.3, 0.82, -0.70619, 0.0888, -0.049169, 0.70071])
    # target_pose = np.array([Bx, -np.abs(By), 0.65, 0.93529, -0.11321, 0.18413, -0.2802])
    target_pose = np.array([0.35, 0., 0.66, -0.8957, 0.1506, -0.2052, 0.366])
    ctrl_point = np.array([np.array([0.3, -0.3, 0.75, -0.8957, 0.1506, -0.2052, 0.366]),
                           np.array([0.35, 0., 0.68, -0.8957, 0.1506, -0.2052, 0.366])])
    cartesian_path = cycleik.generate_cubic_bezier_trajectory(start_pose=start_pose, target_pose=target_pose,
                                                              control_points=ctrl_point, points=25)

    for target in cartesian_path[:,:3]:
        marker_publisher.publish(create_marker(target))
        global marker_publisher_id_counter
        marker_publisher_id_counter += 1
        rospy.sleep(0.01)

    #js_solution, fk_error, _, _ = cycleik.inverse_kinematics(cartesian_path, calculate_error=True)
    #
    #print(js_solution)
    #print(fk_error)
    #
    #for jc in js_solution:
    #    if arm_control.is_right:
    #        arm_control.set_pose(arm_target=tuple(jc), stay_time=0.1, wait_finished=True, move_time=0.5,
    #                             auto_resolve=False)
    #    else:
    #        jc = np.multiply(jc, np.array([-1, -1, -1, -1, -1, -1]))
    #        arm_control.set_pose(arm_target=tuple(jc), stay_time=0.1, wait_finished=True, move_time=0.5,
    #                             auto_resolve=False)
    ## arm_control.set_pose(arm_target=tuple((0.157, -1.57, 1.57, 1.57, -1.39, 0.0)), stay_time=0, wait_finished=True, move_time=0.5)
    #time.sleep(5.0)
    #print(f"detected object: {detected_object}")
    #
    ## backward_ctrl_points= [np.arr]
    #action_future = arm_control.set_pose(hand_target=tuple((-1.3089969389957472, -1.3089969389957472)), stay_time=5,
    #                                     wait_finished=True, move_time=0.5)
    #cartesian_path = cycleik.generate_cubic_bezier_trajectory(start_pose=target_pose, target_pose=start_pose,
    #                                                          control_points=ctrl_point, points=20)
    #js_solution, fk_error, _, _ = cycleik.inverse_kinematics(cartesian_path, calculate_error=True)
    #for jc in js_solution:
    #    if arm_control.is_right:
    #        arm_control.set_pose(arm_target=tuple(jc), stay_time=0.1, wait_finished=True, move_time=0.5,
    #                             auto_resolve=False)
    #    else:
    #        jc = np.multiply(jc, np.array([-1, -1, -1, -1, -1, -1]))
    #        arm_control.set_pose(arm_target=tuple(jc), stay_time=0.1, wait_finished=True, move_time=0.5,
    #                             auto_resolve=False)


if __name__ == '__main__':
    main()