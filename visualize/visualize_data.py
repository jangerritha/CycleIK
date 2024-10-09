import matplotlib.pyplot as plt
import pickle
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Quaternion, Pose, Point, Vector3
from std_msgs.msg import Header, ColorRGBA
import rospy
from tqdm import tqdm

rospy.init_node('data_visualizer', anonymous=True)

marker_publisher = rospy.Publisher("/moveit/visualization_marker", Marker, queue_size=100000)
rospy.sleep(1)


def main():
    data = pickle.load(open("./data/results_nicol_1000_3.p", 'rb'))

    for i in tqdm(range(len(data[0]))):
        pose = data[0][i]
        #if pose[2] > 0.85: continue
        #print(pose)
        marker = Marker(
            type=Marker.SPHERE,
            action=Marker.ADD,
            id=i,
            lifetime=rospy.Duration(),
            pose=Pose(Point(pose[0], pose[1], pose[2]), Quaternion(0, 0, 0, 1)),
            scale=Vector3(0.001, 0.001, 0.001),
            header=Header(frame_id='world'),
            color=ColorRGBA(0.0, 1.0, 0.0, 0.8),
            # lifetime=0,
            frame_locked=False)
        marker_publisher.publish(marker)
        rospy.sleep(0.001)


if __name__ == '__main__':
    main()