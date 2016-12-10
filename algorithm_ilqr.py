import roslib
import sys
import rospy
from std_msgs.msg import Float64
import numpy as np
import PyKDL
from sensor_msgs.msg import JointState

from model import Manipulator_X

class ROS_connection():
	def __init__(self):
		rospy.init_node('tutorial_x_control')
		self.r = rospy.Rate(1)
		self.pub = []
		self.pub.append(rospy.Publisher('/robotis_manipulator_x/joint1_position/command', Float64, queue_size=1))
		self.pub.append(rospy.Publisher('/robotis_manipulator_x/joint2_position/command', Float64, queue_size=1))
		self.pub.append(rospy.Publisher('/robotis_manipulator_x/joint3_position/command', Float64, queue_size=1))
		self.pub.append(rospy.Publisher('/robotis_manipulator_x/joint4_position/command', Float64, queue_size=1))
		self.pub.append(rospy.Publisher('/robotis_manipulator_x/joint5_position/command', Float64, queue_size=1))
		self.pub.append(rospy.Publisher('/robotis_manipulator_x/joint6_position/command', Float64, queue_size=1))
		self.pub.append(rospy.Publisher('/robotis_manipulator_x/joint7_position/command', Float64, queue_size=1))
		self.pub.append(rospy.Publisher('/robotis_manipulator_x/grip_joint_position/command', Float64, queue_size=1))
		self.pub.append(rospy.Publisher('/robotis_manipulator_x/grip_joint_sub_position/command', Float64, queue_size=1))
		self.sub_once = rospy.Subscriber('/robotis_manipulator_x/joint_states', JointState, self.callback)

		self.q_init = PyKDL.JntArray(7)
		
		return

	def callback(self, data):
		self.q_init[0] = data.position[2]
		self.q_init[1] = data.position[3]
		self.q_init[2] = data.position[4]
		self.q_init[3] = data.position[5]
		self.q_init[4] = data.position[6]
		self.q_init[5] = data.position[7]
		self.q_init[6] = data.position[8]	
		#self.sub_once.unregister()
		print "callback"		
		print(self.q_init)
		return

	def move_arm(self, T, trj):
		for i in range(T):
			for j in range(0,7):
				self.pub[j].publish(trj[i][j])			
			self.r.sleep()
			print("ros signal sent")
        	        print(trj[i])
		return

if __name__ == '__main__':
	model = Manipulator_X(T=20,weight=[1.0,5.0])
	model.build_ilqr_problem()
	model.solve_ilqr_problem()
	trajectory = model.res['x_array_opt']
	#model.plot_ilqr_result()
	
	print ('iLQR done')

	ros_agent = ROS_connection()
	ros_agent.move_arm(model.T, trajectory)
	
	print "final position"
	print model.fin_position
	print "result position"
	print model.getPosition(trajectory[-1])
	print model.res['J_hist'][-1]


