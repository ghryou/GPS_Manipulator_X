import roslib
import sys
import rospy
from std_msgs.msg import Float64
import numpy as np
import PyKDL
from sensor_msgs.msg import JointState
from model import Manipulator_X
import tensorflow as tf

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

	model = Manipulator_X(T=20, weight=[1.0,5.0])
	model.build_ilqr_problem()

	N_sample = 10  #number of total sample
	N_sample_init = 1 #number of sample from each ddp solution
	N_ddp_sol = 2  #number of ddp solution
	flag_end_sample = 0 #last index of sample

	sample_policy = [None]*N_ddp_sol
	sample_trajectory = [None]*N_sample
	sample_control = [None]*N_sample

	#Generate DDP solutions and build sample set
	for i in range(N_ddp_sol):
		print "iLQR number ",i
		model.solve_ilqr_problem()
		res_temp = model.res
		res_temp['type'] = 0
		sample_policy[i] = res_temp
		model.generate_dest()
		for j in range(N_sample_init):
			trajectory_temp = []
			control_temp = []
			trj = model.q_init
			x_i = sample_policy[i]['x_array_opt']
			u_i = sample_policy[i]['u_array_opt'] 
			K_i = sample_policy[i]['K_array_opt']
			Cov = sample_policy[i]['Q_array_opt']
			for t in range(model.T):
				next_policy = np.random.multivariate_normal(u_i[t]+K_i[t].dot(trj-x_i[t]),Cov[t])
				trajectory_temp.append(trj)
				control_temp.append(next_policy)
				trj += next_policy #dynamics
			sample_trajectory[i*N_sample_init+j] = trajectory_temp
			sample_control[i*N_sample_init+j] = control_temp

	#Initialize theta
	N_hidden_node = 20
	x_temp = tf.placeholder("float",[None, model.nj])
	W1 = tf.Variable(tf.zeros([model.nj, N_hidden_node]), name="W1")
	W2 = tf.Variable(tf.zeros([N_hidden_node, model.nj]), name="W2")
	h_node = tf.nn.softmax(tf.matmul(x_temp,W1))
	y_node = tf.matmul(h_node,W2)
	y_temp = tf.placeholder("float",[None, model.nj])
	loss = tf.reduce_sum(tf.square(y_temp-y_node))
	train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)
	x = []
	y = []
	for i in range(N_ddp_sol*N_sample_init):
		for t in range(model.T):
			x.append(sample_trajectory[i][t])
			y.append(sample_control[i][t])
	sess.run(train, feed_dict={x_temp:x, y_temp:y})
	
	#Build initial sample set S
	x_eval = tf.placeholder("float",[1,model.nj])
	h_eval = tf.nn.softmax(tf.matmul(x_eval,W1))
	y_eval = tf.matmul(h_eval,W2)

	for j in range(N_sample_init):
		trajectory_temp = []
		trj = [model.q_init]
		for i in range(model.T):
			next_policy = sess.run(y_eval, feed_dict={x_eval:trj})
			next_policy = np.random.multivariate_normal(next_policy[0],np.eye(model.nj))
			trajectory_temp.append(trj[0])
			control_temp.append(next_policy[0])
			trj[0] += next_policy[0]
		sample_trajectory[N_ddp_sol*N_sample_init+j] = trajectory_temp
		sample_control[N_ddp_sol*N_sample_init+j] = control_temp

	print ('initialize done')

	#GPS start!!!!! wow! LOL!

	#Choose current sample set

	#Optimize theta

	ros_agent = ROS_connection()
	ros_agent.move_arm(model.T, trajectory)
	
	print "final position"
	print model.fin_position
	print "result position"
	print model.getPosition(trajectory[-1])
	print model.res['J_hist'][-1]


