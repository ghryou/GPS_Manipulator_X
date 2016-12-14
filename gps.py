import roslib
import sys
import rospy
from std_msgs.msg import Float64
import numpy as np
import PyKDL
from sensor_msgs.msg import JointState
from model import Manipulator_X
import tensorflow as tf
from scipy.stats import multivariate_normal as mul_normal

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
		self.q_status = PyKDL.JntArray(7)
                self.flag_callback = 1
                self.move_arm_once(self.q_init)
		return

	def callback(self, data):
                """
		self.q_init[0] = data.position[2]
		self.q_init[1] = data.position[3]
		self.q_init[2] = data.position[4]
		self.q_init[3] = data.position[5]
		self.q_init[4] = data.position[6]
		self.q_init[5] = data.position[7]
		self.q_init[6] = data.position[8]	
		self.sub_once.unregister()
                """			
		self.q_status[0] = data.position[2]
		self.q_status[1] = data.position[3]
		self.q_status[2] = data.position[4]
		self.q_status[3] = data.position[5]
		self.q_status[4] = data.position[6]
		self.q_status[5] = data.position[7]
		self.q_status[6] = data.position[8]
                if self.flag_callback == 1:
                        print "callback"	
	        	print([self.q_status])
                        self.flag_callback = 0
		return

	def move_arm(self, T, trj):
		for i in range(T):
			for j in range(0,7):
				self.pub[j].publish(trj[i][j])			
			self.r.sleep()
			print("ros signal sent")
        	        print(trj[i])
		return self.q_status

        def move_arm_once(self, target):
                self.flag_callback = 1		
                for j in range(0,7):
			self.pub[j].publish(target[j])			
                while self.flag_callback == 0:		
                        self.r.sleep()
		print("ros signal sent")
     	        print(target)
		return self.q_status

if __name__ == '__main__':

	model = Manipulator_X(T=2, weight=[1.0,5.0])
	model.build_ilqr_problem()

	N_sample = 100  #number of total sample
	N_sample_init = 10 #number of sample from each ddp solution
	N_ddp_sol = 1  #number of ddp solution
	N_max_sol = 100
        flag_end_sol = 0
	flag_end_sample = 0 #last index of sample

	sample_policy = [None]*N_max_sol
	sample_trajectory = [None]*N_sample
	sample_control = [None]*N_sample
        
        sample_dict = [None]*N_sample

	print "test 1 ",model.q_init

	#line 1&2 Generate DDP solutions and build sample set
        print "process 1 Generate DDP solutions"
	for i in range(N_ddp_sol):
		print "iLQR number ",i
		model.solve_ilqr_problem()
		res_temp = model.res
		res_temp['type'] = 0
		sample_policy[i] = res_temp
		model.generate_dest()
                flag_end_sol += 1
		for j in range(N_sample_init):
			trajectory_temp = []
			control_temp = []
                        pb_temp = []
                        c_pb_temp = []
                        cost_temp = []
			trj = np.zeros(model.nj)
			x_i = sample_policy[i]['x_array_opt']
			u_i = sample_policy[i]['u_array_opt'] 
			K_i = sample_policy[i]['K_array_opt']
			Cov = sample_policy[i]['Q_array_opt']
			for t in range(model.T):
                                next_policy = np.random.multivariate_normal(u_i[t]+K_i[t].dot(trj-x_i[t]),Cov[t])
                                pb_temp.append(mul_normal.pdf(next_policy, u_i[t]+K_i[t].dot(trj-x_i[t]),Cov[t]))
                                if len(c_pb_temp) == 0:
                                        c_pb_temp.append(pb_temp[0])
                                else:
                                        c_pb_temp.append(c_pb_temp[-1]*pb_temp[-1])
				trajectory_temp.append(trj)
				control_temp.append(next_policy)
				trj += next_policy #dynamics
                                cost_temp.append(model.instaneous_cost(trj,next_policy,t,0))

			sample_temp = {
                        'trajectory':trajectory_temp,
                        'control':control_temp,
                        'pb':pb_temp,
                        'c_pb':c_pb_temp,
                        'cost':cost_temp,
                        'index':flag_end_sol
                        }
                        sample_dict[flag_end_sample] = sample_temp
                        flag_end_sample += 1


	#line 3 Initialize theta
        print "process 2 Initialize parameter"
	N_hidden_node = 30
	x_temp = tf.placeholder("float",[None, model.nj])
	W1 = tf.Variable(tf.zeros([model.nj, N_hidden_node]), name="W1")
	W2 = tf.Variable(tf.zeros([N_hidden_node, model.nj]), name="W2")
	W1_new = tf.Variable(tf.zeros([model.nj, N_hidden_node]), name="W1")
	W2_new = tf.Variable(tf.zeros([N_hidden_node, model.nj]), name="W2")
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
			x.append(sample_dict[i]['trajectory'][t])
			y.append(sample_dict[i]['control'][t])
	sess.run(train, feed_dict={x_temp:x, y_temp:y})
	
	#line 4 Build initial sample set S
        print "process 3 Build initial sample set"
	x_eval = tf.placeholder("float",[1,model.nj])
	h_eval = tf.nn.softmax(tf.matmul(x_eval,W1))
	y_eval = tf.matmul(h_eval,W2)

       	x_eval_new = tf.placeholder("float",[1,model.nj])
       	h_eval_new = tf.nn.softmax(tf.matmul(x_eval_new,W1_new))
       	y_eval_new = tf.matmul(h_eval_new,W2_new)

	for j in range(N_sample_init):
		trajectory_temp = []
        	control_temp = []
                i_rate_temp = []
                c_i_rate_temp = []
                cost_temp = []
		trj = [np.zeros(model.nj)]
		#print "init ", trj
		for i in range(model.T):
			next_policy_mean = sess.run(y_eval, feed_dict={x_eval:trj})
			next_policy = np.random.multivariate_normal(next_policy_mean[0],np.eye(model.nj))
                        pb_temp.append(mul_normal.pdf(next_policy, next_policy_mean[0],np.eye(model.nj)))
                        if len(c_pb_temp) == 0:
                                c_pb_temp.append(pb_temp[0])
                        else:
                                c_pb_temp.append(c_pb_temp[-1]*pb_temp[-1])
                        trj[0] += next_policy #dynamics
                        
        		trajectory_temp.append(trj[0])
			control_temp.append(next_policy)
			cost_temp.append(model.instaneous_cost(trj[0],next_policy,i,0))
			#print "time ", i, "result", trj[0]

		sample_temp = {
                'trajectory':trajectory_temp,
                'control':control_temp,
                'pb':pb_temp,
                'c_pb':c_pb_temp,
                'cost':cost_temp,
                'index':flag_end_sol
                }
                sample_dict[flag_end_sample] = sample_temp
                flag_end_sample += 1

		#sample_trajectory[flag_end_sample] = trajectory_temp
		#sample_control[flag_end_sample] = control_temp
		#flag_end_sample += 1        
        flag_end_sol += 1

	print ('process 4 Initialize finished')

	print "test 3 ", model.q_init
        print "flag_sample", flag_end_sample
        print "flag_solution", flag_end_sol


	#line 5 GPS start!!!!! wow! LOL!
        print "process 5 GPS start"
        K = 1
        w_reg = 1e-4
        cost_prev = -1
        
        ros_agent = ROS_connection()	
        
	for k in range(K):
                print "GPS iter ",k
                cost_next = 0
                cost_temp = []
                trj = [np.zeros(model.nj)]
                trajectory_temp = [np.zeros(model.nj)]
                policy_temp = []

                #line 6 Choose current sample set
                

		#line 7 Optimize theta
                print "Optimize parameter"
		h_node_new = tf.nn.softmax(tf.matmul(x_temp,W1_new))
		y_node_new = tf.matmul(h_node_new,W2_new)

                z_t = np.zeros(model.T)
                i_rate_temp = np.ones(flag_end_sample)
                i_rate_sel = np.zeros(flag_end_sol)

		opt = tf.train.GradientDescentOptimizer(0.5)
		"""
		print "Compute gradients of policy"
                #v, g = opt.compute_gradients(y_node_new, [W1_new, W2_new])
		print "Compute gradients of z_t and J"
		loss_grad = 0
		z = []
		J = []
		dist = tf.contrib.distributions.MultivariateNormalDiag(mu=y_node_new,diag_stdev=np.ones(model.nj).astype(np.float32))
		i_temp = dist.pdf(y_temp)
		for i in range(model.T):
	        	exp_temp = 0
        	        z_t = 0
			print "time ",i
			for j in range(flag_end_sample):
				z_t += i_temp[i*flag_end_sample+j]/sample_dict[j]['c_pb'][i]
				exp_temp += i_temp[i*flag_end_sample+j]/sample_dict[j]['c_pb'][i]*sample_dict[j]['cost'][i]
			z.append(z_t)				
        	        J.append(exp_temp/z_t + w_reg*tf.log(z_t))

		print "Compute gradients of loss function"
		for t in range(model.T):
			print "time ",t
			for j in range(flag_end_sample):
				exp_temp = 0
				for i in range(t,model.T):
					exp_temp += i_temp[i*flag_end_sample+j]/sample_dict[j]['c_pb'][i]/z[i]*(sample_dict[j]['cost'][i]-J[i]+w_reg)
				for i in range(model.nj):
					g, v = opt.compute_gradients(y_node_new[t*flag_end_sample+j][i], [W1_new, W2_new])
					loss_grad += g*(y_temp[t*flag_end_sample+j][i]-y_node_new[t*flag_end_sample+j][i])*exp_temp
		"""
		loss_new = 0
                #define loss function
		dist = tf.contrib.distributions.MultivariateNormalDiag(mu=y_node_new,diag_stdev=np.ones(model.nj).astype(np.float32))
		i_temp = dist.pdf(y_temp)
		print "Define loss function"
		for i in range (model.T):
                        exp_temp = 0
                        z_t = 0
			print "time ",i
			for j in range(flag_end_sample):
				z_t += i_temp[i*flag_end_sample+j]/sample_dict[j]['c_pb'][i]
				exp_temp += i_temp[i*flag_end_sample+j]/sample_dict[j]['c_pb'][i]*sample_dict[j]['cost'][i]
                        loss_new += exp_temp/z_t + w_reg*tf.log(z_t)
		
		#print "Apply gradients"		
		#train = opt.apply_gradients(loss_grad)
		print "Start training"
		train = opt.minimize(loss_new)

                #execute Optimization
		x = []
		y = []
		for t in range(model.T):
			for i in range(flag_end_sample):
				x.append(sample_dict[i]['trajectory'][t])
				y.append(sample_dict[i]['control'][t])
		sess.run(train, feed_dict={x_temp:x, y_temp:y})


                #line 8 Append samples to current sample set S
                print "Append new samples"
                for j in range(N_sample_init):
		        trajectory_temp = []
                	control_temp = []
                        i_rate_temp = []
                        c_i_rate_temp = []
                        cost_temp = []
        		trj = [np.zeros(model.nj)]
        		#print "init ", trj
        		for i in range(model.T):
        			next_policy_mean = sess.run(y_eval, feed_dict={x_eval:trj})
        			next_policy = np.random.multivariate_normal(next_policy_mean[0],np.eye(model.nj))
                                pb_temp.append(mul_normal.pdf(next_policy, next_policy_mean[0],np.eye(model.nj)))
                                if len(c_pb_temp) == 0:
                                        c_pb_temp.append(pb_temp[0])
                                else:
                                        c_pb_temp.append(c_pb_temp[-1]*pb_temp[-1])
                                trj[0] += next_policy #dynamics
                        
                		trajectory_temp.append(trj[0])
        			control_temp.append(next_policy)
        			cost_temp.append(model.instaneous_cost(trj[0],next_policy,i,0))
        			#print "time ", i, "result", trj[0]
        
        		sample_temp = {
                        'trajectory':trajectory_temp,
                        'control':control_temp,
                        'pb':pb_temp,
                        'c_pb':c_pb_temp,
                        'cost':cost_temp,
                        'index':flag_end_sol
                        }
                        sample_dict[flag_end_sample] = sample_temp
                        flag_end_sample += 1
                flag_end_sol += 1

                #line 10 Estimate the costs of prev and next parameters
                print "Execute and estimate the cost"
                #execute and calculate cost

                trj = [np.zeros(model.nj)]
                cost_temp = []
                cost_next = 0
                for i in range(model.T):
                        next_policy = sess.run(y_eval_new, feed_dict={x_eval_new:trj})
                        trj[0] += next_policy[0]
                        trj_temp = ros_agent.move_arm_once(trj[0])
			for j in range(model.nj):
				trj[0][j] = trj_temp[j]
                        cost_next += model.instaneous_cost(trj[0],next_policy[0],i,0)
		        cost_temp.append(cost_next)
                
                #line 11 Compare and change sample
                if cost_prev == -1 or cost_prev > cost_next:
                        cost_prev = cost_next
                        W1 = W1_new
                        W2 = W2_new
                        w_reg/=10
                        print "Update the parameter"
                else:
                        w_reg*=10
                        print "Keep the parameter"

        print "GPS finished"

	#calculate final cost
        print "###Final Execute###"
        trj = [np.zeros(model.nj)]
        cost_final = 0
        cost_temp = []
        for i in range(model.T):
                next_policy = sess.run(y_eval, feed_dict={x_eval:trj})
                trj[0] += next_policy[0]
                trj_temp = ros_agent.move_arm_once(trj[0])
		for j in range(model.nj):
			trj[0][j] = trj_temp[j]
		cost_final += model.instaneous_cost(trj[0],next_policy[0],i,0)
		cost_temp.append(cost_final)        
	
        #ros_agent.move_arm(model.T, sample_trajectory[flag_end_sample-1])
	
	print "final position"
	print model.fin_position
	print "result position"
	print model.getPosition(sample_trajectory[flag_end_sample-1][-1])
	#print model.res['J_hist'][-1]

