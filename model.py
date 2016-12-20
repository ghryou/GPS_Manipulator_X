import PyKDL
import numpy as np
from sensor_msgs.msg import JointState
from math import pi
import random as rd
import matplotlib.pyplot as plt

from pylqr import PyLQR_iLQRSolver

class Manipulator_X():
	def __init__(self, T=20, weight=[1.0,1.0]):
		#initialize model
		self.chain = PyKDL.Chain()
		self.chain.addSegment(PyKDL.Segment("Base", PyKDL.Joint(PyKDL.Joint.None), PyKDL.Frame(PyKDL.Vector(0.0, 0.0, 0.042)), PyKDL.RigidBodyInertia(0.08659, PyKDL.Vector(0.00025, 0.0, -0.02420), PyKDL.RotationalInertia(1.0, 1.0, 1.0, 0.0, 0.0, 0.0))))
		self.chain.addSegment(PyKDL.Segment("Joint1", PyKDL.Joint(PyKDL.Joint.RotZ), PyKDL.Frame(PyKDL.Vector(0.0, -0.019, 0.028)), PyKDL.RigidBodyInertia(0.00795, PyKDL.Vector(0.0, 0.019, -0.02025), PyKDL.RotationalInertia(1.0, 1.0, 1.0, 0.0, 0.0, 0.0))))
		self.chain.addSegment(PyKDL.Segment("Joint2", PyKDL.Joint(PyKDL.Joint.RotY), PyKDL.Frame(PyKDL.Vector(0.0, 0.019, 0.0405)), PyKDL.RigidBodyInertia(0.09312, PyKDL.Vector(0.00000, -0.00057, -0.02731), PyKDL.RotationalInertia(1.0, 1.0, 1.0, 0.0, 0.0, 0.0))))
		self.chain.addSegment(PyKDL.Segment("Joint3", PyKDL.Joint(PyKDL.Joint.RotZ), PyKDL.Frame(PyKDL.Vector(0.024, -0.019, 0.064)), PyKDL.RigidBodyInertia(0.19398, PyKDL.Vector(-0.02376, 0.01864, -0.02267), PyKDL.RotationalInertia(1.0, 1.0, 1.0, 0.0, 0.0, 0.0))))
		self.chain.addSegment(PyKDL.Segment("Joint4", PyKDL.Joint("minus_RotY", PyKDL.Vector(0,0,0), PyKDL.Vector(0,-1,0), PyKDL.Joint.RotAxis), PyKDL.Frame(PyKDL.Vector(0.064, 0.019, 0.024)), PyKDL.RigidBodyInertia(0.09824, PyKDL.Vector(-0.02099, 0.0, -0.01213), PyKDL.RotationalInertia(1.0, 1.0, 1.0, 0.0, 0.0, 0.0))))
		self.chain.addSegment(PyKDL.Segment("Joint5", PyKDL.Joint(PyKDL.Joint.RotX), PyKDL.Frame(PyKDL.Vector(0.0405, -0.019, 0.0)), PyKDL.RigidBodyInertia(0.09312, PyKDL.Vector(-0.01319, 0.01843, 0.0), PyKDL.RotationalInertia(1.0, 1.0, 1.0, 0.0, 0.0, 0.0))))
		self.chain.addSegment(PyKDL.Segment("Joint6", PyKDL.Joint("minus_RotY", PyKDL.Vector(0,0,0), PyKDL.Vector(0,-1,0), PyKDL.Joint.RotAxis), PyKDL.Frame(PyKDL.Vector(0.064, 0.019, 0.0)), PyKDL.RigidBodyInertia(0.09824, PyKDL.Vector(-0.02099, 0.0, 0.01142), PyKDL.RotationalInertia(1.0, 1.0, 1.0, 0.0, 0.0, 0.0))))
		self.chain.addSegment(PyKDL.Segment("Joint7", PyKDL.Joint(PyKDL.Joint.RotX), PyKDL.Frame(PyKDL.Vector(0.14103, 0.0, 0.0)), PyKDL.RigidBodyInertia(0.26121, PyKDL.Vector(-0.09906, 0.00146, -0.00021), PyKDL.RotationalInertia(1.0, 1.0, 1.0, 0.0, 0.0, 0.0))))

		self.min_position_limit = [-160.0, -90.0, -160.0, -90.0, -160.0, -90.0, -160.0]
		self.max_position_limit = [160.0, 90.0, 160.0, 90.0, 160.0, 90.0, 160.0]
		self.min_joint_position_limit = PyKDL.JntArray(7)
		self.max_joint_position_limit = PyKDL.JntArray(7)
		for i in range (0,7):
			self.min_joint_position_limit[i] = self.min_position_limit[i]*pi/180
			self.max_joint_position_limit[i] = self.max_position_limit[i]*pi/180

		self.fksolver = PyKDL.ChainFkSolverPos_recursive(self.chain)
		self.iksolver = PyKDL.ChainIkSolverVel_pinv(self.chain)
		self.iksolverpos = PyKDL.ChainIkSolverPos_NR_JL(self.chain, self.min_joint_position_limit, self.max_joint_position_limit, self.fksolver, self.iksolver, 100, 1e-6)
		
		#parameter
		self.T = T
		self.weight_u = weight[0]
		self.weight_x = weight[1]

		self.nj = self.chain.getNrOfJoints()
		self.q_init = np.zeros(self.nj)
		self.q_out = np.zeros(self.nj)

		ret, self.dest, self.q_out = self.generate_dest()
		self.fin_position = self.dest.p
		return

	def generate_dest(self):
		dest = PyKDL.Frame()
		ret = -3;
		jointpositions = PyKDL.JntArray(self.nj)
		for i in range(1,7):
			jointpositions[i] = 0
		while dest.p.z() <= 0 or ret < 0:
			"""
        		jointpositions[0] = rd.randrange(0,11)/10*(self.max_joint_position_limit[0]-self.min_joint_position_limit[0])+self.min_joint_position_limit[0]
        		jointpositions[1] = rd.randrange(0,11)/10*(self.max_joint_position_limit[1]-self.min_joint_position_limit[1])+self.min_joint_position_limit[1]
        		jointpositions[2] = rd.randrange(0,11)/10*(self.max_joint_position_limit[2]-self.min_joint_position_limit[2])+self.min_joint_position_limit[2]
        		jointpositions[3] = rd.randrange(0,11)/10*(self.max_joint_position_limit[3]-self.min_joint_position_limit[3])+self.min_joint_position_limit[3]
        		jointpositions[4] = rd.randrange(0,11)/10*(self.max_joint_position_limit[4]-self.min_joint_position_limit[4])+self.min_joint_position_limit[4]
        		jointpositions[5] = rd.randrange(0,11)/10*(self.max_joint_position_limit[5]-self.min_joint_position_limit[5])+self.min_joint_position_limit[5]
        		jointpositions[6] = 0
			"""
			jointpositions[0] = 0.3
			jointpositions[1] = -0.5
			jointpositions[3] = -0.6
			jointpositions[5] = -0.5
			jointpositions[6] = -0.8
        		kinematics_status = self.fksolver.JntToCart(jointpositions, dest)

			ret = self.iksolverpos.CartToJnt(self.NpToJnt(self.q_init), dest, self.NpToJnt(self.q_out))
		print dest
		return ret, dest, jointpositions
	
	def NpToJnt(self, q):
		temp = PyKDL.JntArray(self.nj)
		for j in range(self.nj):
			temp[j] = q[j]
		return temp

	def JntToNp(self, q):
		temp = np.zeros(self.nj)
		for j in range(self.nj):
			temp[j] = q[j]
		return temp

	def plant_dyn(self, x, u, t, aux):
		x_new = x + u
		return x_new

	def getPosition(self, x):
		temp_dest = PyKDL.Frame()
		self.fksolver.JntToCart(self.NpToJnt(x), temp_dest)
		x_position = temp_dest.p
		return x_position	

	def instaneous_cost(self, x, u, t, aux):
		if t < self.T - 1:
			return u.dot(u)*self.weight_u
		else:
			temp_dest = PyKDL.Frame()
			self.fksolver.JntToCart(self.NpToJnt(x), temp_dest)
			x_position = temp_dest.p
			return PyKDL.dot(self.fin_position-x_position, self.fin_position-x_position)*self.weight_x + np.dot(u,u)*self.weight_u

	def build_ilqr_problem(self):
		self.ilqr_solver = PyLQR_iLQRSolver(T=self.T, plant_dyn=self.plant_dyn, cost=self.instaneous_cost)
		return
	
	def solve_ilqr_problem(self):
		u_init = []
		#sum = self.q_init
		sum = np.zeros(self.nj)
		#delta = (self.JntToNp(self.q_out)-self.q_init)/self.T
		delta = (self.JntToNp(self.q_out)-np.zeros(self.nj))/self.T
		print "delta",delta
		for t in range(self.T):
			temp = np.zeros(self.nj)
			param = rd.random()+0.5
			for i in range(self.nj):
				if t < self.T-1:
					temp[i] = param*delta[i]
					sum[i] += temp[i]
				else:
					temp[i] = self.q_out[i] - sum[i]
			u_init.append(temp)
		#x_init = self.q_init
		x_init = np.zeros(self.nj)
		if self.ilqr_solver is not None:
			self.res = self.ilqr_solver.ilqr_iterate(x_init,u_init,n_itrs=150, tol=1e-6)
		return

	def plot_ilqr_result(self):
        	if self.res is not None:
			#draw cost evolution and phase chart
			fig = plt.figure(figsize=(16, 8), dpi=80)
			ax_cost = fig.add_subplot(121)
			n_itrs = len(self.res['J_hist'])
			ax_cost.plot(np.arange(n_itrs), self.res['J_hist'], 'r', linewidth=3.5)
			f = open("log/ilqr_result.txt",'a')
			f.write("ilqr_result\n")
			for i in np.arange(n_itrs):
				f.write(str(self.res['J_hist'][i]))
				f.write("\n")
			f.close()
			ax_cost.set_xlabel('Number of Iterations', fontsize=20)
			ax_cost.set_ylabel('Trajectory Cost')

			ax_phase = fig.add_subplot(122)
			theta = self.res['x_array_opt'][:, 0]
			theta_dot = self.res['x_array_opt'][:, 1]
			ax_phase.plot(theta, theta_dot, 'k', linewidth=3.5)
			ax_phase.set_xlabel('theta (rad)', fontsize=20)
			ax_phase.set_ylabel('theta_dot (rad/s)', fontsize=20)
			ax_phase.set_title('Phase Plot', fontsize=20)
			#draw the destination point
			ax_phase.hold(True)
			ax_phase.plot([theta[-1]], [theta_dot[-1]], 'b*', markersize=16)
			print self.res['x_array_opt']
			plt.show()
		return		

