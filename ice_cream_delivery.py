import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys

class Ice_Cream_Delivery:

	def __init__(self, M, p_min, p_max, p_same, t_drive, t_walk, t_wait, r_no):
		self.M=M
		self.p_min=p_min
		self.p_max=p_max
		self.p_same = p_same
		self.t_drive = t_drive
		self.t_walk = t_walk
		self.t_wait = t_wait
		self.r_no = r_no
		
		self.V = np.zeros(M)
		self.pi = ['walk', 'wait', 'drive', 'walk', 'wait', 'drive', 'walk', 'wait', 'drive', 'walk', 'wait', 'drive']
	
		self.P = np.linspace(p_min, p_max, num=M)
	
		self.gamma = 1.0
		self.actions = ["walk", "wait", "drive"]
		
	def value_iteration(self, sigma):
	
		max_iter = 1000
		count = 0
		values = []
		while True:
			delta = 0.0
			for s in range(0,self.M):
				v = self.V[s]
				vs = self.calculate_values(s)
				self.V[s] = np.max(np.array(vs))
				delta = max(delta, np.abs(v-self.V[s]))
			values.append(np.copy(self.V))
			if delta < sigma:
				break
			count += 1

		values = np.array(values).T
		self.plot_data([int(i)+i for i in range(0, values.shape[1])], values, "Value Iteration Values", "Iteration", "Value V(s)", "values_value_it.png")
		values = np.array(values).T
		self.plot_data([int(i) for i in range(0, self.M)], np.array(values), "State vs. Value in Value Iteration", "State", "Value V(s)", "state_values_value_it.png")
		self.pi = self.calculate_policy()
		policies  = [self.pi]
		self.plot_policy([int(i) for i in range(0, self.M)], policies, "State vs. Policy in Value Iteration", "State (s)", "Value V(s)", "state_policy_value_it.png")
		return self.V, self.pi
	
	def policy_iteration(self, sigma):
		max_iter = 1000
		counta = 0
		countb =0	
		policies =[]
		values = []
		while True:
			while True:
				delta = 0.0
				for s in range(0, self.M):
					v = self.V[s]
					self.V[s] = self.calculate_values_action(s,self.pi[s])
					delta = max(delta, np.abs(v-self.V[s]))
				values.append(np.copy(self.V))
				if delta < sigma:
					break
				counta += 1
			policy_stable = True
			
			old_policy = list.copy(self.pi)
			policies.append(list.copy(self.pi))
			self.pi = self.calculate_policy()
			#$policies.append(list.copy(self.pi))
			if old_policy != self.pi:
				policy_stable = False

			if policy_stable:
				values = np.array(values).T
				self.plot_data([int(i)+1 for i in range(0, values.shape[1])], values, "Policy Iteration Values", "Iteration", "V(s)", "values_policy_it.png")
				values = np.array(values).T
				self.plot_data([int(i) for i in range(0, self.M)], values, "State vs. Value Policy Iteration", "State", "V(s)", "state_values_policy_it.png")
				self.plot_policy([i for i in range(0,M)], policies, "State vs. Policy Policy Iteration", "State", "Policy", "policies_policy_it.png")
				return self.V, self.pi
			countb += 1
		
		self.plot_data(len(values), values, "Policy Iteration Values", 'Iteration', "V(s)", 'values_policy_it.png')
		self.plot_policy([i for i in range(0, M)], policies, "State vs. Policy Policy Iteration", "State", "Policy", "policies_policy_it.png" )		

	def reward(self, state, action):
		# returns rewards given state and action	
		if action == 'walk':
			return -(self.M-state-1)*self.t_walk

		elif action == 'wait':
			return -self.t_wait
		elif action == 'drive':
			if state ==self.M-1:
				return -self.r_no
			else:
				return -self.t_drive
		else:
			print("error")
			exit(0)
				
	def calculate_policy(self):
		# calculates polucy given state values
		for s in range(0,self.M):
			vs = self.calculate_values(s)
			self.pi[s] = self.actions[int(np.argmax(np.array(vs)))]
			
		return self.pi
			
			
	def calculate_values(self, s):
		# calculates values of actions given state
		vs = []
		# walk
		vs.append( (1-self.P[s])*( self.reward(s, 'walk') + self.gamma*0))
		# wait
		vs.append( (self.p_same)* (self.reward(s, 'wait') + self.gamma*self.V[s]))
		# drive
		if s==self.M-1:
			vs.append( (self.P[s])* (self.reward(s, 'drive') + self.gamma*0))	
		else:
			vs.append( (self.P[s])* (self.reward(s, 'drive') + self.gamma*self.V[s+1]))

		return vs

	def calculate_values_action(self, s, a):
		# returns value of a given state and action
		if a == 'walk':
			return (1-self.P[s])*(self.reward(s,'walk') + self.gamma*0)
		elif a == 'wait':
			return self.p_same*(self.reward(s, 'wait')+self.gamma*self.V[s])
		elif a == 'drive':
			if s==self.M-1:
				return self.P[s]*(self.reward(s,'drive')+self.gamma*0)
			else:
				return self.P[s]*(self.reward(s, 'drive')+self.gamma*self.V[s+1])
		else:
			print("Error: Invalid State")



	def plot_data(self, x, y, title, xlabel, ylabel, filename):
		# function for plotting
		for i, y_ in enumerate(y):
			plt.plot(x, y_, label = i)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		#plt.legend()
		plt.title(title)
		plt.show()
		plt.savefig(filename)		
		plt.clf()

	def plot_policy(self, x, y, title, xlabel, ylabel, filename):
		d = {'wait': 0, 'walk': 1, 'drive': 2}
		fig = plt.figure()
		ax = fig.add_subplot(211)
		for i, y_ in enumerate(y):
			ax.plot(x, y_, label=i)
			#ax.plot(x, [d[j] for j in y_], label=i)
		#plt.yticks(['wait', 'walk', 'drive'])
		ax.set_yticks(['wait', 'walk', 'drive'])
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.title(title)
		plt.show()
		plt.legend()
		plt.savefig(filename)
		plt.clf()

# set delivery world parameters with command line arguments
M=int(sys.argv[1])
p_min=float(sys.argv[2])
p_max=float(sys.argv[3])
p_same = float(sys.argv[4])
t_drive = float(sys.argv[5])
t_walk = float(sys.argv[6])
t_wait = float(sys.argv[7])
r_no = float(sys.argv[8])

delivery_world1 = Ice_Cream_Delivery(M, p_min, p_max, p_same, t_drive, t_walk, t_wait, r_no)
vi_value_function, vi_policy = delivery_world1.value_iteration(0.000001)
print("Value Iteration value function: ", vi_value_function)
print("Value Iteration policy: ", vi_policy)
delivery_world2 = Ice_Cream_Delivery(M, p_min, p_max, p_same, t_drive, t_walk, t_wait, r_no)
pi_value_function, pi_policy = delivery_world2.policy_iteration(0.00001)
print("Policy Iteration value function: ", pi_value_function)
print("Policy Iteration policy:", pi_policy)
