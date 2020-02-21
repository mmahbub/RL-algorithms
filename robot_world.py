import numpy as np
import random
#import matplotlib.pyplot as plt 
import time
import sys

class Robot_World:
    def __init__(self, d, alpha, epsilon, n, m):
        self.d = d
        self.alpha = alpha
        self.epsilon = epsilon
        self.n = n
        self.m = m
        self.pi = np.full((d,d,d,d,4), 0.25)
        self.q = np.full((d,d,d,d,4), 0.0)
        self.episodes = []
        self.returns = [[[[[[] for l in range(4)] for m in range(d)] for j in range(d)] for j in range(d)] for i in range(d)]
        self.actions = [0,1,2,3]
        self.total_returns = []
        self.total_time = []
        self.center = d/2.0 - 1
        self.random_episode = random.randint(1,n)
        
    def plot_episode(self, a_i, a_j, b_i, b_j, action, step):
        
        # print a single step of an episode
        grid = np.zeros((self.d,self.d))
        grid[a_i, a_j] = 1
        
        # print non-terminal state
        if not (b_i == self.d or b_i == -1 or b_j == self.d or b_j == -1):
            
            grid[b_i, b_j] = 9
            print()
            print("s"+ str(step))
            print(grid)
            if action == 0:
                print("a"+str(step)+": North")
            elif action == 1:
                print("a"+str(step)+": South")
            elif action == 2:
                print("a"+str(step)+": East")
            else:
                print("a"+str(step)+": West")        
        
        # print terminal state
        else:
            print()
            print("s"+ str(step)+":")           
            print(grid)


    def online_mc(self, episode_n, print_episodes=False, starting_state=None):
        
        rewards = []
        
        #generate episodes
        episode = []
        
        # select random starting state
        if starting_state == None:
            a_i, a_j = [random.randint(0,d-1) for i in range(0,2)]

            while True: 
                b_i, b_j = [random.randint(0,d-1) for i in range(0,2)]  

                if not (a_i == b_i and a_j == b_j):
                    break
                    
        # use preset starting state            
        else:
            a_i, a_j = starting_state[0:2]
            b_i, b_j = starting_state[2:4]            
        
        steps = 0
        curr_r = -1
    
        # loop through steps in episode until bomb is off island or max steps exceeded
        while(b_i >= 0 and b_i < self.d and b_j >= 0 and b_j < self.d and steps < 1000):
            
            prev_a_i, prev_a_j = a_i, a_j
            prev_b_i, prev_b_j = b_i, b_j
            
            s = [a_i, a_j, b_i, b_j]
            
            
            # select random action with epsilon greedy policy
            prob_scores = self.pi[a_i, a_j, b_i, b_j,:]
            
            a = np.random.choice(np.arange(len(prob_scores)), p = prob_scores)
            
            if print_episodes: 
                self.plot_episode(prev_a_i, prev_a_j, prev_b_i, prev_b_j, a, steps)
            
            # move robot based on action
            if a == 0:
                a_i -= 1

            elif a == 1: #N
                a_i += 1

            elif a == 2: #E
                a_j += 1

            else:
                a_j -= 1

            # return robot to previous if out of bounds
            if not (a_i >= 0 and a_i < self.d and a_j >= 0 and a_j < self.d):
                a_i, a_j = prev_a_i, prev_a_j

            # move bomb when robot is on it
            if a_i == b_i and a_j == b_j:
                if a == 0: #S
                    b_i -= 1

                elif a == 1: #N
                    b_i += 1

                elif a == 2: #E
                    b_j += 1

                else: #W
                    b_j -= 1                
                
            
            # calculate reward get state
            curr_r = self.calculate_reward(b_i, b_j, prev_b_i, prev_b_j, a)
            r = curr_r
            
            steps+=1

            # add episode to list of episodes
            episode.append([r,s,a])
            
        # print final step    
        if print_episodes:
            self.plot_episode(a_i, a_j, b_i, b_j, a, steps)

        g = 0
        previous_states_actions = []
        
        for t in range(len(episode)-1, -1, -1):
            
            # increment returns
            g = g+episode[t][0]
            
            w = episode[t][1][0]
            x = episode[t][1][1]
            y = episode[t][1][2]
            z = episode[t][1][3]
            a_t = episode[t][2]
            
            
            # check if state action pair previously visited
            if [episode[t][1], episode[t][2]] not in previous_states_actions:
                
                # save state action pair and returns
                previous_states_actions.append([episode[t][1], episode[t][2]])
                self.returns[w][x][y][z][a_t].append(g)
            
                # update action value to mean return at state action pair
                self.q[w,x,y,z,a_t] = np.mean(self.returns[w][x][y][z][a_t])
                
                # select action at state, action using epsilon-greedy approach
                A = np.argmax(self.q[w,x,y,z])
                for a in self.actions:
                    if a == A:
                        self.pi[w,x,y,z,a] =  1 - self.epsilon + self.epsilon/len(self.actions)
                    else:
                        self.pi[w,x,y,z,a] = self.epsilon/len(self.actions)

        return g

    def q_learning(self, episode_n, e_greedy, print_episodes=False, starting_state=None):
	
        # perform one episode of q-learning
        returns = []    
    
        # select random starting state
        if starting_state == None:
            a_i, a_j = [random.randint(0,d-1) for i in range(0,2)]
        
        
            while True: 
                b_i, b_j = [random.randint(0,d-1) for i in range(0,2)]  

                if not (a_i == b_i and a_j == b_j):
                    break
        
        # use pre-specified starting state
        else:
            a_i, a_j = starting_state[0:2]
            b_i, b_j = starting_state[2:4]

        steps = 0
        curr_r = -1

        # loop through episode until bomb off island or max # steps exceeded
        while(b_i >= 0 and b_i < self.d and b_j >= 0 and b_j < self.d and steps < 1000):
            
            # select next action using epsilon-greedy or greedy policy
            prob_scores = self.pi[a_i, a_j, b_i, b_j,:]
            if e_greedy:
                a = np.random.choice(np.arange(len(prob_scores)), p = prob_scores)
            else:
                a = np.argmax(prob_scores)
            
            # save previous state
            prev_a_i, prev_a_j = a_i, a_j
            prev_b_i, prev_b_j = b_i, b_j
    
            # print current step of episode
            if print_episodes:
                self.plot_episode(prev_a_i, prev_a_j, prev_b_i, prev_b_j, a, steps)

    
            # move robot based on action
            if a == 0: #N
                a_i -= 1

            elif a == 1: #S
                a_i += 1

            elif a == 2: #E
                a_j += 1

            else: #W
                a_j -= 1

            # return robot to previous if out of bounds
            if not (a_i >= 0 and a_i < self.d and a_j >= 0 and a_j < self.d):
                a_i, a_j = prev_a_i, prev_a_j

            # move bomb when robot is on it
            if a_i == b_i and a_j == b_j:
                
                if a == 0: #N
                    b_i -= 1

                elif a == 1: #S
                    b_i += 1

                elif a == 2: #E
                    b_j += 1

                else: #W
                    b_j -= 1

            # calculate reward and update state
            r = self.calculate_reward(b_i, b_j, prev_b_i, prev_b_j, a)
            s = [a_i, a_j, b_i, b_j]
            
            # calculate q at terminal and non-terminal states
            if (b_i >= 0 and b_i < self.d and b_j >= 0 and b_j < self.d):
                part2 = r + np.max(self.q[a_i,a_j,b_i,b_j,:])-self.q[prev_a_i, prev_a_j, prev_b_i, prev_b_j,a]
                self.q[prev_a_i, prev_a_j,prev_b_i, prev_b_j,a] = self.q[prev_a_i, prev_a_j,prev_b_i, prev_b_j,a] + self.alpha*part2
            else:
                part2 = r + 0-self.q[prev_a_i, prev_a_j, prev_b_i, prev_b_j,a]
                self.q[prev_a_i, prev_a_j,prev_b_i, prev_b_j,a] = self.q[prev_a_i, prev_a_j,prev_b_i, prev_b_j,a] + self.alpha*part2
            
            # update action value function q
            A = np.argmax(self.q[prev_a_i,prev_a_j,prev_b_i, prev_b_j,])
            for a in self.actions:
                if a == A:
                    self.pi[prev_a_i,prev_a_j,prev_b_i, prev_b_j,a] =  1 - self.epsilon + self.epsilon/len(self.actions)
                else:
                    self.pi[prev_a_i,prev_a_j,prev_b_i, prev_b_j,a] = self.epsilon/len(self.actions)
            returns.append(r)
            steps += 1
            
        # print terminal state
        if print_episodes:
            self.plot_episode(a_i, a_j, b_i, b_j, a, steps)
        
        return sum(returns)                                                       

    def learning(self):
        
        # use Monte Carlo learning
        if self.m == 1:
            print("Monte Carlo")
            for i in range(0,n):
                start = time.time()
                returns = self.online_mc(i)
                self.total_returns.append(returns)
                self.total_time.append(time.time()-start)  
            if self.d == 8:
                print("Final Episode: ")
                print("Setup Q-values:")
                print(self.q[5,0,4,1])
                self.online_mc(n, print_episodes=True, starting_state=[5,0,4,1])
            
        # use q-learning                
        if self.m == 2:
            print("Q-Learning")
            for i in range(0,n):
                e_greedy = True
                start = time.time()
                returns = self.q_learning(i, e_greedy)
                self.total_returns.append(returns)
                self.total_time.append(time.time()-start)
            if self.d == 8:
                print("Setup Q-values:")
                print(self.q[5,0,4,1])
                print("Final Episode: ")
                self.q_learning(n, e_greedy, print_episodes=True, starting_state=[5,0,4,1])
        
        return self.total_returns

    # get list of episode runtimes
    def get_episode_times(self):
        return self.total_time
    
    # calculate given bomb's previous and current position
    def calculate_reward(self, b_i, b_j, prev_b_i, prev_b_j, action):
        if (b_i == -1 or b_i == self.d or b_j == -1 or b_j == self.d):
            curr_r = 10
        elif(np.abs(self.center-b_i) > np.abs(self.center-prev_b_i) or np.abs(self.center-b_j) > np.abs(self.center-prev_b_j)):
            curr_r = 1
        else:
            curr_r = -1
            
        return curr_r
            
			
d= int(sys.argv[1])
a= float(sys.argv[2])
e= float(sys.argv[3])
n= int(sys.argv[4])
m= int(sys.argv[5])

print(d, a, e, n, m)

robot_world = Robot_World(d, a, e, n, m)
return_total = robot_world.learning()
