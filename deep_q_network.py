#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import cv2
import sys
sys.path.append("game/")
#import wrapped_flappy_bird as game
import t2
import random
import copy
import numpy as np
from collections import deque

GAME = 'snake' # the name of the game being played for log files
ACTIONS = 4 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 100000. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
#EXPLORE = 500000.
#FINAL_EPSILON = 0.0001 # final value of epsilon
#INITIAL_EPSILON = 0.0001 # starting value of epsilon
REPLAY_MEMORY = 100000 # number of previous transitions to remember
BATCH = 32# size of minibatch
FRAME_PER_ACTION = 1
INITIAL_EPSILON = 0.4
FINAL_EPSILON = 0.05
tau = 0.001
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

class createNetwork():
    def __init__(self):
        self.W_conv1 = weight_variable([8, 8, 4, 32])
        self.b_conv1 = bias_variable([32])

        self.W_conv2 = weight_variable([4, 4, 32, 64])
        self.b_conv2 = bias_variable([64])

        self.W_conv3 = weight_variable([3, 3, 64, 64])
        self.b_conv3 = bias_variable([64])

        self.W_fc1 = weight_variable([1600, 512])
        self.b_fc1 = bias_variable([512])

        #W_fc2 = weight_variable([512, 2])
        #b_fc2 = bias_variable([2])
    
        self.W_fc2 = weight_variable([512, ACTIONS])
        self.b_fc2 = bias_variable([ACTIONS])

        # input layer
        self.s = tf.placeholder("float", [None, 80, 80, 4])

        # hidden layers
        self.h_conv1 = tf.nn.relu(conv2d(self.s, self.W_conv1, 4) + self.b_conv1)
        self.h_pool1 = max_pool_2x2(self.h_conv1)

        self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2, 2) + self.b_conv2)
        #h_pool2 = max_pool_2x2(h_conv2)

        self.h_conv3 = tf.nn.relu(conv2d(self.h_conv2, self.W_conv3, 1) + self.b_conv3)
        #h_pool3 = max_pool_2x2(h_conv3)

        #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
        self.h_conv3_flat = tf.reshape(self.h_conv3, [-1, 1600])

        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_conv3_flat, self.W_fc1) + self.b_fc1)

        # readout layer
        self.readout = tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2
        self.predict = tf.argmax(self.readout, 1)
        self.a = tf.placeholder("float", [None, ACTIONS])
        self.y = tf.placeholder("float", [None])
        self.readout_action = tf.reduce_sum(tf.multiply(self.readout, self.a), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y - self.readout_action))
        self.train_step = tf.train.AdamOptimizer(1e-6).minimize(self.cost)

    
    

    #return s, readout, h_fc1

      
def trainNetwork(target_q,current_q,sess):
    # define the cost function
    

    # open up a game state to communicate with emulator
    game_state = t2.game()

    # store the previous observations in replay memory
    D = deque()
    E=deque()
    F=deque()
    #F=deque()
    #G=deque()
    num=0

    # printing
    a_file = open("logs_" + GAME + "/readout.txt", 'w')
    h_file = open("logs_" + GAME + "/hidden.txt", 'w')

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
   
    do_nothing[random.randrange(ACTIONS)] = 1
    #last_state=[100,240]
    t=0
    flag=np.zeros(3)
    x_t, r_0, terminal,flag,_= game_state.frame_step(do_nothing,t,num)
    #print(x_t.shape)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    

    # saving and loading networks
    
    update_w1=tf.assign(target_q.W_conv1,current_q.W_conv1)
    update_h1=tf.assign(target_q.b_conv1,current_q.b_conv1)
    update_w2=tf.assign(target_q.W_conv2,current_q.W_conv2)
    update_h2=tf.assign(target_q.b_conv2,current_q.b_conv2)
    update_w3=tf.assign(target_q.W_conv3,current_q.W_conv3)
    update_h3=tf.assign(target_q.b_conv3,current_q.b_conv3)
    update_w_fc1=tf.assign(target_q.W_fc1,current_q.W_fc1)
    update_b_fc1=tf.assign(target_q.b_fc1,current_q.b_fc1)
    update_w_fc2=tf.assign(target_q.W_fc2,current_q.W_fc2)
    update_b_fc2=tf.assign(target_q.b_fc2,current_q.b_fc2)
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    saver = tf.train.Saver()
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")
    
    
    #trainables = tf.trainable_variables()
    #targetOps = updateTargetGraph(trainables, tau)
    # start training
    epsilon = INITIAL_EPSILON
    t = 0
    #updateTarget(targetOps, sess)
    while (True):
        # choose an action epsilon greedily
        readout_t = current_q.readout.eval(feed_dict={current_q.s : [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        #if t % FRAME_PER_ACTION == 0:
        if random.random() <= epsilon:
            #print("----------Random Action----------")
            action_index = random.randrange(ACTIONS)
            a_t[random.randrange(ACTIONS)] = 1
        else:
            action_index = np.argmax(readout_t)
            a_t[action_index] = 1
        #else:
            #a_t[0] = 1 # do nothing

        # scale down epsilon
        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        #if epsilon > FINAL_EPSILON and t > OBSERVE:
        #    epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal,flag,ac= game_state.frame_step(a_t,t,num)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)
        
        if flag[1]==1 or flag[0]==1:
            a_t=copy.copy(ac)
        if t<1:
            if flag[1]==0 and flag[0]==1:
                num=random.randrange(0,2)
            if terminal:
                num=random.randrange(0,2)
        else:
            num=1
        # store the transition in D
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if flag[0]==1:
            E.append((s_t, a_t, r_t, s_t1, terminal))
        if not terminal:
            F.append((s_t, a_t, r_t, s_t1, terminal))
        if terminal:
            #s_t=copy.copy(s_temp)
            x_t, r_0, terminal,flag,_= game_state.frame_step(do_nothing,t,num)
            x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
            ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
            s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
        else:
            s_t=s_t1
        if len(D) > REPLAY_MEMORY:
            D.popleft()
        if len(E) > REPLAY_MEMORY:
            E.popleft()
        if len(F) > REPLAY_MEMORY:
            F.popleft()
       
        # only train if done observing
        

        if t > OBSERVE:
            if(t %100==0):
                print(t)
            #else:
            if(t>300000):
                minibatch = random.sample(D,8)
                minibatch+= random.sample(E,8)
                minibatch+= random.sample(F,16)
            else:
                minibatch = random.sample(D,24)
                minibatch+= random.sample(E,8)

            
            
            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            A = current_q.predict.eval(feed_dict = {current_q.s : s_j1_batch})
                #print("kkkkkk")
            Q = target_q.readout.eval(feed_dict = {target_q.s : s_j1_batch})
                #print("ppppppp")
            doubleQ = Q[range(32), A]
                #print("hhhhhh")
            doubleQ=r_batch + GAMMA *doubleQ
            current_q.train_step.run(feed_dict = {
                    current_q.y : doubleQ,
                    current_q.a : a_batch,
                    current_q.s : s_j_batch}
            )
                
            if t %500==0:
                sess.run(update_w1)
                sess.run(update_w2)
                sess.run(update_w3)
                sess.run(update_h1)
                sess.run(update_h2)
                sess.run(update_h3)
                sess.run(update_w_fc1)
                sess.run(update_w_fc2)
                sess.run(update_b_fc1)
                sess.run(update_b_fc2)
        t += 1

        #save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = t)
            print("done")
            print(epsilon)

        # print info
        # state = ""
        # if t <= OBSERVE:
        #     state = "observe"
        # elif t > OBSERVE and t <= OBSERVE + EXPLORE:
        #     state = "explore"
        # else:
        #     state = "train"

        #print("TIMESTEP", t, "/ STATE", state, \
        #    "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
        #    "/ Q_MAX %e" % np.max(readout_t))


        # write info to files
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        '''

def playGame():
    target_q= createNetwork()
    current_q= createNetwork()
    sess = tf.InteractiveSession()

#target_q= createNetwork()

    trainNetwork(target_q,current_q,sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
