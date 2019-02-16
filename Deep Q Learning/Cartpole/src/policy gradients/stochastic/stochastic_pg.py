# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 17:55:56 2018

@author: @author: Artem Oppermann
"""


import numpy as np
import tensorflow as tf
import gym
from sklearn.kernel_approximation import RBFSampler
import sklearn.pipeline
import sklearn.preprocessing
import imageio


class Actor:
    
    def __init__(self, env, FLAGS):
        
        """
        This class implements the actor for the stochastic policy gradient model.
        The actor class determines the action that the agent must take in a environment.
        
        :param FLAGS: TensorFlow flags which contain thevalues for hyperparameters
        
        """
        
        self.env=env
        self.FLAGS=FLAGS
        
        mountainCarEnv=env.get_mountain_env()
        
        # Placeholder for the state
        self.state = tf.placeholder(tf.float32, [40], "state")
        
        # Placeholder for the TD-Target
        self.td_error = tf.placeholder(dtype=tf.float32, name="td_error")
        
        # Linear clsasifier for the mean of the gaussian distribution
        self.mu = self._mu_classifier()
        
        # Linear clsasifier for the variance of the gaussian distribution
        self.sigma = self._sigma_classifier()
        
        # Create a Gaussian Distribution
        self.normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
        
        # Sample the action from the distribution and clip it to the range [-1,+1]
        self.action = self.normal_dist._sample_n(1)
        self.action = tf.clip_by_value(self.action, mountainCarEnv.action_space.low[0], mountainCarEnv.action_space.high[0])
        
        # Calculate loss
        self.loss = -self.normal_dist.log_prob(self.action) * self.td_error
                                              
        # Add entropy cost to encourage exploration
        self.loss -= 1e-1 * self.normal_dist.entropy()
        
        # Training operations
        self.optimizer = tf.train.AdamOptimizer(self.FLAGS.learning_rate_Actor)
        self.train_op = self.optimizer.minimize(self.loss)
    
    def _mu_classifier(self):
        ''' Linear clsasifier for the mean of the gaussian distribution '''
        
        with tf.name_scope('mu'):
            W1=tf.get_variable('W_1_mu', shape=(40,1), initializer=tf.zeros_initializer)
            mu=tf.matmul(tf.expand_dims(self.state, 0), W1)
            mu = tf.squeeze(mu)
        return mu  
    
    def _sigma_classifier(self):
        ''' Linear clsasifier for the variance of the gaussian distribution '''
        
        with tf.name_scope('sigma'):
            W2=tf.get_variable('W_1_sigma', shape=(40,1), initializer=tf.zeros_initializer)
            sigma=tf.matmul(tf.expand_dims(self.state, 0), W2)
            sigma = tf.squeeze(sigma)
            sigma = tf.nn.softplus(sigma) + 1e-5                 
        return sigma
           
    def update(self, state, td_error, action):
        ''' Update operation for the weights '''
        
        state = self.env.featurize_state(state)
        feed_dict = { self.state: state, self.td_error: td_error, self.action: action  }
        _, loss = self.session.run([self.train_op, self.loss], feed_dict)
        return loss

    def sample_action(self, state):   
        ''' Sample an action from the gaussian distribution '''
        
        state = self.env.featurize_state(state)
        return self.session.run(self.action, feed_dict={self.state:state})

   
    def set_session(self, session):
        ''' Setter method for the session '''
        self.session=session
    
    
class Critic:
    
    def __init__(self, env, FLAGS):
        """
        This class implements the Critic for the stochastic policy gradient model.
        The critic provides a state-value for the current state environment where 
        the agent operates.
        
        :param env: The openAI Gym instance
        :param FLAGS: TensorFlow flags which contain thevalues for hyperparameters
        
        """
        
        self.env=env
        self.FLAGS=FLAGS
        
        self.state = tf.placeholder(tf.float32, [40], "state")
        self.target = tf.placeholder(dtype=tf.float32, name="target")

        # Linear classifier operation for the estimatation of the state-value
        self.state_value = self._value_estimator()
        self.loss = tf.squared_difference(self.state_value, self.target)

        # Training operations
        self.optimizer = tf.train.AdamOptimizer(self.FLAGS.learning_rate_Critic)
        self.train_op = self.optimizer.minimize(self.loss)  
    
    def _value_estimator(self):
        ''' Linear classifier for the estimatation of the state-value '''
        
        with tf.name_scope('state_value'):
            W=tf.get_variable('W_state_value', shape=(40,1), initializer=tf.zeros_initializer)
            state_value=tf.matmul(tf.expand_dims(self.state, 0), W)
            state_value=tf.squeeze(state_value)
        return state_value
    
    def set_session(self, session):
        ''' Setter method for the session '''
        self.session=session
    
    def predict(self, state):
        
        '''Predict the state-value for the state 
        
        :param state: Current state in the OpenAI Gym Environment
        
        '''
        state = self.env.featurize_state(state)
        return self.session.run(self.state_value, { self.state: state })
    
    def update(self, state, target):
        ''' Update the weights of the critic 
     
        :param state: Current state in the OpenAI Gym Environment
        :param target: TD-Target
        
        '''
        state = self.env.featurize_state(state)
        feed_dict = { self.state: state, self.target: target }
        _, loss = self.session.run([self.train_op, self.loss], feed_dict)
        return loss
    
    


class Environment:
    
    def __init__(self):
        """
        This class implements the OpenAI Gym Environment for MountainCarContinuous-v0

        
        """
        
        self.env = gym.make('MountainCarContinuous-v0')
        self.state_size = len(self.env.observation_space.sample())
          
        observation_examples = np.array([self.env.observation_space.sample() for x in range(10000)])
        
        # Feature the state representation. Leads to better results, 
        # since the dimension of the state increases from 3 to 40 dimensions
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)
        self.featurizer = sklearn.pipeline.FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=10)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=10)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=10)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=10))
            ])
        self.featurizer.fit(self.scaler.transform(observation_examples))
        
    def get_mountain_env(self):
        '''Getter function for the OpenAI Gym instance '''
        return self.env
    
    def get_state_size(self):
        '''Getter function for the state-size in the environment '''
        return self.state_size
    
    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        scaled = self.scaler.transform(state)
        featurized = self.featurizer.transform(scaled)
        return featurized[0]


class Model:
    
    def __init__(self, FLAGS):
        """
        This class build the model that implements the stochstic 
        gradient descent algorithm.
        
        :param FLAGS: TensorFlow flags which contain thevalues for hyperparameters
        
        """
        
        self.env=Environment()
        self.FLAGS=FLAGS
          
        # Build the Actor instance
        self.actor=Actor(self.env, FLAGS)
        # Build the Critic instance
        self.critic=Critic(self.env, FLAGS)
            
        init = tf.global_variables_initializer()
        session = tf.InteractiveSession()
        session.run(init)
        
        self.critic.set_session(session)
        self.actor.set_session(session)
        
        self.num_episodes=1000
        
         

    def playEpisode(self, episode):
        '''Play an episode in the OpenAI Gym
        
        :param episode: Number of the current episode
        '''
        
        # Get the initial state and reshape it
        state=self.env.get_mountain_env().reset()
        state=state.reshape(1,self.env.get_state_size())
        
        done=False
        total_reward=0
        iters = 0
        
        # Only necessary for GIF creation 
        images=[]

        # Loop for the episode
        while not done and iters < 2000:
            
            # Sample an action from the gauss distribution
            
            action=self.actor.sample_action(state)
            prev_state=state
            
            # Obtain a <state, reward, done> tuple from the environment
            state, reward, done, _ = self.env.get_mountain_env().step(action)
            state=state.reshape(1,self.env.get_state_size())
            
            '''
            # Only necessary for GIF creation
            if episode>10:
                image=self.env.get_mountain_env().render(mode='rgb_array')
                images.append(image)
            '''
            
            total_reward=total_reward+reward
            
            # Calculate TD-Target and TD-Error
            value_next = self.critic.predict(state)
            td_target = reward + self.FLAGS.gamma * value_next
            td_error = td_target - self.critic.predict(prev_state)
           
            # Update the critic
            self.critic.update(prev_state, td_target)
            
            # Update the actor
            self.actor.update(state, td_error, action)
        
        '''
        # Only necessary for GIF creation
        if episode>10:
            imageio.mimsave("C:/Users/Admin/Desktop/Deep Learning/" + 'mountain_car_%i.gif'%episode, images, fps=30)
        '''    
            
        return total_reward
            

    def run_model(self):
        '''Run the environment for a particular number of episodes. '''
        
        totalrewards = np.empty(self.num_episodes+1)
        n_steps=1
        
        for n in range(0, self.num_episodes+1):
            
            total_reward=self.playEpisode(n)
            
            totalrewards[n]=total_reward 
            
            if n>0:
                print("episodes: %i, avg_reward (last: %i episodes): %.2f" %(n, n_steps, totalrewards[max(0, n-n_steps):(n+1)].mean()))



tf.app.flags.DEFINE_float('learning_rate_Actor', 0.001, 'Learning rate for the policy estimator')

tf.app.flags.DEFINE_float('learning_rate_Critic', 0.1, 'Learning rate for the state-value estimator')

tf.app.flags.DEFINE_float('gamma', 0.95, 'Future discount factor')

FLAGS = tf.app.flags.FLAGS



if __name__ == "__main__":
    
    pendulum=Model(FLAGS)
    pendulum.run_model()
    






