"""
Basic PGPE for Baxter Robot
"""

import theano
import theano.tensor as T
import lasagne

import numpy as np
from numpy import array
from numpy import ones
from numpy import asarray

import rospy

from pybrain.rl.environments import Environment
from pybrain.rl.environments import EpisodicTask

import argparse

import rospy

import baxter_interface
import baxter_external_devices
from baxter_interface import settings

from baxter_interface import CHECK_VERSION

np.set_printoptions()

# Number of transmitted variables
N_TRANS = 5

# Input features
N_INPUT_FEATURES = 7

# Output Features
N_ACTIONS = 7

# Output Features
N_OUTPUT_FEATURES = 4

# Length of each input sequence of data
N_TIME_STEPS = 1  # in cart pole balancing case, x, x_dot, theta, theta_dot and reward are inputs


# Number of units in the hidden (recurrent) layer
N_HIDDEN = 10

# This means how many sequences you would like to input to the sequence.
N_BATCH = 1

# SGD learning rate
LEARNING_RATE = 2e-1

# Number of iterations to train the net
N_ITERATIONS = 1000000

# Forget rate
FORGET_RATE = 0.9

# Number of reward output
N_REWARD = 1


class BaxterReachEnv(Environment):
    """
    This class takes care of communicating with simulator
    """
    def __init__(self):
        Environment.__init__(self)
        print("Initializing node... ")
        rospy.init_node("test")
        self.rs = baxter_interface.RobotEnable(CHECK_VERSION)
        init_state = self.rs.state().enabled

        def clean_shutdown():
            print("\nExiting example...")
            if not init_state:
                print("Disabling robot...")
                self.rs.disable()
        rospy.on_shutdown(clean_shutdown)

        print("Enabling robot... ")
        self.rs.enable()

        self.left_arm = baxter_interface.Limb('left')
        self.joint_names = self.left_arm.joint_names()


    def getSensors(self):
        """ returns the state one step (dt) ahead in the future. stores the state in
            self.sensors because it is needed for the next calculation. The sensor return
            vector has 4 elements: theta, theta', s, s' (s being the distance from the
            origin).
        """
        return self.left_arm.joint_angles().values() # get reversed order states from wrist to shoulder

    def performAction(self, raw_actions):
        """

        :param action: action is a list of lens 7.
        :return:
        """
        actions = dict(zip(self.left_arm.joint_names(),
                          raw_actions[0].flatten().tolist()))
        self.actions = actions
        self.step()

    def step(self):
        self.left_arm.set_joint_velocities(self.actions)


    def reset(self):
        """ re-initializes the environment, setting the cart back in a random position.
        """
        angles = dict(zip(self.left_arm.joint_names(),
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        self.left_arm.move_to_joint_positions(angles)

    def getEndEffectorPosition(self):
        return self.left_arm.endpoint_pose()["position"]


class BaxterReachTask(EpisodicTask):
    """ The task of balancing some pole(s) on a cart """
    def __init__(self, env=None, maxsteps=1000, desiredValue = 0, tolorance = 0.3):
        """
        :key env: (optional) an instance of a CartPoleEnvironment (or a subclass thereof)
        :key maxsteps: maximal number of steps (default: 1000)
        """
        self.desiredValue = desiredValue
        EpisodicTask.__init__(self, env)
        self.N = maxsteps
        self.t = 0
        self.tolorance = tolorance


        # self.sensor_limits = [None] * 4
        # actor between -10 and 10 Newton
        self.actor_limits = [(-50, 50)]

    def reset(self):
        EpisodicTask.reset(self)
        self.t = 0

    def performAction(self, actions):
        self.t += 1
        EpisodicTask.performAction(self, actions)

    def isFinished(self):

        # Neutral place
        # x=0.6359848748431522, y=0.8278984542845692, z=0.19031037139621507
        end_effector_pose = self.env.getEndEffectorPosition()
        if end_effector_pose.x - 0.6359848748431522 < self.tolorance and \
            end_effector_pose.y - 0.8278984542845692< self.tolorance and \
            end_effector_pose.z - 0.19031037139621507 < self.tolorance:
            return True
        elif self.t == self.N:
            return True
        return False

    def getReward(self):
        end_effector_pose = self.env.getEndEffectorPosition()
        if end_effector_pose.x - 0.6359848748431522 < self.tolorance and \
            end_effector_pose.y - 0.8278984542845692< self.tolorance and \
            end_effector_pose.z - 0.19031037139621507 < self.tolorance:
            reward = (self.N - self.t)
        else:
            reward = -1
        return reward

    def setMaxLength(self, n):
        self.N = n

def theano_form(list, shape):
    """
    This function transfer any list structure to a from that meets theano computation requirement.
    :param list: list to be transformed
    :param shape: output shape
    :return:
    """
    return array(list, dtype=theano.config.floatX).reshape(shape)


def one_iteration(task, all_params, action_prediction, l_action_2_formed):
    """
    Give current value of weights, output all rewards
    :return:
    """
    rewards = []
    _all_params = lasagne.layers.get_all_params(l_action_2_formed)

    _all_params[0].set_value(theano_form(all_params[0:N_HIDDEN*N_INPUT_FEATURES], shape=(N_HIDDEN, N_INPUT_FEATURES)))
    _all_params[1].set_value(theano_form(all_params[N_HIDDEN*N_INPUT_FEATURES::], shape=(N_INPUT_FEATURES, N_HIDDEN)))
    task.reset()
    while not task.isFinished():
        train_inputs = theano_form(task.getObservation(), shape=[N_BATCH, N_TIME_STEPS, N_INPUT_FEATURES])
        model_reward_result = action_prediction(train_inputs)
        task.performAction(model_reward_result)
        rewards.append(task.getReward())
    return sum(rewards)

def sample_parameter(sigma_list):
    """
    sigma_list contains sigma for each parameters
    """
    return np.random.normal(0., sigma_list)

def extract_parameter(params):
    current = array([])
    for param in params:
        current = np.concatenate((current, param.get_value().flatten()), axis=0)

    return current



def main():
    """RSDK Joint Position Example: Keyboard Control

    Use your dev machine's keyboard to control joint positions.

    Each key corresponds to increasing or decreasing the angle
    of a joint on one of Baxter's arms. Each arm is represented
    by one side of the keyboard and inner/outer key pairings
    on each row for each joint.
    """

    # From the network
    l_in = lasagne.layers.InputLayer(shape=(N_BATCH, N_TIME_STEPS, N_INPUT_FEATURES))

    l_action_1 = lasagne.layers.DenseLayer(incoming=l_in,
                                                num_units=N_HIDDEN,
                                                nonlinearity=None,
                                                b=None)
    l_action_1_formed = lasagne.layers.ReshapeLayer(input_layer=l_action_1,
                                        shape=(N_BATCH, N_TIME_STEPS, N_HIDDEN))
    l_action_2 = lasagne.layers.DenseLayer(incoming=l_action_1_formed,
                                                num_units=N_ACTIONS,
                                                nonlinearity=None,
                                                b=None)
    l_action_2_formed = lasagne.layers.ReshapeLayer(input_layer=l_action_2,
                                        shape=(N_BATCH, N_TIME_STEPS, N_ACTIONS))

    # Cost function is mean squared error
    input = T.tensor3('input')
    target_output = T.tensor3('target_output')

    # Make the function
    action_prediction = theano.function([input], l_action_2_formed.get_output(input))
    all_params = lasagne.layers.get_all_params(l_action_2_formed)

    # Parameters for system
    baseline = None
    num_parameters = N_INPUT_FEATURES * N_HIDDEN + N_HIDDEN * N_ACTIONS # five parameters
    epsilon = 1  # initial number sigma
    sigma_list = ones(num_parameters) * epsilon
    best_reward = -2000
    current = extract_parameter(params=all_params)
    arg_reward = []

    env = BaxterReachEnv()
    task = BaxterReachTask(env, 1500, desiredValue=None, tolorance = 0.06)
    for n in xrange(1000):
         # current parameters
        deltas = sample_parameter(sigma_list=sigma_list)
        reward1 = one_iteration(task=task, all_params=current + deltas, l_action_2_formed = l_action_2_formed, action_prediction=action_prediction)
        if reward1 > best_reward:
            best_reward = reward1
        reward2 = one_iteration(task= task, all_params=current - deltas, l_action_2_formed = l_action_2_formed, action_prediction=action_prediction)
        if reward2 > best_reward:
            best_reward = reward2
        mreward = (reward1 + reward2) / 2.

        if baseline is None:
            # first learning step
            baseline = mreward
            fakt = 0.
            fakt2 = 0.
        else:
            #calc the gradients
            if reward1 != reward2:
                #gradient estimate alla SPSA but with likelihood gradient and normalization
                fakt = (reward1 - reward2) / (2. * best_reward - reward1 - reward2)
            else:
                fakt=0.
            #normalized sigma gradient with moving average baseline
            norm = (best_reward - baseline)
            if norm != 0.0:
                fakt2=(mreward-baseline)/(best_reward-baseline)
            else:
                fakt2 = 0.0
        #update baseline
        baseline = (0.9 * baseline + 0.1 * mreward)


        # update parameters and sigmas
        current = current + LEARNING_RATE * fakt * deltas

        if fakt2 > 0.: #for sigma adaption alg. follows only positive gradients
            #apply sigma update locally
            sigma_list = sigma_list + LEARNING_RATE * fakt2 * (deltas * deltas - sigma_list * sigma_list) / sigma_list


        arg_reward.append(mreward)
        if not n%10:
            print baseline
            print "best reward", best_reward, "average reward", sum(arg_reward)/len(arg_reward)
            arg_reward = []



if __name__ == '__main__':
    main()


