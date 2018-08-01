import tensorflow as tf
import gym

env = gym.make("CartPole-v1")

NumGames = 100

for i in range(NumGames):
    obs = env.reset()
    GenReward = 0
    done = False

    while not done:
        env.render()

        action = env.action_space.sample()

        obs, reward, done, info = env.step(action)
        GenReward += reward
    print("your score was: "+str(GenReward))
env.close()


class Agent:
    def __init__(self, num_actions, state_size):
        initializer = tf.contrib.layers.xavier_initializer()

        self.input_layer = tf.placeholder(dtype=tf.float32, shape=[None, state_size])

        # Neural net starts here
        hidden_layer = tf.layers.dense(self.input_layer, 8, activation=tf.nn.relu, kernel_initializer=initializer)#hidden layers
        hidden_layer_2 = tf.layers.dense(hidden_layer, 8, activation=tf.nn.relu, kernel_initializer=initializer)

        out = tf.layers.dense(hidden_layer_2, num_actions, activation=None)#output

        self.outputs = tf.nn.softmax(out)
        self.choice = tf.argmax(self.outputs, axis=1)

        self.rewards = tf.placeholder(shape=[None, ], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None, ], dtype=tf.int32)

        one_hot_actions = tf.one_hot(self.actions, num_actions)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=one_hot_actions)
