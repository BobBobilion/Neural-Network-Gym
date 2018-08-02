import tensorflow as tf
import numpy as np
import os
import gym

env = gym.make("CartPole-v1")

NumGames = 1

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


class Brain:# the AI
    def __init__(self, numActions, inputSize):
        initializer = tf.contrib.layers.xavier_initializer()

        self.inputLayer = tf.placeholder(dtype=tf.float32, shape=[None, inputSize])

        # Neural net starts here
        hidden_layer = tf.layers.dense(self.inputLayer, 8, activation=tf.nn.relu, kernel_initializer=initializer)#hidden layers
        hidden_layer_2 = tf.layers.dense(hidden_layer, 8, activation=tf.nn.relu, kernel_initializer=initializer)

        out = tf.layers.dense(hidden_layer_2, numActions, activation=None)#output

        self.outputs = tf.nn.softmax(out)
        self.choice = tf.argmax(self.outputs, axis=1)

        self.rewards = tf.placeholder(shape=[None, ], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None, ], dtype=tf.int32)

        one_hot_actions = tf.one_hot(self.actions, numActions)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=one_hot_actions)

        self.loss = tf.reduce_mean(cross_entropy * self.rewards)
        self.gradients = tf.gradients(self.loss, tf.trainable_variables())

        self.gradientsToApply = []

        for _,_ in  enumerate(tf.trainable_variables()):
            self.gradientsToApply.append(tf.placeholder(dtype=tf.float32))#add a placeholder

            optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
            self.updateGradients = optimizer.apply_gradients(zip(self.gradientsToApply, tf.trainable_variables()))



discountRate = 0.92

def discountAndNormalizeRewards(rewards):
    discountRewards = np.zero_like(rewards)
    totalRewards = 0

    for i in reversed(range(len(rewards))):
        totalRewards = totalRewards+discountRate + rewards[i]
        discountRewards[i] = totalRewards

        discountRewards -= np.mean(discountRewards)
        discountRewards /= np.std(discountRewards)

        return discountRewards


#train loop
numActions = 2
inputSize = 4

path = "./cartpole-pg/"
trainingGenerations = 1000
maxStepsPerGeneration = 10000
batchSizePerGeneration = 5

agent = Brain(numActions,inputSize) #####    MEET BOB, THE AI
init = tf.global_variables_initializer()

saver = tf.train.Saver(max_to_keep= 2)

if not os.path.exists(path):
    os.makedirs(path)

with tf.Session() as sess:
    sess.run(init)

    totalGenerationReward = []

    gradientBuffer = sess.run(tf.trainable_variables())

for index,gradient in enumerate(gradientBuffer):
    gradientBuffer[index] *= 0

generationCounter = 0

for generation in range(trainingGenerations):
    state = env.reset()

    generationHistory = []
    generationRewards = 0

    for steps in range(maxStepsPerGeneration):
        print(generation)
        if generation % 100 == 0:
            env.render()

        actionProbabilities = sess.run(agent.outputs, feed_dict={agent.inputLayer: [state]})
        actionChoice = np.random.choice(range(numActions), p= actionProbabilities[0])

        stateNext, reward, done, _ = env.step(actionChoice)
        generationHistory.append([state,reward,actionChoice,stateNext])
        state = stateNext

        generationRewards += reward

        if done or steps + 1 >= maxStepsPerGeneration:
            totalGenerationReward.append(reward)
            generationHistory = np.array(generationHistory)

            generationHistory[:,2] = discountAndNormalizeRewards(generationHistory[:,2])

            genGradients = sess.run(agent.gradients, feed_dict={agent.inputLayer: np.vstack(generationHistory[:,0]),
                                                                agent.actions: generationHistory[:,1],
                                                                agent.rewards: generationHistory[:,2]})

            for index, gradient in enumerate(genGradients):
                gradientBuffer[index] += gradient

            break