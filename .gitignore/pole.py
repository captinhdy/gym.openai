import gym
from NeuralNetwork import NeuralNetwork
from random import seed
import random
import numpy as np

l_rate = 0.01
epochs= 100;

neuralNetwork = NeuralNetwork();
steps = 500;
inputs = 4;
n_hidden = 15
outputs = 2;

seed(100)
env = gym.make('CartPole-v1')

neuralNetwork.initialize_network(inputs, n_hidden, outputs)
previousReward = 0
stop = False;
trained = False;
winCount = 0;
episodes = 1;
allReward = 0;
mutationChance = 20
while not stop:

    observation = env.reset()
    totalReward = 0;
    moves = [];

    for t in range(steps):
        env.render()
        mutation = random.random() * 100

        action = 0;
        if mutation <= mutationChance:
            action  = random.random() * outputs
            action = int(action)
            print('mutation occured')
        else:
            action = neuralNetwork.predict(observation);
            
           

        observation, reward, done, info = env.step(action)

        tempArray =[];
        tempArray =observation.tolist()
        tempArray.append(action)
        moves.append(tempArray);
        totalReward += reward

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    
    reward = 1;
    
    mean_reward = np.abs((allReward - totalReward) / episodes)

    if totalReward >= 100:
        trained = True;
        mutationChance = 1
    elif totalReward > previousReward:
        reward = 1
        if mutationChance > 1:
            mutationChance = mutationChance - 1
    elif totalReward < previousReward:
        reward = -1
        
    previousReward = totalReward
    allReward = allReward + totalReward;

    if not trained:
        neuralNetwork.train_network(moves, l_rate, epochs, outputs, reward)
    elif trained:
        if totalReward > 195:
            print("trained in {} episodes".format(str(episodes)))
            stop=True;
        else:
            if totalReward > previousReward:
                reward = 2;
                neuralNetwork.train_network(moves, l_rate, epochs, outputs, reward)


    print(totalReward)
    episodes = episodes + 1