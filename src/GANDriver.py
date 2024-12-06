from generator import Generator 
from Discriminator import Discriminator
import numpy as np
import kagglehub as kaggle
from os import listdir
import os
from PIL import Image
import random
from matplotlib import pyplot as plt
import sys
import json
generational_images = []

# np.set_printoptions(threshold=sys.maxsize)

def parseData():
    folder = './bin/trees'
    data = []
    for images in os.listdir(folder):
        if(images.endswith('.png')):
            img = Image.open(folder+'/'+images).convert('L')
            # print(np.asarray(img))
            arr = np.asarray(img)
            if(arr.shape == (32,32)):
                arr = arr.flatten()
                arr = arr/255
                data.append(arr)

                # print(np.asarray(img))
                # exit()
                print("THIS IS GRAY SCALE" + str(arr.shape))
                
    return data


def view_samples(samples, m, n):
    fig, axes = plt.subplots(figsize=(10, 10), nrows=m, ncols=n, sharey=True, sharex=True)
    generational_images.append(samples[0])
    for ax, img in zip(axes.flatten(), samples):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    #     im = ax.imshow(1-img.reshape((32,32)), cmap='Greys_r')  
    # plt.show()
    return fig, axes


def getImages(G : Generator):
    generated_images = []
    for i in range(4):
        G.z = random.random()
        generated_image = G.forward(G.z)
        generated_images.append(generated_image)
    _ = view_samples(generated_images, 1, 4)
    for i in generated_images:
        print(i)

def save_data(D : Discriminator , G : Generator):
    data = {"Discriminator_wights" : D.weights.tolist(),
        "Discriminator_bias" : D.bias,
        "Generator_weights" : G.weights.tolist(),
        "Generator_bias" : G.biases.tolist(),
        }
    with open('./bin/weights.json' , "w")as file:
        json.dump(data , file)

def load_data():
    with open('./bin/weights.json' , "r")as file:
        data = json.load(file)
    if(len(data) > 0):
        return data
    else:
        return None


if __name__ == '__main__':
    np.random.seed(42)
    learning_rate = 0.1
    amount_of_nodes = 1024
    data = load_data()
    if(data != None):
        Gweight = np.array(data['Generator_weights'])
        Gbias= np.array(data['Generator_bias'])
        Dweight = np.array(data['Discriminator_wights'])
        G = Generator(amount_of_nodes,learning_rate=learning_rate , weights = Gweight , bias = Gbias)
        D = Discriminator(amount_of_nodes,learning_rate=learning_rate , weights = Dweight , bias = data["Discriminator_bias"])
    else:
        G = Generator(amount_of_nodes,learning_rate=learning_rate)
        D = Discriminator(amount_of_nodes,learning_rate=learning_rate )
    epochs = 200000
    training_data = parseData()
# For the error plot
    errors_discriminator = []
    errors_generator = []
    checks = 20000
    for epoch in range(epochs):
        
        for tree in training_data:
            
            # Update the discriminator weights from the real face
            D.update_from_image(tree)
            # print("Finished Update from Image")
            # Pick a random number to generate a fake face
            G.z = random.random()
            # print(f"Random Number chosen ==>> {G.z}")
            # Calculate the discriminator error
            errors_discriminator.append(sum(D.error_from_image(tree) + D.error_from_noise(G.z)))
            # print("Errors for discriminator")
            # Calculate the generator error
            errors_generator.append(G.error(G.z, D))
            # print("Errors for generator")
            # Build a fake face
            noise = G.forward(G.z)
            # print("Found Noise")
            # Update the discriminator weights from the fake face
            D.update_from_noise(noise)
            # print("Finished Update from noise ")
            # Update the generator weights from the fake face
            G.update(G.z, D)
            # print("Finished Loop")
            # exit()
        print(f"Generation : {epoch}")
        if(epoch % checks == 0 and epoch >= checks):
            print(f"This is epoch ==> {epoch}")
            generational_images.append(G.forward(random.random()))
            # getImages(G)
    generational_images.append(G.forward(random.random()))
    count = checks
    rows = int(len(generational_images) / 4) + 1 
    fig, axes = plt.subplots(figsize=(10, 10), nrows=rows, ncols=4, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), generational_images):
        ax.set_title(f"Epoch {count}")
        count+=checks
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(1-img.reshape((32,32)), cmap='Greys_r')  

    save_data(D , G)
    plt.show()
