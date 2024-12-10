from generator import Generator 
from Discriminator import Discriminator
import numpy as np
from os import listdir
import os
from PIL import Image
import random
from matplotlib import pyplot as plt
import sys
import json
import datetime 
import time
generational_images = []

# np.set_printoptions(threshold=sys.maxsize)

def parseData():
    print("here in parse data")
    
    folder = './bin/trees'
    data = []
    for images in os.listdir(folder):
        print(f"In first loop ==> {images}")
        if(images.endswith('.png')):
            img = Image.open(folder+'/'+images)
            img = img.convert('RGB')
            # print(np.asarray(img))
            arr = np.asarray(img)
            if(arr.shape == (32,32,3)):
                # arr = arr.flatten()
                arr = arr/255
                data.append(arr)
            
            # print(arr.shape)
            # exit()
                # print("THIS IS GRAY SCALE" + str(arr.shape))
                
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
        "Discriminator_bias" : D.bias.tolist(),
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
    amount_of_nodes = 32
    data = load_data()
    if(data != None):
        Gweight = np.array(data['Generator_weights'])
        Gbias= np.array(data['Generator_bias'])
        Dweight = np.array(data['Discriminator_wights'])
        DBias = np.array(data["Discriminator_bias"]) 
        # DBias = data["Discriminator_bias"]
        G = Generator(amount_of_nodes,learning_rate=learning_rate , weights = Gweight , bias = Gbias)
        D = Discriminator(amount_of_nodes,learning_rate=learning_rate , weights = Dweight , bias = DBias)
    else:
        G = Generator(amount_of_nodes,learning_rate=learning_rate)
        D = Discriminator(amount_of_nodes,learning_rate=learning_rate )
    epochs = 10000
    training_data = parseData()
# For the error plot
    errors_discriminator = []
    errors_generator = []
    checks = 1000

    for epoch in range(epochs):
        
        for tree in training_data:
            
            # Update the discriminator weights from the real face
            D.update_from_image(tree)
            # print("Finished Update from Image")
            # Pick a random number to generate a fake face
            z = random.random()
            # print(f"Random Number chosen ==>> {G.z}")
            # Calculate the discriminator error
            
            errors_discriminator.append(sum(D.error_from_image(tree) + D.error_from_noise(z)))
            # print("Errors for discriminator")
            # Calculate the generator error
            errors_generator.append(G.error(z, D))
            # print("Errors for generator")
            # Build a fake face
            noise = G.forward(z)
            # print(f"This is noise in Driver {noise.shape}")
            # print("Found Noise")
            # Update the discriminator weights from the fake face
            D.update_from_noise(noise)
            # print("Finished Update from noise ")
            # Update the generator weights from the fake face
            G.update(z, D)
            # print("Finished Loop")
            # exit()
        print(f"Generation : {epoch}")
        if(epoch % checks == 0 and epoch >= checks):
            print(f"This is epoch ==> {epoch}")
            rand = random.random()
            generational_images.append(G.forward(rand))
            # getImages(G)
    generational_images.append(G.forward(random.random()))
    count = checks
    index = 0
    rows = int(len(generational_images) / 4) + 1 
    fig, axes = plt.subplots(figsize=(10, 10), nrows=rows, ncols=4, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), generational_images):
        ax.set_title(f"Epoch {count}")
        index+=1
        count+=checks
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        #print(f"Image size {img[0].shape} and size {img[0].size}")
        # exit()
        if(img.shape == (32,32,3)):
            im = ax.imshow(1-img.reshape((32,32,3)))  

    save_data(D , G)
    plt.savefig(f'./Notes/rgb_Results/{datetime.datetime.now().strftime("%Y-%m-%d")}_{datetime.datetime.now().strftime("%H-%M-%S")}.png')
    plt.show()
