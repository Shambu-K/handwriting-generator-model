import numpy as np
import matplotlib.pyplot as plt
import os
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

def get_strokes(word_strokes) -> list:
    ''' Get strokes from a word with multiple strokes.'''
    word_strokes[-1, 3] = 1  # Set the EoS flag of the last point to 1
    strokes = []
    start = -1
    for cur in range(word_strokes.shape[0]):
        if word_strokes[cur, 2] == 1:
            start = cur
        if word_strokes[cur, 3] == 1 and start != -1:
            strokes.append(word_strokes[start:cur+1])
            start = -1
            
    return strokes

def plot_word_strokes(word_strokes, image_size, target_img_path, num, color = 'black', split_strokes=True):
    ''' Plot the strokes of a word.'''
    plt.ioff()  # Turn off interactive mode
    dpi = 96
    fig = plt.figure(figsize=(image_size[0]/dpi, image_size[1]/dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')  

    if split_strokes: strokes = get_strokes(word_strokes)
    else: strokes = [word_strokes]
    for stroke in strokes:
        ax.plot(stroke[:, 0], -stroke[:, 1], color=color)
        ax.set_xlim(0, image_size[0])  # Set x-axis limit
        ax.set_ylim(-image_size[1], 0)  # Set y-axis limit
    ax.set_aspect('equal')
    plt.axis('off')
    fig.savefig(target_img_path + f'image_{num}.png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def vertical_padding(image, required_height):
    ''' Add padding to the top and bottom of the image to make it the required height.'''
    image_size = image.size
    difference = required_height - image_size[1]
    padding_bottom = difference // 2
    padding_top = difference - padding_bottom  # This accounts for any odd difference
    # add padding to the top and bottom of the image
    image = ImageOps.expand(image, border=(0, padding_bottom, 0, padding_top), fill='white')
    return image

# main function
def main():
    dir_stroke_path = '../../../DataSet/IAM-Online/Resized_Dataset/Train/Strokes/'
    dir_image_path = '../../../DataSet/IAM-Online/Resized_Dataset/Train/Images/'
    # get the number of files in the directory
    num_files = len([f for f in os.listdir(dir_stroke_path)if os.path.isfile(os.path.join(dir_stroke_path, f))])
    target_image_path = '../../../DataSet/IAM-Online/Resized_Dataset/Train/Images_enhanced/'
    required_height = 60
    for num in range(1, num_files + 1):
        input_image_path = dir_image_path + f'image_{num}.png'
        input_image = Image.open(input_image_path)
        input_size = input_image.size
        # get the strokes of the word
        stroke_path = dir_stroke_path + f'stroke_{num}.npy'
        word_strokes = np.load(stroke_path)
        word_strokes = np.delete(word_strokes, 2, axis=1)
        # plot the strokes of the word and save the image
        plot_word_strokes(word_strokes, input_size, target_image_path, num, color = 'black', split_strokes=True)
        # get the image of the word
        image_path = target_image_path + f'image_{num}.png'
        image = Image.open(image_path)
        # if height of image is less than required height
        # add padding to the image
        if image.size[1] < required_height:
            image = vertical_padding(image, required_height)
        
        if image.size[1] != required_height:
            print(f'Image {num} not padded')
            break
        # save the image
        image.save(image_path)
    print('Done!')

if __name__ == '__main__':
    main()
        