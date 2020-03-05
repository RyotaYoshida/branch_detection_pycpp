import sys
sys.path.append('D:/Anaconda3/envs/tf_gpu/Lib/site-packages')

from PIL import Image
import numpy as np

def write_image(image):
    back_ground = [0, 0, 0]
    #branch = [150, 70, 20]
    branch = [0, 0, 255]
    leaf = [0, 150, 0]
    fruit = [255, 0, 0]
    tsukene = [255, 64, 0]
    peduncle = [128, 64, 128]
    r = image.copy()
    g = image.copy()
    b = image.copy()
    label_color = np.array([back_ground, branch, fruit, leaf, tsukene, peduncle])
    for l in range(0, 5):
        r[image == l] = label_color[l, 0]
        g[image == l] = label_color[l, 1]
        b[image == l] = label_color[l, 2]
    rgb = np.zeros((image.shape[0], image.shape[1], 3))
    rgb[:,:,0] = r / 1.0
    rgb[:,:,1] = g / 1.0
    rgb[:,:,2] = b / 1.0
    im = Image.fromarray(np.uint8(rgb))
    return im

def mask_image(input_path, predict_img, width=350, height=310):
    predict_img = predict_img.resize((width, height))

    mask = Image.new("L", (width, height), 128)
    input_image = Image.open(input_path)
    input_image = input_image.resize((width, height))
    im = Image.composite(input_image, predict_img, mask)
    return im

def mask(input_img, predict_img, width=350, height=310):
    input_img = input_img.resize((width, height))
    predict_img = predict_img.resize((width, height))

    mask = Image.new("L", (width, height), 128)
    im = Image.composite(input_img, predict_img, mask)
    return im

