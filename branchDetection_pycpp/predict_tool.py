import sys
sys.path.append('D:/Anaconda3/envs/tf_gpu/Lib/site-packages')
from PIL import Image
import keras
from keras.preprocessing import image
from keras.preprocessing.image import random_rotation, array_to_img, img_to_array
from keras.models import model_from_json

import numpy as np
import image_process
from image_concatenater import ImageConcatenater
import cv2

#import argparse

width = 480
height = 320
model = model_from_json(open('branch_model_99.json').read())
model.load_weights('branch_weights99.h5')
model._make_predict_function() #predictの前に呼んでおくと処理が速くなるらしい

def predict(input_image):
    
    data = []
    data = np.append(data, input_image)
    data = data.reshape(1, height, width, 3)
    
    print("predict")
    predict = model.predict(data)
    print("end")
    predict = np.asarray(predict, dtype = np.uint8)

    #one-hot形式で1がどこにあるか調べてクラス番号を取得
    res = np.argmax(predict[0], axis = 2)
    
    img = image_process.write_image(res)
    #img.show()
    mask = image_process.mask(array_to_img(input_image), img)
    mask.show()

    return res

#config
#input_path = 'D:/dev/PyWork/semantic_segmentation/testdata/color_52113283.jpg'
#input_path = 'test/35.jpg'

#predict tool
#def main():
#    parser = argparse.ArgumentParser(description='枝領域の推論プログラム')
#    parser.add_argument('input_path', help='推論させる画像ファイルのパス')
#    parser.add_argument('-m', '--model', help='モデルファイルのパス', default='unet_random_batch_model/branch_model_99.json')
#    parser.add_argument('-w', '--weight', help='重みファイルのパス', default='unet_random_batch_model/branch_weights_99.h5')
#    args = parser.parse_args() #引数を解析

#    INPUT_PATH = args.input_path
#    MODEL_FILE_PATH = args.model
#    WEIGHT_FILE_PATH = args.weight

#    input_image = Image.open(INPUT_PATH)
#    input_image = input_image.resize((width, height))
#    input_image = np.array(input_image)

#    data = []
#    data = np.append(data, input_image)
#    data = data.reshape(1, height, width, 3)

#    #正規化
#    data = data / 255.

#    #model = model_from_json(open('D:/dev/PyWork/semantic_segmentation/branch_u_model6.json').read())
#    #model.load_weights('D:/dev/PyWork/semantic_segmentation/branch_u_weights6.h5')
#    model = model_from_json(open(MODEL_FILE_PATH).read())
#    model.load_weights(WEIGHT_FILE_PATH)
#    model.compile(loss = 'categorical_crossentropy',
#                  optimizer = 'rmsprop',
#                  metrics = ['accuracy'])
#    #predict = model.predict(data)
#    import time
#    print("predict")
#    t1 = time.time()
#    predict = model.predict(data)
#    t2 = time.time()
#    print("end")
#    print("経過時間:", t2-t1)
#    predict = np.asarray(predict, dtype = np.uint8)

#    #one-hot形式で1がどこにあるか調べてクラス番号を取得
#    res = np.argmax(predict[0], axis = 2)
    
#    img = image_process.write_image(res)
#    img.show()
#    mask = image_process.mask_image(INPUT_PATH, img)
#    mask.show()

#    input_image = array_to_img(input_image)
#    imageConcatenater = ImageConcatenater(input_image)
#    imageConcatenater.append(img)
#    imageConcatenater.append(mask)
#    imageConcatenater.concat_h().save('results/result_random.jpg')

#if __name__ == "__main__":
#    main()
    
