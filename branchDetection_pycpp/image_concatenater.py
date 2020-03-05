from PIL import Image
import numpy as np

class ImageConcatenater():
    def __init__(self, img):
        #結合する画像を格納するリスト
        self.images = []
        self.images.append(img)
        self.width = img.width
        self.height = img.height
        self.num = 1

    #画像を追加する
    def append(self, img):
        self.images.append(img)
        self.num += 1

    #結合画像generator
    def concat_h(self):
        width = self.width * self.num
        dst = Image.new('RGB', (width, self.height))
        for i in range(self.num):
            dst.paste(self.images[i], (i * self.width, 0))

        return dst

if __name__ == "__main__":
    img1 = Image.open('data/train/80.jpg')
    img2 = Image.open('data/teach/80.png') 
    imageConcatenator = ImageConcatenater(img1)
    imageConcatenator.append(img2)
    imageConcatenator.concat_h().save('results/labeling.png')