from PIL import Image
import glob

a = glob.glob('result/case1/*.jpg')


images = []
for img in a :
    temp = Image.open(img)
    images.append(temp)

images[0].save('res.gif', save_all=True, append_images=images, optimize=False , duration=0.00001, fps = 1,loop=0)

print ('finish')