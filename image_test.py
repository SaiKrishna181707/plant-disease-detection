from PIL import Image
import numpy as np

img = Image.open("dataset/leaf.jpg")

img = img.resize((128,128))

arr = np.array(img)

print("Before:")
print(arr[0][0])

arr = arr / 255.0

print("After:")
print(arr[0][0])