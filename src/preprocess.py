import numpy as np

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize((256,256))

    img_array = np.array(image)

    img_array = img_array.reshape(1,256,256,3)

    return img_array