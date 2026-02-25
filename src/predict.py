from src.preprocess import preprocess_image

def predict_image(model,image):

    img_array = preprocess_image(image)

    prediction = model.predict(img_array)

    prob_dog = float(prediction[0][0])

    prob_cat = 1.0 - prob_dog

    return prob_cat, prob_dog