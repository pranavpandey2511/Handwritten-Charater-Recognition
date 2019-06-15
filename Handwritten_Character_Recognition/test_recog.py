from keras.models import model_from_json
import train_model_char

def load_model():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model

model = load_model()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.predict(X_test[0])