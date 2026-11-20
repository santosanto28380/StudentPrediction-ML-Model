import pickle
import numpy as np
with open("models/student_model.pkl", "rb") as file:
    model = pickle.load(file)

print("Model loaded successfully!")
data = input("Enter input values separated by comma: ")

input_data = np.array(
    [float(x) for x in data.split(',')]
).reshape(1, -1)
prediction = model.predict(input_data)

print("Predicted Output:", prediction[0])