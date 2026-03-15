import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    "DATASET/TRAIN",
    target_size=(224,224),
    batch_size=32,
    class_mode="binary"
)

val_data = val_gen.flow_from_directory(
    "DATASET/VALIDATION",
    target_size=(224,224),
    batch_size=32,
    class_mode="binary"
)

model = Sequential([
    Input(shape=(224,224,3)),

    Conv2D(32,(3,3),activation="relu"),
    MaxPooling2D((2,2)),

    Conv2D(64,(3,3),activation="relu"),
    MaxPooling2D((2,2)),

    Conv2D(128,(3,3),activation="relu"),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(128,activation="relu"),
    Dense(1,activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)


plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train","Validation"])

plt.show()