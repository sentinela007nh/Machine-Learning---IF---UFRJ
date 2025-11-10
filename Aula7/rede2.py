import keras

model = keras.Sequential (
    [
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ]
model.compile(adamm="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
