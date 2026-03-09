# train_model.py
import os, json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

DATA_DIR = "dataset"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = (224,224)
BATCH = 64
EPOCHS = 15

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH, subset="training")
val_gen = datagen.flow_from_directory(DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH, subset="validation")

base = MobileNetV2(include_top=False, weights="imagenet", input_shape=IMG_SIZE+(3,))
base.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
out = Dense(train_gen.num_classes, activation="softmax")(x)

model = Model(base.input, out)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# Save
model.save(os.path.join(MODEL_DIR, "sign_model.h5"))
inv_map = {v:k for k,v in train_gen.class_indices.items()}  # index->label
with open(os.path.join(MODEL_DIR, "class_indices.json"), "w") as f:
    json.dump(inv_map, f)

print("✅ Model trained and saved to", MODEL_DIR)
