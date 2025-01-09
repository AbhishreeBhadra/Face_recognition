from keras.models import Model  # type: ignore
from keras.layers import Input, Flatten, Dense  # type: ignore
from keras.applications.vgg16 import VGG16  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from glob import glob

# Paths to datasets
train_path = 'C:/Users/KIIT/Desktop/Face Recognition/Datasets/Train'

# Preprocessing the data
IMAGE_SIZE = [224, 224]  # VGG16 input size

# Load the VGG16 model without the top layer
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# Freeze the pretrained layers
for layer in vgg.layers:
    layer.trainable = False

# Add custom layers
folders = glob(train_path + '/*')
x = Flatten()(vgg.output)
prediction = Dense(len(folders), activation='softmax')(x)

# Create the model
model = Model(inputs=vgg.input, outputs=prediction)
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
model.summary()

# Data augmentation
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# Training set
training_set = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Train the model
model.fit(
    training_set,
    epochs=20,
    steps_per_epoch=len(training_set)
)

# Save the model
model.save('facefeatures_new_model.h5')
