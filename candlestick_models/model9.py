'''
Found 585346 images belonging to 2 classes.
Found 390229 images belonging to 2 classes.
Total params: 317,089

18293/18293 [==============================] - 2597s 142ms/step - loss: 0.6166 - accuracy: 0.5604 - val_loss: 0.6136 - val_accuracy: 0.5783
Elapsed time: 0:43:17.668481

'''

import coremlv2 as core
import importlib
importlib.reload(core)
core._init_ml()
core.os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from sklearn.utils import class_weight

###
dataset_path = './db/v5/'
target_size = (64,64)
batch_size = 32
shuffle = True

datagen = core.tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.4)

train_generator = datagen.flow_from_directory(dataset_path, target_size=target_size, color_mode='grayscale', class_mode='binary', batch_size=batch_size, shuffle=shuffle, subset='training')

validation_generator = datagen.flow_from_directory(dataset_path, target_size=target_size, color_mode='grayscale', class_mode='binary', batch_size=batch_size, shuffle=shuffle, subset='validation')

available_classes = [i for i in range(len(core.np.unique(train_generator.classes)))]
class_weights = class_weight.compute_class_weight('balanced', classes=available_classes, y=train_generator.classes)
class_weights_dict = {class_name:class_weight for class_name, class_weight in enumerate(class_weights)}

###
for tg in train_generator:
    x, y = tg
    break
    
###
input_shape = core.tf.keras.Input(shape=x[0].shape)

filters_mult = 0.25

X = core.tf.keras.layers.Conv2D(filters=16*filters_mult, kernel_size=2, strides=(1,1), padding='same')(input_shape)
X = core.tf.keras.layers.Conv2D(filters=16*filters_mult, kernel_size=2, strides=(1,1), padding='same')(X)
X = core.tf.keras.layers.BatchNormalization()(X)
X = core.tf.keras.layers.Activation('relu')(X)
X = core.tf.keras.layers.Conv2D(filters=16*filters_mult, kernel_size=2, strides=(1,1), padding='same')(X)
X = core.tf.keras.layers.Conv2D(filters=16*filters_mult, kernel_size=2, strides=(1,1), padding='same')(X)
X = core.tf.keras.layers.BatchNormalization()(X)
X = core.tf.keras.layers.Activation('relu')(X)
X = core.tf.keras.layers.MaxPool2D(pool_size=(2,2))(X)

X = core.tf.keras.layers.Conv2D(filters=32*filters_mult, kernel_size=2, strides=(1,1), padding='same')(X)
X = core.tf.keras.layers.Conv2D(filters=32*filters_mult, kernel_size=2, strides=(1,1), padding='same')(X)
X = core.tf.keras.layers.BatchNormalization()(X)
X = core.tf.keras.layers.Activation('relu')(X)
X = core.tf.keras.layers.Conv2D(filters=32*filters_mult, kernel_size=2, strides=(1,1), padding='same')(X)
X = core.tf.keras.layers.Conv2D(filters=32*filters_mult, kernel_size=2, strides=(1,1), padding='same')(X)
X = core.tf.keras.layers.BatchNormalization()(X)
X = core.tf.keras.layers.Activation('relu')(X)
X = core.tf.keras.layers.MaxPool2D(pool_size=(2,2))(X)

X = core.tf.keras.layers.Conv2D(filters=64*filters_mult, kernel_size=2, strides=(1,1), padding='same')(X)
X = core.tf.keras.layers.Conv2D(filters=64*filters_mult, kernel_size=2, strides=(1,1), padding='same')(X)
X = core.tf.keras.layers.BatchNormalization()(X)
X = core.tf.keras.layers.Activation('relu')(X)
X = core.tf.keras.layers.Conv2D(filters=64*filters_mult, kernel_size=2, strides=(1,1), padding='same')(X)
X = core.tf.keras.layers.Conv2D(filters=64*filters_mult, kernel_size=2, strides=(1,1), padding='same')(X)
X = core.tf.keras.layers.BatchNormalization()(X)
X = core.tf.keras.layers.Activation('relu')(X)
X = core.tf.keras.layers.MaxPool2D(pool_size=(2,2))(X)

X = core.tf.keras.layers.Conv2D(filters=128*filters_mult, kernel_size=2, strides=(1,1), padding='same')(X)
X = core.tf.keras.layers.Conv2D(filters=128*filters_mult, kernel_size=2, strides=(1,1), padding='same')(X)
X = core.tf.keras.layers.BatchNormalization()(X)
X = core.tf.keras.layers.Activation('relu')(X)
X = core.tf.keras.layers.Conv2D(filters=128*filters_mult, kernel_size=2, strides=(1,1), padding='same')(X)
X = core.tf.keras.layers.Conv2D(filters=128*filters_mult, kernel_size=2, strides=(1,1), padding='same')(X)
X = core.tf.keras.layers.BatchNormalization()(X)
X = core.tf.keras.layers.Activation('relu')(X)
X = core.tf.keras.layers.MaxPool2D(pool_size=(2,2))(X)

X = core.tf.keras.layers.Conv2D(filters=256*filters_mult, kernel_size=2, strides=(1,1), padding='same')(X)
X = core.tf.keras.layers.Conv2D(filters=256*filters_mult, kernel_size=2, strides=(1,1), padding='same')(X)
X = core.tf.keras.layers.BatchNormalization()(X)
X = core.tf.keras.layers.Activation('relu')(X)
X = core.tf.keras.layers.Conv2D(filters=256*filters_mult, kernel_size=2, strides=(1,1), padding='same')(X)
X = core.tf.keras.layers.Conv2D(filters=256*filters_mult, kernel_size=2, strides=(1,1), padding='same')(X)
X = core.tf.keras.layers.BatchNormalization()(X)
X = core.tf.keras.layers.Activation('relu')(X)
X = core.tf.keras.layers.MaxPool2D(pool_size=(2,2))(X)

X = core.tf.keras.layers.Conv2D(filters=512*filters_mult, kernel_size=2, strides=(1,1), padding='same')(X)
X = core.tf.keras.layers.Conv2D(filters=512*filters_mult, kernel_size=2, strides=(1,1), padding='same')(X)
X = core.tf.keras.layers.BatchNormalization()(X)
X = core.tf.keras.layers.Activation('relu')(X)
X = core.tf.keras.layers.Conv2D(filters=512*filters_mult, kernel_size=2, strides=(1,1), padding='same')(X)
X = core.tf.keras.layers.Conv2D(filters=512*filters_mult, kernel_size=2, strides=(1,1), padding='same')(X)
X = core.tf.keras.layers.BatchNormalization()(X)
X = core.tf.keras.layers.Activation('relu')(X)
X = core.tf.keras.layers.MaxPool2D(pool_size=(2,2))(X)

X = core.tf.keras.layers.Flatten()(X)
X = core.tf.keras.layers.Dense(units=64, activation='relu')(X)
outputs = core.tf.keras.layers.Dense(units=1, activation='sigmoid')(X)

model = core.tf.keras.models.Model(inputs=input_shape, outputs=outputs)
optimizer = core.tf.keras.optimizers.Adam()
# loss = core.tf.keras.losses.CategoricalCrossentropy(from_logits=True)
# loss = core.tf.keras.losses.MeanSquaredError()
loss = core.tf.keras.losses.BinaryCrossentropy(from_logits=False)
precision_metrics = core.tf.keras.metrics.Precision()
recall_metrics = core.tf.keras.metrics.Recall()
accuracy_metrics = core.tf.keras.metrics.Accuracy()
# accuracy_metrics, precision_metrics, recall_metrics, 
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

###
model.summary()


###
tick = core.datetime.datetime.now()
model.fit(train_generator, validation_data=validation_generator, epochs=1, class_weight=class_weights_dict)
tock = core.datetime.datetime.now()
print(f'Elapsed time: {tock-tick}')