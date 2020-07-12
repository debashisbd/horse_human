import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import zipfile
import urllib.request
from google.colab import files
from keras.preprocessing import image


zip_url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip"
urllib.request.urlretrieve(zip_url,'/tmp/horse-or-human.zip')
local_zip= '/tmp/horse-or-human.zip'
zip_ref= zipfile.ZipFile(local_zip,'r')
zip_ref.extractall('/tmp/horse-or-human')
zip_ref.close()



zip_url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip"
urllib.request.urlretrieve(zip_url,'/tmp/validation-horse-or-human.zip')
local_zip= '/tmp/validation-horse-or-human.zip'
zip_ref= zipfile.ZipFile(local_zip,'r')
zip_ref.extractall('/tmp/validation-horse-or-human')
zip_ref.close()





train_horse_dir = os.path.join('/tmp/horse-or-human/horses')
train_human_dir = os.path.join('/tmp/horse-or-human/humans')

validation_horse_dir = os.path.join('/tmp/validation-horse-or-human/horses')
validation_human_dir = os.path.join('/tmp/validation-horse-or-human/humans')


train_horse_names = os.listdir(train_horse_dir)
print(train_horse_names[:10])


train_human_names = os.listdir(train_human_dir)
print(train_human_names[:10])


validation_horse_names = os.listdir(validation_horse_dir)
print(validation_horse_names[:10])


validation_human_names = os.listdir(validation_human_dir)
print(validation_human_names[:10])


print('Total training horse images:',len(os.listdir(train_horse_dir)))
print('Total training human images:',len(os.listdir(train_human_dir)))

print('Total validation horse images:',len(os.listdir(validation_horse_dir)))
print('Total validation human images:',len(os.listdir(validation_human_dir)))

nrows = 4
ncols = 4

pic_index = 0

fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)

pic_index += 8
next_horse_pix = [os.path.join(train_horse_dir, fname)
                  for fname in train_horse_names[pic_index-8:pic_index]]

next_human_pix = [os.path.join(train_human_dir, fname)
                  for fname in train_human_names[pic_index-8:pic_index]]

for i, img_path in enumerate(next_horse_pix+next_human_pix):
    sp = plt.subplot(nrows, ncols, i+1)
    sp.axis('Off')
    img = mpimg.imread(img_path)
    plt.imshow(img)
    
plt.show()



#Model Declear

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3), activation = 'relu', input_shape=(300,300,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(32,(3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(64,(3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(64,(3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(64,(3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(512, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid'),   
    
    ])


model.summary()


model.compile(loss='binary_crossentropy',
              optimizer = RMSprop(lr = 0.001),
              metrics=['acc'])



train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)


train_generator= train_datagen.flow_from_directory('/tmp/horse-or-human',
                                                  target_size=(300,300),
                                                  batch_size=128,
                                                  class_mode='binary')

validation_generator= validation_datagen.flow_from_directory('/tmp/validation-horse-or-human',
                                                  target_size=(300,300),
                                                  batch_size=128,
                                                  class_mode='binary')


history = model.fit_generator(
    train_generator,
    steps_per_epoch=8,
    epochs=15,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=8)





uploaded = files.upload()

for fn in uploaded.keys():
    path = '/content/'+fn
    img = image.load_image(path, target_size=(300,300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    print (classes[0])
    
    if classes[0]>0.5:
        print(fn + 'is a human')
    else:
        print(fn + 'is a horse')







