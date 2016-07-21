from keras.layers import merge, Convolution2D, MaxPooling2D, Input, UpSampling2D, ZeroPadding2D
from keras.models import Model
from keras.utils.visualize_util import model_to_dot, plot

## Create FCN model ##

FCN_CLASSES = 21

input_img = Input(shape=(3, 224, 224))
#(3*224*224)
x = ZeroPadding2D(padding=(1, 1))(input_img)
x = Convolution2D(64, 3, 3, activation='relu')(x)
x = ZeroPadding2D(padding=(1, 1))(x)
x = Convolution2D(64, 3, 3, activation='relu')(x)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)
#(64*112*112)
x = ZeroPadding2D(padding=(1, 1))(x)
x = Convolution2D(128, 3, 3, activation='relu')(x)
x = ZeroPadding2D(padding=(1, 1))(x)
x = Convolution2D(128, 3, 3, activation='relu')(x)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)
#(128*56*56)
x = ZeroPadding2D(padding=(1, 1))(x)
x = Convolution2D(256, 3, 3, activation='relu')(x)
x = ZeroPadding2D(padding=(1, 1))(x)
x = Convolution2D(256, 3, 3, activation='relu')(x)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)
#(256*28*28)

#split layer
p3 = x
p3 = Convolution2D(FCN_CLASSES, 1, 1)(p3)
#(21*28*28)

x = ZeroPadding2D(padding=(1, 1))(x)
x = Convolution2D(512, 3, 3, activation='relu')(x)
x = ZeroPadding2D(padding=(1, 1))(x)
x = Convolution2D(512, 3, 3, activation='relu')(x)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)
#(512*14*14)

#split layer
p4 = x
p4 = Convolution2D(FCN_CLASSES, 1, 1)(p4)
u4 = UpSampling2D(size=(2,2))(p4)
#(21*28*28)

x = ZeroPadding2D(padding=(1, 1))(x)
x = Convolution2D(512, 3, 3, activation='relu')(x)
x = ZeroPadding2D(padding=(1, 1))(x)
x = Convolution2D(512, 3, 3, activation='relu')(x)
x = MaxPooling2D((2, 2), strides=(2, 2))(x)
#(512*7*7)

p5 = x
p5 = Convolution2D(FCN_CLASSES, 1, 1)(p5)
u5 = UpSampling2D(size=(4,4))(p5)
#(21*28*28)

# merge scores
# "merge" sums vecters as default
x = merge([p3, u4, u5], mode='concat', concat_axis=1)
x = Convolution2D(21,1,1)(x)

out = UpSampling2D(size=(8,8))(x)
#(21*224*224)

model = Model(input_img, out)
# visualize model
plot(model, to_file='FCN_model.png',show_shapes=True)


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

## Train ##
batch_size = 10
nb_epoch = 100

result = model.fit(X_train, Y_train,
                   batch_size=batch_size,
                   nb_epoch=nb_epoch,
                   verbose=1,
                   validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

## make graph
x = range(nb_epoch)
plt.plot(x, result.history['acc'], label="train acc")
plt.plot(x, result.history['val_acc'], label="val acc")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

plt.plot(x, result.history['loss'], label="train loss")
plt.plot(x, result.history['val_loss'], label="val loss")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
