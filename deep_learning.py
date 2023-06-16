import tensorflow as tf
import keras.losses
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, GlobalMaxPooling2D
from keras.applications.vgg16 import VGG16
from keras.models import load_model

def build_model():
    input_layer = Input(shape=(120, 120, 3))
    vgg = VGG16(include_top=False)(input_layer)

    f1 = GlobalMaxPooling2D()(vgg)
    class1 = Dense(2048, activation='relu')(f1)
    class2 = Dense(1,activation='sigmoid')(class1)

    f2 = GlobalMaxPooling2D()(vgg)
    regress1 = Dense(2048, activation='relu')(f2)
    regress2 = Dense(4, activation='sigmoid')(regress1)

    chairtracker = Model(inputs=input_layer, outputs=[class2, regress2])
    return chairtracker

def localization_loss(y_true, yhat):
    delta_coord = tf.reduce_sum(tf.square(y_true[:, :2] - yhat[:, :2]))

    h_true = y_true[:, 3] - y_true[:, 1]
    w_true = y_true[:, 2] - y_true[:, 0]

    h_pred = yhat[:, 3] - yhat[:, 1]
    w_pred = yhat[:, 2] - yhat[:, 0]

    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true - h_pred))

    return delta_coord + delta_size


class FaceTracker(Model):
    def __init__(self, eyetracker, **kwargs):
        super().__init__(**kwargs)
        self.model = eyetracker

    def compile(self, opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs)
        self.closs = classloss
        self.lloss = localizationloss
        self.opt = opt

    def train_step(self, batch, **kwargs):

        X, y = batch

        with tf.GradientTape() as tape:
            classes, coords = self.model(X, training=True)

            batch_classloss = self.closs(y[0], classes)
            batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)

            total_loss = batch_localizationloss + 0.5 * batch_classloss

            grad = tape.gradient(total_loss, self.model.trainable_variables)

        self.opt.apply_gradients(zip(grad, self.model.trainable_variables))

        return {"total_loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_localizationloss}

    def test_step(self, batch, **kwargs):
        X, y = batch
        classes, coords = self.model(X, training=False)

        batch_classloss = self.closs(y[0], classes)
        batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
        total_loss = batch_localizationloss + 0.5 * batch_classloss

        return {"total_loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_localizationloss}


    def call(self, X, **kwargs):
        return self.model(X, **kwargs)


def deep_learning_fun(train, val):

    # vgg = VGG16(include_top=False)
    # print(vgg.summary())
    chairtracker = build_model()

    batches_per_epoch = len(train)
    lr_decay = (1. / 0.75 - 1) / batches_per_epoch
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001, decay=lr_decay)

    classloss = keras.losses.BinaryCrossentropy()
    regressloss = localization_loss

    model = FaceTracker(chairtracker)
    model.compile(opt, classloss, regressloss)

    logdir = 'logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    hist = model.fit(train, epochs=3, validation_data=val, callbacks=[tensorboard_callback])

    chairtracker.save('chairtracker.h5')
    chairtracker = load_model('chairtracker.h5')
    return chairtracker
