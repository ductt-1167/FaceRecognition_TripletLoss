import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

import get_data
import general

tf.compat.v1.disable_eager_execution()
tf.keras.backend.clear_session()


def triplet_loss(model_anchor, model_positive, model_negative, margin):
    distance1 = tf.sqrt(tf.reduce_sum(tf.pow(model_anchor - model_positive, 2), 1, keepdims=True))
    distance2 = tf.sqrt(tf.reduce_sum(tf.pow(model_anchor - model_negative, 2), 1, keepdims=True))
    return tf.reduce_mean(tf.maximum(distance1 - distance2 + margin, 0))


def build_network(input_shape, embedding_size):
    '''
    Define the neural network to learn image similarity
    Input :
            input_shape : shape of input images
            embeddingsize : vectorsize used to encode our picture
    '''

    # Convolutional Neural Network
    network = Sequential()
    network.add(
        layers.Conv2D(input_shape=input_shape, filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    network.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    network.add(layers.Dropout(0.2))

    network.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    network.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    network.add(layers.Dropout(0.2))

    network.add(layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
    network.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    network.add(layers.Dropout(0.2))

    network.add(layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
    network.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    network.add(layers.Dropout(0.2))

    network.add(layers.Flatten())
    network.add(layers.Dense(units=4096, activation='relu'))
    network.add(layers.Dropout(0.2))
    network.add(layers.Dense(units=1024, activation='relu'))
    network.add(layers.Dropout(0.2))
    network.add(layers.Dense(units=512, activation='relu'))
    network.add(layers.Dropout(0.2))
    network.add(layers.Dense(units=embedding_size))

    network.summary()

    return network


def train_model_with_session():
    img_rows = general.img_row
    img_cols = general.img_col
    input_shape = (img_rows, img_cols, 3)
    margin = 0.5

    network = build_network(input_shape, embedding_size=general.embedding_size)

    # Define the tensors for the three input images
    anchor_input = tf.keras.layers.Input(input_shape, name="anchor_input")
    positive_input = tf.keras.layers.Input(input_shape, name="positive_input")
    negative_input = tf.keras.layers.Input(input_shape, name="negative_input")

    # Generate the encodings (feature vectors) for the three images
    anchor_output = network(anchor_input)
    positive_output = network(positive_input)
    negative_output = network(negative_input)

    # TripletLoss
    loss = triplet_loss(anchor_output, positive_output, negative_output, margin)

    # Setup Optimizer
    learning_rate = 0.0001
    momentum = 0.9    # giống như cấp cho 1 vận tốc ban đầu, thường là 0.9, giải quyết được vấn đề dừng wor local
                        # minium mà k tới global minium
    batch_size = 64
    train_iter = 200
    step = 50

    global_step = tf.Variable(0, trainable=False)

    train_step = tf.compat.v1.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True) \
        .minimize(loss, global_step=global_step)

    # Start Training
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        # Setup Tensorboard
        tf.summary.scalar('step', global_step)
        tf.summary.scalar('loss', loss)
        for var in tf.compat.v1.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        # Train iter
        for i in range(train_iter):
            batch_anchor, batch_positive, batch_negative = get_data.get_triplets_batch(batch_size)

            _, loss_train = sess.run([train_step, loss],
                                     feed_dict={anchor_input: batch_anchor, positive_input: batch_positive,
                                                negative_input: batch_negative})

            print("\r#%d - Loss" % i, loss_train)

            if (i + 1) % step == 0:
                saver.save(sess, general.model_triplet_path)
        saver.save(sess, general.model_triplet_path)
    print('Training completed successfully.')


train_model_with_session()
