from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout


class FeatureEmbedding:


    def test(self):
        size = (224, 224)
        num_classes = 3

        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=size)
        for layer in base_model.layers:
            layer.trainable = False

        top_model = base_model.output
        top_model = Flatten(name="flatten")(top_model)
        top_model = Dense(1024, activation="relu")(top_model)
        top_model = Dense(1024, activation="relu")(top_model)
        top_model = Dense(512, activation="relu")(top_model)
        top_model = Dense(num_classes, activation="softmax")(top_model)

        model = Model(inputs=base_model.input, outputs=top_model)


        #------------------

        # Classification block
        top_model = GlobalAveragePooling2D(name='AvgPool')(top_model)
        top_model = Dropout(0.2, name='Dropout')(top_model)
        # Bottleneck
        top_model = Dense(128, use_bias=False, name='Bottleneck')(top_model)
        #top_model = BatchNormalization(momentum=0.995, epsilon=0.001, scale=False, name='Bottleneck_BatchNorm')(x)
        #-----------

        x = inc_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        predictions = Dense(2, activation='softmax')(x)


@classmethod
    def get_embeddings(cls, face_pixels: ndarray):
        # convert into an array of samples
        samples = np.expand_dims(face_pixels, axis=0)
        samples = np.asarray(samples, 'float32')
        # prepare the face for the model, e.g. center pixels
        samples = preprocess_input(samples)
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        top_model = base_model.output
        top_model = GlobalAveragePooling2D()(top_model)
        top_model = Flatten(name="flatten")(top_model)

        top_model = Dense(1024, activation="relu")(top_model)
        top_model = Dense(512, activation="relu")(top_model)
        top_model = Dense(128, activation='relu')(top_model)

        model = Model(inputs=base_model.input, outputs=top_model)

        # perform prediction
        yhat = model.predict(samples)
        return yhat


if __name__ == '__main__':
    feature_extractor = FeatureEmbedding()
    feature_extractor.test()