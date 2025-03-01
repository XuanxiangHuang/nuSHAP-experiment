# this is the code from
# https://shap.github.io/shap/notebooks/deep_explainer/Front%20Page%20DeepExplainer%20MNIST%20Example.html
import numpy as np
import pandas as pd
import torch
import shap
import warnings
import time

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# Ignore FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
np.random.seed(119)
NUM_ROWS = 20
batch_size = 128
num_classes = 10
epochs = 12
device = torch.device("cpu")

# input image dimensions
img_rows, img_cols = 28, 28

collected_X = np.empty((0, 28, 28, 1))
collected_y = np.empty((0,), dtype=np.int32)


class CustomSequential(Sequential):
    def predict(self, x, batch_size=None, verbose=0, steps=None, callbacks=None, **kwargs):
        """
        Predict the given inputs and collect them globally.

        :param x: Input data (can be a batch).
        :return: Predictions for the input data.
        """
        global collected_X, collected_y

        # Call the original predict function to get predictions
        pred_prob = super().predict(x, batch_size=batch_size, verbose=verbose, steps=steps, callbacks=callbacks,
                                    **kwargs)

        # Determine predicted labels (using argmax for multi-class classification)
        pred = np.argmax(pred_prob, axis=1)  # pred is now a vector of predicted classes

        # Ensure input data (x) is in the correct shape for stacking
        if x.ndim > 3:  # Batch of inputs
            collected_X = np.vstack((collected_X, x))  # Stack the batch of inputs
        else:  # Single input
            collected_X = np.vstack((collected_X, np.expand_dims(x, axis=0)))  # Add single input to collected_X

        # Stack predictions into collected_y
        collected_y = np.hstack((collected_y, pred))  # Stack predictions

        # Remove duplicate inputs
        combined_Xy = np.hstack((collected_X.reshape(collected_X.shape[0], -1),
                                 collected_y.reshape(-1, 1)))  # Flatten inputs for unique checking
        unique_Xy = np.unique(combined_Xy, axis=0)

        # Update global variables with unique values
        collected_X = unique_Xy[:, :-1].reshape(-1, 28, 28, 1)  # Reshape the unique inputs back to original shape
        collected_y = unique_Xy[:, -1].astype(np.int32)  # Ensure correct dtype for collected_y

        return pred_prob


if __name__ == '__main__':
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Replace the original model with the customized one
    model = CustomSequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    # Train the model as usual
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    data_feature_names = [f'pixel_{i}_{j}' for i in range(28) for j in range(28)]

    background = x_train[np.random.choice(x_train.shape[0], 5000, replace=False)]
    f = lambda in_x: model.predict(
        in_x.reshape(-1, 28, 28, 1) if in_x.size % (28 * 28) == 0 else ValueError("Invalid input size")
    )

    background_flattened = background.reshape(background.shape[0], -1)

    explainer = shap.SamplingExplainer(model=f, data=background_flattened, feature_names=data_feature_names)

    all_shap_scs = []
    all_shap_time = []
    all_nushap_time = []
    all_nushap_samples = []

    for i in range(NUM_ROWS):
        # explain first NUM_ROWS test instances
        x = np.expand_dims(x_test[i], axis=0)
        pred_prob = model.predict(x)
        pred = pred_prob.argmax(axis=1)
        print(f"{i}-th data point, prediction: {pred[0]}")
        time_shap_start = time.perf_counter()
        x_flat = x.flatten()
        shap_sc = explainer(x_flat)
        shap_values_pred_class = shap_sc.values[:, pred[0]]
        print("SHAP scores: ", shap_values_pred_class)

        assert shap_values_pred_class.shape == (784,), "The shape is not (784,)!"
        assert shap_values_pred_class.ndim == 1, "The array is not 1-D!"

        all_shap_scs.append(shap_values_pred_class)
        time_shap_end = time.perf_counter()
        all_shap_time.append(np.round(time_shap_end - time_shap_start, 4))
        print("SHAP runtime: ", np.round(time_shap_end - time_shap_start, 4))

        # collect test instance and samples
        x_df = pd.DataFrame([x_flat], columns=data_feature_names)
        y_df = pd.DataFrame([pred[0]], columns=["target"])
        inst_df = pd.concat([x_df, y_df], axis=1)
        inst_df.to_csv(f"nushap_samples/mnist/shap-inst-{i}.csv", index=False)

        # Flatten collected_X from (n_samples, 28, 28, 1) to (n_samples, 28*28)
        assert collected_X.ndim == 4, "Input array is not 4-dimensional!"
        assert collected_X.shape[1:] == (28, 28, 1), "Shape mismatch: Expected (n_samples, 28, 28, 1)"

        flattened_X = collected_X.reshape(collected_X.shape[0], -1)

        # Assert the shape after flattening
        assert flattened_X.ndim == 2, "Flattened array is not 2-dimensional!"
        assert flattened_X.shape[1] == 28 * 28, f"Shape mismatch: Expected (n_samples, {28 * 28})"

        X_df = pd.DataFrame(flattened_X, columns=data_feature_names)
        Y_df = pd.DataFrame(collected_y, columns=["target"])
        samples_df = pd.concat([X_df, Y_df], axis=1)
        samples_df.to_csv(f"nushap_samples/mnist/shap-sample-{i}.csv", index=False)
        # Reset collected data after saving
        collected_X = np.empty((0, 28, 28, 1))
        collected_y = np.empty((0,), dtype=np.int32)
        all_nushap_samples.append(samples_df.shape[0])
        all_nushap_time.append(-1)

    header_line = ",".join(data_feature_names)
    header_line = header_line.lstrip("#")
    np.savetxt(f"results/shap/mnist.csv", all_shap_scs,
               delimiter=",", header=header_line, comments="", fmt='%.4f')

    runtime_df_shap = pd.DataFrame(all_shap_time, columns=["runtime"])
    runtime_df_shap.to_csv(f"runtime_res/shap/mnist.csv", index=False)
    runtime_df_nushap = pd.DataFrame({
        "runtime": all_nushap_time,
        "n_samples": all_nushap_samples
    })
    runtime_df_nushap.to_csv(f"runtime_res/nushap/mnist.csv", index=False)
