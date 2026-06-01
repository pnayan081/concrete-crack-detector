from __future__ import annotations

from typing import Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Input, layers
from tensorflow.keras.applications import DenseNet121


@keras.utils.register_keras_serializable(package="ConcreteCrack")
class ChannelAverage(layers.Layer):
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)


@keras.utils.register_keras_serializable(package="ConcreteCrack")
class ChannelMaximum(layers.Layer):
    def call(self, inputs):
        return tf.reduce_max(inputs, axis=[1, 2], keepdims=True)


@keras.utils.register_keras_serializable(package="ConcreteCrack")
class SpatialAverage(layers.Layer):
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=-1, keepdims=True)


@keras.utils.register_keras_serializable(package="ConcreteCrack")
class SpatialMaximum(layers.Layer):
    def call(self, inputs):
        return tf.reduce_max(inputs, axis=-1, keepdims=True)


def cbam_block(input_tensor, reduction_ratio: int = 8):
    channel_count = input_tensor.shape[-1]
    if channel_count is None:
        raise ValueError("CBAM requires a known channel dimension.")

    average_pool = ChannelAverage()(input_tensor)
    maximum_pool = ChannelMaximum()(input_tensor)
    shared_mlp = keras.Sequential(
        [
            layers.Dense(channel_count // reduction_ratio, activation="relu"),
            layers.Dense(channel_count, activation="sigmoid"),
        ],
        name="cbam_shared_mlp",
    )

    channel_attention = layers.Add()(
        [shared_mlp(average_pool), shared_mlp(maximum_pool)]
    )
    channel_refined = layers.Multiply()([input_tensor, channel_attention])

    spatial_average = SpatialAverage()(channel_refined)
    spatial_maximum = SpatialMaximum()(channel_refined)
    spatial_attention = layers.Concatenate(axis=-1)(
        [spatial_average, spatial_maximum]
    )
    spatial_attention = layers.Conv2D(
        1, kernel_size=7, padding="same", activation="sigmoid"
    )(spatial_attention)
    return layers.Multiply()([channel_refined, spatial_attention])


def build_enhanced_densenet(
    input_shape: tuple[int, int, int] = (64, 64, 3),
    num_classes: int = 1,
    weights: Optional[str] = "imagenet",
) -> Model:
    """Build the notebook's DenseNet121 + CBAM binary classifier."""
    base_model = DenseNet121(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
    )
    base_model.trainable = False

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = cbam_block(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="sigmoid")(x)
    return Model(inputs, outputs, name="concrete_crack_enhanced_densenet")
