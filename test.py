import tensorflow as tf

# Import custom layers first
from api.model_definition import (
    ChannelAverage,
    ChannelMaximum,
    SpatialAverage,
    SpatialMaximum
)

model = tf.keras.models.load_model(
    "models/concrete_crack.keras",
    compile=False
)