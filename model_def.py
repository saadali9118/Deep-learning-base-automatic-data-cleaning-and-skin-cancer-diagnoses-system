# model_def.py
# ──────────────────────────────────────────────────────────────────────────────
# Rebuilds the EXACT architecture used in training, so the weights-only file
# (best_model_m2.weights.h5) can be loaded into it.
#
# Must match the classifier build_model() precisely:
#   EfficientNetV2S(include_preprocessing=True) → GAP → Dense(128,relu,l2) →
#   Dropout(0.3) → Dense(1,sigmoid). Backbone called with training=False.
# ──────────────────────────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S

INPUT_SHAPE = (224, 224, 3)


def build_model(weights_path=None):
    """
    Recreate the trained network and (optionally) load fine-tuned weights.
    weights=None on the backbone because our .h5 already contains ALL weights
    (backbone + head) saved during training — no ImageNet download needed.
    """
    backbone = EfficientNetV2S(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights=None,                 # our checkpoint supplies all weights
        include_preprocessing=True,   # model normalizes raw [0,255] internally
    )

    inp = keras.Input(INPUT_SHAPE)
    x   = backbone(inp, training=False)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.Dense(128, activation="relu",
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       name="head_dense1")(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid", dtype="float32", name="head_out")(x)

    model = keras.Model(inp, out)

    if weights_path:
        try:
            model.load_weights(weights_path)
            print(f"[model_def] Loaded weights successfully from {weights_path}")
        except Exception as e:
            print(f"[model_def] Standard weights loading failed: {e}")
            print("[model_def] Attempting to load weights by name (by_name=True)...")
            try:
                model.load_weights(weights_path, by_name=True)
                print("[model_def] Loaded weights successfully using by_name=True")
            except Exception as e_inner:
                print(f"[model_def] Failed to load weights by name: {e_inner}")
                raise e

    return model
