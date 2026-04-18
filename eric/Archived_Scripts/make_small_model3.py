#!/usr/bin/env python3
"""
Build, train (30 epochs), and save a small-node-count quantized Model3 H5.

Run with:
    conda activate mlgpu_qkeras
    python make_small_model3.py

Architecture (HLS-friendly sizes):
  - Conv2D: 4 filters, 3x3 kernel → MaxPool(2,2) → Flatten → 240 nodes
  - Scalar branch: nModule + x_local + y_local (3) → Dense(8)
  - Concatenate: 240 + 8 = 248
  - Dense(32) → Dense(16) → Dense(1) → quantized_tanh
  - 6-bit weights/biases, 8-bit activations (QKeras)

Output: model3_small_6w0i.h5
"""

import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.layers import (Input, Concatenate, MaxPooling2D,
                                     Flatten, Reshape)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

EarlyStopping   = tf.keras.callbacks.EarlyStopping    # avoid unresolved-import warning
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint

try:
    from qkeras import QDense, QActivation, QConv2D
    from qkeras.quantizers import quantized_bits, quantized_relu
except ImportError:
    sys.exit("ERROR: QKeras not found — activate the mlgpu_qkeras conda env first.")

# Use the full-featured ODG that supports nModule and x_local
sys.path.insert(0, '/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/MuC_Smartpix_Data_Production/tfRecords/')
import OptimizedDataGenerator4_data_shuffled_bigData_NewFormat as ODG

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TF_RECORDS_DIR = (
    "/local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_2026Feb/"
    "TF_Records/filtering_records16384_data_shuffled_single_bigData"
)
BATCH_SIZE     = 16384
NUM_EPOCHS     = 30

CONV_FILTERS   = 4
KERNEL_SIZE    = (3, 3)
SCALAR_UNITS   = 8
MERGED_DENSE_1 = 32
MERGED_DENSE_2 = 16

WEIGHT_BITS    = 6
INT_BITS       = 0
ACT_BITS       = 8

OUTPUT_H5      = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "model3_small_6w0i.h5")

# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------
def build_model():
    weight_q = quantized_bits(WEIGHT_BITS, INT_BITS, alpha=1.0)
    bias_q   = quantized_bits(WEIGHT_BITS, INT_BITS, alpha=1.0)
    act_q    = quantized_relu(ACT_BITS, 0)

    cluster_input = Input(shape=(13, 21), name="cluster")
    nmodule_input = Input(shape=(1,),     name="nModule")
    x_local_input = Input(shape=(1,),     name="x_local")
    y_local_input = Input(shape=(1,),     name="y_local")

    # Conv2D branch
    x = Reshape((13, 21, 1), name="add_channel")(cluster_input)
    x = QConv2D(
        filters=CONV_FILTERS, kernel_size=KERNEL_SIZE, padding="same",
        kernel_quantizer=weight_q, bias_quantizer=bias_q, name="conv2d"
    )(x)
    x = QActivation(act_q, name="conv2d_act")(x)
    x = MaxPooling2D((2, 2), name="pool2d")(x)
    x = Flatten(name="flatten")(x)   # → 60 * CONV_FILTERS = 240 nodes

    # Scalar branch: pairwise concatenation (hls4ml only supports 2-tensor merges)
    s = Concatenate(name="concat_scalars_1")([nmodule_input, x_local_input])
    s = Concatenate(name="concat_scalars_2")([s, y_local_input])
    s = QDense(SCALAR_UNITS, kernel_quantizer=weight_q, bias_quantizer=bias_q,
               name="dense_scalars")(s)
    s = QActivation(act_q, name="dense_scalars_act")(s)

    # Merge
    m = Concatenate(name="concat_all")([x, s])

    # Head
    m = QDense(MERGED_DENSE_1, kernel_quantizer=weight_q, bias_quantizer=bias_q,
               name="merged_dense1")(m)
    m = QActivation(act_q, name="merged_dense1_act")(m)

    m = QDense(MERGED_DENSE_2, kernel_quantizer=weight_q, bias_quantizer=bias_q,
               name="merged_dense2")(m)
    m = QActivation(act_q, name="merged_dense2_act")(m)

    out = QDense(1, kernel_quantizer=weight_q, bias_quantizer=bias_q,
                 name="output_dense")(m)
    out = QActivation("quantized_tanh", name="output")(out)

    model = Model(
        inputs=[cluster_input, nmodule_input, x_local_input, y_local_input],
        outputs=out,
        name="model3_small_6w0i"
    )
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["binary_accuracy"]
    )
    return model


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data():
    train_dir = f"{TF_RECORDS_DIR}/tfrecords_train/"
    val_dir   = f"{TF_RECORDS_DIR}/tfrecords_validation/"

    print(f"Loading training data from:   {train_dir}")
    print(f"Loading validation data from: {val_dir}")

    features = ['cluster', 'y_local', 'nModule', 'x_local']
    timestamps = [19]   # last timestamp only, matching Model3

    train_gen = ODG.OptimizedDataGeneratorDataShuffledBigData(
        load_records=True,
        tf_records_dir=train_dir,
        x_feature_description=features,
        time_stamps=timestamps,
        batch_size=BATCH_SIZE,
    )
    val_gen = ODG.OptimizedDataGeneratorDataShuffledBigData(
        load_records=True,
        tf_records_dir=val_dir,
        x_feature_description=features,
        time_stamps=timestamps,
        batch_size=BATCH_SIZE,
    )

    print(f"Training batches : {len(train_gen)}")
    print(f"Validation batches: {len(val_gen)}")
    return train_gen, val_gen


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Building small Model3 (4 conv filters, 32/16 dense)")
    print("=" * 60)

    model = build_model()
    model.summary()

    conv_flat   = 60 * CONV_FILTERS
    concat_size = conv_flat + SCALAR_UNITS
    print(f"\nNode counts per stage:")
    print(f"  Conv2D flatten  : {conv_flat}")
    print(f"  Scalar dense    : {SCALAR_UNITS}")
    print(f"  Concat (merge)  : {concat_size}")
    print(f"  merged_dense1   : {MERGED_DENSE_1}")
    print(f"  merged_dense2   : {MERGED_DENSE_2}")
    print(f"  output          : 1")
    print(f"\nTotal parameters: {model.count_params():,}")

    print("\nLoading TFRecord data...")
    train_gen, val_gen = load_data()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True,
                      verbose=1),
        ModelCheckpoint(OUTPUT_H5, monitor="val_loss", save_best_only=True,
                        verbose=1),
    ]

    print(f"\nTraining for up to {NUM_EPOCHS} epochs (early stop patience=5)...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=NUM_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # ModelCheckpoint already saved best weights; save once more to be sure
    model.save(OUTPUT_H5)
    print(f"\nSaved: {OUTPUT_H5}")
