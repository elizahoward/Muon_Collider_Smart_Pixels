"""
Model3 with Quantized Inputs

Subclasses Model3 and overrides the quantized model-building methods so that
every raw input tensor is passed through a QActivation(quantized_bits) layer
before any Conv/Dense computation.

New constructor parameters
--------------------------
input_bits      : total bits for input quantization (default 8)
input_int_bits  : integer bits for input quantization (default 0, purely fractional)
hp_input_bits_min / max / step  : HP search bounds for input_bits
hp_input_int_bits_min / max / step : HP search bounds for input_int_bits

Both makeQuantizedModel and makeQuantizedModelHyperParameterTuning are covered.
The tanh output is also updated to QActivation("quantized_tanh(8,0)") followed
by a Lambda rescale from [-1, 1] → [0, 1], matching the Model2.5_QuantizedInputs
convention.

Author: Eric
Date: 2026
"""

import sys
sys.path.append('/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/MuC_Smartpix_ML/')
sys.path.append('../MuC_Smartpix_ML/')

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Concatenate, Dropout, Conv2D, MaxPooling2D, Flatten, Reshape
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from model3 import Model3

try:
    from qkeras import QDense, QActivation, QConv2D
    from qkeras.quantizers import quantized_bits, quantized_relu
    QKERAS_AVAILABLE = True
except ImportError:
    print("QKeras not available. Please install with: pip install qkeras")
    QKERAS_AVAILABLE = False


class Model3_QuantizedInputs(Model3):
    """
    Model3 variant where every input is quantized via QActivation before any
    downstream Conv2D / Dense computation.  Tanh output is also rescaled from
    [-1, 1] to [0, 1] via Lambda((x + 1) / 2), matching Model2.5_QuantizedInputs.
    """

    def __init__(self,
                 # ── passthrough to Model3 ────────────────────────────────────
                 tfRecordFolder: str = "/local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_2026Feb/TF_Records/filtering_records16384_data_shuffled_single_bigData",
                 conv_filters: int = 8,
                 kernel_rows: int = 3,
                 kernel_cols: int = 3,
                 scalar_dense_units: int = 32,
                 merged_dense_1: int = 200,
                 merged_dense_2: int = 100,
                 dropout_rate: float = 0.1,
                 initial_lr: float = 0.000871145,
                 end_lr: float = 5.3e-05,
                 power: int = 2,
                 # ── input quantization ────────────────────────────────────────
                 input_bits: int = 8,
                 input_int_bits: int = 0,
                 # ── HP search bounds for input precision ─────────────────────
                 hp_input_bits_min: int = 4,
                 hp_input_bits_max: int = 10,
                 hp_input_bits_step: int = 2,
                 hp_input_int_bits_min: int = 0,
                 hp_input_int_bits_max: int = 0,
                 hp_input_int_bits_step: int = 1):

        super().__init__(
            tfRecordFolder=tfRecordFolder,
            conv_filters=conv_filters,
            kernel_rows=kernel_rows,
            kernel_cols=kernel_cols,
            scalar_dense_units=scalar_dense_units,
            merged_dense_1=merged_dense_1,
            merged_dense_2=merged_dense_2,
            dropout_rate=dropout_rate,
            initial_lr=initial_lr,
            end_lr=end_lr,
            power=power,
        )
        self.modelName = "Model3_QuantizedInputs"

        self.input_bits = input_bits
        self.input_int_bits = input_int_bits

        self.hp_input_bits_min  = hp_input_bits_min
        self.hp_input_bits_max  = hp_input_bits_max
        self.hp_input_bits_step = hp_input_bits_step
        self.hp_input_int_bits_min  = hp_input_int_bits_min
        self.hp_input_int_bits_max  = hp_input_int_bits_max
        self.hp_input_int_bits_step = hp_input_int_bits_step

    # ─────────────────────────────────────────────────────────────────────────
    # Helper
    # ─────────────────────────────────────────────────────────────────────────

    def _q_input(self, bits, int_bits, name_suffix):
        """Return a QActivation that quantizes a raw input tensor."""
        return QActivation(f"quantized_bits({bits},{int_bits})", name=f"q_input_{name_suffix}")

    # ─────────────────────────────────────────────────────────────────────────
    # makeQuantizedModel — weights + inputs quantized, tanh rescaled
    # ─────────────────────────────────────────────────────────────────────────

    def makeQuantizedModel(self):
        """
        Build fully-quantized Model3 variants with quantized inputs.
        Replaces the parent's makeQuantizedModel.
        """
        if not QKERAS_AVAILABLE:
            raise ImportError("QKeras is required for quantized models")

        for weight_bits, int_bits in self.bit_configs:
            config_name = f"quantized_{weight_bits}w{int_bits}i"
            print(f"Building Model3_QuantizedInputs {config_name}...")
            print(f"  - Input quantization: {self.input_bits}-bit, {self.input_int_bits} int bits")
            print(f"  - Weight quantization: {weight_bits}-bit, {int_bits} int bits")

            weight_quantizer     = quantized_bits(weight_bits, int_bits, alpha=1.0)
            bias_quantizer       = quantized_bits(weight_bits, int_bits, alpha=1.0)
            activation_quantizer = quantized_relu(8, 0)
            input_q_str          = f"quantized_bits({self.input_bits},{self.input_int_bits})"

            # Raw inputs
            cluster_input = Input(shape=(13, 21), name="cluster")
            nmodule_input = Input(shape=(1,), name="nModule")
            x_local_input = Input(shape=(1,), name="x_local")
            y_local_input = Input(shape=(1,), name="y_local")

            # Quantize every input
            cluster_q  = QActivation(input_q_str, name="q_input_cluster")(cluster_input)
            nmodule_q  = QActivation(input_q_str, name="q_input_nModule")(nmodule_input)
            x_local_q  = QActivation(input_q_str, name="q_input_x_local")(x_local_input)
            y_local_q  = QActivation(input_q_str, name="q_input_y_local")(y_local_input)

            # Conv2D branch
            conv_x = Reshape((13, 21, 1), name="add_channel")(cluster_q)
            conv_x = QConv2D(
                filters=self.conv_filters,
                kernel_size=(self.kernel_rows, self.kernel_cols),
                padding="same",
                kernel_quantizer=weight_quantizer,
                bias_quantizer=bias_quantizer,
                name=f"conv2d_{self.kernel_rows}x{self.kernel_cols}"
            )(conv_x)
            conv_x = QActivation(activation_quantizer, name="conv2d_act")(conv_x)
            conv_x = MaxPooling2D((2, 2), name="pool2d_1")(conv_x)
            conv_x = Flatten(name="flatten_vol")(conv_x)

            # Scalar branch
            scalar_concat_1 = Concatenate(name="concat_scalars_1")([nmodule_q, x_local_q])
            scalar_concat   = Concatenate(name="concat_scalars_2")([scalar_concat_1, y_local_q])
            scalar_x = QDense(
                self.scalar_dense_units,
                kernel_quantizer=weight_quantizer,
                bias_quantizer=bias_quantizer,
                name="dense_scalars"
            )(scalar_concat)
            scalar_x = QActivation(activation_quantizer, name="dense_scalars_act")(scalar_x)

            # Merge
            merged = Concatenate(name="concat_all")([conv_x, scalar_x])

            # Head
            h = QDense(
                self.merged_dense_1,
                kernel_quantizer=weight_quantizer,
                bias_quantizer=bias_quantizer,
                name="merged_dense1"
            )(merged)
            h = QActivation(activation_quantizer, name="merged_dense1_act")(h)
            h = Dropout(self.dropout_rate, name="dropout_1")(h)

            h = QDense(
                self.merged_dense_2,
                kernel_quantizer=weight_quantizer,
                bias_quantizer=bias_quantizer,
                name="merged_dense2"
            )(h)
            h = QActivation(activation_quantizer, name="merged_dense2_act")(h)

            # Output: quantized_tanh(8,0) then rescale [-1,1] -> [0,1]
            output_dense = QDense(
                1,
                kernel_quantizer=weight_quantizer,
                bias_quantizer=bias_quantizer,
                name="output_dense"
            )(h)
            output = QActivation("quantized_sigmoid(8,0)", name="output_activation")(output_dense)

            model = Model(
                inputs=[cluster_input, nmodule_input, x_local_input, y_local_input],
                outputs=output,
                name=f"model3_qi_{config_name}"
            )
            model.compile(
                optimizer=Adam(learning_rate=self.initial_lr),
                loss="binary_crossentropy",
                metrics=["binary_accuracy"],
                run_eagerly=True
            )
            self.models[config_name] = model

        print(f"✓ Built {len(self.bit_configs)} Model3_QuantizedInputs variants")

    # ─────────────────────────────────────────────────────────────────────────
    # makeQuantizedModelHyperParameterTuning — HP search includes input_bits
    # ─────────────────────────────────────────────────────────────────────────

    def makeQuantizedModelHyperParameterTuning(self, hp, weight_bits, int_bits):
        """
        Build fully-quantized Model3 for HP tuning with quantized inputs.
        input_bits / input_int_bits are included in the search space.
        """
        if not QKERAS_AVAILABLE:
            raise ImportError("QKeras is required for quantized models")

        # Input precision HP
        input_bits = hp.Int('input_bits',
                            min_value=self.hp_input_bits_min,
                            max_value=self.hp_input_bits_max,
                            step=self.hp_input_bits_step)
        input_int_bits = hp.Int('input_int_bits',
                                min_value=self.hp_input_int_bits_min,
                                max_value=self.hp_input_int_bits_max,
                                step=self.hp_input_int_bits_step)

        # Architecture HPs (same ranges as the constrained search in model3.py)
        conv_filters       = hp.Int('conv_filters',       min_value=4,  max_value=10,  step=2)
        kernel_rows        = hp.Choice('kernel_rows',     values=[3, 3])
        kernel_cols        = hp.Choice('kernel_cols',     values=[3, 3])
        scalar_dense_units = hp.Int('scalar_dense_units', min_value=8,  max_value=32,  step=8)
        merged_dense_1     = hp.Int('merged_dense_1',     min_value=16, max_value=128, step=16)
        merged_multiplier_2 = hp.Float('merged_multiplier_2', min_value=0.4, max_value=0.8, step=0.2)
        merged_dense_2     = int(round(merged_dense_1 * merged_multiplier_2))
        dropout_rate       = hp.Float('dropout_rate',     min_value=0.1, max_value=0.1, step=0.1)
        learning_rate      = hp.Float('learning_rate',    min_value=1e-4, max_value=1e-2, sampling='log')

        weight_quantizer     = quantized_bits(weight_bits, int_bits, alpha=1.0)
        bias_quantizer       = quantized_bits(weight_bits, int_bits, alpha=1.0)
        activation_quantizer = quantized_relu(8, 0)
        input_q_str          = f"quantized_bits({input_bits},{input_int_bits})"

        # Raw inputs
        cluster_input = Input(shape=(13, 21), name="cluster")
        nmodule_input = Input(shape=(1,), name="nModule")
        x_local_input = Input(shape=(1,), name="x_local")
        y_local_input = Input(shape=(1,), name="y_local")

        # Quantize every input
        cluster_q = QActivation(input_q_str, name="q_input_cluster")(cluster_input)
        nmodule_q = QActivation(input_q_str, name="q_input_nModule")(nmodule_input)
        x_local_q = QActivation(input_q_str, name="q_input_x_local")(x_local_input)
        y_local_q = QActivation(input_q_str, name="q_input_y_local")(y_local_input)

        # Conv2D branch
        conv_x = Reshape((13, 21, 1), name="add_channel")(cluster_q)
        conv_x = QConv2D(
            filters=conv_filters,
            kernel_size=(kernel_rows, kernel_cols),
            padding="same",
            kernel_quantizer=weight_quantizer,
            bias_quantizer=bias_quantizer,
            name="conv2d"
        )(conv_x)
        conv_x = QActivation(activation_quantizer, name="conv2d_act")(conv_x)
        conv_x = MaxPooling2D((2, 2), name="pool2d_1")(conv_x)
        conv_x = Flatten(name="flatten_vol")(conv_x)

        # Scalar branch
        scalar_concat_1 = Concatenate(name="concat_scalars_1")([nmodule_q, x_local_q])
        scalar_concat   = Concatenate(name="concat_scalars_2")([scalar_concat_1, y_local_q])
        scalar_x = QDense(
            scalar_dense_units,
            kernel_quantizer=weight_quantizer,
            bias_quantizer=bias_quantizer,
            name="dense_scalars"
        )(scalar_concat)
        scalar_x = QActivation(activation_quantizer, name="dense_scalars_act")(scalar_x)

        # Merge
        merged = Concatenate(name="concat_all")([conv_x, scalar_x])

        # Head
        h = QDense(
            merged_dense_1,
            kernel_quantizer=weight_quantizer,
            bias_quantizer=bias_quantizer,
            name="merged_dense1"
        )(merged)
        h = QActivation(activation_quantizer, name="merged_dense1_act")(h)
        h = Dropout(dropout_rate, name="dropout_1")(h)

        h = QDense(
            merged_dense_2,
            kernel_quantizer=weight_quantizer,
            bias_quantizer=bias_quantizer,
            name="merged_dense2"
        )(h)
        h = QActivation(activation_quantizer, name="merged_dense2_act")(h)

        # Output: quantized_tanh(8,0) then rescale [-1,1] -> [0,1]
        output_dense = QDense(
            1,
            kernel_quantizer=weight_quantizer,
            bias_quantizer=bias_quantizer,
            name="output_dense"
        )(h)
        output = QActivation("quantized_sigmoid(8,0)", name="output_activation")(output_dense)

        model = Model(
            inputs=[cluster_input, nmodule_input, x_local_input, y_local_input],
            outputs=output,
            name=f"model3_qi_{weight_bits}w{int_bits}i_hp_tuning"
        )
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["binary_accuracy"],
            run_eagerly=True
        )
        return model
