"""
Model2.5 with Fixed Input Bit-Width for Input Quantization Sweep

Subclass of Model2_5_QuantizedInputs that overrides makeQuantizedModelHyperParameterTuning
so that input_bits and input_int_bits are NOT searched as hyperparameters — they are fixed
at construction time via self.input_bits / self.input_int_bits.

This is used by run_input_bits_sweep_model2_5.py to isolate the effect of input precision:
the script loops over a set of fixed input bit-widths (e.g. 4, 6, 8, 10) and for each one
runs a full HP search over architecture params only (layer sizes, dropout, LR).

Usage
-----
    from model2_5_fixed_input_bits import Model2_5_FixedInputBits

    model = Model2_5_FixedInputBits(
        bit_configs=[(6, 0)],
        input_bits=4,
        input_int_bits=0,
    )
    model.runQuantizedHyperparameterTuning(...)
"""

import sys
sys.path.append('/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/MuC_Smartpix_ML/')
sys.path.append('../MuC_Smartpix_ML/')

from tensorflow.keras.layers import Input, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from model2_5_quantized_inputs import Model2_5_QuantizedInputs

try:
    from qkeras import QDense, QActivation
    from qkeras.quantizers import quantized_bits
    QKERAS_AVAILABLE = True
except ImportError:
    print("QKeras not available. Please install with: pip install qkeras")
    QKERAS_AVAILABLE = False


class Model2_5_FixedInputBits(Model2_5_QuantizedInputs):
    """
    Model2.5 variant for input-quantization sweep experiments.

    Identical to Model2_5_QuantizedInputs except that input_bits and
    input_int_bits are fixed at construction time and are never included
    in the hyperparameter search space.  All other HPs (layer sizes,
    dropout, learning rate) are still searched by Keras Tuner.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.modelName = "model2.5.fixedinputbits"

    def makeQuantizedModelHyperParameterTuning(self, hp, weight_bits, int_bits):
        """
        Build fully-quantized Model2.5 for HP tuning with FIXED input precision.

        input_bits / input_int_bits come from self.input_bits / self.input_int_bits
        (set at construction) rather than being sampled by the HP search.
        """
        if not QKERAS_AVAILABLE:
            raise ImportError("QKeras is required for quantized models")

        # ── Architecture hyperparameters — tightly constrained to exactly 15 combinations ──
        # 5 spatial sizes × 3 nmodule sizes = 15 unique models, spanning tiny → large.
        # Dense layers are derived deterministically so they don't add extra combinations.
        spatial_units        = hp.Choice('spatial_units',        [8, 32, 64, 96, 128])
        nmodule_xlocal_units = hp.Choice('nmodule_xlocal_units', [2, 6, 12])

        concat_size  = spatial_units + nmodule_xlocal_units
        dense2_units = max(4, int(concat_size * 0.5 / 4) * 4)   # ~50 % of concat, fixed ratio
        dense3_units = max(4, int(dense2_units * 0.5 / 2) * 2)  # ~50 % of dense2, fixed ratio

        dropout_rate  = 0.05
        learning_rate = 1e-3

        # ── Fixed input precision (not searched) ──────────────────────────────
        input_bits     = self.input_bits
        input_int_bits = self.input_int_bits

        # ── Quantizers ────────────────────────────────────────────────────────
        weight_quantizer = quantized_bits(weight_bits, int_bits, alpha=1.0)
        nmodule_xlocal_weight_quantizer = quantized_bits(
            self.nmodule_xlocal_weight_bits, self.nmodule_xlocal_int_bits, alpha=1.0)
        input_q_str = f"quantized_bits({input_bits},{input_int_bits})"

        # ── Raw inputs ────────────────────────────────────────────────────────
        x_profile_input = Input(shape=(21,), name="x_profile")
        nmodule_input    = Input(shape=(1,),  name="nModule")
        x_local_input    = Input(shape=(1,),  name="x_local")
        y_profile_input = Input(shape=(13,), name="y_profile")
        y_local_input    = Input(shape=(1,),  name="y_local")

        # ── Quantize every input ──────────────────────────────────────────────
        x_profile_q = QActivation(input_q_str, name="q_input_x_profile")(x_profile_input)
        nmodule_q   = QActivation(input_q_str, name="q_input_nModule")(nmodule_input)
        x_local_q   = QActivation(input_q_str, name="q_input_x_local")(x_local_input)
        y_profile_q = QActivation(input_q_str, name="q_input_y_profile")(y_profile_input)
        y_local_q   = QActivation(input_q_str, name="q_input_y_local")(y_local_input)

        # ── Spatial branch ────────────────────────────────────────────────────
        xy_concat      = Concatenate(name="xy_concat")([x_profile_q, y_profile_q])
        other_features = Concatenate(name="other_features")([xy_concat, y_local_q])
        other_dense = QDense(
            spatial_units,
            kernel_quantizer=weight_quantizer,
            bias_quantizer=weight_quantizer,
            name="other_dense"
        )(other_features)
        other_dense = QActivation("quantized_relu(8,0)", name="other_activation")(other_dense)

        # ── nModule_xlocal branch ─────────────────────────────────────────────
        nmodule_xlocal_concat = Concatenate(name="nmodule_xlocal_concat")([nmodule_q, x_local_q])
        nmodule_xlocal_dense = QDense(
            nmodule_xlocal_units,
            kernel_quantizer=nmodule_xlocal_weight_quantizer,
            bias_quantizer=nmodule_xlocal_weight_quantizer,
            name="nmodule_xlocal_dense"
        )(nmodule_xlocal_concat)
        nmodule_xlocal_dense = QActivation("quantized_relu(8,0)",
                                           name="nmodule_xlocal_activation")(nmodule_xlocal_dense)

        # ── Merge ─────────────────────────────────────────────────────────────
        merged = Concatenate(name="merged_features")([other_dense, nmodule_xlocal_dense])

        hidden = QDense(
            dense2_units,
            kernel_quantizer=weight_quantizer,
            bias_quantizer=weight_quantizer,
            name="dense2"
        )(merged)
        hidden = QActivation("quantized_relu(8,0)", name="dense2_activation")(hidden)
        hidden = Dropout(dropout_rate, name="dropout1")(hidden)

        hidden = QDense(
            dense3_units,
            kernel_quantizer=weight_quantizer,
            bias_quantizer=weight_quantizer,
            name="dense3"
        )(hidden)
        hidden = QActivation("quantized_relu(8,0)", name="dense3_activation")(hidden)

        output_dense = QDense(
            1,
            kernel_quantizer=weight_quantizer,
            bias_quantizer=weight_quantizer,
            name="output"
        )(hidden)
        output = QActivation("quantized_sigmoid(8,0)", name="output_activation")(output_dense)

        model = Model(
            inputs=[x_profile_input, nmodule_input, x_local_input,
                    y_profile_input, y_local_input],
            outputs=output,
            name=f"model2_5_fib_{weight_bits}w{int_bits}i_{input_bits}ib_hp_tuning"
        )
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["binary_accuracy"],
            run_eagerly=True
        )
        return model
