"""
Model3_QuantizedInputs_Large

Subclass of Model3_QuantizedInputs that overrides makeQuantizedModelHyperParameterTuning
with a much larger HP search space.  The hypothesis being tested is that Model3
underperforms Model2 because it is capacity-limited.  Every tunable dimension
is pushed into a much bigger range so the tuner can find architectures with
enough parameters to close the gap.

Author: Eric
Date: 2026
"""

import sys
sys.path.append('/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/MuC_Smartpix_ML/')
sys.path.append('../MuC_Smartpix_ML/')

from tensorflow.keras.layers import Input, Concatenate, Dropout, MaxPooling2D, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay

from model3_quantized_inputs import Model3_QuantizedInputs

try:
    from qkeras import QDense, QActivation, QConv2D
    from qkeras.quantizers import quantized_bits, quantized_relu
    QKERAS_AVAILABLE = True
except ImportError:
    print("QKeras not available.")
    QKERAS_AVAILABLE = False


class Model3_QuantizedInputs_Large(Model3_QuantizedInputs):
    """
    Same as Model3_QuantizedInputs but with a much larger HP search space.
    Only makeQuantizedModelHyperParameterTuning is overridden.
    """

    def makeQuantizedModelHyperParameterTuning(self, hp, weight_bits, int_bits):
        if not QKERAS_AVAILABLE:
            raise ImportError("QKeras is required for quantized models")

        # ── Input precision ───────────────────────────────────────────────────
        input_bits = hp.Int(
            'input_bits',
            min_value=self.hp_input_bits_min,
            max_value=self.hp_input_bits_max,
            step=self.hp_input_bits_step,
        )
        input_int_bits = hp.Int(
            'input_int_bits',
            min_value=self.hp_input_int_bits_min,
            max_value=self.hp_input_int_bits_max,
            step=self.hp_input_int_bits_step,
        )

        # ── Architecture — large, skewed-high search space ───────────────────
        # Skewed towards bigger values to test the capacity hypothesis.
        # Learning rate and dropout are fixed (not tuned).
        conv_filters = hp.Choice(
            'conv_filters',
            values=[32, 48, 64, 80, 96],   # max 96, weighted toward large
        )
        kernel_rows = hp.Choice('kernel_rows', values=[3, 3])  # always 3
        kernel_cols = hp.Choice('kernel_cols', values=[3, 3])  # always 3

        scalar_dense_units = hp.Choice(
            'scalar_dense_units',
            values=[128, 192, 256, 320, 384],  # max 384, skewed large
        )

        merged_dense_1 = hp.Choice(
            'merged_dense_1',
            values=[256, 384, 448, 512, 576, 640],  # max 640, skewed large
        )
        merged_multiplier_2 = hp.Float(
            'merged_multiplier_2', min_value=0.4, max_value=0.8, step=0.2,
        )
        merged_dense_2 = int(round(merged_dense_1 * merged_multiplier_2))

        dropout_rate = 0.1  # fixed

        # Polynomial decay — same ranges as the original search that found the
        # best rates (initial_lr=0.000871145, end_lr=5.3e-05, power=2)
        initial_lr = hp.Float('initial_lr', min_value=1e-4, max_value=1e-2, sampling='log')
        end_lr     = hp.Float('end_lr',     min_value=1e-6, max_value=1e-4, sampling='log')
        power      = hp.Float('poly_power', min_value=0.5,  max_value=2.5,  step=0.5)
        decay_steps = 30 * 35  # len(train_gen) * epochs
        lr_schedule = PolynomialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            end_learning_rate=end_lr,
            power=power,
        )

        # ── Quantizers ───────────────────────────────────────────────────────
        weight_quantizer     = quantized_bits(weight_bits, int_bits, alpha=1.0)
        bias_quantizer       = quantized_bits(weight_bits, int_bits, alpha=1.0)
        activation_quantizer = quantized_relu(8, 0)
        input_q_str          = f"quantized_bits({input_bits},{input_int_bits})"

        # ── Inputs ───────────────────────────────────────────────────────────
        cluster_input = Input(shape=(13, 21), name="cluster")
        nmodule_input = Input(shape=(1,),     name="nModule")
        x_local_input = Input(shape=(1,),     name="x_local")
        y_local_input = Input(shape=(1,),     name="y_local")

        # Quantize every input
        cluster_q = QActivation(input_q_str, name="q_input_cluster")(cluster_input)
        nmodule_q = QActivation(input_q_str, name="q_input_nModule")(nmodule_input)
        x_local_q = QActivation(input_q_str, name="q_input_x_local")(x_local_input)
        y_local_q = QActivation(input_q_str, name="q_input_y_local")(y_local_input)

        # ── Conv2D branch ────────────────────────────────────────────────────
        conv_x = Reshape((13, 21, 1), name="add_channel")(cluster_q)
        conv_x = QConv2D(
            filters=conv_filters,
            kernel_size=(kernel_rows, kernel_cols),
            padding="same",
            kernel_quantizer=weight_quantizer,
            bias_quantizer=bias_quantizer,
            name="conv2d",
        )(conv_x)
        conv_x = QActivation(activation_quantizer, name="conv2d_act")(conv_x)
        conv_x = MaxPooling2D((2, 2), name="pool2d_1")(conv_x)
        conv_x = Flatten(name="flatten_vol")(conv_x)

        # ── Scalar branch ────────────────────────────────────────────────────
        scalar_concat_1 = Concatenate(name="concat_scalars_1")([nmodule_q, x_local_q])
        scalar_concat   = Concatenate(name="concat_scalars_2")([scalar_concat_1, y_local_q])
        scalar_x = QDense(
            scalar_dense_units,
            kernel_quantizer=weight_quantizer,
            bias_quantizer=bias_quantizer,
            name="dense_scalars",
        )(scalar_concat)
        scalar_x = QActivation(activation_quantizer, name="dense_scalars_act")(scalar_x)

        # ── Merge & head ─────────────────────────────────────────────────────
        merged = Concatenate(name="concat_all")([conv_x, scalar_x])

        h = QDense(
            merged_dense_1,
            kernel_quantizer=weight_quantizer,
            bias_quantizer=bias_quantizer,
            name="merged_dense1",
        )(merged)
        h = QActivation(activation_quantizer, name="merged_dense1_act")(h)
        h = Dropout(dropout_rate, name="dropout_1")(h)

        h = QDense(
            merged_dense_2,
            kernel_quantizer=weight_quantizer,
            bias_quantizer=bias_quantizer,
            name="merged_dense2",
        )(h)
        h = QActivation(activation_quantizer, name="merged_dense2_act")(h)

        # ── Output ───────────────────────────────────────────────────────────
        output_dense = QDense(
            1,
            kernel_quantizer=weight_quantizer,
            bias_quantizer=bias_quantizer,
            name="output_dense",
        )(h)
        output = QActivation("quantized_sigmoid(8,0)", name="output_activation")(output_dense)

        model = Model(
            inputs=[cluster_input, nmodule_input, x_local_input, y_local_input],
            outputs=output,
            name=f"model3_qi_large_{weight_bits}w{int_bits}i_hp_tuning",
        )
        model.compile(
            optimizer=Adam(learning_rate=lr_schedule),
            loss="binary_crossentropy",
            metrics=["binary_accuracy"],
            run_eagerly=True,
        )
        return model
