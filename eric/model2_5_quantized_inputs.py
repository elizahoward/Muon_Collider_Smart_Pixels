"""
Model2.5 with Quantized Inputs

This module subclasses Model2_5 and overrides every model-building method
so that every raw input tensor is passed through a QActivation layer before
any Dense/QDense computation.  The quantization applied to the inputs is
controlled by two new constructor parameters:

    input_bits    – total bits used to represent each input value (default 8)
    input_int_bits – number of integer bits (default 0, i.e. purely fractional)

All four model-building entry points are covered:
  • makeUnquantizedModel
  • makeUnquatizedModelHyperParameterTuning
  • makeQuantizedModel
  • makeQuantizedModelHyperParameterTuning

Usage
-----
    from model2_5_quantized_inputs import Model2_5_QuantizedInputs

    model = Model2_5_QuantizedInputs(
        input_bits=8,
        input_int_bits=0,
        ...  # all other Model2_5 kwargs
    )
    model.makeQuantizedModel()
"""

import sys
sys.path.append('/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/MuC_Smartpix_ML/')
sys.path.append('../MuC_Smartpix_ML/')

from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from model2_5 import Model2_5

try:
    from qkeras import QDense, QActivation
    from qkeras.quantizers import quantized_bits, quantized_relu
    QKERAS_AVAILABLE = True
except ImportError:
    print("QKeras not available. Please install with: pip install qkeras")
    QKERAS_AVAILABLE = False


class Model2_5_QuantizedInputs(Model2_5):
    """
    Model2.5 variant where every input is quantized via a QActivation layer
    before being consumed by any downstream Dense / QDense layer.

    New parameters
    --------------
    input_bits : int
        Total bit-width used for input quantization (default 8).
    input_int_bits : int
        Number of integer bits for input quantization (default 0).
    """

    def __init__(self,
                 tfRecordFolder: str = "/local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_2026Feb/TF_Records/filtering_records16384_data_shuffled_single_bigData",
                 nBits: list = None,
                 loadModel: bool = False,
                 modelPath: str = None,
                 dense_units: int = 128,
                 nmodule_xlocal_units: int = 128,
                 dense2_units: int = 64,
                 dense3_units: int = 32,
                 dropout_rate: float = 0.1,
                 initial_lr: float = 1e-3,
                 end_lr: float = 1e-4,
                 power: int = 2,
                 bit_configs=None,
                 nmodule_xlocal_weight_bits: int = 8,
                 nmodule_xlocal_int_bits: int = 0,
                 # ── input quantization ───────────────────────────────────────
                 input_bits: int = 8,       # total bits for every input (e.g. 8)
                 input_int_bits: int = 0,   # integer bits (0 = purely fractional)
                 # ── HP-search range for input_bits ────────────────────────────
                 # used by makeUnquatizedModelHyperParameterTuning and
                 # makeQuantizedModelHyperParameterTuning to bound the search
                 hp_input_bits_min: int = 4,
                 hp_input_bits_max: int = 16,
                 hp_input_bits_step: int = 2,
                 hp_input_int_bits_min: int = 0,
                 hp_input_int_bits_max: int = 4,
                 hp_input_int_bits_step: int = 1):

        if bit_configs is None:
            bit_configs = [(16, 0), (8, 0), (6, 0), (4, 0), (3, 0), (2, 0)]

        super().__init__(
            tfRecordFolder=tfRecordFolder,
            nBits=nBits,
            loadModel=loadModel,
            modelPath=modelPath,
            dense_units=dense_units,
            nmodule_xlocal_units=nmodule_xlocal_units,
            dense2_units=dense2_units,
            dense3_units=dense3_units,
            dropout_rate=dropout_rate,
            initial_lr=initial_lr,
            end_lr=end_lr,
            power=power,
            bit_configs=bit_configs,
            nmodule_xlocal_weight_bits=nmodule_xlocal_weight_bits,
            nmodule_xlocal_int_bits=nmodule_xlocal_int_bits
        )
        self.modelName = "Model2.5_QuantizedInputs"

        # Fixed input precision (used when NOT running HP search)
        self.input_bits = input_bits
        self.input_int_bits = input_int_bits

        # Bounds for the HP search over input precision
        self.hp_input_bits_min  = hp_input_bits_min
        self.hp_input_bits_max  = hp_input_bits_max
        self.hp_input_bits_step = hp_input_bits_step
        self.hp_input_int_bits_min  = hp_input_int_bits_min
        self.hp_input_int_bits_max  = hp_input_int_bits_max
        self.hp_input_int_bits_step = hp_input_int_bits_step

    # ──────────────────────────────────────────────────────────────────────────
    # Helper
    # ──────────────────────────────────────────────────────────────────────────

    def _make_input_quantizer(self, bits, int_bits, name_suffix):
        """Return a QActivation layer that quantizes an input tensor."""
        quantizer_str = f"quantized_bits({bits},{int_bits})"
        return QActivation(quantizer_str, name=f"q_input_{name_suffix}")

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Unquantized model (weights unquantized; inputs are still quantized)
    # ──────────────────────────────────────────────────────────────────────────

    def makeUnquantizedModel(self):
        """
        Build Model2.5 with quantized inputs but unquantized weights.
        Each raw input is passed through a QActivation(quantized_bits) layer
        before any Dense computation.
        """
        if not QKERAS_AVAILABLE:
            raise ImportError("QKeras is required for input quantization")

        print("Building unquantized Model2.5 with quantized inputs...")
        print(f"  - Input quantization: {self.input_bits}-bit, {self.input_int_bits} integer bits")
        print(f"  - Spatial features branch: {self.dense_units} units")
        print(f"  - nModule_xlocal branch: {self.nmodule_xlocal_units} units")
        print(f"  - Merged dense layers: {self.dense2_units} -> {self.dense3_units}")

        # ── Raw inputs ────────────────────────────────────────────────────────
        x_profile_input   = Input(shape=(21,), name="x_profile")
        nmodule_input      = Input(shape=(1,),  name="nModule")
        x_local_input      = Input(shape=(1,),  name="x_local")
        y_profile_input   = Input(shape=(13,), name="y_profile")
        y_local_input      = Input(shape=(1,),  name="y_local")

        # ── Quantize every input ──────────────────────────────────────────────
        x_profile_q  = self._make_input_quantizer(self.input_bits, self.input_int_bits, "x_profile")(x_profile_input)
        nmodule_q    = self._make_input_quantizer(self.input_bits, self.input_int_bits, "nModule")(nmodule_input)
        x_local_q    = self._make_input_quantizer(self.input_bits, self.input_int_bits, "x_local")(x_local_input)
        y_profile_q  = self._make_input_quantizer(self.input_bits, self.input_int_bits, "y_profile")(y_profile_input)
        y_local_q    = self._make_input_quantizer(self.input_bits, self.input_int_bits, "y_local")(y_local_input)

        # ── Spatial branch (two-stage concat for HLS compatibility) ───────────
        xy_concat      = Concatenate(name="xy_concat")([x_profile_q, y_profile_q])
        other_features = Concatenate(name="other_features")([xy_concat, y_local_q])
        other_dense    = Dense(self.dense_units, activation="relu", name="other_dense")(other_features)

        # ── nModule_xlocal branch ─────────────────────────────────────────────
        nmodule_xlocal_concat = Concatenate(name="nmodule_xlocal_concat")([nmodule_q, x_local_q])
        nmodule_xlocal_dense  = Dense(self.nmodule_xlocal_units, activation="relu",
                                      name="nmodule_xlocal_dense")(nmodule_xlocal_concat)

        # ── Merge ─────────────────────────────────────────────────────────────
        merged = Concatenate(name="merged_features")([other_dense, nmodule_xlocal_dense])
        hidden = Dense(self.dense2_units, activation="relu", name="dense2")(merged)
        hidden = Dropout(self.dropout_rate, name="dropout1")(hidden)
        hidden = Dense(self.dense3_units, activation="relu", name="dense3")(hidden)
        output = Dense(1, activation="tanh", name="output")(hidden)

        self.models["Unquantized"] = Model(
            inputs=[x_profile_input, nmodule_input, x_local_input,
                    y_profile_input, y_local_input],
            outputs=output,
            name="model2_5_qi_unquantized"
        )
        print("✓ Unquantized Model2.5 (quantized inputs) built successfully")

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Unquantized hyperparameter-tuning model
    # ──────────────────────────────────────────────────────────────────────────

    def makeUnquatizedModelHyperParameterTuning(self, hp):
        """
        Build Model2.5 for hyperparameter tuning with quantized inputs and
        unquantized weights.  input_bits / input_int_bits are also tunable.
        """
        if not QKERAS_AVAILABLE:
            raise ImportError("QKeras is required for input quantization")

        # ── Hyperparameters ───────────────────────────────────────────────────
        # Input precision – bounds driven by constructor args so you can narrow
        # or widen the search without editing this method.
        input_bits     = hp.Int('input_bits',     min_value=self.hp_input_bits_min,
                                                  max_value=self.hp_input_bits_max,
                                                  step=self.hp_input_bits_step)
        input_int_bits = hp.Int('input_int_bits', min_value=self.hp_input_int_bits_min,
                                                  max_value=self.hp_input_int_bits_max,
                                                  step=self.hp_input_int_bits_step)

        spatial_units          = hp.Int('spatial_units',          min_value=32, max_value=256, step=32)
        nmodule_xlocal_units   = hp.Int('nmodule_xlocal_units',   min_value=16, max_value=128, step=16)

        concat_size  = spatial_units + nmodule_xlocal_units
        dense2_ratio = hp.Float('dense2_ratio', min_value=0.2, max_value=1.0, step=0.05)
        dense2_units = max(32, int(concat_size * dense2_ratio / 16) * 16)

        dense3_ratio = hp.Float('dense3_ratio', min_value=0.2, max_value=1.0, step=0.05)
        dense3_units = max(16, int(dense2_units * dense3_ratio / 8) * 8)

        dropout_rate   = hp.Float('dropout_rate',   min_value=0.0, max_value=0.3, step=0.05)
        learning_rate  = hp.Float('learning_rate',  min_value=1e-4, max_value=1e-2, sampling='log')

        # ── Raw inputs ────────────────────────────────────────────────────────
        x_profile_input  = Input(shape=(21,), name="x_profile")
        nmodule_input     = Input(shape=(1,),  name="nModule")
        x_local_input     = Input(shape=(1,),  name="x_local")
        y_profile_input  = Input(shape=(13,), name="y_profile")
        y_local_input     = Input(shape=(1,),  name="y_local")

        # ── Quantize every input ──────────────────────────────────────────────
        q_str = f"quantized_bits({input_bits},{input_int_bits})"
        x_profile_q  = QActivation(q_str, name="q_input_x_profile")(x_profile_input)
        nmodule_q    = QActivation(q_str, name="q_input_nModule")(nmodule_input)
        x_local_q    = QActivation(q_str, name="q_input_x_local")(x_local_input)
        y_profile_q  = QActivation(q_str, name="q_input_y_profile")(y_profile_input)
        y_local_q    = QActivation(q_str, name="q_input_y_local")(y_local_input)

        # ── Spatial branch ────────────────────────────────────────────────────
        xy_concat      = Concatenate(name="xy_concat")([x_profile_q, y_profile_q])
        other_features = Concatenate(name="other_features")([xy_concat, y_local_q])
        other_dense    = Dense(spatial_units, activation="relu", name="other_dense")(other_features)

        # ── nModule_xlocal branch ─────────────────────────────────────────────
        nmodule_xlocal_concat = Concatenate(name="nmodule_xlocal_concat")([nmodule_q, x_local_q])
        nmodule_xlocal_dense  = Dense(nmodule_xlocal_units, activation="relu",
                                      name="nmodule_xlocal_dense")(nmodule_xlocal_concat)

        # ── Merge ─────────────────────────────────────────────────────────────
        merged = Concatenate(name="merged_features")([other_dense, nmodule_xlocal_dense])
        hidden = Dense(dense2_units, activation="relu", name="dense2")(merged)
        hidden = Dropout(dropout_rate, name="dropout1")(hidden)
        hidden = Dense(dense3_units, activation="relu", name="dense3")(hidden)
        output = Dense(1, activation="tanh", name="output")(hidden)

        model = Model(
            inputs=[x_profile_input, nmodule_input, x_local_input,
                    y_profile_input, y_local_input],
            outputs=output,
            name="model2_5_qi_hp_tuning"
        )
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["binary_accuracy"]
        )
        return model

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Quantized model (weights + inputs quantized)
    # ──────────────────────────────────────────────────────────────────────────

    def makeQuantizedModel(self):
        """
        Build fully-quantized Model2.5 variants (weights *and* inputs quantized).
        Input quantization uses self.input_bits / self.input_int_bits.
        Weight quantization iterates over self.bit_configs as in the parent.
        """
        if not QKERAS_AVAILABLE:
            raise ImportError("QKeras is required for quantized models")

        for weight_bits, int_bits in self.bit_configs:
            config_name = f"quantized_{weight_bits}w{int_bits}i"
            print(f"Building Model2.5_QuantizedInputs {config_name}...")
            print(f"  - Input quantization:       {self.input_bits}-bit, {self.input_int_bits} integer bits")
            print(f"  - Spatial branch weights:   {weight_bits}-bit")
            print(f"  - nModule_xlocal weights:   {self.nmodule_xlocal_weight_bits}-bit")

            # ── Quantizers ────────────────────────────────────────────────────
            weight_quantizer              = quantized_bits(weight_bits, int_bits, alpha=1.0)
            nmodule_xlocal_weight_quantizer = quantized_bits(
                self.nmodule_xlocal_weight_bits, self.nmodule_xlocal_int_bits, alpha=1.0)
            input_q_str = f"quantized_bits({self.input_bits},{self.input_int_bits})"

            # ── Raw inputs ────────────────────────────────────────────────────
            x_profile_input  = Input(shape=(21,), name="x_profile")
            nmodule_input     = Input(shape=(1,),  name="nModule")
            x_local_input     = Input(shape=(1,),  name="x_local")
            y_profile_input  = Input(shape=(13,), name="y_profile")
            y_local_input     = Input(shape=(1,),  name="y_local")

            # ── Quantize every input ──────────────────────────────────────────
            x_profile_q  = QActivation(input_q_str, name="q_input_x_profile")(x_profile_input)
            nmodule_q    = QActivation(input_q_str, name="q_input_nModule")(nmodule_input)
            x_local_q    = QActivation(input_q_str, name="q_input_x_local")(x_local_input)
            y_profile_q  = QActivation(input_q_str, name="q_input_y_profile")(y_profile_input)
            y_local_q    = QActivation(input_q_str, name="q_input_y_local")(y_local_input)

            # ── Spatial branch ────────────────────────────────────────────────
            xy_concat      = Concatenate(name="xy_concat")([x_profile_q, y_profile_q])
            other_features = Concatenate(name="other_features")([xy_concat, y_local_q])
            other_dense = QDense(
                self.dense_units,
                kernel_quantizer=weight_quantizer,
                bias_quantizer=weight_quantizer,
                name="other_dense"
            )(other_features)
            other_dense = QActivation("quantized_relu(8,0)", name="other_activation")(other_dense)

            # ── nModule_xlocal branch ─────────────────────────────────────────
            nmodule_xlocal_concat = Concatenate(name="nmodule_xlocal_concat")([nmodule_q, x_local_q])
            nmodule_xlocal_dense = QDense(
                self.nmodule_xlocal_units,
                kernel_quantizer=nmodule_xlocal_weight_quantizer,
                bias_quantizer=nmodule_xlocal_weight_quantizer,
                name="nmodule_xlocal_dense"
            )(nmodule_xlocal_concat)
            nmodule_xlocal_dense = QActivation("quantized_relu(8,0)",
                                                name="nmodule_xlocal_activation")(nmodule_xlocal_dense)

            # ── Merge ─────────────────────────────────────────────────────────
            merged = Concatenate(name="merged_features")([other_dense, nmodule_xlocal_dense])

            hidden = QDense(
                self.dense2_units,
                kernel_quantizer=weight_quantizer,
                bias_quantizer=weight_quantizer,
                name="dense2"
            )(merged)
            hidden = QActivation("quantized_relu(8,0)", name="dense2_activation")(hidden)
            hidden = Dropout(self.dropout_rate, name="dropout1")(hidden)

            hidden = QDense(
                self.dense3_units,
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
            output = QActivation("quantized_tanh(8,0)", name="output_activation")(output_dense)

            model = Model(
                inputs=[x_profile_input, nmodule_input, x_local_input,
                        y_profile_input, y_local_input],
                outputs=output,
                name=f"model2_5_qi_{config_name}"
            )
            model.compile(
                optimizer=Adam(learning_rate=self.initial_lr),
                loss="binary_crossentropy",
                metrics=["binary_accuracy"],
                run_eagerly=True
            )
            self.models[config_name] = model

        print(f"✓ Built {len(self.bit_configs)} fully-quantized Model2.5_QuantizedInputs variants")

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Quantized hyperparameter-tuning model
    # ──────────────────────────────────────────────────────────────────────────

    def makeQuantizedModelHyperParameterTuning(self, hp, weight_bits, int_bits):
        """
        Build fully-quantized Model2.5 for hyperparameter tuning.
        input_bits / input_int_bits are included in the hyperparameter search space.
        """
        if not QKERAS_AVAILABLE:
            raise ImportError("QKeras is required for quantized models")

        # ── Hyperparameters ───────────────────────────────────────────────────
        # Input precision – bounds driven by constructor args.
        input_bits     = hp.Int('input_bits',     min_value=self.hp_input_bits_min,
                                                  max_value=self.hp_input_bits_max,
                                                  step=self.hp_input_bits_step)
        input_int_bits = hp.Int('input_int_bits', min_value=self.hp_input_int_bits_min,
                                                  max_value=self.hp_input_int_bits_max,
                                                  step=self.hp_input_int_bits_step)

        spatial_units        = hp.Int('spatial_units',        min_value=8,  max_value=128, step=8)
        nmodule_xlocal_units = hp.Int('nmodule_xlocal_units', min_value=2,  max_value=12,  step=2)

        concat_size  = spatial_units + nmodule_xlocal_units
        dense2_ratio = hp.Float('dense2_ratio', min_value=0.2, max_value=0.7, step=0.1)
        dense2_units = max(4, int(concat_size * dense2_ratio / 4) * 4)

        dense3_ratio = hp.Float('dense3_ratio', min_value=0.2, max_value=0.7, step=0.1)
        dense3_units = max(4, int(dense2_units * dense3_ratio / 2) * 2)

        dropout_rate  = hp.Float('dropout_rate',  min_value=0.0, max_value=0.1, step=0.05)
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

        # ── Quantizers ────────────────────────────────────────────────────────
        weight_quantizer              = quantized_bits(weight_bits, int_bits, alpha=1.0)
        nmodule_xlocal_weight_quantizer = quantized_bits(
            self.nmodule_xlocal_weight_bits, self.nmodule_xlocal_int_bits, alpha=1.0)
        input_q_str = f"quantized_bits({input_bits},{input_int_bits})"

        # ── Raw inputs ────────────────────────────────────────────────────────
        x_profile_input  = Input(shape=(21,), name="x_profile")
        nmodule_input     = Input(shape=(1,),  name="nModule")
        x_local_input     = Input(shape=(1,),  name="x_local")
        y_profile_input  = Input(shape=(13,), name="y_profile")
        y_local_input     = Input(shape=(1,),  name="y_local")

        # ── Quantize every input ──────────────────────────────────────────────
        x_profile_q  = QActivation(input_q_str, name="q_input_x_profile")(x_profile_input)
        nmodule_q    = QActivation(input_q_str, name="q_input_nModule")(nmodule_input)
        x_local_q    = QActivation(input_q_str, name="q_input_x_local")(x_local_input)
        y_profile_q  = QActivation(input_q_str, name="q_input_y_profile")(y_profile_input)
        y_local_q    = QActivation(input_q_str, name="q_input_y_local")(y_local_input)

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
        output = QActivation("quantized_tanh(8,0)", name="output_activation")(output_dense)

        model = Model(
            inputs=[x_profile_input, nmodule_input, x_local_input,
                    y_profile_input, y_local_input],
            outputs=output,
            name=f"model2_5_qi_{weight_bits}w{int_bits}i_hp_tuning"
        )
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["binary_accuracy"],
            run_eagerly=True
        )
        return model


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity-check / example usage
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """Example usage of Model2.5_QuantizedInputs"""
    print("=== Model2.5_QuantizedInputs Example Usage ===")

    model = Model2_5_QuantizedInputs(
        dense_units=128,
        nmodule_xlocal_units=32,
        dense2_units=128,
        dense3_units=64,
        dropout_rate=0.1,
        initial_lr=1e-3,
        end_lr=1e-4,
        power=2,
        bit_configs=[(4, 0)],
        nmodule_xlocal_weight_bits=4,
        nmodule_xlocal_int_bits=0,
        # ── input quantization ──
        input_bits=8,
        input_int_bits=0
    )

    model.makeUnquantizedModel()
    model.models["Unquantized"].summary()

    model.makeQuantizedModel()
    model.models["quantized_4w0i"].summary()

    print("Model2.5_QuantizedInputs build test completed successfully!")


if __name__ == "__main__":
    main()
