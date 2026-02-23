"""
Model2.5 Implementation based on Model2 with a single dense fusion layer
and an 8-bit dedicated projection for the nModule_xlocal feature (concatenation of nModule and x_local).
"""

from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, Add, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from model2 import Model2

try:
    from qkeras import QDense, QActivation
    from qkeras.quantizers import quantized_bits, quantized_relu, quantized_tanh
    QKERAS_AVAILABLE = True
except ImportError:
    print("QKeras not available. Please install with: pip install qkeras")
    QKERAS_AVAILABLE = False


class Model2_5(Model2):
    """
    Model2.5: Single-hidden-layer variant of Model2 with dedicated 8-bit processing
    for the nModule_xlocal feature (concatenation of nModule and x_local) during quantized training.
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
                 bit_configs = [(16, 0), (8, 0), (6, 0), (4, 0), (3, 0), (2, 0)],
                 nmodule_xlocal_weight_bits: int = 8,
                 nmodule_xlocal_int_bits: int = 0):
        super().__init__(
            tfRecordFolder=tfRecordFolder,
            nBits=nBits,
            loadModel=loadModel,
            modelPath=modelPath,
            xz_units=dense_units,
            yl_units=nmodule_xlocal_units,
            merged_units_1=dense2_units,
            merged_units_2=dense3_units,
            merged_units_3=32,
            dropout_rate=dropout_rate,
            initial_lr=initial_lr,
            end_lr=end_lr,
            power=power,
            bit_configs=bit_configs
        )
        self.modelName = "Model2.5"
        self.dense_units = dense_units
        self.nmodule_xlocal_units = nmodule_xlocal_units
        self.dense2_units = dense2_units
        self.dense3_units = dense3_units
        self.nmodule_xlocal_weight_bits = nmodule_xlocal_weight_bits
        self.nmodule_xlocal_int_bits = nmodule_xlocal_int_bits
        
        # Override feature description to use nModule and x_local instead of z_global
        self.x_feature_description = ['x_profile', 'nModule', 'x_local', 'y_profile', 'y_local']

    def makeUnquantizedModel(self):
        """Build the unquantized Model2.5 architecture."""
        print("Building unquantized Model2.5...")
        print(f"  - Spatial features branch: {self.dense_units} units")
        print(f"  - nModule_xlocal branch: {self.nmodule_xlocal_units} units")
        print(f"  - Merged dense layers: {self.dense2_units} -> {self.dense3_units}")
        
        # Input layers
        x_profile_input = Input(shape=(21,), name="x_profile")
        nmodule_input = Input(shape=(1,), name="nModule")
        x_local_input = Input(shape=(1,), name="x_local")
        y_profile_input = Input(shape=(13,), name="y_profile")
        y_local_input = Input(shape=(1,), name="y_local")

        # Spatial features branch - concatenate in two stages for HLS compatibility
        xy_concat = Concatenate(name="xy_concat")([x_profile_input, y_profile_input])
        other_features = Concatenate(name="other_features")([xy_concat, y_local_input])
        other_dense = Dense(self.dense_units, activation="relu", name="other_dense")(other_features)

        # nModule_xlocal branch - concatenate nModule and x_local first
        nmodule_xlocal_concat = Concatenate(name="nmodule_xlocal_concat")([nmodule_input, x_local_input])
        nmodule_xlocal_dense = Dense(self.nmodule_xlocal_units, activation="relu", name="nmodule_xlocal_dense")(nmodule_xlocal_concat)

        # Merge both branches
        merged = Concatenate(name="merged_features")([other_dense, nmodule_xlocal_dense])
        
        # Two more dense layers
        hidden = Dense(self.dense2_units, activation="relu", name="dense2")(merged)
        hidden = Dropout(self.dropout_rate, name="dropout1")(hidden)
        hidden = Dense(self.dense3_units, activation="relu", name="dense3")(hidden)
        
        # Output layer
        output = Dense(1, activation="tanh", name="output")(hidden)

        self.models["Unquantized"] = Model(
            inputs=[x_profile_input, nmodule_input, x_local_input, y_profile_input, y_local_input],
            outputs=output,
            name="model2_5_unquantized"
        )
        print("✓ Unquantized Model2.5 built successfully")

    def makeUnquatizedModelHyperParameterTuning(self, hp):
        """Build Model2.5 for hyperparameter tuning with progressive layer constraints."""
        # Hyperparameter search space with progressive constraints
        # First layer sizes
        # Layer 1: Sample directly
        spatial_units = hp.Int('spatial_units', min_value=32, max_value=256, step=32)
        nmodule_xlocal_units = hp.Int('nmodule_xlocal_units', min_value=16, max_value=128, step=16)
        
        # Layer 2: Use ratio of concat_size (avoids clipping bias)
        # This ensures dense2_units <= concat_size by construction
        concat_size = spatial_units + nmodule_xlocal_units
        dense2_ratio = hp.Float('dense2_ratio', min_value=0.2, max_value=1.0, step=0.05)
        dense2_units = max(32, int(concat_size * dense2_ratio / 16) * 16)  # Round to multiple of 16
        
        # Layer 3: Use ratio of dense2_units (avoids clipping bias)
        # This ensures dense3_units <= dense2_units by construction
        dense3_ratio = hp.Float('dense3_ratio', min_value=0.2, max_value=1.0, step=0.05)
        dense3_units = max(16, int(dense2_units * dense3_ratio / 8) * 8)  # Round to multiple of 8
        
        dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.3, step=0.05)
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        
        # Input layers
        x_profile_input = Input(shape=(21,), name="x_profile")
        nmodule_input = Input(shape=(1,), name="nModule")
        x_local_input = Input(shape=(1,), name="x_local")
        y_profile_input = Input(shape=(13,), name="y_profile")
        y_local_input = Input(shape=(1,), name="y_local")

        # Spatial features branch - concatenate in two stages for HLS compatibility
        xy_concat = Concatenate(name="xy_concat")([x_profile_input, y_profile_input])
        other_features = Concatenate(name="other_features")([xy_concat, y_local_input])
        other_dense = Dense(spatial_units, activation="relu", name="other_dense")(other_features)

        # nModule_xlocal branch - concatenate nModule and x_local first
        nmodule_xlocal_concat = Concatenate(name="nmodule_xlocal_concat")([nmodule_input, x_local_input])
        nmodule_xlocal_dense = Dense(nmodule_xlocal_units, activation="relu", name="nmodule_xlocal_dense")(nmodule_xlocal_concat)

        # Merge both branches
        merged = Concatenate(name="merged_features")([other_dense, nmodule_xlocal_dense])
        
        # Two more dense layers
        hidden = Dense(dense2_units, activation="relu", name="dense2")(merged)
        hidden = Dropout(dropout_rate, name="dropout1")(hidden)
        hidden = Dense(dense3_units, activation="relu", name="dense3")(hidden)
        
        # Output layer
        output = Dense(1, activation="tanh", name="output")(hidden)

        model = Model(
            inputs=[x_profile_input, nmodule_input, x_local_input, y_profile_input, y_local_input],
            outputs=output,
            name="model2_5_hyperparameter_tuning"
        )

        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["binary_accuracy"]
        )

        return model

    def makeQuantizedModel(self):
        """Build quantized Model2.5 variants."""
        if not QKERAS_AVAILABLE:
            raise ImportError("QKeras is required for quantized models")

        for weight_bits, int_bits in self.bit_configs:
            config_name = f"quantized_{weight_bits}w{int_bits}i"
            print(f"Building Model2.5 {config_name}...")
            print(f"  - Spatial features branch: {self.dense_units} units ({weight_bits}-bit)")
            print(f"  - nModule_xlocal branch: {self.nmodule_xlocal_units} units ({self.nmodule_xlocal_weight_bits}-bit)")

            # Quantizers
            weight_quantizer = quantized_bits(weight_bits, int_bits, alpha=1.0)
            nmodule_xlocal_weight_quantizer = quantized_bits(self.nmodule_xlocal_weight_bits, self.nmodule_xlocal_int_bits, alpha=1.0)

            # Input layers
            x_profile_input = Input(shape=(21,), name="x_profile")
            nmodule_input = Input(shape=(1,), name="nModule")
            x_local_input = Input(shape=(1,), name="x_local")
            y_profile_input = Input(shape=(13,), name="y_profile")
            y_local_input = Input(shape=(1,), name="y_local")

            # Spatial features branch - concatenate in two stages for HLS compatibility
            xy_concat = Concatenate(name="xy_concat")([x_profile_input, y_profile_input])
            other_features = Concatenate(name="other_features")([xy_concat, y_local_input])
            other_dense = QDense(
                self.dense_units,
                kernel_quantizer=weight_quantizer,
                bias_quantizer=weight_quantizer,
                name="other_dense"
            )(other_features)
            other_dense = QActivation("quantized_relu(8,0)", name="other_activation")(other_dense)

            # nModule_xlocal branch - concatenate nModule and x_local first
            nmodule_xlocal_concat = Concatenate(name="nmodule_xlocal_concat")([nmodule_input, x_local_input])
            nmodule_xlocal_dense = QDense(
                self.nmodule_xlocal_units,
                kernel_quantizer=nmodule_xlocal_weight_quantizer,
                bias_quantizer=nmodule_xlocal_weight_quantizer,
                name="nmodule_xlocal_dense"
            )(nmodule_xlocal_concat)
            nmodule_xlocal_dense = QActivation("quantized_relu(8,0)", name="nmodule_xlocal_activation")(nmodule_xlocal_dense)

            # Merge both branches
            merged = Concatenate(name="merged_features")([other_dense, nmodule_xlocal_dense])
            
            # Two more dense layers
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

            # Output layer
            output_dense = QDense(
                1,
                kernel_quantizer=weight_quantizer,
                bias_quantizer=weight_quantizer,
                name="output"
            )(hidden)
            output = QActivation("quantized_tanh(8,0)", name="output_activation")(output_dense)

            model = Model(
                inputs=[x_profile_input, nmodule_input, x_local_input, y_profile_input, y_local_input],
                outputs=output,
                name=f"model2_5_{config_name}"
            )

            model.compile(
                optimizer=Adam(learning_rate=self.initial_lr),
                loss="binary_crossentropy",
                metrics=["binary_accuracy"],
                run_eagerly=True
            )

            self.models[config_name] = model

        print(f"✓ Built {len(self.bit_configs)} quantized Model2.5 variants")

    def makeQuantizedModelHyperParameterTuning(self, hp, weight_bits, int_bits):
        """Build quantized Model2.5 for hyperparameter tuning with progressive layer constraints."""
        if not QKERAS_AVAILABLE:
            raise ImportError("QKeras is required for quantized models")

        # Layer 1: Sample directly
        spatial_units = hp.Int('spatial_units', min_value=8, max_value=128, step=8)
        nmodule_xlocal_units = hp.Int('nmodule_xlocal_units', min_value=2, max_value=12, step=2)
        
        # Layer 2: Use ratio of concat_size (avoids clipping bias)
        # This ensures dense2_units <= concat_size by construction
        concat_size = spatial_units + nmodule_xlocal_units
        dense2_ratio = hp.Float('dense2_ratio', min_value=0.2, max_value=0.7, step=0.1)
        dense2_units = max(4, int(concat_size * dense2_ratio / 4) * 4)  # Round to multiple of 4
        
        # Layer 3: Use ratio of dense2_units (avoids clipping bias)
        # This ensures dense3_units <= dense2_units by construction
        dense3_ratio = hp.Float('dense3_ratio', min_value=0.2, max_value=0.7, step=0.1)
        dense3_units = max(4, int(dense2_units * dense3_ratio / 2) * 2)  # Round to multiple of 2
        
        dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.1, step=0.05)
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

        # Quantizers
        weight_quantizer = quantized_bits(weight_bits, int_bits, alpha=1.0)
        nmodule_xlocal_weight_quantizer = quantized_bits(self.nmodule_xlocal_weight_bits, self.nmodule_xlocal_int_bits, alpha=1.0)

        # Input layers
        x_profile_input = Input(shape=(21,), name="x_profile")
        nmodule_input = Input(shape=(1,), name="nModule")
        x_local_input = Input(shape=(1,), name="x_local")
        y_profile_input = Input(shape=(13,), name="y_profile")
        y_local_input = Input(shape=(1,), name="y_local")

        # Spatial features branch - concatenate in two stages for HLS compatibility
        xy_concat = Concatenate(name="xy_concat")([x_profile_input, y_profile_input])
        other_features = Concatenate(name="other_features")([xy_concat, y_local_input])
        other_dense = QDense(
            spatial_units,
            kernel_quantizer=weight_quantizer,
            bias_quantizer=weight_quantizer,
            name="other_dense"
        )(other_features)
        other_dense = QActivation("quantized_relu(8,0)", name="other_activation")(other_dense)

        # nModule_xlocal branch - concatenate nModule and x_local first
        nmodule_xlocal_concat = Concatenate(name="nmodule_xlocal_concat")([nmodule_input, x_local_input])
        nmodule_xlocal_dense = QDense(
            nmodule_xlocal_units,
            kernel_quantizer=nmodule_xlocal_weight_quantizer,
            bias_quantizer=nmodule_xlocal_weight_quantizer,
            name="nmodule_xlocal_dense"
        )(nmodule_xlocal_concat)
        nmodule_xlocal_dense = QActivation("quantized_relu(8,0)", name="nmodule_xlocal_activation")(nmodule_xlocal_dense)

        # Merge both branches
        merged = Concatenate(name="merged_features")([other_dense, nmodule_xlocal_dense])
        
        # Two more dense layers
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

        # Output layer
        output_dense = QDense(
            1,
            kernel_quantizer=weight_quantizer,
            bias_quantizer=weight_quantizer,
            name="output"
        )(hidden)
        output = QActivation("quantized_tanh(8,0)", name="output_activation")(output_dense)

        model = Model(
            inputs=[x_profile_input, nmodule_input, x_local_input, y_profile_input, y_local_input],
            outputs=output,
            name=f"model2_5_quantized_{weight_bits}w{int_bits}i_hyperparameter_tuning"
        )

        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["binary_accuracy"],
            run_eagerly=True
        )

        return model

    def _calculate_model_parameters(self, hyperparams):
        """Calculate parameter counts for Model2.5 architecture."""
        spatial_units = hyperparams.get('spatial_units', self.dense_units)
        nmodule_xlocal_units = hyperparams.get('nmodule_xlocal_units', self.nmodule_xlocal_units)
        dense2_units = hyperparams.get('dense2_units', self.dense2_units)
        dense3_units = hyperparams.get('dense3_units', self.dense3_units)
        
        other_dim = 21 + 13 + 1  # x_profile + y_profile + y_local
        nmodule_xlocal_dim = 2  # nModule + x_local

        # First layer: spatial features and nModule_xlocal
        spatial_dense_params = (other_dim * spatial_units) + spatial_units  # weights + bias
        nmodule_xlocal_dense_params = (nmodule_xlocal_dim * nmodule_xlocal_units) + nmodule_xlocal_units  # weights + bias
        
        # Second layer: concatenated (spatial_units + nmodule_xlocal_units) -> dense2_units
        concat_dim = spatial_units + nmodule_xlocal_units
        dense2_params = (concat_dim * dense2_units) + dense2_units
        
        # Third layer: dense2_units -> dense3_units
        dense3_params = (dense2_units * dense3_units) + dense3_units
        
        # Output layer: dense3_units -> 1
        output_params = dense3_units + 1

        total_params = spatial_dense_params + nmodule_xlocal_dense_params + dense2_params + dense3_params + output_params

        layer_structure = [
            {'name': 'x_profile', 'type': 'Input', 'shape': 21},
            {'name': 'y_profile', 'type': 'Input', 'shape': 13},
            {'name': 'y_local', 'type': 'Input', 'shape': 1},
            {'name': 'nModule', 'type': 'Input', 'shape': 1},
            {'name': 'x_local', 'type': 'Input', 'shape': 1},
            {'name': 'other_dense', 'type': 'Dense/QDense', 'units': spatial_units, 'parameters': spatial_dense_params},
            {'name': 'nmodule_xlocal_dense', 'type': 'Dense/QDense (8-bit)', 'units': nmodule_xlocal_units, 'parameters': nmodule_xlocal_dense_params},
            {'name': 'concatenate', 'type': 'Concatenate', 'units': concat_dim, 'parameters': 0},
            {'name': 'dense2', 'type': 'Dense/QDense', 'units': dense2_units, 'parameters': dense2_params},
            {'name': 'dense3', 'type': 'Dense/QDense', 'units': dense3_units, 'parameters': dense3_params},
            {'name': 'output', 'type': 'Dense/QDense', 'units': 1, 'parameters': output_params}
        ]

        return {
            'total_parameters': int(total_params),
            'trainable_parameters': int(total_params),
            'non_trainable_parameters': 0,
            'num_layers': len(layer_structure),
            'layer_structure': layer_structure
        }


def main():
    """Example usage of Model2.5"""
    print("=== Model2.5 Example Usage ===")

    model25 = Model2_5(
        tfRecordFolder="/local/d1/smartpixML/2026Datasets/Data_Files/Data_Set_2026Feb/TF_Records/filtering_records16384_data_shuffled_single_bigData",
        dense_units=128,
        nmodule_xlocal_units=32,
        dense2_units=128,
        dense3_units=64,
        dropout_rate=0.1,
        initial_lr=1e-3,
        end_lr=1e-4,
        power=2,
        bit_configs=[(4, 0)],
        # nmodule_xlocal_weight_bits=8,  # Original: 8-bit nModule_xlocal
        nmodule_xlocal_weight_bits=4,    # Changed to 4-bit nModule_xlocal
        nmodule_xlocal_int_bits=0
    )

    results = model25.runAllStuff(numEpochs = 20)

    print("Model2.5 quantization testing completed successfully!")


if __name__ == "__main__":
    main()

