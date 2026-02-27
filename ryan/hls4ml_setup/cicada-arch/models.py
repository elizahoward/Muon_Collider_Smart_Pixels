import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Activation,
    Add,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Conv3D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Lambda, 
    Reshape,
    UpSampling2D,
    Conv2DTranspose,
    ZeroPadding1D,
    Cropping2D, 
)
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import HeNormal
from qkeras import QActivation, QConv2D, QDense, QDenseBatchnorm, quantized_bits
from larq.layers import QuantConv2D, QuantConv3D, QuantDense
from larq.quantizers import SteSign


class TeacherAutoencoder:
    def __init__(self, input_shape: tuple):
        self.input_shape = input_shape

    def get_model(self):
        inputs = Input(shape=self.input_shape, name="teacher_inputs_")
        x = Reshape((18, 14, 1), name="teacher_reshape")(inputs)
        x = Conv2D(20, (3, 3), strides=1, padding="same", name="teacher_conv2d_1")(x)
        x = Activation("relu", name="teacher_relu_1")(x)
        x = AveragePooling2D((2, 2), name="teacher_pool_1")(x)
        x = Conv2D(30, (3, 3), strides=1, padding="same", name="teacher_conv2d_2")(x)
        x = Activation("relu", name="teacher_relu_2")(x)
        x = Flatten(name="teacher_flatten")(x)
        x = Dense(80, activation="relu", name="teacher_latent")(x)
        x = Dense(9 * 7 * 30, name="teacher_dense")(x)
        x = Reshape((9, 7, 30), name="teacher_reshape2")(x)
        x = Activation("relu", name="teacher_relu_3")(x)
        x = Conv2D(30, (3, 3), strides=1, padding="same", name="teacher_conv2d_3")(x)
        x = Activation("relu", name="teacher_relu_4")(x)
        x = UpSampling2D((2, 2), name="teacher_upsampling")(x)
        x = Conv2D(20, (3, 3), strides=1, padding="same", name="teacher_conv2d_4")(x)
        x = Activation("relu", name="teacher_relu_5")(x)
        outputs = Conv2D(
            1,
            (3, 3),
            activation="relu",
            strides=1,
            padding="same",
            name="teacher_outputs",
        )(x)
        return Model(inputs, outputs, name="teacher")


class TeacherAutoencoderRevised:
    def __init__(self, input_shape: tuple):
        self.input_shape = input_shape

    def get_model(self):
        inputs = Input(shape=self.input_shape, name="teacher_inputs_")
        x = Reshape((18, 14, 1), name="teacher_reshape")(inputs)
        x = Conv2D(20, (3, 3), strides=1, padding="same", name="teacher_conv2d_1")(x)
        x = Activation("relu", name="teacher_relu_1")(x)
        x = AveragePooling2D((2, 2), name="teacher_pool_1")(x)
        x = Conv2D(30, (3, 3), strides=1, padding="same", name="teacher_conv2d_2")(x)
        x = Activation("relu", name="teacher_relu_2")(x)
        x = Flatten(name="teacher_flatten")(x)
        x = Dense(80, activation="relu", name="teacher_latent")(x)
        x = Dense(9 * 7 * 30, name="teacher_dense")(x)
        x = Reshape((9, 7, 30), name="teacher_reshape2")(x)
        x = Activation("relu", name="teacher_relu_3")(x)
        x = Conv2D(30, (3, 3), strides=1, padding="same", name="teacher_conv2d_3")(x)
        x = Activation("relu", name="teacher_relu_4")(x)
        x = Conv2DTranspose(30, (3, 3), strides=2, padding="same", name="teacher_conv_transpose")(x)
        x = Conv2D(20, (3, 3), strides=1, padding="same", name="teacher_conv2d_4")(x)
        x = Activation("relu", name="teacher_relu_5")(x)
        outputs = Conv2D(
            1,
            (3, 3),
            activation="relu",
            strides=1,
            padding="same",
            name="teacher_outputs",
        )(x)
        return Model(inputs, outputs, name="teacher-transpose")


class CicadaV1:
    def __init__(self, input_shape: tuple):
        self.input_shape = input_shape

    def get_model(self):
        inputs = Input(shape=self.input_shape, name="inputs_")
        x = QDenseBatchnorm(
            16,
            kernel_quantizer=quantized_bits(8, 1, 1, alpha=1.0),
            bias_quantizer=quantized_bits(8, 3, 1, alpha=1.0),
            name="dense1",
        )(inputs)
        x = QActivation("quantized_relu(10, 6)", name="relu1")(x)
        x = Dropout(1 / 8)(x)
        x = QDense(
            1,
            kernel_quantizer=quantized_bits(12, 3, 1, alpha=1.0),
            use_bias=False,
            name="dense2",
        )(x)
        outputs = QActivation("quantized_relu(16, 8)", name="outputs")(x)
        return Model(inputs, outputs, name="cicada-v1")


class CicadaV2:
    def __init__(self, input_shape: tuple):
        self.input_shape = input_shape

    def get_model(self):
        inputs = Input(shape=self.input_shape, name="inputs_")
        x = Reshape((18, 14, 1), name="reshape")(inputs)
        x = QConv2D(
            4,
            (2, 2),
            strides=2,
            padding="valid",
            use_bias=False,
            kernel_quantizer=quantized_bits(12, 3, 1, alpha=1.0),
            name="conv",
        )(x)
        x = QActivation("quantized_relu(10, 6)", name="relu0")(x)
        x = Flatten(name="flatten")(x)
        x = Dropout(1 / 9)(x)
        x = QDenseBatchnorm(
            16,
            kernel_quantizer=quantized_bits(8, 1, 1, alpha=1.0),
            bias_quantizer=quantized_bits(8, 3, 1, alpha=1.0),
            name="dense1",
        )(x)
        x = QActivation("quantized_relu(10, 6)", name="relu1")(x)
        x = Dropout(1 / 8)(x)
        x = QDense(
            1,
            kernel_quantizer=quantized_bits(12, 3, 1, alpha=1.0),
            use_bias=False,
            name="dense2",
        )(x)
        outputs = QActivation("quantized_relu(16, 8)", name="outputs")(x)
        return Model(inputs, outputs, name="cicada-v2")


class CNN_Trial:
    def __init__(self, input_shape: tuple, id: int):
        self.params = {
            "input_shape": input_shape, 
            "n_conv_layers": 1, 
            "n_dense_layers": 1, 
            "n_layers": 2, 
            "n_filters": [4], 
            "kernel_dims": [[2, 2]], 
            "stride_dims": [[2, 2]],
            "use_bias_conv": False, 
            "n_dense_units": [16], 
            "q_kernel_conv_bits": 12, 
            "q_kernel_conv_ints": 3, 
            "q_bias_conv_bits": 12, 
            "q_bias_conv_ints": 3, 
            "q_kernel_dense_bits": 8, 
            "q_kernel_dense_ints": 1, 
            "q_bias_dense_bits": 8, 
            "q_bias_dense_ints": 3, 
            "q_activation": "quantized_relu(10, 6)", 
            "shortcut": False, 
            "dropout": 0., 
            "id": id, 
        }

    def get_trial(self, trial):

        # Layers
        n_conv_layers = trial.suggest_int('n_conv_layers', 0, 2)
        n_dense_layers = trial.suggest_int('n_dense_layers', 0, 4)
        n_layers = n_conv_layers + n_dense_layers

        # Conv
        n_filters = [trial.suggest_int(f'n_filters_{i}', 1, 8) for i in range(n_conv_layers)]
        kernel_width = [trial.suggest_int(f'kernel_width_{i}', 1, 4) for i in range(n_conv_layers)]
        kernel_height = [trial.suggest_int(f'kernel_height_{i}', 1, 4) for i in range(n_conv_layers)]
        kernel_dims = np.stack((kernel_width, kernel_height), axis=-1)
        stride_width, stride_height = [], []
        for i in range(n_conv_layers):
            stride_width.append(trial.suggest_int(f'stride_width_{i}', 1, kernel_width[i]))
            stride_height.append(trial.suggest_int(f'stride_height_{i}', 1, kernel_height[i]))
        stride_dims = np.stack((stride_width, stride_height), axis=-1)
        if n_conv_layers>0: use_bias_conv = trial.suggest_categorical("use_bias_conv", [True, False])

        # Dense
        conv_in_dims = [18, 14]
        conv_out_dims = []
        for i in range(len(conv_in_dims)):
            conv_out_dim = conv_in_dims[i]
            for j in range(n_conv_layers):
                conv_out_dim = ((conv_out_dim-kernel_dims[j][i])//stride_dims[j][i])+1
            conv_out_dims.append(conv_out_dim)
        conv_out_size = np.prod(conv_out_dims)
        if n_conv_layers > 0:
            conv_out_size = conv_out_size * n_filters[-1]

        if conv_out_size < 2: return None, 0
        max_size = 0
        while 2 ** max_size <= conv_out_size:
            max_size = max_size + 1
        if n_dense_layers > 0:
            n_dense_units = [trial.suggest_int(f'n_dense_units_0', 1, max_size-1)]
            for i in range(1, n_dense_layers):
                n_dense_units.append(trial.suggest_int(f'n_dense_units_{i}', 1, n_dense_units[-1]))
            n_dense_units = [2 ** n_dense_unit for n_dense_unit in n_dense_units]

        # Quantizations
        if n_conv_layers > 0:
            q_kernel_conv_bits = trial.suggest_int('q_kernel_conv_bits', 1, 32)
            q_kernel_conv_ints = trial.suggest_int('q_kernel_conv_ints', 1, q_kernel_conv_bits)
            if use_bias_conv:
                q_bias_conv_bits = trial.suggest_int('q_bias_conv_bits', 1, 32)
                q_bias_conv_ints = trial.suggest_int('q_bias_conv_ints', 1, q_bias_conv_bits)
        if n_dense_layers > 0:
            q_kernel_dense_bits = trial.suggest_int('q_kernel_dense_bits', 1, 32)
            q_kernel_dense_ints = trial.suggest_int('q_kernel_dense_ints', 1, q_kernel_dense_bits)
            q_bias_dense_bits = trial.suggest_int('q_bias_dense_bits', 1, 32)
            q_bias_dense_ints = trial.suggest_int('q_bias_dense_ints', 1, q_bias_dense_bits)
        q_activation_bits = trial.suggest_int('q_activation_bits', 1, 32)
        q_activation_ints = trial.suggest_int('q_activation_ints', 1, q_activation_bits)
        q_activation = f"quantized_relu({q_activation_bits}, {q_activation_ints})"

        # Shortcut, dropout
        shortcut = trial.suggest_categorical('shortcut', [True, False])
        dropout = trial.suggest_float('dropout', 0., 0.5)

        for key, value in [
            ["n_conv_layers", n_conv_layers],
            ["n_dense_layers", n_dense_layers],
            ["n_layers", n_layers],
            ["q_activation", q_activation],
            ["shortcut", shortcut],
            ["dropout", dropout],
            ]:
            self.params.update({key: value})
        
        if n_conv_layers > 0:
            for key, value in [
                ["n_filters", n_filters],
                ["kernel_dims", kernel_dims],
                ["stride_dims", stride_dims],
                ["use_bias_conv", use_bias_conv], 
                ["q_kernel_conv_bits", q_kernel_conv_bits],
                ["q_kernel_conv_ints", q_kernel_conv_ints],
                ]:
                self.params.update({key: value})

            if use_bias_conv:
                for key, value in [
                    ["q_bias_conv_bits", q_bias_conv_bits],
                    ["q_bias_conv_ints", q_bias_conv_ints],
                    ]:
                    self.params.update({key: value})
        
        if n_dense_layers > 0:
            for key, value in [
                ["n_dense_units", n_dense_units],
                ["q_kernel_dense_bits", q_kernel_dense_bits],
                ["q_kernel_dense_ints", q_kernel_dense_ints],
                ["q_bias_dense_bits", q_bias_dense_bits],
                ["q_bias_dense_ints", q_bias_dense_ints],
                ]:
                self.params.update({key: value})

        return self.create()

    def get_model(self, params):

        for key in params.keys():
            self.params[key] = params[key]

        # Layers
        n_conv_layers = self.params['n_conv_layers']
        n_dense_layers = self.params['n_dense_layers']
        n_layers = n_conv_layers + n_dense_layers

        # Conv
        n_filters = [self.params[f'n_filters_{i}'] for i in range(n_conv_layers)]
        kernel_width = [self.params[f'kernel_width_{i}'] for i in range(n_conv_layers)]
        kernel_height = [self.params[f'kernel_height_{i}'] for i in range(n_conv_layers)]
        kernel_dims = np.stack((kernel_width, kernel_height), axis=-1)
        stride_width, stride_height = [], []
        for i in range(n_conv_layers):
            stride_width.append(self.params[f'stride_width_{i}'])
            stride_height.append(self.params[f'stride_height_{i}'])
        stride_dims = np.stack((stride_width, stride_height), axis=-1)
        if n_conv_layers>0: use_bias_conv = self.params["use_bias_conv"]

        # Dense
        conv_in_dims = [18, 14]
        conv_out_dims = []
        for i in range(len(conv_in_dims)):
            conv_out_dim = conv_in_dims[i]
            for j in range(n_conv_layers):
                conv_out_dim = ((conv_out_dim-kernel_dims[j][i])//stride_dims[j][i])+1
            conv_out_dims.append(conv_out_dim)
        conv_out_size = np.prod(conv_out_dims)
        if n_conv_layers > 0:
            conv_out_size = conv_out_size * n_filters[-1]

        if conv_out_size < 2: return None, 0
        max_size = 0
        while 2 ** max_size <= conv_out_size:
            max_size = max_size + 1
        if n_dense_layers > 0:
            n_dense_units = []
            for i in range(n_dense_layers):
                n_dense_units.append(self.params[f'n_dense_units_{i}'])
            n_dense_units = [2 ** n_dense_unit for n_dense_unit in n_dense_units]

        # Quantizations
        if n_conv_layers > 0:
            q_kernel_conv_bits = self.params['q_kernel_conv_bits']
            q_kernel_conv_ints = self.params['q_kernel_conv_ints']
            if use_bias_conv:
                q_bias_conv_bits = self.params['q_bias_conv_bits']
                q_bias_conv_ints = self.params['q_bias_conv_ints']
        if n_dense_layers > 0:
            q_kernel_dense_bits = self.params['q_kernel_dense_bits']
            q_kernel_dense_ints = self.params['q_kernel_dense_ints']
            q_bias_dense_bits = self.params['q_bias_dense_bits']
            q_bias_dense_ints = self.params['q_bias_dense_ints']
        q_activation_bits = self.params['q_activation_bits']
        q_activation_ints = self.params['q_activation_ints']
        q_activation = f"quantized_relu({q_activation_bits}, {q_activation_ints})"

        # Shortcut, dropout
        shortcut = self.params['shortcut']
        dropout = self.params['dropout']

        for key, value in [
            ["n_conv_layers", n_conv_layers],
            ["n_dense_layers", n_dense_layers],
            ["n_layers", n_layers],
            ["q_activation", q_activation],
            ["shortcut", shortcut],
            ["dropout", dropout],
            ]:
            self.params.update({key: value})
        
        if n_conv_layers > 0:
            for key, value in [
                ["n_filters", n_filters],
                ["kernel_dims", kernel_dims],
                ["stride_dims", stride_dims],
                ["use_bias_conv", use_bias_conv], 
                ["q_kernel_conv_bits", q_kernel_conv_bits],
                ["q_kernel_conv_ints", q_kernel_conv_ints],
                ]:
                self.params.update({key: value})

            if use_bias_conv:
                for key, value in [
                    ["q_bias_conv_bits", q_bias_conv_bits],
                    ["q_bias_conv_ints", q_bias_conv_ints],
                    ]:
                    self.params.update({key: value})
        
        if n_dense_layers > 0:
            for key, value in [
                ["n_dense_units", n_dense_units],
                ["q_kernel_dense_bits", q_kernel_dense_bits],
                ["q_kernel_dense_ints", q_kernel_dense_ints],
                ["q_bias_dense_bits", q_bias_dense_bits],
                ["q_bias_dense_ints", q_bias_dense_ints],
                ]:
                self.params.update({key: value})

        return self.create()

    def convblock(self, x, it):
        if self.params["shortcut"]:
            cut = QConv2D(
                self.params["n_filters"][it], 
                (1, 1), 
                strides=(self.params["stride_dims"][it][0], self.params["stride_dims"][it][1]), 
                padding='valid', 
                use_bias=False,
                kernel_initializer=self.initializer, 
                kernel_quantizer=quantized_bits(self.params["q_kernel_conv_bits"], self.params["q_kernel_conv_ints"], 1, alpha=1.0),
                name=f'shortcut{it}', 
            )(x)

        x = QConv2D(
            self.params["n_filters"][it], 
            (self.params["kernel_dims"][it][0], self.params["kernel_dims"][it][1]), 
            strides=(self.params["stride_dims"][it][0], self.params["stride_dims"][it][1]), 
            padding='valid', 
            use_bias=self.params["use_bias_conv"], 
            kernel_initializer=self.initializer, 
            bias_initializer=self.initializer, 
            kernel_quantizer=quantized_bits(self.params["q_kernel_conv_bits"], self.params["q_kernel_conv_ints"], 1, alpha=1.0), 
            bias_quantizer=quantized_bits(self.params["q_bias_conv_bits"], self.params["q_bias_conv_ints"], 1, alpha=1.0), 
            name=f'conv{it}',
        )(x)
        
        if self.params["shortcut"]:
            x = BatchNormalization()(x) # should be folded into QConv2D
            cut = Cropping2D(cropping=((cut.shape[1]-x.shape[1], 0), (cut.shape[2]-x.shape[2], 0)))(cut)
            cut = BatchNormalization()(cut) # should be folded into QConv2D
            x = Add()([x, cut])

        x = QActivation(self.params["q_activation"], name=f'relu{it}')(x)
        x = Dropout(self.params["dropout"])(x)

        size=self.params["n_filters"][it] * self.params["kernel_dims"][it][0] * self.params["kernel_dims"][it][1] * self.params["q_kernel_conv_bits"]
        if self.params["use_bias_conv"]:
            size=size+self.params["n_filters"][it] * self.params["kernel_dims"][it][0] * self.params["kernel_dims"][it][1] * self.params["q_bias_conv_bits"]
        return x, size

    def denseblock(self, x, it):
        input_dim = x.shape[-1]
        if self.params["shortcut"]:
            output_dim = self.params["n_dense_units"][it]
            cut = Lambda(lambda t: t[:, :output_dim])(x)
            cut = BatchNormalization()(cut) # should be folded

        x = QDenseBatchnorm(
            self.params["n_dense_units"][it], 
            kernel_initializer=self.initializer, 
            kernel_quantizer=quantized_bits(self.params["q_kernel_dense_bits"], self.params["q_kernel_dense_ints"], 1, alpha=1.0),
            bias_quantizer=quantized_bits(self.params["q_bias_dense_bits"], self.params["q_bias_dense_ints"], 1, alpha=1.0),  
            name=f'dense{it}',
        )(x)

        if self.params["shortcut"]:
            x = Add()([x, cut])

        x = QActivation(self.params["q_activation"], name=f'relu{it+self.params["n_conv_layers"]}')(x)
        x = Dropout(self.params["dropout"])(x)
        output_dim = x.shape[-1]
        return x, (input_dim * output_dim) * (self.params["q_kernel_dense_bits"] + self.params["q_bias_dense_bits"])

    def create(self):

        total_bits = 0

        # Initializer
        self.initializer = HeNormal(seed=self.params["id"])

        # Input layer
        inputs = Input(shape=(self.params["input_shape"]), name="input")
        x = Reshape((18, 14, 1), name='reshape')(inputs)

        # Convolutional layers
        for i in range(self.params["n_conv_layers"]):
            x, bits = self.convblock(x, i)
            total_bits = total_bits + bits
            
        x = Flatten(name='flatten')(x)

        # Dense layers
        for i in range(self.params["n_dense_layers"]):
            x, bits = self.denseblock(x, i)
            total_bits = total_bits + bits
        total_bits = total_bits + (x.shape[-1]*12)

        # Output layer
        x = QDense(
            1, 
            kernel_initializer=self.initializer, 
            kernel_quantizer=quantized_bits(12, 3, 1, alpha=1.0),
            use_bias=False,
            name='dense_output',
        )(x)
        outputs = QActivation("quantized_relu(16, 8)", name='outputs')(x)

        return Model(inputs, outputs, name='cnn'), total_bits


class Binary_Trial:
    def __init__(self, input_shape: tuple, binary_type: str, id: int):
        self.params = {
            "input_shape": input_shape, 
            "binary_type": binary_type, 
            "n_conv_layers": 1, 
            "n_dense_layers": 1, 
            "n_layers": 2, 
            "n_filters": [4], 
            "kernel_dims": [[2, 2]], 
            "stride_dims": [[2, 2]],
            "use_bias_conv": False, 
            "n_dense_units": [16], 
            "q_activation": "quantized_relu(10, 6)", 
            "shortcut": False, 
            "dropout": 0., 
            "id": id, 
        }

    def get_trial(self, trial):

        # Type
        binary_type = trial.suggest_categorical('binary_type', ['bnn', 'ban', 'bwn'])

        # Layers
        n_conv_layers = trial.suggest_int('n_conv_layers', 0, 2)
        n_dense_layers = trial.suggest_int('n_dense_layers', 0, 4)
        n_layers = n_conv_layers + n_dense_layers

        # Conv
        n_filters = [trial.suggest_int(f'n_filters_{i}', 1, 8) for i in range(n_conv_layers)]
        kernel_width = [trial.suggest_int(f'kernel_width_{i}', 1, 4) for i in range(n_conv_layers)]
        kernel_height = [trial.suggest_int(f'kernel_height_{i}', 1, 4) for i in range(n_conv_layers)]
        kernel_dims = np.stack((kernel_width, kernel_height), axis=-1)
        stride_width, stride_height = [], []
        for i in range(n_conv_layers):
            stride_width.append(trial.suggest_int(f'stride_width_{i}', 1, kernel_width[i]))
            stride_height.append(trial.suggest_int(f'stride_height_{i}', 1, kernel_height[i]))
        stride_dims = np.stack((stride_width, stride_height), axis=-1)
        use_bias_conv = trial.suggest_categorical("use_bias_conv", [True, False])

        # Dense
        conv_in_dims = [18, 14]
        conv_out_dims = []
        for i in range(len(conv_in_dims)):
            conv_out_dim = conv_in_dims[i]
            for j in range(n_conv_layers):
                conv_out_dim = ((conv_out_dim-kernel_dims[j][i])//stride_dims[j][i])+1
            conv_out_dims.append(conv_out_dim)
        conv_out_size = np.prod(conv_out_dims)
        if n_conv_layers > 0:
            conv_out_size = conv_out_size * n_filters[-1]

        if conv_out_size < 2: return None, 0
        max_size = 0
        while 2 ** max_size <= conv_out_size:
            max_size = max_size + 1
        if n_dense_layers > 0:
            n_dense_units = [trial.suggest_int(f'n_dense_units_0', 1, max_size+3)] # +4 relative to CNN_Trial; 8 bits per byte, extra factor of 2 for flexibility
            for i in range(1, n_dense_layers):
                n_dense_units.append(trial.suggest_int(f'n_dense_units_{i}', 1, n_dense_units[-1]))
            n_dense_units = [2 ** n_dense_unit for n_dense_unit in n_dense_units]

        # Quantizations
        q_activation_bits = trial.suggest_int('q_activation_bits', 1, 32)
        q_activation_ints = trial.suggest_int('q_activation_ints', 1, q_activation_bits)
        q_activation = f"quantized_relu({q_activation_bits}, {q_activation_ints})"

        # Shortcut, dropout
        shortcut = trial.suggest_categorical('shortcut', [True, False])
        dropout = trial.suggest_float('dropout', 0., 0.5)

        for key, value in [
            ["binary_type", binary_type], 
            ["n_conv_layers", n_conv_layers],
            ["n_dense_layers", n_dense_layers],
            ["n_layers", n_layers],
            ["q_activation", q_activation],
            ["shortcut", shortcut],
            ["dropout", dropout],
            ]:
            self.params.update({key: value})
        
        if n_conv_layers > 0:
            for key, value in [
                ["n_filters", n_filters],
                ["kernel_dims", kernel_dims],
                ["stride_dims", stride_dims],
                ["use_bias_conv", use_bias_conv], 
                ]:
                self.params.update({key: value})
        
        if n_dense_layers > 0:
            for key, value in [
                ["n_dense_units", n_dense_units],
                ]:
                self.params.update({key: value})

        return self.create()

    def get_model(self, params):

        for key in params.keys():
            self.params[key] = params[key]
        
        # Type
        binary_type = self.params['binary_type']

        # Layers
        n_conv_layers = self.params['n_conv_layers']
        n_dense_layers = self.params['n_dense_layers']
        n_layers = n_conv_layers + n_dense_layers

        # Conv
        n_filters = [self.params[f'n_filters_{i}'] for i in range(n_conv_layers)]
        kernel_width = [self.params[f'kernel_width_{i}'] for i in range(n_conv_layers)]
        kernel_height = [self.params[f'kernel_height_{i}'] for i in range(n_conv_layers)]
        kernel_dims = np.stack((kernel_width, kernel_height), axis=-1)
        stride_width, stride_height = [], []
        for i in range(n_conv_layers):
            stride_width.append(self.params[f'stride_width_{i}'])
            stride_height.append(self.params[f'stride_height_{i}'])
        stride_dims = np.stack((stride_width, stride_height), axis=-1)
        use_bias_conv = self.params["use_bias_conv"]

        # Dense
        conv_in_dims = [18, 14]
        conv_out_dims = []
        for i in range(len(conv_in_dims)):
            conv_out_dim = conv_in_dims[i]
            for j in range(n_conv_layers):
                conv_out_dim = ((conv_out_dim-kernel_dims[j][i])//stride_dims[j][i])+1
            conv_out_dims.append(conv_out_dim)
        conv_out_size = np.prod(conv_out_dims)
        if n_conv_layers > 0:
            conv_out_size = conv_out_size * n_filters[-1]

        if conv_out_size < 2: return None, 0
        max_size = 0
        while 2 ** max_size <= conv_out_size:
            max_size = max_size + 1
        if n_dense_layers > 0:
            n_dense_units = [] # +4 relative to CNN_Trial; 8 bits per byte, extra factor of 2 for flexibility
            for i in range(n_dense_layers):
                n_dense_units.append(self.params[f'n_dense_units_{i}'])
            n_dense_units = [2 ** n_dense_unit for n_dense_unit in n_dense_units]

        # Quantizations
        q_activation_bits = self.params['q_activation_bits']
        q_activation_ints = self.params['q_activation_ints']
        q_activation = f"quantized_relu({q_activation_bits}, {q_activation_ints})"

        # Shortcut, dropout
        shortcut = self.params['shortcut']
        dropout = self.params['dropout']

        for key, value in [
            ["binary_type", binary_type], 
            ["n_conv_layers", n_conv_layers],
            ["n_dense_layers", n_dense_layers],
            ["n_layers", n_layers],
            ["q_activation", q_activation],
            ["shortcut", shortcut],
            ["dropout", dropout],
            ]:
            self.params.update({key: value})
        
        if n_conv_layers > 0:
            for key, value in [
                ["n_filters", n_filters],
                ["kernel_dims", kernel_dims],
                ["stride_dims", stride_dims],
                ["use_bias_conv", use_bias_conv], 
                ]:
                self.params.update({key: value})
        
        if n_dense_layers > 0:
            for key, value in [
                ["n_dense_units", n_dense_units],
                ]:
                self.params.update({key: value})

        return self.create()

    def convblock(self, x, it):
        if self.params["binary_type"] == 'ban':
            kernel_quantizer = None
        else:
            kernel_quantizer = SteSign()
        if self.params["binary_type"] == 'bwn' or it == 0: # Don't binarize if first layer
            input_quantizer = None
        else:
            input_quantizer = SteSign()

        if self.params["shortcut"]:
            cut = QConv2D(
                self.params["n_filters"][it], 
                (1, 1), 
                strides=(self.params["stride_dims"][it][0], self.params["stride_dims"][it][1]), 
                padding='valid', 
                use_bias=False,
                kernel_initializer=self.initializer, 
                kernel_quantizer=quantized_bits(12, 3, 1, alpha=1.0),
                name=f'shortcut{it}', 
            )(x)

        x = QuantConv2D(
            self.params["n_filters"][it], 
            (self.params["kernel_dims"][it][0], self.params["kernel_dims"][it][1]), 
            strides=(self.params["stride_dims"][it][0], self.params["stride_dims"][it][1]), 
            padding='valid', 
            use_bias=self.params["use_bias_conv"], 
            input_quantizer=input_quantizer, 
            kernel_quantizer=kernel_quantizer, 
            kernel_constraint="weight_clip", 
            kernel_initializer=self.initializer, 
            name=f'conv{it}',
        )(x)
        x = BatchNormalization(momentum=0.9)(x) # Should be folded into QuantConv2D after training
        
        if self.params["shortcut"]:
            cut = Cropping2D(cropping=((cut.shape[1]-x.shape[1], 0), (cut.shape[2]-x.shape[2], 0)))(cut)
            cut = BatchNormalization(momentum=0.9)(cut) # should be folded into QConv2D
            x = Add()([x, cut])

        x = QActivation(self.params["q_activation"], name=f'relu{it}')(x)
        x = Dropout(self.params["dropout"])(x)

        size=self.params["n_filters"][it] * self.params["kernel_dims"][it][0] * self.params["kernel_dims"][it][1]
        if self.params["use_bias_conv"]:
            size=size+self.params["n_filters"][it] * self.params["kernel_dims"][it][0] * self.params["kernel_dims"][it][1] * 32
        return x, size
    
    def denseblock(self, x, it):
        if self.params["binary_type"] == 'ban':
            kernel_quantizer = None
        else:
            kernel_quantizer = SteSign()
        if self.params["binary_type"] == 'bwn' or it == 0:
            input_quantizer = None
        else:
            input_quantizer = SteSign()

        input_dim = x.shape[-1]
        if self.params["shortcut"]:
            output_dim = self.params["n_dense_units"][it]
            if input_dim >= output_dim:
                cut = Lambda(lambda t: t[:, :output_dim])(x)
            else:
                cut = Reshape((input_dim, 1), name=f'shortcut_reshape_prepad{it}')(x)
                cut = ZeroPadding1D(padding=(0, output_dim-input_dim))(cut)
                cut = Reshape((output_dim, ), name=f'shortcut_reshape_postpad{it}')(cut)

            cut = BatchNormalization(momentum=0.9)(cut) # should be folded

        #print(type(self.params["n_dense_units"][it]))
        x = QuantDense(
            self.params["n_dense_units"][it], 
            input_quantizer=input_quantizer, 
            kernel_quantizer=kernel_quantizer,
            kernel_initializer=self.initializer, 
            kernel_constraint="weight_clip", 
            name=f'dense{it}',
        )(x)
        x = BatchNormalization(momentum=0.9)(x) # Should be folded into QuantConv2D after training

        if self.params["shortcut"]:
            x = Add()([x, cut])

        x = QActivation(self.params["q_activation"], name=f'relu{it+self.params["n_conv_layers"]}')(x)
        x = Dropout(self.params["dropout"])(x)

        output_dim = x.shape[-1]
        if self.params["binary_type"] == "bnn" or self.params["binary_type"] == "bwn":
            size = input_dim * output_dim
        elif self.params["binary_type"] == "ban":
            size = input_dim * output_dim * 32
        return x, size

    def create(self):

        total_bits = 0

        # Initializer
        self.initializer = HeNormal(seed=self.params["id"])

        # Input layer
        inputs = Input(shape=(self.params["input_shape"]), name="input")

        x = Reshape((18, 14, 1), name='reshape')(inputs)

        # Convolutional layers
        for i in range(self.params["n_conv_layers"]):
            x, bits = self.convblock(x, i)
            total_bits = total_bits + bits
            
        x = Flatten(name='flatten')(x)

        # Dense layers
        for i in range(self.params["n_dense_layers"]):
            x, bits = self.denseblock(x, i)
            total_bits = total_bits + bits
        total_bits = total_bits + (x.shape[-1]*12)

        # Output layer
        x = QDense(
            1, 
            kernel_initializer=self.initializer, 
            kernel_quantizer=quantized_bits(12, 3, 1, alpha=1.0),
            use_bias=False,
            name='dense_output',
        )(x)
        outputs = QActivation("quantized_relu(16, 8)", name='outputs')(x)

        return Model(inputs, outputs, name=self.params['binary_type']), total_bits
    

class Bit_Binary_Trial:
    def __init__(self, input_shape: tuple, binary_type: str):
        self.params = {
            "input_shape": input_shape, 
            "binary_type": binary_type, 
            "n_filters": [], 
            "n_dense_units": [10], 
            "n_conv_layers": 0, 
            "n_dense_layers": 1, 
            "n_layers": 1, 
            "shortcut": True, 
            "dropout": 0, 
        }

    def get_trial(self, trial):
        n_filters = [trial.suggest_int(f'n_dense_units_{i}', 0, 10) for i in range(trial.suggest_int('n_dense_layers', 0, 1))]
        n_dense_units = [trial.suggest_int(f'n_dense_units_{i}', 0, 10) for i in range(trial.suggest_int('n_dense_layers', 1, 5))]
        n_dense_units = [2 ** n_dense_unit for n_dense_unit in n_dense_units]
        n_conv_layers = len(n_filters)
        n_dense_layers = len(n_dense_units)
        n_layers = n_conv_layers + n_dense_layers
        shortcut = trial.suggest_categorical('shortcut', [True, False])
        dropout = 1/trial.suggest_int('dropout', 0, 20)
        self.params = {
            "n_filters": n_filters, 
            "n_dense_units": n_dense_units, 
            "n_layers": n_layers, 
            "n_conv_layers": n_conv_layers, 
            "n_dense_layers": n_dense_layers, 
            "n_layers": n_layers, 
            "shortcut": shortcut, 
            "dropout": dropout, 
        }
        return self.create()

    def get_model(self, params):
        for key in list(self.params.keys()):
            if key in list(params.keys()):
                self.params[key] = params[key]

        self.params['n_dense_units'] = [2 ** n_dense_unit for n_dense_unit in self.params['n_dense_units']]

        return self.create()
    
    def conv2Dblock(self, x, it):
        if self.params["shortcut"]:
            cut = QConv2D(
                self.params["n_filters"][it], 
                (1, 1), 
                strides=2, 
                padding='valid', 
                use_bias=False,
                kernel_quantizer=quantized_bits(12, 3, 1, alpha=1.0),
                name=f'shortcut{it}', 
            )(x)
            #cut = BatchNormalization(momentum=0.9)(cut) # should be folded into QConv2D

        x = QConv2D(
            self.params["n_filters"][it], 
            (2, 2), 
            strides=2, 
            padding='valid', 
            use_bias=False, 
            kernel_quantizer=quantized_bits(12, 3, 1, alpha=1.0), 
            name=f'conv{it}',
        )(x)
        #x = BatchNormalization(momentum=0.9)(x) # should be folded into QConv2D
        
        if self.params["shortcut"]:
            x = Add()([x, cut]) 

        x = QActivation("quantized_relu(10, 6)", name=f'relu{it}')(x)
        x = Dropout(self.params["dropout"])(x)
        return x

    def conv3Dblock(self, x, it):
        if it == 0: input_quantizer = None
        else: input_quantizer = 'ste_sign'
        kernel_quantizer = 'ste_sign'

        x = tf.expand_dims(x, -1)

        if self.params["shortcut"]:
            cut = Conv3D(
                self.params["n_filters"][it], 
                (1, 1, 10), 
                strides=2, 
                padding='valid', 
                use_bias=False,
                name=f'shortcut{it}', 
            )(x)
            cut = BatchNormalization(momentum=0.9)(cut) # should be folded into QConv2D

        x = QuantConv3D(
            self.params["n_filters"][it], 
            (2, 2, 10), 
            strides=2, 
            padding='valid', 
            input_quantizer=input_quantizer,
            kernel_quantizer=kernel_quantizer, 
            kernel_constraint="weight_clip", 
            name=f'conv{it}',
        )(x)
        x = BatchNormalization(momentum=0.9)(x) # should be folded into QConv2D
        
        if self.params["shortcut"]:
            x = Add()([x, cut]) 

        x = QActivation("quantized_relu(10, 6)", name=f'relu{it}')(x)
        x = Dropout(self.params["dropout"])(x)
        return x

    def denseblock(self, x, it):
        it_dense = it - self.params["n_conv_layers"]
        if it == 0: input_quantizer = None
        else: input_quantizer = 'ste_sign'
        kernel_quantizer = 'ste_sign'

        if self.params["shortcut"]:
            #input_dim = K.int_shape(x)[-1]
            output_dim = self.params["n_dense_units"][it_dense]
            if input_dim < output_dim:
                pad_size = output_dim - input_dim
                cut = Lambda(lambda t: tf.pad(t, [[0, 0], [0, pad_size]]))(x)
            elif input_dim > output_dim:
                cut = Lambda(lambda t: t[:, :output_dim])(x)
            else:
                cut = x  # No change needed
            cut = BatchNormalization(momentum=0.9)(cut)

        x = QuantDense(
            self.params["n_dense_units"][it_dense], 
            input_quantizer=input_quantizer, 
            kernel_quantizer=kernel_quantizer,
            kernel_constraint="weight_clip", 
            name=f'dense{it_dense}',
        )(x)
        x = BatchNormalization(momentum=0.9)(x) # Should be folded into QuantConv2D after training

        if self.params["shortcut"]:
            x = Add()([x, cut])

        x = QActivation("quantized_relu(10, 6)", name=f'relu{it}')(x)
        x = Dropout(self.params["dropout"])(x)
        return x
    
    def bitplane(self, x):
        # Scale up to preserve information from fractional bits if coming from conv2d
        #x = x * (2**4)

        # Round to ensure ints
        x = tf.cast(x, tf.int32)

        # Generate 10 binary channels using bitwise operations
        x = [(tf.bitwise.right_shift(x, i) & 1) for i in range(10)]

        # Concatenate along the last axis to form a 10-channel binary tensor
        x = tf.concat(x, axis=-1)

        # Convert to float32 since Larq expects float inputs
        x = tf.cast(x, tf.float32)

        return x

    def create(self):

        # Input layer
        inputs = Input(shape=(self.params["input_shape"]), name="input")
        x = Reshape((18, 14, 1), name='reshape')(inputs)

        # Bitplane
        x = self.bitplane(x)

        # Convolutional layers
        for i in range(self.params["n_conv_layers"]):
            x = self.conv3Dblock(x, i)

        # Flatten
        x = Flatten(name='flatten')(x)

        # Dense layers
        for i in range(self.params["n_dense_layers"]):
            x = self.denseblock(x, i+self.params["n_conv_layers"])

        # Output layer
        x = QuantDense(
            1, 
            input_quantizer=None, 
            kernel_quantizer=None, 
            name='dense_output',
        )(x)
        outputs = QActivation("quantized_relu(16, 8)", name='outputs')(x)

        return Model(inputs, outputs, name=self.params["binary_type"])
    
class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, patch_size, n_hidden):
        super().__init__()
        self.patch_size = patch_size
        self.n_hidden = n_hidden
        self.flatten = Flatten()
        self.dense = QDense(
            n_hidden, 
            kernel_quantizer=quantized_bits(8, 1, 1, alpha=1.0), 
            bias_quantizer=quantized_bits(8, 3, 1, alpha=1.0)
        )

    def call(self, x):
        batch_size = tf.shape(x)[0]
        patches = tf.image.extract_patches(
            x, sizes=[1, self.patch_size[0], self.patch_size[1], 1],
            strides=[1, self.patch_size[0], self.patch_size[1], 1],
            rates=[1, 1, 1, 1], padding='VALID'
        )
        patches = self.flatten(patches)
        return self.dense(patches)
