import sys
sys.path.append('../MuC_Smartpix_ML/')
sys.path.append('../daniel/')
sys.path.append('../ryan/')
sys.path.append('../eric/')
import argparse
import os
import tensorflow as tf

from Model_Classes import SmartPixModel
from model1 import Model1
from model2 import Model2
from model3 import Model3
from utils import buildFromConfig, load_config
from qkeras import QDense

def main():

    parser = argparse.ArgumentParser(description="Evaluate SmartPix ML Models with Quantization")
    add_arg = parser.add_argument
    add_arg('--config', type=str, default=None, help='Path to model config')
    add_arg('--training_dir', type=str, default=None, help='Path to training directory')
    add_arg('--quantization', type=int, default=-1, help='Which version of quantization to use. -1 for no quantization')

    args = parser.parse_args()

    config = load_config(args.config)
    if args.training_dir is not None:
        config['TrainingDirectory'] = args.training_dir

    quantization_tag = 'unquantized' 
    if args.quantization > 0:
        quantization_tag = f'quantized_{args.quantization}bit'

    config['Model']['args']['loadModel'] = True
    config['Model']['args']['modelPath'] = os.path.join(config['TrainingDirectory'], f"models/{config['Model']['class']}_{quantization_tag}.h5")

    model  = buildFromConfig(config['Model'])
    model.runEval(False)

    # ChatGPT special starting here.
    old_model = model.models["Unquantized"]
    def clone_functional_model(old_model, kernel_add=0):
        """
        Deep-clone a Functional model, preserving:
        - Input names
        - Layer names
        - Weights
        Optionally modifies QDense kernels by adding `kernel_add`.
        Works for any multi-input, multi-output Functional model.
        """
        tensor_map = {}

        for layer in old_model.layers:

            # --- Input layers ---
            if isinstance(layer, tf.keras.layers.InputLayer):
                input_shape = layer.input_shape[1:]
                # Ensure rank >=2 for Dense/QDense
                if len(input_shape) == 0:
                    input_shape = (1,)
                new_tensor = tf.keras.Input(
                    shape=input_shape,
                    dtype=layer.dtype,
                    name=layer.name
                )
                tensor_map[layer.output.ref()] = new_tensor
                continue

            # --- Other layers ---
            new_layer = layer.__class__.from_config(layer.get_config())
            new_layer._name = layer._name

            # Map inbound tensors
            inbound = layer.input if isinstance(layer.input, list) else [layer.input]
            new_inputs = [tensor_map[t.ref()] for t in inbound]

            # Call layer
            if len(new_inputs) == 1:
                new_output = new_layer(new_inputs[0])
            else:
                new_output = new_layer(new_inputs)

            # Copy and optionally modify weights
            weights = layer.get_weights()
            if isinstance(layer, QDense) and kernel_add != 0:
                weights[0] = weights[0] + kernel_add
            if weights:
                new_layer.set_weights(weights)

            # Map outputs
            if isinstance(new_output, list):
                for o_old, o_new in zip(layer.output, new_output):
                    tensor_map[o_old.ref()] = o_new
            else:
                tensor_map[layer.output.ref()] = new_output

        # Build cloned model
        new_model = tf.keras.Model(
            inputs=[tensor_map[i.ref()] for i in old_model.inputs],
            outputs=[tensor_map[o.ref()] for o in old_model.outputs],
            name=old_model.name + "_cloned"
        )

        return new_model

    new_model = clone_functional_model(old_model, kernel_add=0.01)
    new_model.compile(metrics=['accuracy'], loss='binary_crossentropy', optimizer='adam')
    model.runEval(False)
    model.models["Unquantized"] = new_model
    model.runEval(False)


if __name__== "__main__":
    main()