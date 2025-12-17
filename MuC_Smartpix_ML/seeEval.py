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
import qkeras
from qkeras.quantizers import see_quantizer

def build_see_config(config, random_flip_rate=0.0, flip_tensor=None):
    config['module'] = 'qkeras.quantizers'
    config['class_name'] = 'see_quantizer'
    config['registered_name'] = 'see_quantizer'
    config['config']['random_flip_rate'] = random_flip_rate
    config['config']['flip_tensor'] = flip_tensor
    return config

# ~80% ChatGPT. Just makes a copy of the model with modified quantizers.
def clone_functional_model(old_model):
    """
    Deep-clone a Functional model, preserving:
    - Input names
    - Layer names
    - Weights
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
        layer_config = layer.get_config()
        if isinstance(layer, QDense):
            print("Replacing quantized_bits with see_quantizer.")
            layer_config['kernel_quantizer'] = build_see_config(layer_config['kernel_quantizer'], random_flip_rate=0.001)
            layer_config['bias_quantizer']   = build_see_config(layer_config['bias_quantizer'],   random_flip_rate=0.001)

        print(f"Cloning layer: {layer.name} ({layer.__class__.__name__})")
        # print(f"  Config: {layer_config}")

        new_layer = layer.__class__.from_config(layer_config)
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

    old_model = model.models["Unquantized"]
    new_model = clone_functional_model(old_model)
    new_model.compile(metrics=['accuracy'], loss='binary_crossentropy', optimizer='adam', run_eagerly=True)

    print("Running evaluation on original model:")
    model.runEval(False)

    print("Running evaluation on cloned model with modified kernels:")
    model.models["Unquantized"] = new_model
    model.runEval(False)


if __name__== "__main__":
    main()