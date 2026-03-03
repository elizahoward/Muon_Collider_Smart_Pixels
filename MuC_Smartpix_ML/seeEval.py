import sys
sys.path.append('../MuC_Smartpix_ML/')
sys.path.append('../daniel/')
sys.path.append('../ryan/')
sys.path.append('../eric/')
import argparse
import os
import tensorflow as tf
import numpy as np

from Model_Classes import SmartPixModel, WarmupThenDecay
import Model_Classes
from model1 import Model1
from model2 import Model2
from model3 import Model3
from utils import buildFromConfig, load_config
from qkeras import QDense
import qkeras
import keras
from qkeras.quantizers import see_quantizer
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def build_see_config(config, random_flip_rate=0.0, flip_tensor=None):
    config['module'] = 'qkeras.quantizers'
    config['class_name'] = 'see_quantizer'
    config['registered_name'] = 'see_quantizer'
    config['config']['random_flip_rate'] = random_flip_rate
    config['config']['flip_tensor'] = flip_tensor
    return config

# ~70% ChatGPT. Just makes a copy of the model with modified quantizers.
def clone_functional_model(old_model, random_flip_rate=0.0, flip_bits=[]):
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
            input_shape = layer.input_shape[0][1:]
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
            kernel_flip_tensor = None
            bias_flip_tensor   = None
            if [bit for bit in flip_bits if bit['layer_name'] == layer.name]:
                kernel_flip_tensor = np.zeros([layer.input_shape[-1], layer.units], dtype=np.int32)
                bias_flip_tensor   = np.zeros([layer.units], dtype=np.int32)
                for bit in flip_bits:
                    if bit['layer_name'] == layer.name:
                        if bit['type'] == 'kernel':
                            if bit['bit_position'] == layer.kernel_quantizer['config']['bits'] - 1:
                                kernel_flip_tensor[bit['indices']] ^= - (1 << bit['bit_position'])
                            else:
                                kernel_flip_tensor[bit['indices']] ^= 1 << bit['bit_position']
                        elif bit['type'] == 'bias':
                            if bit['bit_position'] == layer.bias_quantizer['config']['bits'] - 1:
                                bias_flip_tensor[bit['indices']] ^= - (1 << bit['bit_position'])
                            else:
                                bias_flip_tensor[bit['indices']] ^= 1 << bit['bit_position']
                kernel_flip_tensor = tf.constant(kernel_flip_tensor, dtype=tf.int32)
                bias_flip_tensor   = tf.constant(bias_flip_tensor,   dtype=tf.int32)
            layer_config['kernel_quantizer'] = build_see_config(layer_config['kernel_quantizer'], random_flip_rate=random_flip_rate, flip_tensor=kernel_flip_tensor)
            layer_config['bias_quantizer']   = build_see_config(layer_config['bias_quantizer'],   random_flip_rate=random_flip_rate, flip_tensor=bias_flip_tensor)
        print(f"Cloning layer: {layer.name} ({layer.__class__.__name__})")

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

def compare_results(initial_results, new_results, print_mismatches=False):
    init_thresholds_99_idx = np.searchsorted(initial_results['tpr'], 0.99) 
    init_thresholds_98_idx = np.searchsorted(initial_results['tpr'], 0.98) 
    init_thresholds_99 = initial_results['thresholds'][init_thresholds_99_idx] 
    init_thresholds_98 = initial_results['thresholds'][init_thresholds_98_idx]

    fpr_i_99 = np.sum(initial_results['predictions'][initial_results['true_labels'] == 0] > init_thresholds_99) / np.sum(initial_results['true_labels'] == 0)
    tpr_i_99 = np.sum(initial_results['predictions'][initial_results['true_labels'] == 1] > init_thresholds_99) / np.sum(initial_results['true_labels'] == 1)
    fpr_i_98 = np.sum(initial_results['predictions'][initial_results['true_labels'] == 0] > init_thresholds_98) / np.sum(initial_results['true_labels'] == 0)
    tpr_i_98 = np.sum(initial_results['predictions'][initial_results['true_labels'] == 1] > init_thresholds_98) / np.sum(initial_results['true_labels'] == 1)
    fpr_n_99 = np.sum(new_results['predictions'][new_results['true_labels'] == 0] > init_thresholds_99) / np.sum(new_results['true_labels'] == 0)
    tpr_n_99 = np.sum(new_results['predictions'][new_results['true_labels'] == 1] > init_thresholds_99) / np.sum(new_results['true_labels'] == 1)
    fpr_n_98 = np.sum(new_results['predictions'][new_results['true_labels'] == 0] > init_thresholds_98) / np.sum(new_results['true_labels'] == 0)
    tpr_n_98 = np.sum(new_results['predictions'][new_results['true_labels'] == 1] > init_thresholds_98) / np.sum(new_results['true_labels'] == 1)

    print(f"Initial Threshold set at TPR=0.99.")
    print(f"Threshold: {init_thresholds_99}")
    print(f"TPR: {tpr_n_99} (Initial: {tpr_i_99})")
    print(f"FPR: {fpr_n_99} (Initial: {fpr_i_99})")
    print(f"Initial Threshold set at TPR=0.98.")
    print(f"Threshold: {init_thresholds_98}")
    print(f"TPR: {tpr_n_98} (Initial: {tpr_i_98})")
    print(f"FPR: {fpr_n_98} (Initial: {fpr_i_98})")

    if print_mismatches:
        for i in range(len(new_results['predictions'])):
            pred = new_results['predictions'][i]
            truth = new_results['true_labels'][i]
            pred_i = initial_results['predictions'][i]
            truth_i = initial_results['true_labels'][i]
            if truth != truth_i:
                print(f"Mismatch in true labels at index {i}: Initial {truth}, New {truth_i}")
            if abs(pred - pred_i) > 1e-1:
                print(f"Significant mismatch in predictions at index {i}: Initial {pred}, New {pred_i}")

    return tpr_i_99, fpr_i_99, tpr_n_99, fpr_n_99, tpr_i_98, fpr_i_98, tpr_n_98, fpr_n_98

def main():

    parser = argparse.ArgumentParser(description="Evaluate SmartPix ML Models with Quantization")
    add_arg = parser.add_argument
    add_arg('--config', type=str, default=None, help='Path to model config')
    add_arg('--training_dir', type=str, default=None, help='Path to training directory')
    add_arg('--quantization', type=int, default=-1, help='Which version of quantization to use. -1 for no quantization')
    add_arg('--random_flip_rate', type=float, default=0.0, help='Random bit flip rate to apply to cloned model')
    add_arg('--flip', type=str, nargs='*', help='Specific bits to flip in the format layer_name:type:indices:bit_position. Example: q_dense_5:kernel:8,0:2')
    add_arg('--skip_eval', action='store_true', help='Skip initial evaluation of original model')
    add_arg('--check_all_bits', type=str, default=None, help='Check all bits in the specified layer for sensitivity')
    add_arg('--stub', type=str, default='', help='Base filepath for output plots.')
    add_arg('--repeat', type=int, default=1, help='Number of times to repeat the evaluation with random flips.')

    args = parser.parse_args()

    config = load_config(args.config)
    if args.training_dir is not None:
        config['TrainingDirectory'] = args.training_dir

    quantization_tag = 'unquantized' 
    if args.quantization > 0:
        quantization_tag = f'quantized_{args.quantization}bit'

    config['Model']['args']['loadModel'] = True
    config['Model']['args']['modelPath'] = os.path.join(config['TrainingDirectory'], f"models/{config['Model']['class'].replace('_', '.')}_{quantization_tag}.h5")

    model = buildFromConfig(config['Model'])
    if not args.skip_eval:
        print("Running evaluation on original model:")
        initial_eval_results = model.evaluate(config_name="Unquantized")

    old_model = model.models["Unquantized"]
    flip_bits = []
    if args.flip is not None:
        for flip_str in args.flip:
            layer_name, type_str, indices_str, bit_position_str = flip_str.split(':')
            indices = tuple(int(i) for i in indices_str.split(','))
            bit_position = int(bit_position_str)
            flip_bits.append({
                'layer_name': layer_name,
                'type': type_str,
                'indices': indices,
                'bit_position': bit_position
            })
    if flip_bits:
        print("Cloning model with the following bit flips:")
    for bit in flip_bits:
        print(bit)

    if args.check_all_bits is None and (args.random_flip_rate != 0.0 or flip_bits):
        new_model = clone_functional_model(old_model, random_flip_rate=args.random_flip_rate, flip_bits=flip_bits)
        new_model.compile(metrics=['accuracy'], loss='binary_crossentropy', optimizer='adam')

        print("Running evaluation on cloned model with modified kernels:")
        model.models["Unquantized"] = new_model
        eval_results = []
        for repeat_idx in range(args.repeat):
            print(f"--- Repeat {repeat_idx + 1} of {args.repeat} ---")
            eval_results.append(model.evaluate(config_name="Unquantized"))

        eval_results = {
            'predictions': np.concatenate([res['predictions'] for res in eval_results]),
            'true_labels': np.concatenate([res['true_labels'] for res in eval_results]),
            'test_loss': np.mean([res['test_loss'] for res in eval_results])
        }
        eval_results['roc_auc'] = tf.keras.metrics.AUC()(eval_results['true_labels'], eval_results['predictions']).numpy()
        eval_results['test_accuracy'] = np.mean((eval_results['predictions'] > 0.5) == eval_results['true_labels'])
        fpr, tpr, thresholds = roc_curve(eval_results['true_labels'], eval_results['predictions'])
        eval_results['fpr'] = fpr
        eval_results['tpr'] = tpr
        eval_results['thresholds'] = thresholds
        eval_results['roc_auc'] = auc(fpr, tpr)

        print("Evaluation Results:")
        print(f"AUC: {eval_results['roc_auc']:.4f} (Initial: {initial_eval_results['roc_auc'] if not args.skip_eval else 'N/A':.4f})")
        print(f"Test Accuracy: {eval_results['test_accuracy']:.4f} (Initial: {initial_eval_results['test_accuracy'] if not args.skip_eval else 'N/A':.4f})")

        logfile = open(f'{args.training_dir}/{args.quantization}bit{args.stub}_see_evaluation.txt', 'w')
        logfile.write("Metric,Initial,New\n")
        logfile.write(f"AUC,{initial_eval_results['roc_auc'] if not args.skip_eval else 'N/A'},{eval_results['roc_auc']}\n")
        logfile.write(f"Test Accuracy,{initial_eval_results['test_accuracy'] if not args.skip_eval else 'N/A'},{eval_results['test_accuracy']}\n")

        if not args.skip_eval:
            tpr_i_99, fpr_i_99, tpr_n_99, fpr_n_99, tpr_i_98, fpr_i_98, tpr_n_98, fpr_n_98 = compare_results(initial_eval_results, eval_results)

            logfile.write(f"TPR_99,{tpr_i_99},{tpr_n_99}\n")
            logfile.write(f"FPR_99,{fpr_i_99},{fpr_n_99}\n")
            logfile.write(f"TPR_98,{tpr_i_98},{tpr_n_98}\n")
            logfile.write(f"FPR_98,{fpr_i_98},{fpr_n_98}\n")
            logfile.close()

            plt.figure()
            plt.hist([initial_eval_results['predictions'][initial_eval_results['true_labels'] == 0], eval_results['predictions'][eval_results['true_labels'] == 0], initial_eval_results['predictions'][initial_eval_results['true_labels'] == 1], eval_results['predictions'][eval_results['true_labels'] == 1]], range=(0,1), histtype='step', bins=50, label=['Background', 'Background with SEE', 'Signal', 'Signal with SEE'], density=True)
            plt.xlabel('Model Prediction')
            plt.ylabel('Normalized Entries')
            plt.legend()
            plt.title('Prediction Distributions Before and After Bit Flips')
            plt.savefig(f'{args.training_dir}/{args.quantization}bit{args.stub}_prediction_distributions.png')
            plt.close()

    elif args.check_all_bits is not None:
        layer_to_check = args.check_all_bits
        layer = None
        for l in old_model.layers:
            print(f"Checking layer: {l.name}")
            if l.name == layer_to_check:
                layer = l
                break
        if layer:
            weight_shapes = {}
            if hasattr(layer, 'kernel'):
                weight_shapes['kernel'] = layer.kernel.shape
            if hasattr(layer, 'bias'):
                weight_shapes['bias'] = layer.bias.shape
            layers = {layer_to_check: weight_shapes}
        elif layer_to_check == 'MSB': # Most significant bit of all quantized layers
            layers = {}
            for l in old_model.layers:
                if isinstance(l, QDense):
                    layers[l.name] = {}
                    if hasattr(l, 'kernel'):
                        layers[l.name][f'kernel'] = l.kernel.shape
                    if hasattr(l, 'bias'):
                        layers[l.name][f'bias'] = l.bias.shape

        else:
            print(f"Layer {layer_to_check} not found in model.")
            return

        fprs99 = []
        tprs99 = []
        fprs98 = []
        tprs98 = []
        tpr_i_99 = 0
        fpr_i_99 = 0
        tpr_i_98 = 0
        fpr_i_98 = 0

        logfile = open(f'{args.training_dir}/{args.quantization}bit{args.stub}_bit_sensitivity_{layer_to_check}.txt', 'w')
        logfile.write("bit_position,weight_name,indices,tpr_99,fpr_99,tpr_98,fpr_98\n")

        for layer, weight_shapes in layers.items():
            for weight_name, shape in weight_shapes.items():
                total_elements = np.prod(shape)
                for index in range(total_elements):
                    indices = np.unravel_index(index, shape)
                    for bit_position in range(args.quantization): 
                        if layer_to_check == 'MSB' and bit_position != args.quantization - 1:
                            continue
                        flip_bits = [{
                            'layer_name': layer,
                            'type': weight_name,
                            'indices': indices,
                            'bit_position': bit_position
                        }]
                        new_model = clone_functional_model(old_model, random_flip_rate=0.0, flip_bits=flip_bits)
                        new_model.compile(metrics=['accuracy'], loss='binary_crossentropy', optimizer='adam')

                        print(f"Flipping bit {bit_position} of {weight_name} at index {indices} in layer {layer}")
                        model.models["Unquantized"] = new_model
                        eval_results = model.evaluate(config_name="Unquantized")
                        del new_model
                        tf.keras.backend.clear_session()

                        print(f"AUC: {eval_results['roc_auc']:.4f} (Initial: {initial_eval_results['roc_auc'] if not args.skip_eval else 'N/A':.4f})")
                        print(f"Test Accuracy: {eval_results['test_accuracy']:.4f} (Initial: {initial_eval_results['test_accuracy'] if not args.skip_eval else 'N/A':.4f})")
                        comp = compare_results(initial_eval_results, eval_results, print_mismatches=False)
                        tpr_i_99, fpr_i_99, tpr_n_99, fpr_n_99, tpr_i_98, fpr_i_98, tpr_n_98, fpr_n_98 = comp
                        tprs99.append(tpr_n_99)
                        fprs99.append(fpr_n_99)
                        tprs98.append(tpr_n_98)
                        fprs98.append(fpr_n_98)
                        logfile.write(f"{layer},{bit_position},{weight_name},{indices},{tpr_n_99},{fpr_n_99},{tpr_n_98},{fpr_n_98}\n")
        logfile.close()

        # FPR vs TPR Scatter
        plt.figure()
        plt.scatter(tprs99, fprs99, label='TPR=0.99', alpha=0.6)
        plt.scatter(tprs98, fprs98, label='TPR=0.98', alpha=0.6)
        plt.plot([tpr_i_99], [fpr_i_99], color='red', label='Initial TPR=0.99', marker='x', markersize=10)
        plt.plot([tpr_i_98], [fpr_i_98], color='blue', label='Initial TPR=0.98', marker='x', markersize=10)
        plt.xlabel('True Positive Rate')
        plt.ylabel('False Positive Rate')
        plt.title(f'Bit Sensitivity for Layer {layer_to_check}')
        plt.legend()
        plt.savefig(f'{args.training_dir}/{args.quantization}bit{args.stub}_bit_sensitivity_{layer_to_check}.png')
        plt.close()

        # FPR Histogram
        plt.hist([np.array(fprs99), np.array(fprs98)], bins=50, label=['TPR=0.99', 'TPR=0.98'], histtype='step')
        plt.xlabel('False Positive Rate')
        plt.ylabel('Bits Count')
        plt.title(f'FPR Distribution for Bit Flips in Layer {layer_to_check}')
        plt.legend()
        plt.savefig(f'{args.training_dir}/{args.quantization}bit{args.stub}_bit_sensitivity_{layer_to_check}_fpr_hist.png')
        plt.close()

        # TPR Histogram
        plt.hist([np.array(tprs99), np.array(tprs98)], bins=50, label=['TPR=0.99', 'TPR=0.98'], histtype='step')
        plt.xlabel('True Positive Rate')
        plt.ylabel('Bits Count')
        plt.title(f'TPR Distribution for Bit Flips in Layer {layer_to_check}')
        plt.legend()
        plt.savefig(f'{args.training_dir}/{args.quantization}bit{args.stub}_bit_sensitivity_{layer_to_check}_tpr_hist.png')
        plt.close()

if __name__== "__main__":
    main()