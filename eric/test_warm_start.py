"""
Test script to verify warm-start functionality works correctly
"""
import sys
sys.path.append('/home/youeric/PixelML/SmartpixReal/Muon_Collider_Smart_Pixels/MuC_Smartpix_ML/')
sys.path.append('../MuC_Smartpix_ML/')

from model2 import Model2
import numpy as np

print("="*60)
print("Testing Warm-Start Functionality")
print("="*60)

# Initialize Model2 with small architecture for quick testing
model2 = Model2(
    tfRecordFolder="/local/d1/smartpixML/filtering_models/shuffling_data/all_batches_shuffled_bigData_try3_eric/filtering_records16384_data_shuffled_single_bigData",
    xz_units=32,
    yl_units=32,
    merged_units_1=128,
    merged_units_2=64,
    merged_units_3=32,
    dropout_rate=0.1,
    initial_lr=1e-3,
    end_lr=1e-4,
    power=2,
    bit_configs=[(4, 0)]
)

print("\n1. Building unquantized model...")
model2.buildModel("unquantized")

print("\n2. Building quantized model...")
model2.buildModel("quantized")

print("\n3. Checking layer names...")
print("\nUnquantized model layers:")
for layer in model2.models["Unquantized"].layers:
    if len(layer.get_weights()) > 0:
        print(f"  {layer.name}: {layer.__class__.__name__}, weights shape: {layer.get_weights()[0].shape}")

print("\nQuantized model layers:")
for layer in model2.models["quantized_4w0i"].layers:
    if len(layer.get_weights()) > 0:
        print(f"  {layer.name}: {layer.__class__.__name__}, weights shape: {layer.get_weights()[0].shape}")

print("\n4. Getting initial quantized weights (should be random)...")
quant_layer = None
for layer in model2.models["quantized_4w0i"].layers:
    if layer.name == "xz_dense1":
        quant_layer = layer
        break

if quant_layer:
    initial_weights = quant_layer.get_weights()[0].copy()
    print(f"Initial xz_dense1 weights (first 5): {initial_weights[0, :5]}")
else:
    print("ERROR: Could not find xz_dense1 layer!")
    sys.exit(1)

print("\n5. Setting some known values in unquantized model...")
unquant_layer = None
for layer in model2.models["Unquantized"].layers:
    if layer.name == "xz_dense1":
        unquant_layer = layer
        break

if unquant_layer:
    # Set first row to known values
    weights = unquant_layer.get_weights()
    weights[0][0, :5] = [1.0, 2.0, 3.0, 4.0, 5.0]
    unquant_layer.set_weights(weights)
    print(f"Set unquantized xz_dense1 weights (first 5): {weights[0][0, :5]}")
else:
    print("ERROR: Could not find xz_dense1 layer in unquantized!")
    sys.exit(1)

print("\n6. Running warm-start...")
try:
    copied, skipped = model2.warmStartQuantizedModel("quantized_4w0i", "Unquantized")
    
    print(f"\n7. Verifying weights were copied...")
    final_weights = quant_layer.get_weights()[0]
    print(f"Final xz_dense1 weights (first 5): {final_weights[0, :5]}")
    
    if np.allclose(final_weights[0, :5], [1.0, 2.0, 3.0, 4.0, 5.0]):
        print("\n✓ SUCCESS: Warm-start worked! Weights were copied correctly.")
    else:
        print("\n✗ FAILURE: Weights were NOT copied correctly!")
        print(f"Expected: [1.0, 2.0, 3.0, 4.0, 5.0]")
        print(f"Got: {final_weights[0, :5]}")
        
except Exception as e:
    print(f"\n✗ ERROR during warm-start: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Test Complete")
print("="*60)





