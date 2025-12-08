"""
Generate ASIC-style block diagram for Model 2.5
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

def draw_model2_5_asic(output_path='model2_5_asic_diagram.png'):
    """Draw ASIC-style block diagram for Model 2.5"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    input_color = '#E8F4F8'  # Light blue for inputs
    concat_color = '#FFF4E6'  # Light orange for concatenation
    dense_color = '#E8F5E9'  # Light green for dense layers
    z_branch_color = '#F3E5F5'  # Light purple for z_global branch (8-bit)
    activation_color = '#FFE0B2'  # Light orange for activations
    output_color = '#FFCDD2'  # Light red for output
    
    # Define layer positions
    y_top = 9.5
    y_inputs = 8.0
    y_concat1 = 6.5
    y_dense1 = 5.0
    y_concat2 = 3.5
    y_dense2 = 2.5
    y_dense3 = 1.5
    y_output = 0.5
    
    x_left = 1.0
    x_center = 5.0
    x_right = 9.0
    
    # Title
    ax.text(5, 9.8, 'Model 2.5 ASIC Architecture', 
            fontsize=20, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=2))
    
    # Input ports
    inputs = [
        ('x_profile', 21, 1.5),
        ('y_profile', 13, 3.0),
        ('y_local', 1, 4.5),
        ('z_global', 1, 6.0)
    ]
    
    input_boxes = []
    for i, (name, size, x_pos) in enumerate(inputs):
        box = FancyBboxPatch((x_pos-0.3, y_inputs-0.3), 0.6, 0.6,
                            boxstyle='round,pad=0.05', 
                            facecolor=input_color, 
                            edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x_pos, y_inputs, f'{name}\n[{size}]', 
                ha='center', va='center', fontsize=9, fontweight='bold')
        input_boxes.append((x_pos, y_inputs))
    
    # First concatenation (xy_concat)
    concat1_box = FancyBboxPatch((2.0, y_concat1-0.4), 1.2, 0.8,
                                 boxstyle='round,pad=0.05',
                                 facecolor=concat_color,
                                 edgecolor='black', linewidth=1.5)
    ax.add_patch(concat1_box)
    ax.text(2.6, y_concat1, 'CONCAT\nx+y', ha='center', va='center', 
            fontsize=9, fontweight='bold')
    
    # Arrows from inputs to concat1
    ax.arrow(1.5, y_inputs-0.3, 0.5, -0.3, head_width=0.08, head_length=0.08, 
             fc='black', ec='black', linewidth=1.5)
    ax.arrow(3.0, y_inputs-0.3, -0.3, -0.3, head_width=0.08, head_length=0.08, 
             fc='black', ec='black', linewidth=1.5)
    
    # Second concatenation (other_features)
    concat2_box = FancyBboxPatch((2.0, y_concat1-1.2), 1.2, 0.8,
                                 boxstyle='round,pad=0.05',
                                 facecolor=concat_color,
                                 edgecolor='black', linewidth=1.5)
    ax.add_patch(concat2_box)
    ax.text(2.6, y_concat1-0.8, 'CONCAT\n+ y_local', ha='center', va='center', 
            fontsize=9, fontweight='bold')
    
    # Arrow from concat1 to concat2
    ax.arrow(2.6, y_concat1-0.4, 0, -0.4, head_width=0.08, head_length=0.08, 
             fc='black', ec='black', linewidth=1.5)
    
    # Arrow from y_local to concat2
    ax.arrow(4.5, y_inputs-0.3, -1.7, -0.5, head_width=0.08, head_length=0.08, 
             fc='black', ec='black', linewidth=1.5)
    
    # Spatial features branch - Dense layer
    dense1_box = FancyBboxPatch((2.0, y_dense1-0.5), 1.2, 1.0,
                                boxstyle='round,pad=0.05',
                                facecolor=dense_color,
                                edgecolor='black', linewidth=2)
    ax.add_patch(dense1_box)
    ax.text(2.6, y_dense1, 'DENSE\n128 units\n(4-bit)', ha='center', va='center', 
            fontsize=9, fontweight='bold')
    
    # Arrow from concat2 to dense1
    ax.arrow(2.6, y_concat1-1.2, 0, -0.3, head_width=0.08, head_length=0.08, 
             fc='black', ec='black', linewidth=1.5)
    
    # ReLU activation after dense1
    relu1_box = FancyBboxPatch((2.0, y_dense1-1.5), 1.2, 0.5,
                               boxstyle='round,pad=0.05',
                               facecolor=activation_color,
                               edgecolor='black', linewidth=1.5)
    ax.add_patch(relu1_box)
    ax.text(2.6, y_dense1-1.25, 'ReLU\n(8-bit)', ha='center', va='center', 
            fontsize=8, fontweight='bold')
    
    # Arrow from dense1 to relu1
    ax.arrow(2.6, y_dense1-0.5, 0, -0.5, head_width=0.08, head_length=0.08, 
             fc='black', ec='black', linewidth=1.5)
    
    # z_global branch - Dense layer (8-bit)
    z_dense_box = FancyBboxPatch((6.5, y_dense1-0.5), 1.2, 1.0,
                                 boxstyle='round,pad=0.05',
                                 facecolor=z_branch_color,
                                 edgecolor='purple', linewidth=2.5)
    ax.add_patch(z_dense_box)
    ax.text(7.1, y_dense1, 'DENSE\n32 units\n(8-bit)', ha='center', va='center', 
            fontsize=9, fontweight='bold')
    
    # Arrow from z_global input to z_dense
    ax.arrow(6.0, y_inputs-0.3, 0.5, -0.2, head_width=0.08, head_length=0.08, 
             fc='purple', ec='purple', linewidth=2)
    
    # ReLU activation after z_dense
    z_relu_box = FancyBboxPatch((6.5, y_dense1-1.5), 1.2, 0.5,
                                boxstyle='round,pad=0.05',
                                facecolor=activation_color,
                                edgecolor='purple', linewidth=1.5)
    ax.add_patch(z_relu_box)
    ax.text(7.1, y_dense1-1.25, 'ReLU\n(8-bit)', ha='center', va='center', 
            fontsize=8, fontweight='bold')
    
    # Arrow from z_dense to z_relu
    ax.arrow(7.1, y_dense1-0.5, 0, -0.5, head_width=0.08, head_length=0.08, 
             fc='purple', ec='purple', linewidth=2)
    
    # Merge concatenation
    merge_box = FancyBboxPatch((3.5, y_concat2-0.4), 3.0, 0.8,
                               boxstyle='round,pad=0.05',
                               facecolor=concat_color,
                               edgecolor='black', linewidth=2)
    ax.add_patch(merge_box)
    ax.text(5.0, y_concat2, 'CONCATENATE\n(Spatial + z_global)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows to merge
    ax.arrow(2.6, y_dense1-1.5, 0.9, -0.6, head_width=0.1, head_length=0.1, 
             fc='black', ec='black', linewidth=2)
    ax.arrow(7.1, y_dense1-1.5, -0.6, -0.6, head_width=0.1, head_length=0.1, 
             fc='purple', ec='purple', linewidth=2)
    
    # Dense2 layer
    dense2_box = FancyBboxPatch((3.5, y_dense2-0.4), 3.0, 0.8,
                               boxstyle='round,pad=0.05',
                               facecolor=dense_color,
                               edgecolor='black', linewidth=2)
    ax.add_patch(dense2_box)
    ax.text(5.0, y_dense2, 'DENSE2\n128 units\n(4-bit)', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    
    # Arrow from merge to dense2
    ax.arrow(5.0, y_concat2-0.4, 0, -0.5, head_width=0.1, head_length=0.1, 
             fc='black', ec='black', linewidth=2)
    
    # ReLU activation after dense2
    relu2_box = FancyBboxPatch((3.5, y_dense2-1.2), 3.0, 0.5,
                               boxstyle='round,pad=0.05',
                               facecolor=activation_color,
                               edgecolor='black', linewidth=1.5)
    ax.add_patch(relu2_box)
    ax.text(5.0, y_dense2-0.95, 'ReLU (8-bit)', ha='center', va='center', 
            fontsize=9, fontweight='bold')
    
    # Arrow from dense2 to relu2
    ax.arrow(5.0, y_dense2-0.4, 0, -0.3, head_width=0.08, head_length=0.08, 
             fc='black', ec='black', linewidth=1.5)
    
    # Dropout
    dropout_box = FancyBboxPatch((3.5, y_dense2-1.8), 3.0, 0.4,
                                 boxstyle='round,pad=0.05',
                                 facecolor='#FFCCBC',
                                 edgecolor='black', linewidth=1.5)
    ax.add_patch(dropout_box)
    ax.text(5.0, y_dense2-1.6, 'DROPOUT (0.1)', ha='center', va='center', 
            fontsize=9, fontweight='bold')
    
    # Arrow from relu2 to dropout
    ax.arrow(5.0, y_dense2-1.2, 0, -0.3, head_width=0.08, head_length=0.08, 
             fc='black', ec='black', linewidth=1.5)
    
    # Dense3 layer
    dense3_box = FancyBboxPatch((3.5, y_dense3-0.4), 3.0, 0.8,
                               boxstyle='round,pad=0.05',
                               facecolor=dense_color,
                               edgecolor='black', linewidth=2)
    ax.add_patch(dense3_box)
    ax.text(5.0, y_dense3, 'DENSE3\n64 units\n(4-bit)', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    
    # Arrow from dropout to dense3
    ax.arrow(5.0, y_dense2-1.8, 0, -0.4, head_width=0.08, head_length=0.08, 
             fc='black', ec='black', linewidth=1.5)
    
    # ReLU activation after dense3
    relu3_box = FancyBboxPatch((3.5, y_dense3-1.0), 3.0, 0.5,
                               boxstyle='round,pad=0.05',
                               facecolor=activation_color,
                               edgecolor='black', linewidth=1.5)
    ax.add_patch(relu3_box)
    ax.text(5.0, y_dense3-0.75, 'ReLU (8-bit)', ha='center', va='center', 
            fontsize=9, fontweight='bold')
    
    # Arrow from dense3 to relu3
    ax.arrow(5.0, y_dense3-0.4, 0, -0.3, head_width=0.08, head_length=0.08, 
             fc='black', ec='black', linewidth=1.5)
    
    # Output layer
    output_box = FancyBboxPatch((3.5, y_output-0.3), 3.0, 0.6,
                               boxstyle='round,pad=0.05',
                               facecolor=output_color,
                               edgecolor='red', linewidth=2.5)
    ax.add_patch(output_box)
    ax.text(5.0, y_output, 'OUTPUT\n1 unit\nTanh (8-bit)', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    
    # Arrow from relu3 to output
    ax.arrow(5.0, y_dense3-1.0, 0, -0.2, head_width=0.1, head_length=0.1, 
             fc='red', ec='red', linewidth=2)
    
    # Add quantization legend
    legend_elements = [
        mpatches.Patch(facecolor=dense_color, edgecolor='black', label='Dense Layer (4-bit weights)'),
        mpatches.Patch(facecolor=z_branch_color, edgecolor='purple', label='z_global Branch (8-bit weights)'),
        mpatches.Patch(facecolor=activation_color, edgecolor='black', label='Activation (8-bit)'),
        mpatches.Patch(facecolor=concat_color, edgecolor='black', label='Concatenation'),
        mpatches.Patch(facecolor=output_color, edgecolor='red', label='Output')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)
    
    # Add data flow annotations
    ax.text(0.2, 5.0, 'Data Flow', fontsize=12, fontweight='bold', rotation=90)
    
    # Add quantization notes
    note_text = ('Quantization Notes:\n'
                 '• Spatial branch: 4-bit weights, 8-bit activations\n'
                 '• z_global branch: 8-bit weights, 8-bit activations\n'
                 '• Output: 8-bit quantized tanh')
    ax.text(0.5, 2.0, note_text, fontsize=8, 
            bbox=dict(boxstyle='round', facecolor='#FFF9C4', edgecolor='black', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"ASIC diagram saved to: {output_path}")
    plt.close()

if __name__ == '__main__':
    draw_model2_5_asic('model2_5_asic_diagram.png')


