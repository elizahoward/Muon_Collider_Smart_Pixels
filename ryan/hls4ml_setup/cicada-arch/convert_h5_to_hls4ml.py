import argparse
import os
import tensorflow as tf
import hls4ml

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_h5", required=True, help="Path to Keras .h5 model")
    ap.add_argument("--outdir", default=None, help="Output hls4ml project directory")
    ap.add_argument("--part", default="xc7vx690tffg1927-2", help="FPGA part (edit if needed)")
    ap.add_argument("--clock", type=float, default=6.25, help="Clock period in ns")
    args = ap.parse_args()

    model_path = args.model_h5
    if args.outdir is None:
        base = os.path.splitext(os.path.basename(model_path))[0]
        outdir = f"hls4ml_{base}"
    else:
        outdir = args.outdir

    # IMPORTANT: compile=False avoids needing optimizer classes like "Custom>Adam"
    model = tf.keras.models.load_model(model_path, compile=False)

    # Create default config
    hls_config = hls4ml.utils.config_from_keras_model(model, granularity="name")
    hls_config["Model"]["Strategy"] = "Latency"
    hls_config["Model"]["ReuseFactor"] = 1

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        backend="Vitis",
        clock_period=args.clock,
        io_type="io_parallel",
        hls_config=hls_config,
        output_dir=outdir,
        project_name="proj",
        part=args.part,
    )

    hls_model.compile()
    print(f"✅ Wrote hls4ml project to: {outdir}")

if __name__ == "__main__":
    main()