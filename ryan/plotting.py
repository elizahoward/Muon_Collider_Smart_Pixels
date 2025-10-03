import numpy as np
import matplotlib.pyplot as plt

def plot_weight_distributions(models, per_layer=True, include_bias=True, bins="auto", max_cols=3):
    """
    Plot weight distributions for a dict of Keras/QKeras models.
    
    Args:
        models: dict[str, tf.keras.Model]   # e.g., your `models` mapping names -> models
        per_layer: bool                     # True = one subplot per layer; False = one histogram per model (all weights)
        include_bias: bool                  # include bias vectors when present
        bins: int or "auto"                 # matplotlib bins
        max_cols: int                       # max subplot columns for per_layer mode
    """
    for model_name, model in models.items():
        # Collect weights
        layer_payloads = []   # list of tuples: (layer_title, 1D_array_of_weights)
        all_weights = []

        for li, layer in enumerate(model.layers):
            # Skip layers without weights (e.g., activations)
            if not layer.weights:
                continue

            try:
                w_list = layer.get_weights()
            except Exception:
                continue

            # Common patterns: [kernel] or [kernel, bias]
            entries = []
            if len(w_list) >= 1 and w_list[0] is not None:
                k = np.asarray(w_list[0]).ravel()
                if k.size:
                    entries.append(("kernel", k))
                    all_weights.append(k)
            if include_bias and len(w_list) >= 2 and w_list[1] is not None:
                b = np.asarray(w_list[1]).ravel()
                if b.size:
                    entries.append(("bias", b))
                    all_weights.append(b)

            for kind, arr in entries:
                title = f"L{li} {layer.__class__.__name__} • {kind}"
                layer_payloads.append((title, arr))

        if not layer_payloads:
            print(f"[{model_name}] No weights to plot.")
            continue

        if per_layer:
            # One figure per model, multiple subplots (one per layer entry)
            n = len(layer_payloads)
            ncols = min(max_cols, n)
            nrows = int(np.ceil(n / ncols))
            plt.figure(figsize=(4*ncols, 3*nrows))
            plt.suptitle(f"Weight distributions • {model_name}", y=1.02)

            for i, (title, arr) in enumerate(layer_payloads, 1):
                mu = float(np.mean(arr))
                sigma = float(np.std(arr))
                uniq = np.unique(arr)
                # If lots of unique values, just show the count in title
                uniq_count = len(uniq)

                ax = plt.subplot(nrows, ncols, i)
                ax.hist(arr, bins=bins)  # no explicit colors (keeps defaults)
                ax.set_title(f"{title}\nμ={mu:.4g}, σ={sigma:.4g}, unique={uniq_count}")
                ax.set_xlabel("weight value")
                ax.set_ylabel("count")

            plt.tight_layout()
            plt.show()

        else:
            # Aggregate all weights across layers and plot a single histogram per model
            all_weights = np.concatenate(all_weights, axis=0)
            mu = float(np.mean(all_weights))
            sigma = float(np.std(all_weights))
            uniq_count = len(np.unique(all_weights))

            plt.figure(figsize=(6, 4))
            plt.hist(all_weights, bins=bins)  # no explicit colors
            plt.title(f"Weight distribution (ALL layers) • {model_name}\nμ={mu:.4g}, σ={sigma:.4g}, unique={uniq_count}")
            plt.xlabel("weight value")
            plt.ylabel("count")
            plt.tight_layout()
            plt.show()



def plot_weights(
    models,
    per_layer=True,
    include_bias=True,
    mode="stem",         # "stem" (counts) or "rug" (jittered dots)
    max_cols=3,
    jitter=0.04,         # only used in rug mode
    limit_layers=None,   # e.g., range(0, 10) to restrict by layer index
):
    """
    Plot exact weight locations (no binning) for each model/layer.

    Args:
        models: dict[str, tf.keras.Model]
        per_layer: if True, one subplot per (layer, kernel/bias). If False, aggregate all weights per model.
        include_bias: include bias vectors if present.
        mode: "stem" for spikes with heights = counts, or "rug" for raw points with jitter.
        max_cols: max columns when per_layer=True.
        jitter: vertical jitter for "rug" mode to avoid overplotting.
        limit_layers: optional iterable of layer indices to include (after Keras indexing).
    """
    for model_name, model in models.items():
        layer_payloads = []  # (title, 1D array)

        for li, layer in enumerate(model.layers):
            if limit_layers is not None and li not in limit_layers:
                continue
            if not layer.weights:
                continue

            try:
                w_list = layer.get_weights()
            except Exception:
                continue

            # kernel
            if len(w_list) >= 1 and w_list[0] is not None:
                k = np.asarray(w_list[0]).ravel()
                if k.size:
                    layer_payloads.append((f"L{li} {layer.__class__.__name__} • kernel", k))
            # bias
            if include_bias and len(w_list) >= 2 and w_list[1] is not None:
                b = np.asarray(w_list[1]).ravel()
                if b.size:
                    layer_payloads.append((f"L{li} {layer.__class__.__name__} • bias", b))

        if not layer_payloads:
            print(f"[{model_name}] No weights found.")
            continue

        if per_layer:
            n = len(layer_payloads)
            ncols = min(max_cols, n)
            nrows = int(np.ceil(n / ncols))
            plt.figure(figsize=(4*ncols, 3*nrows))
            plt.suptitle(f"Exact weight locations • {model_name}", y=1.02)

            for i, (title, arr) in enumerate(layer_payloads, 1):
                ax = plt.subplot(nrows, ncols, i)
                if mode == "stem":
                    # spikes at each unique value with height = count
                    xs, counts = np.unique(arr, return_counts=True)
                    ax.vlines(xs, 0, counts)  # default color/style
                    ax.set_ylabel("count")
                else:
                    # rug: plot every point at its exact x, with small y jitter
                    y = np.zeros_like(arr, dtype=float)
                    if jitter:
                        y = y + (np.random.rand(arr.size) - 0.5) * jitter
                    ax.plot(arr, y, ".", markersize=2)  # default style
                    ax.set_yticks([])
                    ax.set_ylabel("")

                ax.set_title(f"{title}\nmin={arr.min():.6g}, max={arr.max():.6g}, unique={len(np.unique(arr))}")
                ax.set_xlabel("weight value")

            plt.tight_layout()
            plt.show()

        else:
            # Aggregate all weights across layers for one plot per model
            all_w = np.concatenate([arr for _, arr in layer_payloads], axis=0)
            plt.figure(figsize=(6, 3.5))
            if mode == "stem":
                xs, counts = np.unique(all_w, return_counts=True)
                plt.vlines(xs, 0, counts)
                plt.ylabel("count")
            else:
                y = np.zeros_like(all_w, dtype=float)
                if jitter:
                    y = y + (np.random.rand(all_w.size) - 0.5) * jitter
                plt.plot(all_w, y, ".", markersize=2)
                plt.yticks([])
                plt.ylabel("")

            plt.title(f"Exact weight locations (ALL layers) • {model_name}\nmin={all_w.min():.6g}, max={all_w.max():.6g}, unique={len(np.unique(all_w))}")
            plt.xlabel("weight value")
            plt.tight_layout()
            plt.show()
