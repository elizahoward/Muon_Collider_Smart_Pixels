#GAAA

#plot acceptance rates on a log scale.

filepaths=[
    "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/paperPlots/evaluationPlots/mdl_ial_057.h5/dataRates_adjusted_hit_time_30ps_gaussian.txt",
    "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/paperPlots/evaluationPlots/mdl_ial_057.h5/dataRates_adjusted_hit_time_60ps_gaussian.txt",
    "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/paperPlots/evaluationPlots/mdl_ial_057.h5/dataRates_adjusted_hit_time_100ps_gaussian.txt",
    "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/paperPlots/evaluationPlots/mdl_ial_057.h5/dataRates_adjusted_hit_time_200ps_gaussian.txt",
    "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/paperPlots/evaluationPlots/mdl_ial_057.h5/dataRates_adjusted_hit_time_500ps_gaussian.txt",
    "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/paperPlots/evaluationPlots/mdl_ial_057.h5/dataRates_adjusted_hit_time_1000ps_gaussian.txt",
    # "/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/paperPlots/evaluationPlots/mdl_ial_057.h5/dataRates_adjusted_hit_time_200ps_gaussian.txt",
    ]
def readDataRates(path: str) -> dict:
    results = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("="):
                continue
            key, _, valueStr = line.partition(":")
            key = key.strip()
            valueStr = valueStr.strip()
            try:
                value = int(valueStr)
            except ValueError:
                value = float(valueStr)
            results[key] = value
    return results

with open("./ahhh.txt","w") as f:
    for filepath in filepaths:
        results=readDataRates(filepath)
        for key in ['1-clusterRateModel','1-clusterRateTiming','1-clusterRateModelAndTiming','1-dataRateModel','1-dataRateTiming','1-dataRateModelAndTiming']:
            print(key,",",results[key],",",filepath[-18:-13],file=f)
keys = ['1-clusterRateModel', '1-clusterRateTiming', '1-clusterRateModelAndTiming',
        '1-dataRateModel', '1-dataRateTiming', '1-dataRateModelAndTiming']
headers = [r"Cluster Model", r"Cluster Timing", r"Cluster Model+Timing",
           r"Data Model", r"Data Timing", r"Data Model+Timing"]

with open("./ahhh.tex", "w") as f:
    print(r"\begin{table}[htbp]", file=f)
    print(r"\centering", file=f)
    print(r"\begin{tabular}{l" + "c" * len(keys) + "}", file=f)
    print(r"\toprule", file=f)
    print(r"Config & " + " & ".join(headers) + r" \\", file=f)
    print(r"\midrule", file=f)
    for filepath in filepaths:
        results = readDataRates(filepath)
        label = filepath[-18:-13].replace("_", r"\_")
        values = " & ".join(f"{results[key]:.4f}" for key in keys)
        print(f"{label} & {values} \\\\", file=f)
    print(r"\bottomrule", file=f)
    print(r"\end{tabular}", file=f)
    print(r"\caption{Fraction of clusters and data rejected for each selection.}", file=f)
    print(r"\label{tab:rejection}", file=f)
    print(r"\end{table}", file=f)