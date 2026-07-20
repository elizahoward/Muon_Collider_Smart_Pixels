#Random script to do that parameter vs. luts and ffs quickly

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

df = pd.read_csv("/home/dabadjiev/smartpixels_ml_dsabadjiev/Muon_Collider_Smart_Pixels/eric/combined_all_models_pareto_newJune2026/combined_all_detailed.csv")
# print(df)
# df = df.query("run_name == 'model25_3bit'")
df = df.query("run_name == 'model3_3bit'")
# df = df.query("luts_plus_ff < 400000")
df = df.query("luts_plus_ff < 4000000")
x = df["parameters"]
y = df["luts_plus_ff"]
# plt.plot(x,y,".")
# plt.savefig("randomGuy.png")


# Calculate the fit metrics
res = stats.linregress(x, y)

# Print the R-squared value
print(f"R-squared: {res.rvalue**2:.4f}")
print(f"slope: {res.slope:.4f}")
print(f"intercept: {res.intercept:.4f}")

# Plot using the slope and intercept from the result
plt.plot(x, y, ".")
plt.plot(x, res.slope * x + res.intercept, "-", label=f"R² = {res.rvalue**2:.3f}\nslope={res.slope}")
plt.legend()
plt.savefig("randomGuy2.png")
