import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import mplhep
from matplotlib.colors import LinearSegmentedColormap

mplhep.style.use(mplhep.style.CMS)

# Define bin labels
pt_labels = ['10-20', '20-45', '45-75', '75-500']
eta_labels = ['barrel_1', 'barrel_2', 'endcap']
eta_pretty_labels = ['|η| ≤ 0.8', '0.8 < |η| ≤ 1.442', '1.442 < |η| ≤ 2.5']

# Load Excel file
excel_path = "final_gold_blp_SF_heatmap.xlsx"
df = pd.read_excel(excel_path, sheet_name='ScaleFactors')

# Filter only DATA rows (or MC if you want those efficiencies)
df_mc = df[df['Type'] == 'DATA'].copy()

df_mc['bin_num'] = df_mc['bin'].str.extract(r'bin(\d+)').astype(int)

heatmap_values = np.full((4, 3), np.nan)
heatmap_errors = np.full((4, 3), np.nan)

# Fill heatmap with SF values and errors
for _, row in df_mc.iterrows():
    bin_num = row['bin_num'] - 2
    eta_region = row['barrel']
    type = row['Type']
    
    try:
        col_idx = eta_labels.index(eta_region)
    except ValueError:
        continue
    if bin_num < 0 or bin_num >= len(pt_labels):
        continue

    heatmap_values[bin_num, col_idx] = row['epsilon']
    heatmap_errors[bin_num, col_idx] = row['epsilon_err']


colors_list = [
    (0, "red"),    
    (1, "yellow"), 
]

cmap = LinearSegmentedColormap.from_list("spike_at_one", colors_list)

cmap.set_bad(color='white')

# Set normalization
norm = colors.Normalize(vmin=np.nanmin(heatmap_values), vmax=np.nanmax(heatmap_values))

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))  # wide layout
cmap.set_bad(color='white')
im = ax.imshow(heatmap_values, cmap=cmap, norm=norm, aspect='auto')

# Axis labels
ax.set_xticks(np.arange(len(eta_labels)))
ax.set_yticks(np.arange(len(pt_labels)))
ax.set_xticklabels(eta_pretty_labels, fontsize=12)
ax.set_yticklabels(pt_labels, fontsize=12)
ax.set_xlabel('η Region', fontsize=14)
ax.set_ylabel('$p_T$ [GeV]', fontsize=14)
plt.title("DATA Efficiencies vs. $p_T$ and η Region", fontsize=16)
ax.invert_yaxis()

# Add values to cells
for i in range(len(pt_labels)):
    for j in range(len(eta_labels)):
        val = heatmap_values[i, j]
        err = heatmap_errors[i, j]
        if np.isnan(val):
            ax.text(j, i, "NaN", ha="center", va="center", color="black", fontsize=10)
        else:
            text = f"{val:.3f} ± {err:.3f}"
            ax.text(j, i, text, ha="center", va="center", color="black", fontsize=10)

# Colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Eficiency", fontsize=12)

# Save
plt.tight_layout()
plt.savefig("final_gold_blp_DATA_EFF_heatmap.png")
plt.show()