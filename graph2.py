import matplotlib.pyplot as plt
import numpy as np

# Data
methods = ["KPD", "KeyGrid", "KeyGridOrig", "SC3K", "SM", "Ours"]
DAS = np.array([0.69, 0.63, 0.67, 0.62, 0.72, 0.79])
DAS_err = np.array([0.13, 0.10, 0.12, 0.033, 0.13, 0.077])
Corr = np.array([0.69, 0.93, 0.92, 0.90, 0.91, 0.98])
Corr_err = np.array([0.16, 0.038, 0.047, 0.04, 0.031, 0.019])

metrics = ["DAS\n(Align with Keypoint Labels)", "Correspondence\n(Align with Segmentation Labels)"]
x = np.arange(len(metrics))  # 2 metric groups
width = 0.12
group_gap = 0.35           # extra gap between groups (⬅ change this to adjust spacing)

fig, ax = plt.subplots()

bars = []

pastel_colors = [
    "#FF9999", # KPD
    "#66B2FF", # KeyGrid
    "#99FF99", # KeyGridOrig
    "#FFF799", # SC3K
    "#C299FF", # SM
    "#FFCC99"  # Ours
]



# For each method, plot its DAS and Correspondence bars next to each other
for i, method in enumerate(methods):
    values = [DAS[i] * 100, Corr[i] * 100]
    errors = [DAS_err[i] * 100, Corr_err[i] * 100]
    # positions = x+ (i - len(methods)/2) * width + width/2
    positions = x + (i - len(methods)/2) * width + width/2
    positions[1] += group_gap  # push second group to create bigger spacing

    if method == "Ours":
        bar = ax.bar(
            positions, values, width, yerr=errors, 
            error_kw=dict(capsize=6, capthick=2, linewidth=1.5),
            label=method, edgecolor="black", linewidth=2.5, color=pastel_colors[i]
        )
    else:
        bar = ax.bar(
            positions, values, width, yerr=errors, 
            error_kw=dict(capsize=6, capthick=2, linewidth=1.5),
            label=method, color=pastel_colors[i]
        )
    bars.append(bar)

ax.set_xticks([x[0], x[1] + group_gap])
ax.set_xticklabels(metrics)
ax.set_ylabel("Metric Value (%)")
ax.set_ylim(0, 100)

# Legend: one entry per algorithm
ax.legend(title="Methods", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig("graph2.png")
