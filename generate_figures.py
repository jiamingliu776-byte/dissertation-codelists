"""Generate dissertation figures as SVG files."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT = r"D:\Dissertation\file\figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.bbox': 'tight',
    'savefig.dpi': 150,
})

# ══════════════════════════════════════════════════════════
# Figure 1: Pipeline Workflow (hand-drawn SVG, not matplotlib)
# ══════════════════════════════════════════════════════════

pipeline_svg = '''<?xml version="1.0" encoding="UTF-8"?>
<svg viewBox="0 0 660 530" xmlns="http://www.w3.org/2000/svg" width="660" height="530">
  <style>
    .box { rx: 8; ry: 8; }
    .box-primary { fill: #E8F0FE; stroke: #4285F4; stroke-width: 1.5; }
    .box-secondary { fill: #FFF3E0; stroke: #F9A825; stroke-width: 1.5; }
    .box-success { fill: #E8F5E9; stroke: #43A047; stroke-width: 1.5; }
    .box-neutral { fill: #F5F5F5; stroke: #9E9E9E; stroke-width: 1.5; }
    .box-danger { fill: #FFEBEE; stroke: #E53935; stroke-width: 1.5; }
    .label-bold { font-family: Calibri, Arial, sans-serif; font-size: 14px; font-weight: 600; fill: #333; text-anchor: middle; dominant-baseline: middle; }
    .label-small { font-family: Calibri, Arial, sans-serif; font-size: 11px; fill: #666; text-anchor: middle; dominant-baseline: middle; }
    .arrow { fill: none; stroke: #888; stroke-width: 1.5; marker-end: url(#ah); }
    .arrow-loop { fill: none; stroke: #4285F4; stroke-width: 1.5; stroke-dasharray: 6 3; marker-end: url(#ah2); }
    .section-label { font-family: Calibri, Arial, sans-serif; font-size: 11px; font-weight: 600; fill: #999; text-anchor: start; text-transform: uppercase; letter-spacing: 1px; }
  </style>
  <defs>
    <marker id="ah" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#888"/></marker>
    <marker id="ah2" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#4285F4"/></marker>
  </defs>

  <text x="30" y="28" class="section-label">1. Data Preparation</text>
  <rect x="30" y="42" width="160" height="50" class="box box-neutral"/>
  <text x="110" y="60" class="label-bold">Clinical Dictionary</text>
  <text x="110" y="76" class="label-small">diagnosis + product codes</text>
  <path d="M190,67 L240,67" class="arrow"/>
  <rect x="240" y="42" width="160" height="50" class="box box-primary"/>
  <text x="320" y="60" class="label-bold">Text Embedding</text>
  <text x="320" y="76" class="label-small">dense vector representation</text>
  <path d="M400,67 L450,67" class="arrow"/>
  <rect x="450" y="42" width="170" height="50" class="box box-primary"/>
  <text x="535" y="60" class="label-bold">Vector Index</text>
  <text x="535" y="76" class="label-small">semantic similarity search</text>

  <text x="30" y="130" class="section-label">2. Agentic Loop</text>
  <rect x="25" y="140" width="610" height="200" rx="12" ry="12" fill="none" stroke="#4285F4" stroke-width="1" stroke-dasharray="8 4" opacity="0.4"/>
  <text x="50" y="162" class="label-small" style="text-anchor:start; font-style:italic;">LLM decides autonomously when to search and when to stop</text>

  <rect x="60" y="185" width="150" height="55" class="box box-secondary"/>
  <text x="135" y="215" class="label-bold">LLM Agent</text>
  <path d="M210,210 L270,210" class="arrow"/>
  <text x="240" y="198" class="label-small">query</text>

  <rect x="270" y="185" width="170" height="55" class="box box-primary"/>
  <text x="355" y="205" class="label-bold">search_dictionary</text>
  <text x="355" y="222" class="label-small">top-K codes returned</text>
  <path d="M355,185 L355,135 L535,135 L535,92" class="arrow"/>

  <path d="M355,240 L355,270 L135,270 L135,240" class="arrow-loop"/>
  <text x="245" y="262" class="label-small" style="font-style:italic; fill:#4285F4;">review results, refine query or stop</text>

  <path d="M210,230 L270,305 L440,305" class="arrow"/>
  <rect x="440" y="278" width="170" height="50" class="box box-success"/>
  <text x="525" y="296" class="label-bold">submit_codelist</text>
  <text x="525" y="313" class="label-small">final code IDs</text>

  <text x="30" y="370" class="section-label">3. Evaluation</text>
  <rect x="80" y="388" width="170" height="50" class="box box-success"/>
  <text x="165" y="406" class="label-bold">AI Codelist</text>
  <text x="165" y="423" class="label-small">generated codes</text>
  <path d="M525,328 L525,365 L165,365 L165,388" class="arrow"/>

  <rect x="390" y="388" width="170" height="50" class="box box-danger"/>
  <text x="475" y="406" class="label-bold">Reference</text>
  <text x="475" y="423" class="label-small">expert codelists</text>

  <rect x="235" y="468" width="170" height="50" class="box box-neutral"/>
  <text x="320" y="486" class="label-bold">Compare</text>
  <text x="320" y="503" class="label-small">similarity metrics</text>

  <line x1="165" y1="438" x2="235" y2="493" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>
  <line x1="475" y1="438" x2="405" y2="493" stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>
</svg>'''

with open(os.path.join(OUT, "fig1_pipeline_workflow.svg"), "w", encoding="utf-8") as f:
    f.write(pipeline_svg)
print("Figure 1: Pipeline workflow SVG saved")


# ══════════════════════════════════════════════════════════
# Figure 2: F1 by Condition and Model
# ══════════════════════════════════════════════════════════

conditions = ["PAD", "CeVD", "HF", "Asthma", "COPD", "HTN", "MI", "AF", "Insulin", "Metformin"]
gpt_f1 =    [0.081, 0.262, 0.286, 0.362, 0.358, 0.377, 0.437, 0.500, 0.575, 0.748]
claude_f1 = [0.180, 0.259, 0.279, 0.391, 0.464, 0.485, 0.419, 0.628, 0.713, 0.859]
gemini_f1 = [0.169, 0.235, 0.317, 0.360, 0.488, 0.508, 0.516, 0.653, 0.709, 0.859]

x = np.arange(len(conditions))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 5.5))
bars1 = ax.bar(x - width, gpt_f1, width, label='GPT-5.4 Mini', color='#5B9BD5', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x, claude_f1, width, label='Claude Sonnet 5', color='#ED7D31', edgecolor='white', linewidth=0.5)
bars3 = ax.bar(x + width, gemini_f1, width, label='Gemini 3.6 Flash', color='#70AD47', edgecolor='white', linewidth=0.5)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.2f}',
                ha='center', va='bottom', fontsize=7, color='#555')

ax.axhline(y=0.399, color='#5B9BD5', linestyle='--', linewidth=0.8, alpha=0.6)
ax.axhline(y=0.468, color='#ED7D31', linestyle='--', linewidth=0.8, alpha=0.6)
ax.axhline(y=0.481, color='#70AD47', linestyle='--', linewidth=0.8, alpha=0.6)

ax.text(9.6, 0.399, 'GPT mean 0.399', fontsize=7, color='#5B9BD5', va='bottom')
ax.text(9.6, 0.468, 'Claude mean 0.468', fontsize=7, color='#ED7D31', va='bottom')
ax.text(9.6, 0.485, 'Gemini mean 0.481', fontsize=7, color='#70AD47', va='bottom')

ax.set_ylabel('F1 Score', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(conditions, fontsize=10)
ax.set_ylim(0, 1.0)
ax.set_yticks(np.arange(0, 1.1, 0.2))
ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax.grid(axis='y', alpha=0.3)
ax.set_title('F1 Score by Condition and Model', fontsize=14, fontweight='bold', pad=12)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig2_f1_by_condition.svg"), format='svg')
plt.close()
print("Figure 2: F1 bar chart SVG saved")


# ══════════════════════════════════════════════════════════
# Figure 3: FP/FN Error Classification (stacked bars)
# ══════════════════════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

models = ['GPT-5.4 Mini', 'Claude Sonnet 5', 'Gemini 3.6 Flash']

# FP data — unsure categories merged
# Unsure = Valid subtype (269,627,624) + Semantic drift (51,283,248)
# Identifiable: History/resolved, Risk/screening, Admin, Exclusion
fp_cats = ['Unsure (clinical review)', 'History/resolved', 'Risk/screening', 'Admin', 'Exclusion']
fp_colors = ['#FF9800', '#9E9E9E', '#2196F3', '#9C27B0', '#F44336']
fp_data = np.array([
    [320, 7, 10, 6, 3],
    [910, 15, 0, 0, 0],
    [872, 0, 0, 1, 0],
], dtype=float)
fp_totals = fp_data.sum(axis=1)
fp_pct = fp_data / fp_totals[:, None] * 100

x_fp = np.arange(3)
bottom = np.zeros(3)
for i, (cat, col) in enumerate(zip(fp_cats, fp_colors)):
    bars = ax1.bar(x_fp, fp_pct[:, i], 0.55, bottom=bottom, label=cat, color=col, edgecolor='white', linewidth=0.5)
    for j, bar in enumerate(bars):
        if fp_pct[j, i] > 5:
            cy = bottom[j] + fp_pct[j, i]/2
            ax1.text(bar.get_x() + bar.get_width()/2, cy + 1.5,
                    f'{fp_pct[j, i]:.0f}%', ha='center', va='center', fontsize=8, color='white', fontweight='bold')
            ax1.text(bar.get_x() + bar.get_width()/2, cy - 2.5,
                    f'({int(fp_data[j, i])})', ha='center', va='center', fontsize=7, color='white')
    bottom += fp_pct[:, i]

ax1.set_xticks(x_fp)
ax1.set_xticklabels(models, fontsize=9)
for j, t in enumerate(fp_totals):
    ax1.text(j, -8, f'n={int(t)}', ha='center', fontsize=8, color='#666')
ax1.set_ylabel('Percentage (%)', fontsize=10)
ax1.set_ylim(0, 110)
ax1.set_title('False Positive Classification', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right', fontsize=7.5, framealpha=0.9)

# FN data — unsure categories merged
# Unsure = Missed valid dx (738,515,494) + Sequelae (16,12,12) + Complication (9,7,8)
# Identifiable: Procedural, Admin, History+Cause of death
fn_cats = ['Unsure (clinical review)', 'Procedural', 'Admin', 'Other identifiable']
fn_colors = ['#FF9800', '#2196F3', '#9C27B0', '#9E9E9E']
fn_data = np.array([
    [763, 243, 174, 15],
    [534, 242, 183, 18],
    [514, 242, 182, 22],
], dtype=float)
fn_totals = fn_data.sum(axis=1)
fn_pct = fn_data / fn_totals[:, None] * 100

bottom = np.zeros(3)
for i, (cat, col) in enumerate(zip(fn_cats, fn_colors)):
    bars = ax2.bar(x_fp, fn_pct[:, i], 0.55, bottom=bottom, label=cat, color=col, edgecolor='white', linewidth=0.5)
    for j, bar in enumerate(bars):
        if fn_pct[j, i] > 5:
            cy = bottom[j] + fn_pct[j, i]/2
            ax2.text(bar.get_x() + bar.get_width()/2, cy + 1.5,
                    f'{fn_pct[j, i]:.0f}%', ha='center', va='center', fontsize=8, color='white', fontweight='bold')
            ax2.text(bar.get_x() + bar.get_width()/2, cy - 2.5,
                    f'({int(fn_data[j, i])})', ha='center', va='center', fontsize=7, color='white')
    bottom += fn_pct[:, i]

ax2.set_xticks(x_fp)
ax2.set_xticklabels(models, fontsize=9)
for j, t in enumerate(fn_totals):
    ax2.text(j, -8, f'n={int(t)}', ha='center', fontsize=8, color='#666')
ax2.set_ylabel('Percentage (%)', fontsize=10)
ax2.set_ylim(0, 110)
ax2.set_title('False Negative Classification', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', fontsize=7.5, framealpha=0.9, bbox_to_anchor=(0.95, 1.0))

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig3_error_classification.svg"), format='svg')
plt.close()
print("Figure 3: Error classification SVG saved")


# ══════════════════════════════════════════════════════════
# Figure 4: Condition Group Comparison
# ══════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(8, 5.5))

groups = ['Cardiovascular\n(AF, HF, CeVD, HTN, MI, PAD)', 'Respiratory\n(COPD, Asthma)', 'Drug Products\n(Insulin, Metformin)']
gpt_g = [0.324, 0.360, 0.662]
claude_g = [0.375, 0.428, 0.786]
gemini_g = [0.400, 0.424, 0.784]

x = np.arange(3)
width = 0.22

bg_colors = ['#E8EAF6', '#FFF3E0', '#E8F5E9']
for i, (bg, gx) in enumerate(zip(bg_colors, x)):
    ax.axvspan(gx - 0.42, gx + 0.42, color=bg, alpha=0.5, zorder=0)

bars1 = ax.bar(x - width, gpt_g, width, label='GPT-5.4 Mini', color='#5B9BD5', edgecolor='white', linewidth=0.5, zorder=3)
bars2 = ax.bar(x, claude_g, width, label='Claude Sonnet 5', color='#ED7D31', edgecolor='white', linewidth=0.5, zorder=3)
bars3 = ax.bar(x + width, gemini_g, width, label='Gemini 3.6 Flash', color='#70AD47', edgecolor='white', linewidth=0.5, zorder=3)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.012, f'{h:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='#444')

ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=10)
ax.set_ylabel('Mean F1 Score', fontsize=11)
ax.set_ylim(0, 0.95)
ax.set_yticks(np.arange(0, 1.0, 0.2))
ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.set_title('Mean F1 by Condition Group', fontsize=14, fontweight='bold', pad=12)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig4_condition_groups.svg"), format='svg')
plt.close()
print("Figure 4: Condition groups SVG saved")

# ══════════════════════════════════════════════════════════
# Figure 5: AF Inter-Expert Pairwise Heatmap (5×5)
# ══════════════════════════════════════════════════════════

names_5 = ["AF-1", "AF-2", "AF-3", "AF-4", "AF-5"]
matrix = np.array([
    [1.000, 0.366, 0.500, 0.387, 0.531],
    [0.366, 1.000, 0.360, 0.340, 0.353],
    [0.500, 0.360, 1.000, 0.410, 0.849],
    [0.387, 0.340, 0.410, 1.000, 0.436],
    [0.531, 0.353, 0.849, 0.436, 1.000],
])

fig, ax = plt.subplots(figsize=(6, 5.5))

im = ax.imshow(matrix, cmap='YlOrRd', vmin=0.2, vmax=1.0, aspect='equal')

ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.set_xticklabels(names_5, fontsize=11)
ax.set_yticklabels(names_5, fontsize=11)
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

for i in range(5):
    for j in range(5):
        val = matrix[i, j]
        if i == j:
            ax.text(j, i, '—', ha='center', va='center', fontsize=14, color='#999')
        else:
            color = 'white' if val > 0.65 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=11, color=color,
                    fontweight='bold' if val > 0.5 else 'normal')

cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
cbar.set_label('Jaccard Index', fontsize=11)

ax.set_title('AF Inter-Expert Pairwise Agreement', fontsize=14, fontweight='bold', pad=12)
ax.text(2, 4.85, f'Mean Jaccard = 0.453 (range 0.340–0.849)', ha='center', fontsize=10, color='#666', fontstyle='italic')

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig5_af_heatmap.svg"), format='svg')
plt.close()
print("Figure 5: AF heatmap SVG saved")

print(f"\nAll figures saved to {OUT}")
