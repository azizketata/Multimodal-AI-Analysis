"""Generate TAM construct bar chart for thesis Figure tam_constructs."""
import matplotlib.pyplot as plt
import numpy as np

# Data
participants = ['P1\n(Post-Doc, PM=4)', 'P2\n(PhD, PM=2)', 'P3\n(MSc, PM=4)', 'P4\n(PhD, PM=2)']
constructs = ['PEOU', 'PU', 'ATU']

# Per-participant construct means
peou = [4.67, 4.00, 4.00, 3.67]
pu   = [4.40, 4.00, 4.00, 4.00]
atu  = [5.00, 4.00, 3.33, 4.00]

x = np.arange(len(participants))
width = 0.22

fig, ax = plt.subplots(figsize=(8, 4.5))

bars1 = ax.bar(x - width, peou, width, label='Perceived Ease of Use (PEOU)', color='#4472C4', edgecolor='white')
bars2 = ax.bar(x, pu, width, label='Perceived Usefulness (PU)', color='#ED7D31', edgecolor='white')
bars3 = ax.bar(x + width, atu, width, label='Attitude Toward Use (ATU)', color='#70AD47', edgecolor='white')

# Neutral midpoint line
ax.axhline(y=3.0, color='gray', linestyle='--', linewidth=0.8, label='Neutral midpoint (3.0)')

# Overall mean line
ax.axhline(y=4.09, color='black', linestyle=':', linewidth=1.0, label='Overall mean (4.09)')

ax.set_ylabel('Mean Rating (1–5 Likert Scale)', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(participants, fontsize=9)
ax.set_ylim(1, 5.5)
ax.set_yticks([1, 2, 3, 4, 5])
ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
ax.grid(axis='y', alpha=0.3)

# Value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7.5)

plt.tight_layout()
plt.savefig('thesis/figures/tam_constructs.png', dpi=300, bbox_inches='tight')
print("Saved thesis/figures/tam_constructs.png")
plt.close()
