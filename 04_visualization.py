"""
Figures 
  fig1_scatter_lmd_vs_return.png: L-MD sentiment vs 1-day post-earnings return
  fig2_beat_vs_miss.png: Mean sentiment (L-MD & VADER) by earnings outcome
  fig3_sentiment_trend.png: L-MD sentiment trajectory across quarters per ticker
  fig4_heatmap_correlations.png: Correlation heatmap: all sentiment*return windows
  fig5_return_distribution.png: 1-day return distribution by ticker

Visualization approach (per Lecture 15 notes)
  - seaborn
  - matplotlib
  - plotnine
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns
import plotnine as p9

# Paths
PROCESSED_DIR = 'data/processed'
FIGURES_DIR   = 'report/figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

# Loading the data
df = pd.read_csv(
    os.path.join(PROCESSED_DIR, 'merged_final.csv'),
    parse_dates=['date']
)
print(f'Loaded {len(df)} rows')
print(df[['ticker', 'quarter', 'lmd_net_score', 'vader_compound',
          'return_1d', 'beat_miss']].to_string(index=False))

# Global theme
sns.set_theme(style='whitegrid', font_scale=1.15)

# Stock Brands Pallette 
PALETTE = {
    'AAPL': '#2196F3',
    'MSFT': '#4CAF50',   
    'NVDA': '#9C27B0',   
    'JPM':  '#FF9800',   
}
OUTCOME_COLORS = {'Beat': '#4CAF50', 'Miss': '#F44336'}


# Figure 1 — L-MD Net Sentiment vs. 1-Day Post-Earnings Return (Scatter Plot)
# This plot checks whether more positive earnings-call language tends to align with stronger next-day stock returns.

# Each point = one earnings quarter for one ticker
# A pooled OLS trend line tests the aggregate directional relationship.

fig, ax = plt.subplots(figsize=(9, 5.5))

for ticker, group in df.groupby('ticker'):
    ax.scatter(
        group['lmd_net_score'],
        group['return_1d'],
        color=PALETTE[ticker],
        label=ticker,
        s=100,
        edgecolors='white',
        linewidths=0.9,
        zorder=3,
    )
    # Annotate every point with its fiscal quarter (e.g. "Q2") for traceability
    for _, row in group.iterrows():
        ax.annotate(
            row['quarter'],
            xy=(row['lmd_net_score'], row['return_1d']),
            xytext=(7, 4),
            textcoords='offset points',
            fontsize=7.5,
            color=PALETTE[ticker],
            alpha=0.85,
        )

# Reference lines at zero emphasise quadrant: positive/negative sentiment × return
ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5, zorder=1)
ax.axvline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5, zorder=1)

# Pooled OLS trend line — np.polyfit returns [slope, intercept] for degree-1 fit
z = np.polyfit(df['lmd_net_score'], df['return_1d'], 1)
p_line = np.poly1d(z)
x_range = np.linspace(df['lmd_net_score'].min(), df['lmd_net_score'].max(), 200)

ax.plot(
    x_range, p_line(x_range),
    color='crimson', linewidth=1.8, linestyle='--', alpha=0.85,
    label='OLS trend (pooled)',
)

# Pearson r in title — summarizes the linear relationship strength
corr = df['lmd_net_score'].corr(df['return_1d'])
ax.set_title(
    f'L-MD Sentiment vs. 1-Day Post-Earnings Return\n'
    f'Pearson r = {corr:+.3f}  |  n = {len(df)} quarters',
    fontsize=13, fontweight='bold', pad=12,
)
ax.set_xlabel('L-MD Net Sentiment Score', fontsize=10.5)
ax.set_ylabel('1-Day Return', fontsize=10.5)
ax.legend(framealpha=0.9, fontsize=9)
sns.despine() 

plt.tight_layout()
fig.savefig(
    os.path.join(FIGURES_DIR, 'fig1_scatter_lmd_vs_return.png'),
    dpi=150, bbox_inches='tight',
)
plt.show()
plt.close()
print('Figure 1 saved.')


# Figure 2 — Mean Sentiment: Beat vs. Miss (L-MD and VADER)
# Goal: compare whether companies that beat earnings estimates used measurably more positive language than those that missed.
# Two panels for direct L-MD vs VADER cross-method comparison.

fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)

for ax, col, ylabel, title_method in zip(
    axes,
    ['lmd_net_score', 'vader_compound'],
    ['L-MD Net Score', 'VADER Compound Score'],
    ['L-MD', 'VADER'],
):

# Aggregate: mean sentiment score grouped by earnings outcome
    means  = df.groupby('beat_miss')[col].mean().reset_index()
    colors = [OUTCOME_COLORS[b] for b in means['beat_miss']]

    bars = ax.bar(
        means['beat_miss'], means[col],
        color=colors, edgecolor='white', linewidth=1.2, width=0.45,
    )

# Data labels above each bar
    for bar, val in zip(bars, means[col]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + means[col].max() * 0.02,
            f'{val:.4f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold',
        )

# Error bars showing standard deviation
    stds = df.groupby('beat_miss')[col].std()
    for bar, outcome in zip(bars, means['beat_miss']):
        ax.errorbar(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            yerr=stds[outcome],
            fmt='none', color='black', capsize=5, linewidth=1.2,
        )

    ax.set_title(f'Mean {title_method} Sentiment\nBeat vs. Miss', fontweight='bold')
    ax.set_xlabel('Earnings Outcome', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(0, means[col].max() * 1.35)
    sns.despine(ax=ax)

fig.suptitle(
    'Sentiment Comparison: Beat vs. Miss  —  L-MD vs. VADER',
    fontsize=13, fontweight='bold', y=1.02,
)

# Legend for the colors
beat_patch = mpatches.Patch(color=OUTCOME_COLORS['Beat'], label='Beat')
miss_patch = mpatches.Patch(color=OUTCOME_COLORS['Miss'], label='Miss')
fig.legend(handles=[beat_patch, miss_patch], loc='upper right',
           bbox_to_anchor=(1.0, 1.0), fontsize=9)

plt.tight_layout()
fig.savefig(
    os.path.join(FIGURES_DIR, 'fig2_beat_vs_miss.png'),
    dpi=150, bbox_inches='tight',
)
plt.show()
plt.close()
print('Figure 2 saved.')


# Figure 3 — L-MD Sentiment Trajectory Across Quarters (Line Plot)
# Goal: reveal within-year sentiment trends for each company.
# Uses seaborn lineplot and overlays beat/miss marker

QUARTER_ORDER = ['Q1', 'Q2', 'Q3', 'Q4']

# Make quarter categorical so the x-axis sorts correctly
df['quarter'] = pd.Categorical(df['quarter'], categories=QUARTER_ORDER, ordered=True)

fig, ax = plt.subplots(figsize=(9, 5.5))

#  seaborn lineplot: marker='o' adds a point at each observed quarter

sns.lineplot(
    data=df.sort_values('quarter'),
    x='quarter', y='lmd_net_score',
    hue='ticker', style='ticker',
    marker='o', markersize=9, linewidth=2.2,
    palette=PALETTE,
    ax=ax,
)

# Overlay beat/miss markers
MARKER_MAP = {'Beat': '^', 'Miss': 'v'}
for _, row in df.iterrows():
    ax.scatter(
        row['quarter'], row['lmd_net_score'],
        marker=MARKER_MAP.get(row['beat_miss'], 'o'),
        color=OUTCOME_COLORS.get(row['beat_miss'], 'gray'),
        s=60, zorder=5, alpha=0.6,
    )

# Ticker names
for ticker, group in df.groupby('ticker'):
    last = group.dropna(subset=['lmd_net_score']).sort_values('quarter').iloc[-1]
    ax.annotate(
        ticker,
        xy=(last['quarter'], last['lmd_net_score']),
        xytext=(10, 2), textcoords='offset points',
        fontsize=9, fontweight='bold', color=PALETTE[ticker],
    )

ax.axhline(0, color='gray', linewidth=0.9, linestyle='--', alpha=0.55)
ax.set_title(
    'L-MD Sentiment Score Across Quarters by Company\n'
    '(▲ Beat  ▼ Miss)',
    fontsize=13, fontweight='bold',
)
ax.set_xlabel('Fiscal Quarter', fontsize=11)
ax.set_ylabel('L-MD Net Sentiment Score', fontsize=11)

# Seaborn Legend
beat_patch = mpatches.Patch(color=OUTCOME_COLORS['Beat'], label='Beat', alpha=0.7)
miss_patch = mpatches.Patch(color=OUTCOME_COLORS['Miss'], label='Miss', alpha=0.7)
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles + [beat_patch, miss_patch],
    labels + ['Beat', 'Miss'],
    title='Ticker / Outcome', framealpha=0.9, fontsize=9,
)

sns.despine()
plt.tight_layout()
fig.savefig(
    os.path.join(FIGURES_DIR, 'fig3_sentiment_trend.png'),
    dpi=150, bbox_inches='tight',
)
plt.show()
plt.close()
print('Figure 3 saved.')


# Figure 4 — Correlation Heatmap (plotnine / ggplot-style)
# Goal: present the full sentiment × return-window correlation matrix.
SENTIMENT_COLS = ['lmd_net_score', 'prep_lmd_net_score', 'vader_compound']
RETURN_COLS    = ['return_1d', 'return_3d', 'return_5d']

# Build tidy long-form DataFrame for plotnine — it expects tidy input (like seaborn)
corr_matrix = (
    df[SENTIMENT_COLS + RETURN_COLS]
    .corr()
    .loc[SENTIMENT_COLS, RETURN_COLS]
    .round(3)
)

# Pretty axis labels
SENT_LABELS   = {
    'lmd_net_score':      'L-MD Net',
    'prep_lmd_net_score': 'L-MD Prep',
    'vader_compound':     'VADER',
}
RETURN_LABELS = {
    'return_1d': '1-Day Return',
    'return_3d': '3-Day Return',
    'return_5d': '5-Day Return',
}

# Melt into long form: columns = sentiment | return_window | corr
corr_long = (
    corr_matrix
    .reset_index()
    .rename(columns={'index': 'sentiment'})
    .melt(id_vars='sentiment', var_name='return_window', value_name='corr')
)
corr_long['sentiment']     = corr_long['sentiment'].map(SENT_LABELS)
corr_long['return_window'] = corr_long['return_window'].map(RETURN_LABELS)

#   plotnine grammar-of-graphics heatmap
#   aes(x, y, fill) — declares axis and fill aesthetics declaratively
#   geom_tile()     — renders each cell as a rectangle
#   geom_text()     — overlays the numeric correlation value in each tile
heat_plot = (
    p9.ggplot(corr_long, p9.aes(x='return_window', y='sentiment', fill='corr'))
    + p9.geom_tile(color='white', size=0.8)
    + p9.geom_text(
        p9.aes(label='corr'),
        format_string='{:.3f}',
        size=11, color='white', fontweight='bold',
    )
    + p9.scale_fill_gradient2(
        low='#D32F2F', mid='#F5F5F5', high='#1976D2',
        midpoint=0, limits=(-1, 1),
        name='Pearson R',
    )
    + p9.labs(
        title='Sentiment × Return Correlation Matrix',
        x='Return Window',
        y='Sentiment Method',
    )
    + p9.theme_minimal(base_size=12)
    + p9.theme(
        plot_title=p9.element_text(weight='bold', size=13),
        axis_text_x=p9.element_text(angle=15, hjust=1),
        figure_size=(7, 4),
    )
)

heat_plot.save(
    os.path.join(FIGURES_DIR, 'fig4_heatmap_correlations.png'),
    dpi=150,
)
print(heat_plot)
print('Figure 4 saved.')

# Figure 5 — Boxplot + stripplot: 1-day return distribution by ticker
# This figure shows how post-earnings returns vary across companies.
# The boxplot gives the median and spread, while the overlaid points to show the individual observations so the sample size remains visible.

fig5_df = df[["ticker", "return_1d", "beat_miss"]].dropna().copy()

fig, ax = plt.subplots(figsize=(9, 5.5))

sns.boxplot(
    data=fig5_df,
    x="ticker",
    y="return_1d",
    hue="ticker",
    palette=PALETTE,
    dodge=False,
    width=0.55,
    fliersize=0,
    linewidth=1.2,
    ax=ax,
)
# Seaborn legend
if ax.get_legend() is not None:
    ax.get_legend().remove()

sns.stripplot(
    data=fig5_df,
    x="ticker",
    y="return_1d",
    hue="beat_miss",
    palette=OUTCOME_COLORS,
    size=7,
    jitter=True,
    alpha=0.9,
    edgecolor="white",
    linewidth=0.6,
    ax=ax,
)

# Zero line separates positive and negative post-earnings reactions
ax.axhline(0, color="gray", linewidth=0.9, linestyle="--", alpha=0.6)

ax.set_title(
    "1-Day Post-Earnings Return Distribution by Ticker",
    fontsize=13,
    fontweight="bold",
)
ax.set_xlabel("Ticker", fontsize=11)
ax.set_ylabel("1-Day Return", fontsize=11)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))

beat_patch = mpatches.Patch(color=OUTCOME_COLORS["Beat"], label="Beat")
miss_patch = mpatches.Patch(color=OUTCOME_COLORS["Miss"], label="Miss")
ax.legend(
    handles=[beat_patch, miss_patch],
    title="Outcome",
    framealpha=0.9,
    fontsize=9,
)

sns.despine()
plt.tight_layout()
fig.savefig(
    os.path.join(FIGURES_DIR, "fig5_return_distribution.png"),
    dpi=150,
    bbox_inches="tight",
)
plt.close()
print("Figure 5 saved.")

# Correlation summary table
print('\n── Correlation Summary ───────────────────────────────────────')
print(
    df[SENTIMENT_COLS + RETURN_COLS]
    .corr()
    .loc[SENTIMENT_COLS, RETURN_COLS]
    .round(3)
    .to_string()
)

print(f'\nAll figures saved to: {FIGURES_DIR}')