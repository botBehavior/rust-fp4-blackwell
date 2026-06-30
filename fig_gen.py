r"""Data figures for the cuda-oxide paper. Numbers are hardcoded from the measured tables in the
paper (decode head-to-head; FP4 GEMV bandwidth) so the figures match the text exactly.
Run:  python fig_gen.py   (writes figs/*.pdf and figs/*.png next to this script)
Self-contained: matplotlib only.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
STYLE = os.path.join(HERE, 'paperstyle.mplstyle')
if os.path.exists(STYLE):
    plt.style.use(STYLE)
OUT = os.path.join(HERE, 'figs'); os.makedirs(OUT, exist_ok=True)
C_OURS, C_REF, C_BAR = '#0072B2', '#D55E00', '#009E73'


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, '%s.%s' % (name, ext)), dpi=160, bbox_inches='tight')
    plt.close(fig)


# ---------- Fig: decode throughput head-to-head (viability, not supremacy) ----------
fig, ax = plt.subplots(figsize=(5.2, 4.0))
labels = ['ours\n(pure-Rust cuda-oxide,\nMXFP4)', 'llama.cpp\n(Q4_K_M, hand-tuned\nCUDA + graphs)']
vals = [181, 664]
# asymmetric ranges: ours 176-193 (median 181); llama 634-664 (we report ~664)
err = [[181 - 176, 664 - 634], [193 - 181, 664 - 664]]
bars = ax.bar([0, 1], vals, width=0.6, color=[C_OURS, C_REF], yerr=err, capsize=5,
              error_kw=dict(ecolor='0.3', lw=1.2))
for x, v in zip([0, 1], vals):
    ax.text(x, v + 14, '%d tok/s' % v, ha='center', fontweight='bold', fontsize=10)
ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel('decode throughput (tokens/s, steady state)')
ax.set_ylim(0, 760)
ax.annotate('', xy=(0, 181), xytext=(1, 664),
            arrowprops=dict(arrowstyle='<->', color='0.4', lw=1.1))
ax.text(0.5, 430, '$\\approx$27% of llama.cpp\n($\\approx$3.7$\\times$ gap)', ha='center',
        fontsize=9.5, color='0.2',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.6', lw=0.8))
ax.set_title('Decode throughput, same GPU + model (RTX 5070 Ti, TinyLlama-1.1B)\nviability, not supremacy: we lose on raw speed', fontsize=9.5)
plt.tight_layout(); save(fig, 'fig_throughput')

# ---------- Fig: FP4 GEMV bandwidth runway (toward the 896 GB/s ceiling) ----------
fig, ax = plt.subplots(figsize=(6.2, 4.0))
steps = ['naive\nwarp-per-row,\nu32 LUT', 'arithmetic E2M1\ndecode, shared-mem,\n16-iter stream', 'same kernel,\nlarger matrix\n(16384$\\times$4096)']
bw = [184, 270, 370]
pct = [20.5, 30.2, 41.4]
PEAK = 896
x = range(len(steps))
ax.bar(x, bw, width=0.6, color=C_BAR, zorder=3)
for xi, (b, p) in enumerate(zip(bw, pct)):
    ax.text(xi, b + 14, '%d GB/s\n%.1f%%' % (b, p), ha='center', fontweight='bold', fontsize=9)
ax.axhline(PEAK, color=C_REF, ls='--', lw=1.4, zorder=2)
ax.text(len(steps) - 1, PEAK - 40, 'memory ceiling 896 GB/s', ha='right', color=C_REF, fontsize=8.5)
ax.set_xticks(list(x)); ax.set_xticklabels(steps, fontsize=8)
ax.set_ylabel('FP4 weight-streaming bandwidth (GB/s)')
ax.set_ylim(0, 980)
ax.set_title('FP4 GEMV bandwidth: a first cut, with runway to peak\n(weights-only traffic; not peak-tuned)', fontsize=9.5)
plt.tight_layout(); save(fig, 'fig_bandwidth')

print('saved fig_throughput, fig_bandwidth to', OUT)
