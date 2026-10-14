# Your Diffusion Model is Just a Distribution Matching Model

Live page: https://xiaoxiaoyang2003.github.io/diffusion-is-matching/

Source: `src/diffusion_is_matching.md` → `index.html`

## Build the site

Edit `src/diffusion_is_matching.md`, then:

```bash
python3 build_site.py
```

## Toy experiments

**Image PE (text-conditioned keys, train-vs-decode grids):**

```bash
python3 experiments/toy_image_pe.py
```

**2D geometry suite** (small circle ablations):

```bash
CUDA_VISIBLE_DEVICES=0 python3 experiments/polish_2d.py
```

**Larger lattice suite** (5×5 grid; exact / interp / extrap):

```bash
python3 experiments/toy_render_suite.py
# smoke: TOY_SMOKE=1 python3 experiments/toy_render_suite.py --smoke
```

**Schema ↔ natural language** (matching drawers along a language-distance axis; 1×8):

```bash
python3 experiments/toy_nl_vs_pe.py
# smoke: TOY_SMOKE=1 python3 experiments/toy_nl_vs_pe.py --smoke
```

Charts and picture grids can be redrawn from the saved results, with no training:

```bash
python3 experiments/polish_2d.py --redraw
python3 experiments/toy_render_suite.py --redraw
python3 experiments/plot_nl_metrics.py
python3 experiments/plot_nl_gallery.py
```

Then rebuild:

```bash
python3 build_site.py
```

Dependencies: `torch` (CUDA recommended), `numpy`, `matplotlib`.
