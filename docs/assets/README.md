# Hero artwork

`hpgsbs-hero.png` is the GitHub README banner, generated on 2026-10-16 with the built-in image_gen tool (imagegen skill). It depicts multichannel signals passing through CPU/PyTorch/Triton routes into aligned state boundaries. This is conceptual promotional artwork, not a measured signal, benchmark plot, or anatomical claim. No private EEG was supplied to the image generator.

The image is included under this repository's MIT license. No external vendor logos are used.

## Final generation prompt

```text
Use case: ads-marketing. Asset type: wide GitHub README hero banner for the open-source scientific Python project HP-GSBS (High-Performance Greedy State Boundary Search).
Create an inventive, sophisticated scientific editorial illustration, landscape about 2.5:1. Concept: a flowing bundle of fine multichannel neural signal ribbons enters a crystalline prism of computation, then resolves into beautifully aligned translucent temporal state blocks with clean luminous vertical boundary planes. Three subtle parallel routes through the prism suggest CPU, PyTorch, and Triton computation, converging to the same state boundaries. The scientific metaphor is faster computation with preserved segmentation decisions; this is conceptual art, not a measured plot. Dark midnight/navy backdrop, restrained electric cyan, warm amber and mint light, elegant glass and fine engraving, precise geometry, ample negative space, premium open-source research identity. No humanoid brain, no robots, no stock photo, no fake chart axes, no performance numbers, no vendor logos.
Typography: crisp, exact, large white title 'HP-GSBS' on the left, below it the small readable line 'High-Performance Greedy State Boundary Search'. Lower left small line 'CPU  /  PyTorch  /  Triton'. Signal sculpture fills the center and right without obscuring text. Strong hierarchy, readable at GitHub page width, restrained glow, clean professional composition. All text must be spelled exactly as provided. Opaque background.
```

## Measured performance figures

`performance-v0.2.0.png/.svg` and the derived JSON come from `results/v02/merge_9baff92/revisions/trials.json`, rendered by `scripts/plot_v02_performance.py`. Median of three timed fits after two warmups, actual min–max, workload-specific ratio of unrounded medians. The v0.1.0 and 9baff92 snapshots were measured in the same interleaved experiment. BBBD0 is one recording with repeated fits.

`performance-comparison.png/.svg` remains the historical v0.1 official-GSBS comparison, reproduced by `scripts/plot_performance.py`. Its BBBD0 result was one measurement per backend. These are separate protocols; neither figure is generated artwork.
