# ISOPE-2027 Conference Paper: Meta-Optimization for Constrained Aerodynamic Shape Design

**Title:** *Enhancing Aerodynamic Shape Optimization via Reinforcement Meta-Learning, Large Language Models, and Likelihood-Free Bayesian Meta-Learning*

**Authors:** Huang Ximing$^a$, Xing Xiuqing$^b$, Zhang Xiaoqin$^b$, Kang Chang Wei$^b$
- $^a$ College of Computing and Data Science, Nanyang Technological University, 50 Nanyang Ave, Singapore 639798
- $^b$ A*STAR Institute of Advanced Intelligence and Computing, 1 Fusionopolis Way, #16-16, Connexis North Tower, Singapore 138632

## Repository Structure

```text
ISOPE2027/
├── main.tex                              # Camera-ready ISOPE-2027 LaTeX manuscript (US Letter, 2-column)
├── main.pdf                              # Compiled camera-ready PDF
├── deffile.tex                           # ISOPE LaTeX style definitions (pdfLaTeX / XeLaTeX / Tectonic compatible)
├── figures/                              # 7 publication-grade vector PDF figures
│   ├── fig1_framework.pdf                # Cross-dimensional (2D D=18 -> 3D D=63) meta-optimization architecture
│   ├── fig2_2d_rlmeta_convergence.pdf    # 2D 6-optimizer Pure RL-Meta vs. Vanilla convergence curves
│   ├── fig3_2d_paradigms_perf.pdf        # 2D 4-paradigm convergence curves & 10-seed distributions
│   ├── fig4_2d_airfoil_shapes.pdf        # 2D representative median optimized airfoils vs. NACA 0012
│   ├── fig5_3d_blade_perf.pdf            # 3D NREL Phase VI 10-seed convergence & distributions
│   ├── fig6_3d_blade_aero.pdf            # 3D chord, twist, effective AoA & 3-station S809 profiles
│   └── fig7_3d_blade_smoothness.pdf      # 3D lofted orthographic top/side smoothness verification
├── scripts/
│   └── generate_figures.py               # Reproducible figure generation from experimental JSON/NPZ outputs
└── Template-LaTeX-Kashiwagi-2011-Final/  # Original official ISOPE LaTeX template archive
```

## Build Instructions

Compile with `pdflatex`, `xelatex`, or `tectonic`:

```bash
pdflatex main.tex && pdflatex main.tex
# or
tectonic main.tex
```
