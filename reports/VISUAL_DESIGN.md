# Memory flows: editorial figure system

The initial design below was produced on 2026-10-08. Experimental and palette changes requested on 2026-10-09 are documented in `REVISION_20261009.md`, which supersedes the affected settings and the earlier unchanged-input claims. The original QA snapshot is retained in `visual_qa_20261008/`.

This revision replaces the ten active plotting scripts with a consistent publication design and adds one mechanistic reduced-flow figure. It supplies 19 figures in PDF, editable-text SVG, and 300 dpi PNG. The bundled numerical examples use the original smoke datasets; they are explicitly labeled and are not paper-scale experimental estimates. The flow portrait is an analytical evaluation of the existing reduced equations, not another full-network experiment.

## Regenerate without simulating

From the repository root:

```bash
python scripts/render_figures.py --mode smoke
python scripts/render_figures.py --mode smoke --quick
python scripts/render_figures.py --mode paper
```

The last command requires the saved paper datasets under `data/raw/paper/`. The default render destination is `figures/editorial/<mode>/`. Use `--input-root PATH` or `--output-dir PATH` to override either root. A complete simulation run remains `./run_all.sh --workers 8 --skip-tex`; it now invokes the same renderer, writing the final figures to `figures/paper/`. No paper-scale simulation was run for this visual revision.

Every complete render writes a manifest containing SHA-256 hashes of input files. `--quick` writes only 110 dpi PNGs. Full renders write PDF, SVG and 300 dpi PNG. Run quick renders in a separate output directory when preserving older full-resolution exports, as quick mode does not delete existing PDF/SVG files.

## Visual grammar

- Typography: bundled Liberation Sans regular/bold, a Helvetica-compatible TrueType publication face; STIX mathematics. Font licenses are included under `plotting/assets/`. Bundling fixes font availability rather than silently changing metrics between machines.
- Default figure width: 7.2 inches for multipanel figures, approximately a two-column journal figure. Labels are 8.5 pt, ticks 7.5 pt, legends usually 6.8-7 pt. Final journal layout must preserve these intended dimensions; squeezing a two-column figure into a single column would make text too small.
- Surface: warm off-white `#FAFAF8`; midnight ink `#16324F`; restrained teal, coral, indigo, ochre and cyan for identities.
- Categorical identity uses color **and** dashes; sampled comparisons also use distinct marker shapes. Memory IDs always map to the same style, independent of amplitude ranking. The five most active traces across each comparison figure are emphasized; every other trace remains visible in the labeled gray context layer.
- Lines join saved samples without spline fitting, smoothing or resampling. Dense data are not silently decimated. High-frequency behavior remains data limited.
- Axes: thin neutral spines, sparse horizontal guides, no enclosing box. Main temporal plots are wide. Return maps and reduced saliency portraits deliberately use equal unit scales because their coordinate geometry has meaning.
- Repeated legends in trajectory panels are collected above the figure, clear of peaks. Panel letters and concise descriptive titles replace long case IDs.

## Continuous color

The `memory_tide` map is built in `plotting/common.py` from midnight blue, desaturated teal, cyan, warm ivory/gold and a pale amber/coral finish. Its samples are equally spaced along an Oklab polyline. Measured relative luminance strictly increases from approximately 0.030 to 0.892; the coefficient of variation of adjacent Oklab distances is approximately 0.0103.

A saturated dark coral endpoint after ivory would reverse lightness and create a false extremum in grayscale. The warm endpoint is therefore pale; stronger coral is reserved for categorical event markers. The six requested palette accents are not all individually distinguishable under every color-vision condition, so colors never carry categorical meaning alone.

`python scripts/figure_qa.py --figures figures/editorial/smoke --output reports/visual_qa` generates grayscale and approximate protan/deutan/tritan review images, palette metrics and contact sheets. These are diagnostic approximations, not a guarantee for every viewer, printer or projector. The diagnostics require Pillow; numerical rendering itself uses the original NumPy, SciPy, pandas and Matplotlib dependencies.

## Scientific encodings improved

| Figure family | Design and scientific interpretation |
|---|---|
| Model comparison | Wide stacked traces; consistent memory identity; all memories retained; same overlap range. |
| Retrieval regimes | Short activation/pattern/gain titles; common saliency range; original case IDs remain in the exported data. Labels do not infer a bifurcation classification from filenames. |
| Escape validation | Theory, stability loss and visible switching remain separate; a companion relative-error panel exposes differences hidden by nearly coincident curves. Undefined events are not bridged in the timing panel. |
| Saliency trajectory | Event times are unobtrusive vertical guides; memory legends sit outside the data. |
| Return maps | Equal coordinate scales; identity subordinate; stable points filled, unstable points open diamonds using the original `stable` field. These are discrete return-map fixed points, not equilibria of the two-dimensional reduced drift. |
| Gain sweep | Theory foreground; simulation markers; error bars explicitly labeled as terminal-window standard deviation, not confidence intervals; unsettled windows are open diamonds. |
| Noise sweep | Mean confidence band belongs to the mean curve; median remains a separate dashed comparison. Negative delays remain visible. Zero-noise inputs use a symlog axis if present rather than being dropped. |
| Capacity and correlation curves | Reported 95% intervals are now visible; N/rule use redundant shapes and dashes. Load axes say realized P/N because they use `alpha_actual`. |
| Fanout heatmaps | Actual sampled coordinates define midpoint cell edges, including nonuniform grids; no smoothing; shared probability range [0,1]. Missing or infeasible cells are patterned. |
| Correlation/gain heatmap | Conditional escape time and transition probability appear together. No-correct-transition cells are hatched; other undefined cells are dotted. Sparse grids display sampled values and no interpolated threshold contour. The time colorbar spans observed finite values and is labeled explicitly. |
| Correlation dynamics | Excluded asymptotic values stay NaN on the original x grid. Empty estimates are explicitly explained, rather than appearing as unmarked empty axes. Unestimated gain thresholds use a baseline cross, not a fabricated gain value. |

Heatmap cells show bins around sampled coordinates, not resolved behavior throughout every bin. Midpoint boundaries are a layout convention, not an inferred bifurcation boundary. Confidence bands linearly join reported interval endpoints; no extra trials or continuous uncertainty model is introduced. Small smoke samples can produce broad intervals and sparse failures, which are preserved.

## Mechanistic reduced-flow portrait

`experiments/export_reduced_phase_plane.py` writes analytical figure data under `figures/data/<mode>/reduced_phase_plane/`. The companion plotter reads those saved fields and paths. For unit gains, orthogonal memories and out-degree at most one, on a stable source-memory branch, the existing theory gives

```text
s = |z_source|,  r = |z_successor|,  u = t/tau_z

ds/du = -s
dr/du = -r + kappa * branch_response(s)

valid branch: s > 1 and r < stability_threshold(s)
```

The background is Euclidean speed in these saliency coordinates. It is **not** energy, a Lyapunov function, density, or the full high-dimensional vector field. The common speed normalization supports comparisons across activations at the same gain. Excluded regions have a neutral fill. Low-weight streamlines show local direction only in the valid region, while explicit reference paths carry higher contrast and direction arrows. They stop at the first stability loss or branch-existence limit.

The selected entry amplitude is the stable return-map fixed point when one exists; otherwise it is 80% of the plotted source range. Three subordinate paths show different entry amplitudes. Start circles and escape squares distinguish progression without suggesting a continuous-time equilibrium. No saddle or stable-equilibrium markers are invented in a domain where the reduced source component is strictly decreasing.

The source/target dynamics and thresholds call the existing activation functions. The independent `solve_ivp` rendering paths agree with the original quadrature-based escape-time routine within 3e-6 for six activation/entry combinations. This added analytical export does not change the simulation solver or its Euler integration.

## Vector and LaTeX delivery

PDF is the publication master, with embedded fonts. Curves, axes, labels and boundaries remain vector; heatmap fields are selectively rasterized at 300 dpi. SVG preserves editable text and includes the same visual layout. Import the bundled fonts when editing SVGs elsewhere.

The renderer also creates standalone LaTeX wrappers under `plotting/latex/generated/<mode>/`. They include the same PDF master, so TeX inclusion cannot silently revert to an older palette or layout. `run_all.sh` compiles these wrappers when LaTeX is requested. Original `plotting/tikz/` sources are preserved as historical editable pgfplots alternatives and are no longer the final visual masters. They retain their older appearance and selected-curve conventions.

## Verification performed

- All ten original plotting entry points plus the reduced-flow plotter rendered successfully from the supplied tables: 19 PDF/SVG/PNG triplets.
- Input bytes were checked against the upload: 11 model-source files, 22 configuration files, 41 raw-data files, 15 reference artifacts and 8 legacy files are unchanged.
- All 23 CSV output contracts passed.
- Seventeen test functions passed by direct invocation, including four new scientific-encoding tests. The environment lacks pytest and tqdm; the three capacity protocol tests could not be imported, and installing those packages was unavailable. A complete pytest run is therefore **not** claimed. Run `python -m pytest -q` in the documented project environment for the full suite.
- Palette luminance, approximate color-vision diagnostics, final contact sheets and vector exports were reviewed. The new figure's mathematical domain and event termination were checked numerically.
- Original simulation runners were not changed. Only an additional analytical exporter, plotting helpers/plotters, render/QA tools, tests, documentation and the plotting stage of `run_all.sh` were added or revised.

The delivered atlas is a design proof using available data. Submission-quality statistical evidence still requires the configured paper experiments; the visual system is ready to render their outputs without another redesign.

The final PDF review detected CFF/OpenType embedding warnings with the initial Nimbus Sans font choice. Final masters use bundled Liberation Sans TrueType instead; font embeddings and rasterized PDF output were checked after regeneration.
