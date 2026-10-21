#!/usr/bin/env python3
"""Build a vector-preserving review atlas from editorial PDF masters.

Optional authoring dependencies: reportlab and pypdf. The individual figures
remain the submission assets; the atlas adds review captions and provenance.
"""
from __future__ import annotations
import argparse,io
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import simpleSplit
from pypdf import PdfReader,PdfWriter,Transformation

ITEMS=[
 ('reduced_phase_plane_kappa_15','Source-branch flow','The reduced drift makes the escape mechanism explicit: source saliency decays while successor saliency builds. Reference paths terminate at the stability boundary. The background encodes speed, not energy.','Analytical reduced model; orthogonal memories, unit gains, singular limit.'),
 ('return_map','From escape to repeated retrieval','The return map relates one entry saliency to the next. Filled circles and open diamonds distinguish the stable and unstable fixed points recorded in the source table. Equal coordinate scales preserve the geometry.','return_map/return_map.csv; return_map_fixed_points.csv'),
 ('tanh_saliency_trajectory','Sequential saliency transfer','Memory identities retain the same colors and line patterns throughout the atlas. Vertical guides mark stability loss. All memories remain present; low-amplitude traces form a lighter context layer.','escape_time/escape_time_timeseries.csv; escape_time_events.csv'),
 ('escape_time_validation','Stability loss precedes visible switching','Theory, empirical stability loss and the later visible switch are separate observables. The companion error panel reveals the timing discrepancy that is difficult to see when the first two curves nearly coincide.','escape_time/escape_time_events.csv'),
 ('model_comparison','Comparing temporal organization','Stacked, wide panels preserve time resolution and a shared overlap scale. The orthogonal two-timescale case now uses fast-state noise sigma = 1e-4 per step; the other cases retain their previous noise settings.','model_comparison/model_comparison_timeseries.csv'),
 ('limit_cycles','Retrieval regimes','A common saliency range makes amplitudes comparable across activation, pattern type and gain. Titles show the measured configuration without treating a case filename as proof of a bifurcation regime.','limit_cycles/limit_cycles_timeseries.csv'),
 ('activation_functions','Activation response','Tanh and SmoothF use redundant line patterns. Both curves are evaluated analytically on the same x grid; their shapes are displayed without decorative interpolation.','Analytical activation functions, evaluated on 500 points.'),
 ('smoothf_implicit_curves','Implicit escape and fixed-point conditions','The model functions are foregrounded against the subdued inverse-gain reference. These panels display the supplied analytical tables for the escape and fixed-point conditions.','return_map/smoothf_implicit_G_Z4.csv; smoothf_implicit_G_fixed.csv'),
 ('kappa_sweep','Feedback gain and residence time','The stable fixed-point prediction provides context for terminal-window simulation estimates. Error bars report window standard deviation. Any unsettled estimates retain a distinct open marker.','kappa_sweep/kappa_sweep_summary.csv; kappa_sweep_theory.csv'),
 ('noise_sweep','Noise, ejection delay and switching outcomes','Mean and median are separate curves; the shaded confidence interval belongs to the mean. Negative switching gaps remain visible. The second panel separates post-escape and premature outcomes.','noise_sweep/noise_sweep_summary.csv'),
 ('capacity_scaled_cycle','Reliability as cycle load increases','Network sizes share a consistent shape and line code. The requested load range is now P/N = 0.15-0.50. Realized loads follow integer memory counts; reported confidence intervals remain visible.','capacity/capacity_probability.csv; scaled_cycle subset'),
 ('capacity_fixed_cycle_L5','Reliability for a fixed sequence','The fixed-cycle protocol is kept distinct from the scaled-cycle experiment. The requested load range is now P/N = 100-140. The refreshed smoke preview tests its endpoints with N = 8; the paper protocol retains N = 256.','capacity/capacity_probability.csv; fixed_cycle subset'),
 ('capacity_fanout_tanh','Branching reliability: Tanh','Small multiples isolate out-degree while preserving the same probability scale. Each network size carries the same marker and dash across panels. Shading shows reported confidence intervals.','capacity/capacity_probability.csv; fanout, tanh subset'),
 ('capacity_fanout_smoothf','Branching reliability: SmoothF','The paired activation uses the same composition and statistical encoding. Every displayed point comes from the supplied summary table, including unsuccessful transitions.','capacity/capacity_probability.csv; fanout, smoothf subset'),
 ('capacity_fanout_phase_plane_tanh','The branching parameter map: Tanh','Midpoint cell edges follow actual sample coordinates. Probability is normalized to [0,1]. Sparse-grid cell labels make the limited sampling explicit; no smooth transition boundary is inferred.','capacity/capacity_probability.csv; fanout_phase, tanh subset'),
 ('capacity_fanout_phase_plane_smoothf','The branching parameter map: SmoothF','The same continuous palette and scale make activation comparisons possible. Missing or infeasible graph cells are patterned when present, rather than encoded as numerical zero.','capacity/capacity_probability.csv; fanout_phase, smoothf subset'),
 ('correlation','Correlation and memory selectivity','Raw and generalized-pseudoinverse decoding are distinguished by both color and line/marker style. Retrieval reliability and unintended-coordinate activation appear as complementary observables.','correlation/correlation_summary.csv'),
 ('correlation_dynamics','Correlation-dependent dynamical limits','Gain and sustained-retrieval estimates are separated from asymptotic observables. This smoke dataset has no qualified asymptotic estimates: those panels say so explicitly, without inventing a trend.','correlation_dynamics/correlation_dynamics_summary.csv; theory.json'),
 ('correlation_phase_plane','Timing and reliability must be read together','Conditional escape time is paired with transition probability. Hatched cells indicate no correct transition. Escape-based deadlines and same-seed extensions reduce premature timeouts. Dotted reliability cells include trials still unfinished at their final horizon.','correlation_phase_plane/correlation_phase_plane_summary.csv')]

ROOT=Path(__file__).resolve().parents[1]
pdfmetrics.registerFont(TTFont('AtlasSans',str(ROOT/'plotting/assets/LiberationSans-Regular.ttf')))
pdfmetrics.registerFont(TTFont('AtlasSans-Bold',str(ROOT/'plotting/assets/LiberationSans-Bold.ttf')))
W,H=792,612;M=42;INK=HexColor('#16324F');MUTED=HexColor('#697781');PAPER=HexColor('#FAFAF8')

def paragraph(c,text,x,y,width,size=9.5,leading=14,color=INK):
    c.setFillColor(color);c.setFont('AtlasSans',size)
    for line in simpleSplit(text,'AtlasSans',size,width):c.drawString(x,y,line);y-=leading
    return y

def background(c,page,title=None):
    c.setFillColor(PAPER);c.rect(0,0,W,H,fill=1,stroke=0)
    c.setStrokeColor(HexColor('#D4DCD9'));c.setLineWidth(.6);c.line(M, H-48,W-M,H-48)
    c.setFillColor(MUTED);c.setFont('AtlasSans',8);c.drawString(M,H-33,'MEMORY FLOWS / SCIENTIFIC FIGURE ATLAS');c.drawRightString(W-M,H-33,f'{page:02d} / 20')
    c.setFont('AtlasSans',7);c.drawString(M,20,'Editorial design proof | 09 October 2026');c.drawRightString(W-M,20,'Individual PDF / SVG / PNG masters included')

def merge_figure(page,path,box):
    source=PdfReader(path).pages[0];sw=float(source.mediabox.width);sh=float(source.mediabox.height)
    x,y,w,h=box;scale=min(w/sw,h/sh)
    x+=(w-sw*scale)/2;y+=(h-sh*scale)/2
    page.merge_transformed_page(source,Transformation().scale(scale).translate(x,y),over=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--figures',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    writer=PdfWriter();stream=io.BytesIO();c=canvas.Canvas(stream,pagesize=(W,H),initialFontName='AtlasSans');background(c,1)
    c.setFillColor(INK);c.setFont('AtlasSans-Bold',34);c.drawString(M,507,'Memory flows')
    c.setFont('AtlasSans',18);c.drawString(M,476,'A coherent visual language for sequential retrieval')
    paragraph(c,'19 figure masters. One reproducible design system.',M,442,700,size=11)
    c.setFillColor(INK);c.setFont('AtlasSans-Bold',10);c.drawString(M,185,'READ THE MECHANISM');c.drawString(410,185,'READ THE EVIDENCE')
    paragraph(c,'Begin with source-branch flow, return maps and saliency transfer. Then inspect timing, gain, noise, capacity and correlation.',M,165,320,size=10)
    paragraph(c,'Simulation figures use the supplied smoke datasets. They are design proofs, not paper-scale estimates. The flow field is an analytical reduction.',410,165,335,size=10)
    c.setFont('AtlasSans-Bold',9);c.drawString(M,87,'REPRODUCE')
    paragraph(c,'python scripts/render_figures.py --mode smoke',M,69,700,size=9)
    paragraph(c,'Use --mode paper after generating paper datasets. See reports/VISUAL_DESIGN.md for encodings and validation.',M,51,700,size=8,color=MUTED)
    c.showPage();c.save();page=PdfReader(stream).pages[0]
    merge_figure(page,a.figures/'reduced_phase_plane_kappa_15.pdf',(40,210,350,210));merge_figure(page,a.figures/'tanh_saliency_trajectory.pdf',(406,235,344,160));writer.add_page(page);writer.add_outline_item('Overview',0)
    for index,(stem,title,caption,source) in enumerate(ITEMS,2):
        stream=io.BytesIO();c=canvas.Canvas(stream,pagesize=(W,H),initialFontName='AtlasSans');background(c,index)
        c.setFillColor(INK);c.setFont('AtlasSans-Bold',18);c.drawString(M,H-78,f'{index-1:02d}  {title}')
        paragraph(c,caption,M,94,W-2*M,size=9.3,leading=13)
        paragraph(c,'Source: '+source,M,41,W-2*M,size=7,color=MUTED)
        c.showPage();c.save();page=PdfReader(stream).pages[0]
        merge_figure(page,a.figures/(stem+'.pdf'),(M,118,W-2*M,390));writer.add_page(page);writer.add_outline_item(title,index-1)
    writer.add_metadata({'/Title':'Memory flows - Scientific figure atlas','/Subject':'19 editorial figures; smoke data and analytical reduced flow','/Author':'Memory flows research project'})
    with a.output.open('wb') as out:writer.write(out)
    print(f'Wrote {len(writer.pages)} pages to {a.output}')
if __name__=='__main__':main()
