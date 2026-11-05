#!/usr/bin/env python3
"""Render the documented CPU architecture; development-only matplotlib dependency."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/tesslot-matplotlib')
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({'font.family': 'DejaVu Serif', 'mathtext.fontset': 'dejavuserif',
                     'font.size': 10, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
fig, ax = plt.subplots(figsize=(18, 11))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set(xlim=(0,18), ylim=(0,11)); ax.axis('off')
ink='#26323d'; blue='#edf3f9'; green='#edf5ef'; gray='#f5f5f5'
def box(x,y,w,h,text,fill=blue,fontsize=10):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=0.06',
                              edgecolor=ink,facecolor=fill,linewidth=0.85))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fontsize,linespacing=1.65,color=ink)
def arrow(points,dashed=False):
    for a,b in zip(points[:-2],points[1:-1]):
        ax.plot([a[0],b[0]],[a[1],b[1]],color=ink,lw=.9,ls='--' if dashed else '-')
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=10,
                                lw=.9,color=ink,linestyle='--' if dashed else '-'))
def label(x,y,s,size=10):ax.text(x,y,s,ha='center',va='center',fontsize=size,color=ink)
label(9,10.62,'Kidra CPU reference: multi-window transformer with shared global exchange',17)
label(9,10.22,r'Architecture kidra-reference/2  •  one synchronized decision snapshot per tick',11)
for x in [4.35,13.15]:ax.plot([x,x],[.85,9.85],color='#b9c1c9',lw=.6)
label(2.2,9.68,'(a) Model overview',13)
label(8.75,9.68,r'(b) Expanded layer $\ell = 1,\ldots,L$',13)
label(15.55,9.68,'(c) Tick and memory semantics',13)
# Overview: token and prediction streams stay distinct.
box(.45,8.35,3.45,.85,r'Open windows $w \in \mathcal{W}_t$'+'\n'+r'Local positions $p_w$; slots $s_w < S$',gray)
box(.45,6.95,3.45,1.0,r'Memory: $E_{\mathrm{tok}}[z] + b_{\mathrm{id}} E_{\mathrm{win}}[s_w]$'+'\n'+r'Query: $e_q + b_{\mathrm{id}} E_{\mathrm{win}}[s_w]$')
arrow([(2.18,8.35),(2.18,7.99)])
box(.45,5.25,3.45,1.1,r'$L$ local blocks + global exchanges'+'\n'+r'Row states $h^{(\ell)} \in \mathbb{R}^{d}$'+'\n'+'Memory and queries obey the same masks')
arrow([(2.18,6.95),(2.18,6.39)])
box(.45,3.85,3.45,.8,r'RMSNorm $\rightarrow W_{\mathrm{vocab}}$'+'\n'+r'Query logits $\in \mathbb{R}^{V}$')
arrow([(2.18,5.25),(2.18,4.69)])
box(.45,2.1,3.45,1.1,'Unshifted teacher-forced targets\n'+r'$\mathcal{L}=\frac{\sum_r \omega_r\,\mathrm{CE}(a_r,z_r)}{\max(\sum_r\omega_r,1)}$',green,11)
arrow([(2.18,3.85),(2.18,3.24)])
label(2.18,1.4,'Prediction queries never embed their targets.\nOnly durable tokens populate memory.',10)
# Expanded layer and cache side branch.
box(4.75,8.4,4.5,.65,r'Input row state $h^{(\ell-1)} \in \mathbb{R}^{d}$',gray,11)
box(4.75,6.8,4.5,1.0,'Local attention + residual\n'+r'$Q=\mathrm{RoPE}_{p_w}(W_Q^{\ell}\mathrm{RMSNorm}(h))$'+'\n'+r'$u=h+W_O^{\ell}\,\mathrm{Attn}(Q,K_w,V_w;M_w)$')
arrow([(7,8.4),(7,7.84)])
box(4.75,5.25,4.5,.95,'SwiGLU feed-forward + residual\n'+r'$f=u+W_{\downarrow}^{\ell}[\mathrm{SiLU}(W_g^{\ell}\bar u)\odot W_{\uparrow}^{\ell}\bar u]$'+'\n'+r'$\bar u=\mathrm{RMSNorm}(u),\quad d_{\mathrm{ff}}=2d$')
arrow([(7,6.8),(7,6.24)])
box(4.75,3.1,4.5,1.5,'Shared global exchange + residual\n'+r'$Q_g=W_{Q,g}\mathrm{LayerNorm}(f)$'+'\n'+r'$h^{(\ell)}=f+W_{O,g}\mathrm{Attn}(Q_g,K_g,V_g;M_g)$'+'\n'+'No global K/V projections or query RoPE',green)
arrow([(7,5.25),(7,4.64)])
box(4.75,1.8,4.5,.65,r'Output $h^{(\ell)}$ to layer $\ell+1$ (or head)',gray,11)
arrow([(7,3.1),(7,2.49)])
box(9.8,8.25,2.85,.95,'Durable memory layer inputs\n'+r'$\{h_j^{(\ell-1)},p_j,s_j\}$',gray)
box(9.8,6.65,2.85,1.05,'Reuse local projections\n'+r'$K_j=\mathrm{RoPE}_{p_j}(W_K^{\ell}\bar h_j)$'+'\n'+r'$V_j=W_V^{\ell}\bar h_j$')
arrow([(11.23,8.25),(11.23,7.74)],True)
arrow([(9.8,7.15),(9.29,7.15)],True)
box(9.8,4.8,2.85,1.05,'Global memory bank\n'+r'$K_g=\mathrm{cat}_j(K_j+b_{\mathrm{id}}E_{\mathrm{win}}[s_j])$'+'\n'+r'$V_g=\mathrm{cat}_j(V_j)$',green,9)
arrow([(11.23,6.65),(11.23,5.89)],True)
arrow([(11.23,4.8),(11.23,3.85),(9.29,3.85)],True)
label(11.23,2.35,'Own + other visible windows\nLocal RoPE is retained on K.\nIdentity is added only to K.',9)
label(8.7,1.15,r'$\mathrm{Attn}(Q,K,V;M)=\mathrm{softmax}(QK^T/\sqrt{d_h}+M)V$',11)
# Tick lifecycle.
steps=[(8.3,'A  Open windows\nInsert BOS and headers'),(6.85,'B  Ordered observations\nPredict supervised B before insertion'),(5.15,'C  Shared after-B snapshot\nCompute all D logits and memory states'),(3.65,'D  Atomic commit\nDurable targets enter memory; NOOP does not'),(2.2,'Evict / advance\nRelease closed / replaced cache entries')]
for i,(y,text) in enumerate(steps):
    box(13.55,y,3.95,.95,text,gray if i in [0,4] else blue,10)
    if i:arrow([(15.52,steps[i-1][0]),(15.52,y+.99)])
label(15.52,1.3,'No current-tick D target is a memory key.\nEvery layer enforces causal visibility.',10)
label(9,.42,'Solid arrows: state flow.  Dashed arrows: memory reads.  Exchange weights are shared across layers; local-block weights are layer-specific.',10)
root=Path(__file__).resolve().parents[1]/'docs/figures'
root.mkdir(exist_ok=True)
for fmt in ['svg','pdf','png']:
    fig.savefig(root/f'kidra-architecture.{fmt}',dpi=200,facecolor='white',metadata={'Creator':'Tesslot architecture figure'})
plt.close(fig)
