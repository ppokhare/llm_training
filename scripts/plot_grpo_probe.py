"""Chart a saved post-hoc fixed-question checkpoint probe, without GPUs."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--probe',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();data=json.loads(Path(args.probe).read_text());out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    points=data['checkpoints'];before,after=points[0],points[-1];steps=[r['step'] for r in points]
    blue,gray,orange='#2563eb','#64748b','#c2410c'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'figure.dpi':160})
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    axes[0,0].plot(steps,[r['greedy_accuracy'] for r in points],'o-',color=blue)
    axes[0,0].set(title='Greedy correctness on fixed questions',xlabel='Saved checkpoint step',ylim=(0,1))
    axes[0,0].yaxis.set_major_formatter(PercentFormatter(1))
    for r in points:axes[0,0].annotate(f"{round(r['greedy_accuracy']*64)}/64",(r['step'],r['greedy_accuracy']),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
    axes[0,1].plot(steps,[r['greedy_mean_tokens'] for r in points],'o-',color=blue)
    axes[0,1].set(title='Greedy length: modest subset increase',xlabel='Saved checkpoint step',ylabel='Mean tokens',ylim=(0,140))
    for ax,key,title in [(axes[0,2],'sample_mean_tokens','Sampled length: no growth'),(axes[1,2],'mean_distinct_final_answers','Distinct valid final answers per question')]:
        y=[before[key],after[key]]
        bars=ax.bar(['SFT','RLVR 300'],y,color=[gray,blue],width=.55)
        ax.set(title=title,ylim=(0,max(y)*1.25))
        for bar,v in zip(bars,y):ax.annotate(f'{v:.2f}',(bar.get_x()+bar.get_width()/2,v),xytext=(0,5),textcoords='offset points',ha='center')
    x=np.arange(2);width=.34
    for i,(r,label,color) in enumerate([(before,'SFT',gray),(after,'RLVR 300',blue)]):
        values=[r['sampled_accuracy'],r['pass_at_8']]
        bars=axes[1,0].bar(x+(i-.5)*width,values,width,label=label,color=color)
        for bar,v in zip(bars,values):axes[1,0].annotate(f'{v:.1%}',(bar.get_x()+bar.get_width()/2,v),xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)
    axes[1,0].set(title='Sampling improves answer reliability',xticks=x,xticklabels=['Sample accuracy','Observed pass@8'],ylim=(0,1.12))
    axes[1,0].yaxis.set_major_formatter(PercentFormatter(1));axes[1,0].legend(frameon=False,fontsize=8)
    bottom=np.zeros(2)
    for key,label,color in [('all_wrong_fraction','All 8 wrong',orange),('mixed_fraction','Mixed rewards',gray),('all_correct_fraction','All 8 correct',blue)]:
        values=np.array([before[key],after[key]])
        bars=axes[1,1].bar(['SFT','RLVR 300'],values,bottom=bottom,color=color,width=.55,label=label)
        for j,(bar,v) in enumerate(zip(bars,values)):
            if v>.08:axes[1,1].text(bar.get_x()+bar.get_width()/2,bottom[j]+v/2,f'{v:.1%}',ha='center',va='center',color='white',fontsize=8)
        bottom+=values
    axes[1,1].set(title='Zero-variance groups: more all-correct',ylim=(0,1));axes[1,1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1,1].legend(frameon=False,fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.08),ncol=1)
    fig.suptitle('Post-hoc fixed validation probe · same 64 questions · 8 samples/question at endpoints',fontsize=13)
    fig.supxlabel('Temperature 1 · top-p 1 · 512-token cap · descriptive small-sample diagnostics, not full validation scores',fontsize=9)
    for ext in ['png','svg']:fig.savefig(out/f'checkpoint_probe.{ext}')
    plt.close(fig)
    compact={k:v for k,v in data.items() if k!='checkpoints'}
    compact['checkpoints']=[{k:v for k,v in r.items() if k not in ['greedy','samples']} for r in points]
    # Bootstrap whole prompt groups, not correlated completions as independent examples.
    a=np.array([sum(s['correct'] for s in before['samples'][i:i+8])/8 for i in range(0,512,8)])
    b=np.array([sum(s['correct'] for s in after['samples'][i:i+8])/8 for i in range(0,512,8)])
    indices=np.random.default_rng(42).integers(0,64,size=(20000,64))
    compact['sampled_accuracy_delta_paired_prompt_bootstrap_95']=np.quantile((b-a)[indices].mean(1),[.025,.975]).tolist()
    compact['sampled_accuracy_delta']=float((b-a).mean())
    (out/'fixed_probe.json').write_text(json.dumps(compact,indent=2)+'\n')
    print(f'Probe plots saved in {out}')


if __name__=='__main__':main()
