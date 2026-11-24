"""Plot GRPO learning and length diagnostics from saved artifacts, without GPUs."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

BLUE, ORANGE, GRAY = '#2563eb', '#c2410c', '#64748b'


def smooth(values, window=20):
    return np.convolve(values, np.ones(window)/window, mode='valid')


def wilson(k,n):
    z=1.959963984540054;p=k/n;d=1+z*z/n
    c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [c-h,c+h]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--history',required=True)
    parser.add_argument('--audit',required=True)
    parser.add_argument('--sft',required=True)
    parser.add_argument('--grpo',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    state=json.loads(Path(args.history).read_text())
    audit=json.loads(Path(args.audit).read_text())
    history=[r for r in state['log_history'] if 'reward' in r]
    sft,rl=[json.loads(Path(p).read_text()) for p in [args.sft,args.grpo]]
    for k in ['dataset_sha256','n','do_sample','max_new_tokens','batch_size']:
        if sft[k]!=rl[k]:raise ValueError(f'Evaluation mismatch: {k}')
    if sft['environment']['dtype'] != rl['environment']['dtype']:
        raise ValueError('Evaluation dtype differs')
    for a,b in zip(sft['samples'],rl['samples'],strict=True):
        if any(a[k]!=b[k] for k in ['id','question','gold']):raise ValueError('Unpaired evaluation')
    if len(history)!=state['global_step']:raise ValueError('Missing per-step logs')
    steps=np.array([r['step'] for r in history])
    def vals(key):return np.array([r[key] for r in history])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,
        'grid.alpha':.18,'figure.dpi':160})
    def save(fig,name):
        for ext in ['png','svg']:fig.savefig(out/f'{name}.{ext}')
        plt.close(fig)
    def curve(ax,key,title,ylabel,percent=False,transform=None):
        y=vals(key);y=transform(y) if transform else y
        ax.plot(steps,y,color=BLUE,alpha=.20,lw=.7,label='Each step: 32 completions')
        ax.plot(steps[19:],smooth(y),color=BLUE,lw=2,label='Trailing 20-step mean')
        ax.set(title=title,xlabel='Optimizer step',ylabel=ylabel)
        if percent:ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    curve(axes[0,0],'loss','GRPO policy loss (not SFT cross-entropy)','Policy loss')
    axes[0,0].axhline(0,color=GRAY,lw=.7)
    curve(axes[0,1],'reward','Training sampled correctness — changing questions','Binary reward / sampled accuracy',True)
    axes[0,1].set_ylim(0,1);axes[0,1].legend(frameon=False,fontsize=8)
    axes[1,0].plot(steps,vals('learning_rate'),color=BLUE)
    axes[1,0].set(title='Learning-rate schedule',xlabel='Optimizer step',ylabel='Learning rate')
    axes[1,0].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    curve(axes[1,1],'grad_norm','Adapter gradient norms','Gradient norm')
    fig.suptitle('Qwen3-0.6B · BF16 LoRA GRPO · 300 updates · 8 answers/question',fontsize=14)
    fig.supxlabel('Validation was evaluated only at the SFT and final RLVR endpoints; no intermediate validation curve.',fontsize=9)
    save(fig,'grpo_training')

    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    curve(axes[0,0],'completions/mean_length','Sampled response length: decreases, then stays flat','Mean completion tokens')
    axes[0,0].plot(steps[19:],smooth(vals('completions/mean_terminated_length')),color=ORANGE,ls='--',lw=1.3,label='EOS-terminated only, 20-step mean')
    axes[0,0].legend(frameon=False,fontsize=8)
    curve(axes[0,1],'entropy','Token entropy: policy becomes more concentrated','Mean next-token entropy (nats)')
    curve(axes[1,0],'frac_reward_zero_std','Informative prompt groups (mixed correct/wrong)','Fraction of groups',True,lambda y:1-y)
    axes[1,0].set_ylim(0,1.05)
    for key,label,color in [('verifier/invalid_fraction','Unparseable final answer',ORANGE),('completions/clipped_ratio','Hit 512-token cap',GRAY)]:
        axes[1,1].plot(steps,vals(key),color=color,alpha=.18,lw=.7)
        axes[1,1].plot(steps[19:],smooth(vals(key)),color=color,lw=2,label=label+' · 20-step mean')
    axes[1,1].set(title='Format and truncation remain rare',xlabel='Optimizer step',ylabel='Fraction of completions')
    axes[1,1].yaxis.set_major_formatter(PercentFormatter(1));axes[1,1].legend(frameon=False,fontsize=8)
    fig.suptitle('GRPO diagnostics · logged every update · four prompt groups per update',fontsize=14)
    fig.supxlabel('Training questions change each step: smoothed trends are observational, not fixed-prompt policy comparisons.',fontsize=9)
    save(fig,'grpo_diagnostics')

    fig,axes=plt.subplots(1,4,figsize=(13,4.1),layout='constrained')
    comparisons=[('accuracy_greedy','Numerical accuracy ↑'),('invalid_format_rate','Unparseable answer ↓'),('token_limit_rate','Token-limit hits ↓'),('mean_output_tokens','Mean output length')]
    for ax,(key,title) in zip(axes,comparisons):
        y=[sft[key],rl[key]];bars=ax.bar(['SFT','RLVR 300'],y,color=[GRAY,BLUE],width=.55)
        ax.set(title=title);ax.set_axisbelow(True)
        if key=='accuracy_greedy':ax.set_ylim(0,.8)
        elif key=='mean_output_tokens':ax.set_ylim(0,130)
        else:ax.set_ylim(0,max(y)*1.65+.001)
        if key!='mean_output_tokens':ax.yaxis.set_major_formatter(PercentFormatter(1))
        for bar,v in zip(bars,y):
            label=f'{v:.2%}\n{round(v*sft["n"])}/{sft["n"]}' if key!='mean_output_tokens' else f'{v:.2f}'
            ax.annotate(label,(bar.get_x()+bar.get_width()/2,v),xytext=(0,5),textcoords='offset points',ha='center',va='bottom',fontsize=9)
    fig.suptitle('Matched H100 validation · 748 questions · greedy · 512-token cap · BF16',fontsize=13)
    save(fig,'sft_vs_grpo')

    x=np.array([r['generated_tokens'] for r in sft['samples']]);y=np.array([r['generated_tokens'] for r in rl['samples']])
    old=np.array([r['correct'] for r in sft['samples']],dtype=bool);new=np.array([r['correct'] for r in rl['samples']],dtype=bool)
    wins=(~old)&new;losses=old&(~new);same=~(wins|losses)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    bins=np.arange(0,541,20)
    for data,label,color in [(x,'SFT',GRAY),(y,'RLVR 300',BLUE)]:
        axes[0].hist(data,bins=bins,histtype='step',lw=2,label=f'{label}: mean {data.mean():.2f}',color=color)
    axes[0].set(title='Same validation questions: length distributions',xlabel='Generated tokens',ylabel='Questions');axes[0].legend(frameon=False)
    for mask,label,color in [(same,'Same correctness',GRAY),(wins,'Wrong → correct',BLUE),(losses,'Correct → wrong',ORANGE)]:
        axes[1].scatter(x[mask],y[mask],s=13,alpha=.38,label=f'{label} ({mask.sum()})',color=color,edgecolors='none')
    axes[1].plot([0,512],[0,512],ls='--',color=GRAY,lw=1)
    axes[1].set(title='Paired lengths: above diagonal means longer',xlabel='SFT tokens',ylabel='RLVR tokens',xlim=(0,525),ylim=(0,525));axes[1].legend(frameon=False,fontsize=8)
    save(fig,'response_lengths')

    def endpoint(d):
        rows=d['samples'];n=len(rows);k=sum(r['correct'] for r in rows);tokens=[r['generated_tokens'] for r in rows]
        return dict(correct=k,n=n,accuracy=k/n,accuracy_wilson_95=wilson(k,n),unparseable=sum(r['prediction'] is None for r in rows),
            token_limit_hits=sum(r['hit_token_limit'] for r in rows),multiple_final_tags=sum(r['final_tag_count']>1 for r in rows),
            mean_output_tokens=float(np.mean(tokens)),median_output_tokens=float(np.median(tokens)),p95_output_tokens=float(np.quantile(tokens,.95)))
    n=len(x);w=int(wins.sum());l=int(losses.sum());discordant=w+l
    boot=np.random.default_rng(42).multinomial(n,[w/n,l/n,1-discordant/n],size=20000)
    pair=dict(improved=w,regressed=l,delta_accuracy=(w-l)/n,paired_bootstrap_95=np.quantile((boot[:,0]-boot[:,1])/n,[.025,.975]).tolist(),
        bootstrap_resamples=20000,bootstrap_seed=42,mcnemar_exact_two_sided_p=min(1,2*sum(math.comb(discordant,i) for i in range(min(w,l)+1))/2**discordant),
        improved_from_unparseable=sum(not a['correct'] and b['correct'] and a['prediction'] is None for a,b in zip(sft['samples'],rl['samples'])),
        regressed_to_unparseable=sum(a['correct'] and not b['correct'] and b['prediction'] is None for a,b in zip(sft['samples'],rl['samples'])),
        mean_token_delta=float((y-x).mean()),median_token_delta=float(np.median(y-x)),longer=int((y>x).sum()),shorter=int((y<x).sum()),same_length=int((y==x).sum()))
    fields=['completions/mean_length','completions/mean_terminated_length','reward','entropy','frac_reward_zero_std','completions/clipped_ratio','verifier/invalid_fraction','grad_norm','learning_rate']
    windows=[]
    for start in range(0,len(history),50):
        chunk=history[start:start+50]
        windows.append(dict(start=chunk[0]['step'],end=chunk[-1]['step'],**{k:float(np.mean([r[k] for r in chunk])) for k in fields}))
    summary=dict(run=Path(args.history).parent.name,training={k:v for k,v in audit.items() if k!='trainable'},
        dataset_sha256=rl['dataset_sha256'],sft=endpoint(sft),grpo=endpoint(rl),paired=pair,training_windows=windows,
        tracking=dict(frequency='every optimizer step',records=len(history),completions_per_step=32,prompt_groups_per_step=4,
            fixed_prompt_evaluations_during_training=False,raw_training_completions_saved=False),
        rollouts=dict(completions=len(history)*32,prompt_groups=len(history)*4,train_questions_available=6725,
            completion_tokens=round(float(vals('completions/mean_length').sum()*32)),
            mean_reward=float(vals('reward').mean()),mixed_group_fraction=float((1-vals('frac_reward_zero_std')).mean()),
            invalid_completions=round(float(vals('verifier/invalid_fraction').sum()*32)),
            truncated_completions=round(float(vals('completions/clipped_ratio').sum()*32))),
        paired_lengths_by_outcome={name:dict(n=int(mask.sum()),sft_mean=float(x[mask].mean()),grpo_mean=float(y[mask].mean()),mean_delta=float((y[mask]-x[mask]).mean()))
            for name,mask in [('both_correct',old&new),('improved',wins),('regressed',losses),('both_wrong',(~old)&(~new))]},
        training_history=history)
    (out/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['sft','grpo','paired','rollouts','paired_lengths_by_outcome']},indent=2))
    print(f'Plots saved in {out}')


if __name__=='__main__':main()
