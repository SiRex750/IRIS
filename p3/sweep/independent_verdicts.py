"""Independent re-computation of the PREREGISTRATION.md verdicts from results.csv (written by the chat auditor, not by the sweep code). Run from p3/sweep/: python independent_verdicts.py"""
import csv, statistics as st
r=list(csv.DictReader(open('results.csv')))
idx={(x['pass'],x['encode_id'],x['group'],x['sequence'],float(x['tau'])):x for x in r}
seqs=sorted(set(x['sequence'] for x in r))
def v(eid,grp,s,k='stale_of_valid',p='final',tau=1.0):
    x=idx.get((p,eid,grp,s,tau)); 
    if x is None or x[k] in('','nan'): return None
    return float(x[k])
missing=[]
def need(eid,grp):
    out={}
    for s in seqs:
        a=v(eid,grp,s)
        if a is None: missing.append((eid,grp,s))
        out[s]=a
    return out
def rank(a):
    o=sorted(range(len(a)),key=lambda i:a[i]); rk=[0]*len(a); i=0
    while i<len(a):
        j=i
        while j+1<len(a) and a[o[j+1]]==a[o[i]]: j+=1
        for k in range(i,j+1): rk[o[k]]=(i+j)/2+1
        i=j+1
    return rk
def spearman(x,y):
    rx,ry=rank(x),rank(y); mx,my=st.mean(rx),st.mean(ry)
    num=sum((a-mx)*(b-my) for a,b in zip(rx,ry)); den=(sum((a-mx)**2 for a in rx)*sum((b-my)**2 for b in ry))**.5
    return num/den if den else float('nan')
res=[]
def cnt(worse,better,th=0.001):
    d={s:(worse[s]-better[s]) for s in seqs if worse[s] is not None and better[s] is not None}
    return sum(1 for s in d if d[s]>th), len(d), d
# P1
for name,w,b,need_n in [('P1 x264 CRF45-CRF12','x264_crf45','x264_crf12',18),('P1 NVENC QP45-QP18','nvenc_qp45','nvenc_qp18',18),('P1 mpeg4 q31-q2 (not kill)','mpeg4_q31','mpeg4_q2',16)]:
    n,N,d=cnt(need(w,'d1'),need(b,'d1')); res.append((name,f'{n}/{N}',f'>= {need_n}/23','PASS' if n>=need_n else 'FAIL'))
    fails=[s for s in seqs if d.get(s) is not None and d[s]<=0.001]; print(name,'not passing:',[(s,round(d[s],4)) for s in fails])
crfs=[12,18,23,28,33,38,45]; ser={c:need(f'x264_crf{c}','d1') for c in crfs}
rhos={s:spearman(crfs,[ser[c][s] for c in crfs]) for s in seqs}
n=sum(1 for s in seqs if rhos[s]>=0.8); res.append(('P1 Spearman(CRF, stale) >= 0.8',f'{n}/23','>= 16/23','PASS' if n>=16 else 'FAIL'))
print('rho<0.8:',[(s,round(rhos[s],2)) for s in seqs if not rhos[s]>=0.8])
# P2
xb=need('x264_qp24_bf2_pb1','B_naive'); x0=need('x264_qp24','d1'); n,N,dx=cnt(xb,x0); res.append(('P2 x264 QP24 B - bf0',f'{n}/{N}','>= 18/23','PASS' if n>=18 else 'FAIL'))
print('P2 x264 not passing:',[(s,round(dx[s],4)) for s in seqs if dx[s]<=0.001])
nb=need('nvenc_qp28_bf2_bqeq','B_naive'); n0=need('nvenc_qp28','d1'); n,N,dn=cnt(nb,n0); res.append(('P2 NVENC QP28 B=P - bf0',f'{n}/{N}','>= 16/23','PASS' if n>=16 else 'FAIL'))
print('P2 nvenc not passing:',[(s,round(dn[s],4)) for s in seqs if dn[s]<=0.001])
for nm,eid in [('x264','x264_qp24_bf2_pb1'),('NVENC','nvenc_qp28_bf2_bqeq')]:
    a=need(eid,'B_scaled'); b=need(eid,'B_naive'); m=st.median(abs(a[s]-b[s]) for s in seqs)
    res.append((f'P2b {nm} median |scaled-naive|',f'{m:.4f}','< 0.005','PASS' if m<0.005 else 'FAIL'))
# P3
c23=need('x264_crf23','d1'); c33=need('x264_crf33','d1'); ref_m=st.median(abs(c33[s]-c23[s]) for s in seqs); th=0.5*ref_m
print(f'CRF23->33 median |change| {ref_m:.5f}; P3 threshold {th:.5f}')
p3m={}
for nm,eid,grp,base,bgrp in [('ref 2','x264_crf23_ref2','d1(assumed)','x264_crf23','d1'),('ref 3','x264_crf23_ref3','d1(assumed)','x264_crf23','d1'),('keyint 30','x264_crf23_keyint30','d1','x264_crf23','d1'),('umh 32','x264_crf23_umh32','d1','x264_crf23_umh16','d1'),('umh 64','x264_crf23_umh64','d1','x264_crf23_umh16','d1')]:
    a=need(eid,grp); b=need(base,bgrp); m=st.median(abs(a[s]-b[s]) for s in seqs); p3m[nm]=m
    res.append((f'P3 {nm}',f'{m:.5f}',f'< {th:.5f}','PASS' if m<th else 'FAIL'))
mb=st.median(dx[s] for s in seqs)
res.append(('P2c median x264 B effect > every P3 median',f'{mb:.5f} vs max {max(p3m.values()):.5f}','>','PASS' if mb>max(p3m.values()) else 'FAIL'))
print('\nMISSING cells:',missing[:10], len(missing))
print()
for row in res: print(f'{row[0]:45s} {row[1]:28s} {row[2]:14s} {row[3]}')

print('\n--- effect sizes (median over 23, stale/valid, final, tau=1)')
x12=need('x264_crf12','d1'); x45=need('x264_crf45','d1')
print('x264 CRF12 median', round(st.median(x12.values()),4), ' CRF45 median', round(st.median(x45.values()),4))
print('NVENC QP18 median', round(st.median(need('nvenc_qp18','d1').values()),4), ' QP45', round(st.median(need('nvenc_qp45','d1').values()),4))
print('x264 QP24 bf0 median', round(st.median(x0.values()),4), ' B median', round(st.median(xb.values()),4), ' CRF33', round(st.median(c33.values()),4), ' CRF38', round(st.median(need('x264_crf38','d1').values()),4))
print('NVENC QP28 bf0 median', round(st.median(n0.values()),4), ' B=P median', round(st.median(nb.values()),4))
print('\n--- shaman_3 NVENC detail')
for eid,grp in [('nvenc_qp28','d1'),('nvenc_qp28_bf2_bqeq','B_naive'),('nvenc_qp28_bf2','B_naive'),('x264_qp24','d1'),('x264_qp24_bf2_pb1','B_naive')]:
    x=idx[('final',eid,grp,'shaman_3',1.0)]
    print(f"{eid:22s} {grp:8s} stale {float(x['stale_of_valid']):.4f} reuse {x['reuse_cells']} valid {x['valid_cells']} cov {x['mv_coverage_pct'][:5]} zero {x['zero_mv_share'][:5]} epe {x['epe_median'][:5]} nfr {x['n_group_frames']} psnr {x['psnr_y'][:5]} kbps {x['bitrate_kbps'][:6]}")
print('\n--- mpeg4 exceptions: coverage (share of valid area with a vector) and stale')
for s in ['ambush_2','ambush_6','temple_3','mountain_1','alley_1']:
    a=idx[('final','mpeg4_q2','d1',s,1.0)]; b=idx[('final','mpeg4_q31','d1',s,1.0)]
    print(f"{s:10s} q2 stale {float(a['stale_of_valid']):.3f} cov {a['mv_coverage_pct'][:5]} | q31 stale {float(b['stale_of_valid']):.3f} cov {b['mv_coverage_pct'][:5]}")
