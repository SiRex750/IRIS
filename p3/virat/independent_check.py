"""Independent re-computation of the VIRAT verdicts from results.csv (chat auditor), plus an EXPLORATORY,
NOT pre-registered view: stale share of MOVING cells (stale_cells / moving_cells) on the eligible clips."""
import csv, statistics as st
r=list(csv.DictReader(open('results.csv')))
I={(x['clip'],x['arm'],x['group'],float(x['tau'])):x for x in r}
clips=sorted(set(x['clip'] for x in r)); elig=sorted(set(x['clip'] for x in r if x['eligible'] in('True','true','1')))
def g(c,a,grp,k,tau=1.0):
    x=I.get((c,a,grp,tau)); return None if x is None or x[k] in ('','nan') else float(x[k])
def cnt(cs,f): v=[f(c) for c in cs]; return sum(v),len(v)
print('eligible',len(elig),'of',len(clips))
print('PRE-REGISTERED (recomputed):')
print(' V1-flip x264 CRF45>CRF18', cnt(clips, lambda c: g(c,'x264_crf45','d1','flip')>g(c,'x264_crf18','d1','flip')))
print(' V1-flip NVENC QP45>QP18', cnt(clips, lambda c: g(c,'nvenc_qp45','d1','flip')>g(c,'nvenc_qp18','d1','flip')))
print(' V1-stale x264 diff>0.0005', cnt(elig, lambda c: g(c,'x264_crf45','d1','stale')-g(c,'x264_crf12','d1','stale')>0.0005))
print(' V1-stale NVENC diff>0.0005', cnt(elig, lambda c: g(c,'nvenc_qp45','d1','stale')-g(c,'nvenc_qp18','d1','stale')>0.0005))
print(' V2 x264 diff>0.0005', cnt(elig, lambda c: g(c,'x264_qp24_bf2_pb1','B','stale')-g(c,'x264_qp24','d1','stale')>0.0005))
print(' V2 NVENC diff>0.0005', cnt(elig, lambda c: g(c,'nvenc_qp28_bf2_bqeq','B','stale')-g(c,'nvenc_qp28','d1','stale')>0.0005))
def sm(c,a,grp):
    x=I[(c,a,grp,1.0)]; m=float(x['moving_cells']); return float(x['stale_cells'])/m
print('\nEXPLORATORY, NOT PRE-REGISTERED: stale / moving cells, eligible clips, tau=1')
pairs=[('x264 CRF12->45','x264_crf12','d1','x264_crf45','d1'),('NVENC QP18->45','nvenc_qp18','d1','nvenc_qp45','d1'),
       ('x264 QP24 bf0->B','x264_qp24','d1','x264_qp24_bf2_pb1','B'),('NVENC QP28 bf0->B=P','nvenc_qp28','d1','nvenc_qp28_bf2_bqeq','B')]
for nm,a,ga,b,gb in pairs:
    A=[sm(c,a,ga) for c in elig]; B=[sm(c,b,gb) for c in elig]
    print(f' {nm:22s} increases in {sum(y>x for x,y in zip(A,B))}/{len(elig)} clips; median {st.median(A):.3f} -> {st.median(B):.3f}')
