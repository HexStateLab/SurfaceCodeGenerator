# Optimized BP decoder for [[3200,1756,20]] 40x40 Torus
import random, math, time

r=s=40; n=r*s
g=[(0,0),(2,0),(0,2),(2,2)]
sm_X=[[] for _ in range(n)]
ch=[[] for _ in range(n)]
for q in range(n):
    qi=q//s; qj=q%s
    for ri in range(r):
        for rj in range(s):
            if ((qi-ri)%r,(qj-rj)%s) in g:
                c=ri*s+rj; sm_X[q].append(c); ch[c].append(q)

def syndrome(err):
    syn=[0]*n
    for q in range(n):
        if err[q]:
            for c in sm_X[q]: syn[c]^=1
    return syn

deg=4; prior=math.log((1-0.1)/0.1)  # p_err=0.1 (mild prior for BP on sparse errors)
# Precompute: for variable q, check c at local position i: what is c's local position for q?
# q2c_qpos[q][i] = position of qubit q in check sm_X[q][i]'s qubit list
q2c_cpos=[[0]*deg for _ in range(n)]
for q in range(n):
    for i,c in enumerate(sm_X[q]):
        q2c_cpos[q][i]=ch[c].index(q)

def decode_bp(syn, max_iter=20):
    q2c=[[prior]*deg for _ in range(n)]
    c2q=[[0.0]*deg for _ in range(n)]
    for it in range(max_iter):
        # Check nodes
        for c in range(n):
            d=len(ch[c])
            if d==0: continue
            av=[]; sg=[]
            for qi,q in enumerate(ch[c]):
                pi=q2c_cpos[q][sm_X[q].index(c)]
                msg=q2c[q][pi]
                av.append(abs(msg)); sg.append(1.0 if msg>=0 else -1.0)
            ts=1.0
            for s in sg: ts*=s
            if syn[c]: ts*=-1.0
            sv=sorted(av); m1=sv[0]; m2=sv[1] if d>1 else 1e9
            for qi,q in enumerate(ch[c]):
                mv=m2 if av[qi]<=m1+1e-6 else m1
                c2q[c][qi]=ts*sg[qi]*mv
        # Variable nodes
        for q in range(n):
            total=prior
            for i,c in enumerate(sm_X[q]): total+=c2q[c][ch[c].index(q)]
            for i,c in enumerate(sm_X[q]): q2c[q][i]=total-c2q[c][ch[c].index(q)]
    dec=[0]*n
    for q in range(n):
        total=prior
        for i,c in enumerate(sm_X[q]): total+=c2q[c][ch[c].index(q)]
        if total<0: dec[q]=1
    return dec

random.seed(42)
print('BP Min-Sum Decoder — [[3200,1756,20]] 40x40 Torus (Theorem D=20)')
print()
for w in [1,2,3,5,7,10,12,15,17,18,19,20]:
    ok=0; trials=200
    t0=time.time()
    for _ in range(trials):
        qs=random.sample(range(n),w)
        err=[0]*n
        for q in qs: err[q]=1
        d=decode_bp(syndrome(err))
        if d==err: ok+=1
    dt=time.time()-t0
    bar='#'*int(ok/trials*30)+'-'*(30-int(ok/trials*30))
    print(f'  weight-{w:2d}: {ok:3d}/{trials} ({ok/trials*100:5.1f}%) {bar} {dt:.1f}s')
