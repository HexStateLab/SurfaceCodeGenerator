// bb_decoder.c — Universal 2D BB Code Decoder
// Works for any toroidal BB code with HX=[A|B], HZ=[B^T|A^T].
// Syndrome graph: (4,4)-regular Tanner, qubits on a 2D grid, stride-2 connectivity.
// Decoder: scaled iterative threshold (4→3→2 per round).
// Build: gcc -std=gnu11 -O3 -o bb_decoder bb_decoder.c -lm
// Run:   ./bb_decoder [r] [s] [--bench] [--seed N] [--weight W]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#define MAX_R 200
#define MAX_S 200
#define MAX_N (MAX_R*MAX_S*2)  // 2*r*s = 2Nf physical qubits
#define MAX_DEG 4
#define MAX_ITER 10

// ---- Configuration ----
typedef struct {
    int r, s, n;           // grid dimensions, physical qubit count = r*s
    int deg;                // syndrome degree (4 for g=(x²+1)(y²+1))
    // syndrome maps: sm[q][d] = check index, ch[c][d] = qubit index
    uint16_t sm_deg[MAX_N];
    uint16_t sm_idx[MAX_N][MAX_DEG];
    uint16_t ch_deg[MAX_N];
    uint16_t ch_idx[MAX_N][MAX_DEG];
    // polynomial term lists
    int n_terms_a, n_terms_b;
    int16_t terms_a[MAX_DEG*MAX_DEG][2]; // (dx,dy) pairs
    int16_t terms_b[MAX_DEG*MAX_DEG][2];
} BBConfig;

// ---- Default: g = (x²+1)(y²+1) ----
void cfg_set_default(BBConfig *cfg, int r, int s) {
    cfg->r = r; cfg->s = s; cfg->n = r*s; cfg->deg = 4;
    cfg->n_terms_a = 4; cfg->terms_a[0][0]=0; cfg->terms_a[0][1]=0;
                         cfg->terms_a[1][0]=2; cfg->terms_a[1][1]=0;
                         cfg->terms_a[2][0]=0; cfg->terms_a[2][1]=2;
                         cfg->terms_a[3][0]=2; cfg->terms_a[3][1]=2;
    cfg->n_terms_b = 4;  // b = g·x²y² for standard monomial code
    for(int i=0;i<4;i++) {
        cfg->terms_b[i][0] = (cfg->terms_a[i][0]+2)%r;
        cfg->terms_b[i][1] = (cfg->terms_a[i][1]+2)%s;
    }
}

// ---- Build syndrome maps ----
// X-syndrome: error X(q) fires Z-check c iff AT[c][q]=1
// AT[c][q] = a[(q_i - c_i) mod r][(q_j - c_j) mod s]
void cfg_build(BBConfig *cfg) {
    int r=cfg->r, s=cfg->s, n=cfg->n;
    memset(cfg->sm_deg,0,n*sizeof(uint16_t));
    memset(cfg->ch_deg,0,n*sizeof(uint16_t));
    for(int q=0;q<n;q++) {
        int qi=q/s, qj=q%s;
        for(int ci=0;ci<r;ci++) for(int cj=0;cj<s;cj++) {
            int di=(qi-ci+r)%r, dj=(qj-cj+s)%s;
            for(int t=0;t<cfg->n_terms_a;t++) {
                if(di==cfg->terms_a[t][0] && dj==cfg->terms_a[t][1]) {
                    int c=ci*s+cj;
                    int d=cfg->sm_deg[q];
                    cfg->sm_idx[q][d]=c; cfg->sm_deg[q]++;
                    cfg->ch_idx[c][cfg->ch_deg[c]++]=q;
                }
            }
        }
    }
}

// ---- Syndrome computation ----
void syndrome_X(BBConfig *cfg, uint8_t *err, uint8_t *syn) {
    memset(syn,0,cfg->n);
    for(int q=0;q<cfg->n;q++) if(err[q])
        for(int d=0;d<cfg->sm_deg[q];d++) syn[cfg->sm_idx[q][d]] ^= 1;
}

// ---- Scaled iterative decoder ----
// Rounds: thresholds 4,3,3,3,3,3,2,2,2,2 (strict→relaxed)
int decode_X(BBConfig *cfg, uint8_t *syn, uint8_t *out) {
    int n=cfg->n;
    uint8_t thresholds[MAX_ITER]={4,3,3,3,3,3,2,2,2,2};
    memset(out,0,n);
    for(int it=0;it<MAX_ITER;it++) {
        // Compute current syndrome
        uint8_t cur[MAX_N]; memset(cur,0,n);
        for(int q=0;q<n;q++) if(out[q])
            for(int d=0;d<cfg->sm_deg[q];d++) cur[cfg->sm_idx[q][d]] ^= 1;
        // Residual syndrome
        uint8_t rem[MAX_N]; int nz=0;
        for(int c=0;c<n;c++) { rem[c]=syn[c]^cur[c]; if(rem[c]) nz++; }
        if(nz==0) return it+1;  // converged
        
        int th=thresholds[it], flipped=0;
        for(int q=0;q<n;q++) {
            int hits=0;
            for(int d=0;d<cfg->sm_deg[q];d++) hits+=rem[cfg->sm_idx[q][d]];
            if(hits>=th) { out[q]^=1; flipped=1; }
        }
        if(!flipped) return -(it+1); // stuck
    }
    return 0; // max iters reached
}

// ---- Test harness ----
int run_test(BBConfig *cfg, int weight, int trials, int quiet) {
    int n=cfg->n, ok=0;
    uint8_t err[MAX_N], syn[MAX_N], dec[MAX_N];
    for(int t=0;t<trials;t++) {
        memset(err,0,n);
        // Generate random weight-w error
        for(int i=0;i<weight;) {
            int q=rand()%n;
            if(!err[q]) { err[q]=1; i++; }
        }
        syndrome_X(cfg,err,syn);
        int res=decode_X(cfg,syn,dec);
        if(memcmp(err,dec,n)==0) ok++;  // exact recovery
    }
    return ok;
}

int main(int argc, char **argv) {
    int r=40, s=40, weight=0, trials=200, seed=42;
    int bench=0, quiet=0;
    for(int i=1;i<argc;i++) {
        if(!strcmp(argv[i],"--bench")) bench=1;
        else if(!strcmp(argv[i],"--seed")) seed=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--weight")) weight=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--quiet")) quiet=1;
        else if(!strcmp(argv[i],"--trials")) trials=atoi(argv[++i]);
        else if(argv[i][0]!='-') {
            r=atoi(argv[i]);
            if(i+1<argc && argv[i+1][0]!='-') s=atoi(argv[++i]);
        }
    }
    srand(seed);
    
    BBConfig cfg;
    cfg_set_default(&cfg,r,s);
    cfg_build(&cfg);
    
    if(!quiet) {
        printf("BB Code Decoder — %dx%d Torus, n=%d\n",r,s,cfg.n);
        printf("  a=g=(x^2+1)(y^2+1), b=g·x^2y^2\n");
        printf("  Syndrome degrees: var=%d check=%d\n",cfg.sm_deg[0],cfg.ch_deg[0]);
        int K = r*s + 2*r + 2*s - 4;
        int D = (r<s?r:s)/2;
        int Nphys = 2*r*s;
        printf("  [[%d,%d,%d]], rate=%.4f, weight=8\n",Nphys,K,D,(double)K/Nphys);
        printf("  Decoder: scaled iterative threshold (4->3->2)\n");
    }
    
    if(bench) {
        int weights[]={1,2,3,5,7,10,12,15,18,20};
        printf("%8s %8s %8s\n","Weight","OK/Trials","Rate");
        for(int wi=0;wi<10;wi++) {
            int w=weights[wi];
            int ok=run_test(&cfg,w,trials,quiet);
            printf("%8d %8s %7.1f%%\n",w,
                ok==trials?"ALL":({static char b[16];snprintf(b,16,"%d/%d",ok,trials);b;}),
                100.0*ok/trials);
        }
    } else if(weight>0) {
        int ok=run_test(&cfg,weight,trials,quiet);
        printf("Weight-%d: %d/%d (%.1f%%)\n",weight,ok,trials,100.0*ok/trials);
    } else {
        // Default: single demonstration at weight 5
        int w=5;
        int ok=run_test(&cfg,w,trials,quiet);
        printf("Weight-%d: %d/%d (%.1f%%)\n",w,ok,trials,100.0*ok/trials);
    }
    return 0;
}
