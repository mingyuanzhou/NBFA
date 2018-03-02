/*==========================================================
 * BNBP_mex_collapsed.c -
 *
 * The calling syntax is:
 *
 *		[WSZS,DSZS,n_dot_k,ZS] = PFA_BNBP_collapsed(WSZS,DSZS,n_dot_k,ZS,WS,DS,r_i,c,gamma0,eta);
 *
 * This is a MEX-file for MATLAB.
 * Copyright 2014 Mingyuan Zhou
 *
 *========================================================*/
/* $Revision: 0.1 $ */

#include "mex.h"
#include "string.h"
#include <math.h>
#include <stdlib.h>
/*//#include "util.c"*/
/*#include "digamma.c"*/
/*//#include "cokus.c"
 * //#define RAND_MAX_32 4294967295.0*/



mwIndex BinarySearch(double probrnd, double *prob_cumsum, mwSize Ksize) {
    mwIndex k, kstart, kend;
    if (probrnd <=prob_cumsum[0])
        return(0);
    else {
        for (kstart=1, kend=Ksize-1; ; ) {
            if (kstart >= kend) {
                /*//k = kend;*/
                return(kend);
            }
            else {
                k = kstart+ (kend-kstart)/2;
                if (prob_cumsum[k-1]>probrnd && prob_cumsum[k]>probrnd)
                    kend = k-1;
                else if (prob_cumsum[k-1]<probrnd && prob_cumsum[k]<probrnd)
                    kstart = k+1;
                else
                    return(k);
            }
        }
    }
    return(k);
}



/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{
    double *end_index;
    double *DSZS, *WSZS, *n_dot_k;
    double *ZS,*WS, *DS, *r_i, c, eta, gamma0, *prob_cumsum;
    mwSize Vsize, Nsize, Ksize, WordNum;
    
    mwIndex j,v,k,i;
    double cum_sum, probrnd, sumr;
    void *newptr;
    
    plhs[0] = mxDuplicateArray(prhs[0]);
    plhs[1] = mxDuplicateArray(prhs[1]);
    plhs[2] = mxDuplicateArray(prhs[2]);
    plhs[3] = mxDuplicateArray(prhs[3]);
    
    
    WSZS = mxGetPr(plhs[0]);
    DSZS = mxGetPr(plhs[1]);
    n_dot_k = mxGetPr(plhs[2]);
    ZS = mxGetPr(plhs[3]);
    
    WS = mxGetPr(prhs[4]);
    DS = mxGetPr(prhs[5]);
    r_i = mxGetPr(prhs[6]);
    c = mxGetScalar(prhs[7]);
    gamma0 = mxGetScalar(prhs[8]);
    eta = mxGetScalar(prhs[9]);
    
    Vsize = mxGetM(prhs[0]);
    Ksize = mxGetN(prhs[0]);
    Nsize = mxGetM(prhs[1]);
    
    WordNum = mxGetM(prhs[3])*mxGetN(prhs[3]);
    
    prob_cumsum = (double *) mxCalloc(Ksize,sizeof(double));
    
    
   /* [WSZS,DSZS,n_dot_k,ZS] = PFA_BNBP_collapsed(WSZS,DSZS,n_dot_k,ZS,WS,DS,r_i,c,gamma0,eta);*/
    
    for (sumr=0,j=0;j<Nsize;j++){
        sumr += r_i[j];
    }

    for (i=0;i<WordNum;i++){
        v = (mwIndex) WS[i] -1;
        j = (mwIndex) DS[i] -1;
        k = (mwIndex) ZS[i] -1;
        if(ZS[i]>0){
            DSZS[j+Nsize*k]--;
            WSZS[v+Vsize*k]--;
            n_dot_k[k]--;
        }
        for (cum_sum=0, k=0; k<Ksize; k++) {
            cum_sum += (eta+ WSZS[v+Vsize*k])/((double)Vsize *eta + n_dot_k[k])*n_dot_k[k]*(DSZS[j+Nsize*k] + r_i[j])/(c+n_dot_k[k]+sumr);
            prob_cumsum[k] = cum_sum;
        }
        
        if ( ((double) rand() / RAND_MAX * (cum_sum +1.0/((double)Vsize)*gamma0/(c+sumr)*r_i[j])) < cum_sum){
            probrnd = (double)rand()/(double)RAND_MAX*cum_sum;
            k = BinarySearch(probrnd, prob_cumsum, Ksize);
        }
        else{
            for (k=0; k<Ksize; k++){
                if ((mwIndex)n_dot_k[k]==0){
                    break;
                }
            }
            if (k==Ksize){
                
                
                
                Ksize++;
                newptr = mxRealloc(WSZS,sizeof(*WSZS)*Vsize*Ksize);
                /*memset (newptr + Vsize*(Ksize-1), 0, Vsize*sizeof(*WSZS)) ;*/
                mxSetPr(plhs[0], newptr);
                mxSetM(plhs[0], Vsize);
                mxSetN(plhs[0], Ksize);
                WSZS = mxGetPr(plhs[0]);
                memset (WSZS + Vsize*(Ksize-1), 0, Vsize*sizeof(*WSZS)) ;

                newptr = mxRealloc(DSZS,sizeof(*DSZS)*Nsize*Ksize);
                /*memset (newptr + Vsize*(Ksize-1), 0, Vsize*sizeof(*WSZS)) ;*/
                mxSetPr(plhs[1], newptr);
                mxSetM(plhs[1], Nsize);
                mxSetN(plhs[1], Ksize);
                DSZS = mxGetPr(plhs[1]);
                memset (DSZS + Nsize*(Ksize-1), 0, Nsize*sizeof(*DSZS)) ;

                newptr = mxRealloc(n_dot_k,sizeof(*n_dot_k)*Ksize);
                mxSetPr(plhs[2], newptr);
                mxSetM(plhs[2], Ksize);
                mxSetN(plhs[2], 1);
                n_dot_k = mxGetPr(plhs[2]);
                n_dot_k[Ksize-1]=0;
                
                prob_cumsum =  mxRealloc(prob_cumsum,sizeof(*prob_cumsum)*Ksize);
            }
        }
        ZS[i] = k+1;
        DSZS[j+Nsize*k]++;
        WSZS[v+Vsize*k]++;
        n_dot_k[k]++;
    }
    mxFree(prob_cumsum);
}
