/*==========================================================
 * BNBP_mex_collapsed.c -
 *
 * The calling syntax is:
 *
 *		 [WSZS,DSZS,ell_dot_k,n_dot_k,ZS,TS] = PFA_GNBP_fully(XtrainSparse,WSZS,DSZS,ell_dot_k,n_dot_k,ZS,TS,gamma0,eta,c_q_dot);
 *
 *
 * This is a MEX-file for MATLAB.
 * Copyright 2014 Mingyuan Zhou
 *
 *========================================================*/
/* $Revision: 0.2 $ */

#include "mex.h"
#include "string.h"
#include <math.h>
#include <stdlib.h>
/*//#include "cokus.c"
//#define RAND_MAX_32 4294967295.0*/



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
    mwSize Vsize, Nsize, Ksize;
    mwIndex *ir, *jc;
    double *pr;
    
    double *WSZS,*DSZS,*ell_dot_k, *n_dot_k, *ZS, *TS;
    double gamma0, eta, c_q_dot;
    double *prob_cumsum;
    double cum_sum, probrnd,temp;
    
    mwIndex starting_row_index, stopping_row_index, current_row_index, token;
    mwIndex k, j, v, t, Tsize, ji, ji_begin,ji_end, kk,tt, jji;
    
    double *ZSTSj,*l_j,*prob_cumsum_t, *ZSTSjnew;
    void *newptr;
    
    
  /*  //  [WSZS,DSZS,ell_dot_k,n_dot_k,ZS,TS] = PFA_GNBP_fully(XtrainSparse,WSZS,DSZS,ell_dot_k,n_dot_k,ZS,TS,gamma0,eta,c_q_dot);*/
    
    pr = mxGetPr(prhs[0]);
    ir = mxGetIr(prhs[0]);
    jc = mxGetJc(prhs[0]);
    Nsize = mxGetN(prhs[0]);
    Vsize = mxGetM(prhs[1]);
    Ksize = mxGetN(prhs[1]);
    
    plhs[0] = mxDuplicateArray(prhs[1]);
    plhs[1] = mxDuplicateArray(prhs[2]);
    plhs[2] = mxDuplicateArray(prhs[3]);
    plhs[3] = mxDuplicateArray(prhs[4]);
    plhs[4] = mxDuplicateArray(prhs[5]);
    plhs[5] = mxDuplicateArray(prhs[6]);
    
    WSZS = mxGetPr(plhs[0]);
    DSZS = mxGetPr(plhs[1]);
    ell_dot_k = mxGetPr(plhs[2]);
    n_dot_k = mxGetPr(plhs[3]);
    ZS = mxGetPr(plhs[4]);
    TS = mxGetPr(plhs[5]);
    
    gamma0 = mxGetScalar(prhs[7]);
    eta = mxGetScalar(prhs[8]);
    c_q_dot = mxGetScalar(prhs[9]);
    prob_cumsum = (double *) mxCalloc(Ksize,sizeof(double));
    for (j=0, ji_begin=0;j<Nsize;j++) {
        starting_row_index = jc[j];
        stopping_row_index = jc[j+1];
        if (starting_row_index == stopping_row_index)
            continue;
        else {
            ji = ji_begin;
            Tsize=1;
            for (current_row_index =  starting_row_index; current_row_index<stopping_row_index; current_row_index++) {
                for (token=0;token<(mwIndex) pr[current_row_index]; token++) {
                    if (Tsize < (mwIndex) TS[ji+token])
                        Tsize = (mwIndex) TS[ji+token];
                }
                ji+=token;
            }
            ji_end=ji;
            
            ZSTSj = (double *) mxCalloc(Ksize*Tsize,sizeof(double));
            l_j = (double *) mxCalloc(Ksize,sizeof(double));
            memset (l_j, 1, Ksize*sizeof(*l_j)) ;
/*//             for (k=0;k<Ksize;k++)
//                 l_j[k]=1;*/
            prob_cumsum_t = (double *) mxCalloc(Tsize,sizeof(double));
            
            for (ji=ji_begin;ji<ji_end;ji++){
                k = (mwIndex) ZS[ji] -1;
                t = (mwIndex) TS[ji] -1;
                if (TS[ji]>0){
                    ZSTSj[k+Ksize*t]++;
                    if ((mwIndex)ZSTSj[k+Ksize*t]==1){
                        if ( (mwIndex)l_j[k]<t+1 )
                            l_j[k] = t+1;
                    }
                }
            }
            
            
            ji=ji_begin;
            for (current_row_index =  starting_row_index; current_row_index<stopping_row_index; current_row_index++) {
                for (token=0;token< (mwIndex) pr[current_row_index];token++,ji++) {
                    v = ir[current_row_index];
                    k = (mwIndex) ZS[ji] -1;
                    t = (mwIndex) TS[ji] -1;
                    if((mwIndex) ZS[ji]>0){
                        DSZS[j+Nsize*k]--;
                        WSZS[v+Vsize*k]--;
                        n_dot_k[k]--;
                    }
                    if((mwIndex) TS[ji]>0){
                        ZSTSj[k+Ksize*t]--;
                        if ( (mwIndex) ZSTSj[k+Ksize*t]==0)
                        {
                            if (t==(mwIndex)l_j[k]-1){
                                for (t=(mwIndex)l_j[k]-1;t>0;t--){
                                    if ((mwIndex) ZSTSj[k+Ksize*t] >0)
                                        break;
                                }
                                l_j[k]=t+1;
                            }
                            ell_dot_k[k] --;
                        }
                    }
                    
                    
                    for (cum_sum=0, k=0; k<Ksize; k++) {
                        cum_sum += (eta+ WSZS[v+Vsize*k])/((double)Vsize *eta + n_dot_k[k])*(DSZS[j+Nsize*k] + (ell_dot_k[k])/c_q_dot);
                        prob_cumsum[k] = cum_sum;
                    }
                    if ( ((double) rand() / RAND_MAX * (cum_sum +1.0/((double)Vsize)*gamma0/(c_q_dot))) < cum_sum){
                        /* K will not increase */
                        probrnd = (double)rand()/(double)RAND_MAX*cum_sum;
                        k = BinarySearch(probrnd, prob_cumsum, Ksize);
                        
                        
                        for (cum_sum=0, t=0; t<(mwIndex)l_j[k]; t++)
                        /*    //for (cum_sum=0, t=0; t<Tsize; t++)*/
                        {
                            cum_sum += ZSTSj[k+Ksize*t];
                            prob_cumsum_t[t] = cum_sum;
                        }
                        
                        if ( ((double) rand() / RAND_MAX * (cum_sum +ell_dot_k[k]/c_q_dot)) < cum_sum){
                            probrnd = (double)rand()/(double)RAND_MAX*cum_sum;
                            t = BinarySearch(probrnd, prob_cumsum_t, (mwIndex)l_j[k]);
                           /* //t = BinarySearch(probrnd, prob_cumsum_t, Tsize);*/
                            ZSTSj[k+Ksize*t]++;
                        }
                        else{
                            for (t=1;t<Tsize;t++){
                                if (ZSTSj[k+Ksize*t]==0)
                                    break;
                            }
                            if (t>=(mwIndex)l_j[k])
                                l_j[k] = t+1;
                            
                            if (t==Tsize){
                                Tsize++;
                                ZSTSj = mxRealloc(ZSTSj,sizeof(*ZSTSj)*Ksize*Tsize);
                                memset (ZSTSj + Ksize*(Tsize-1), 0, Ksize*sizeof(*ZSTSj));
                                prob_cumsum_t =  mxRealloc(prob_cumsum_t,Tsize*sizeof(*prob_cumsum_t));
                                prob_cumsum_t[Tsize-1]=0;
                            }
                            l_j[k] = Tsize;
                            ZSTSj[k+Ksize*t]=1;
                            ell_dot_k[k] ++;
                        }
                    }
                    else{
                        /* K = K+1*/
                        t=0;
                        for (kk=0; kk<Ksize; kk++){
                            if ((mwIndex)ell_dot_k[kk]==0){
                                break;
                            }
                        }
                        
                        k=kk;
                        if (k==Ksize){
                            Ksize++;
                            
                            newptr = mxRealloc(WSZS,sizeof(*WSZS)*Vsize*Ksize);
                          /*  //memset (newptr + Vsize*(Ksize-1), 0, Vsize*sizeof(*WSZS)) ;*/
                            mxSetPr(plhs[0], newptr);
                            mxSetM(plhs[0], Vsize);
                            mxSetN(plhs[0], Ksize);
                            WSZS = mxGetPr(plhs[0]);
                            memset (WSZS + Vsize*(Ksize-1), 0, Vsize*sizeof(*WSZS)) ;
                            
                            newptr = mxRealloc(DSZS,sizeof(*DSZS)*Nsize*Ksize);
                         /*   //memset (newptr + Vsize*(Ksize-1), 0, Vsize*sizeof(*WSZS)) ;*/
                            mxSetPr(plhs[1], newptr);
                            mxSetM(plhs[1], Nsize);
                            mxSetN(plhs[1], Ksize);
                            DSZS = mxGetPr(plhs[1]);
                            memset (DSZS + Nsize*(Ksize-1), 0, Nsize*sizeof(*DSZS)) ;
                            
                            newptr = mxRealloc(ell_dot_k,sizeof(*ell_dot_k)*Ksize);
                            mxSetPr(plhs[2], newptr);
                            mxSetM(plhs[2], Ksize);
                            mxSetN(plhs[2], 1);
                            ell_dot_k = mxGetPr(plhs[2]);
                            ell_dot_k[Ksize-1]=0;
                            
                            newptr = mxRealloc(n_dot_k,sizeof(*n_dot_k)*Ksize);
                            mxSetPr(plhs[3], newptr);
                            mxSetM(plhs[3], Ksize);
                            mxSetN(plhs[3], 1);
                            n_dot_k = mxGetPr(plhs[3]);
                            n_dot_k[Ksize-1]=0;
                            
                            l_j =  mxRealloc(l_j,sizeof(*l_j)*Ksize);
                            l_j[Ksize-1]=1;
                            
                            mxFree(ZSTSj);
                            ZSTSj = (double *) mxCalloc(Ksize*Tsize,sizeof(double));
                            
                            prob_cumsum =  mxRealloc(prob_cumsum,sizeof(*prob_cumsum)*Ksize);
                            
                            ell_dot_k[k] ++;
                            l_j[k] =1;
                            TS[ji] = t+1;
                            ZS[ji] = k+1;
                            for (jji=ji_begin;jji<ji_end;jji++){
                                kk = (mwIndex) ZS[jji] -1;
                                tt = (mwIndex) TS[jji] -1;
                                if ((mwIndex) TS[jji]>0){
                                    ZSTSj[kk+Ksize*tt]++;
                                    if ((mwIndex)ZSTSj[kk+Ksize*tt]==1){
                                        if ( (mwIndex)l_j[kk]<tt+1 )
                                            l_j[kk] = tt+1;
                                    }
                                }
                            }
                        }
                        else{
                            ell_dot_k[k] ++;
                            l_j[k] =1;
                            ZSTSj[k+Ksize*t]=1;
                        }
                    }
                    TS[ji] = t+1;
                    ZS[ji] = k+1;
                    DSZS[j+Nsize*k]++;
                    WSZS[v+Vsize*k]++;
                    n_dot_k[k]++;
                }
            }
            ji_begin=ji;
            mxFree(ZSTSj);
            mxFree(l_j);
            mxFree(prob_cumsum_t);
        }
    }
    mxFree(prob_cumsum);
}