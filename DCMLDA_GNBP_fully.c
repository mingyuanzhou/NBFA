/*==========================================================
 *
 *
 * The calling syntax is:
 *
 *		[WSZS,ell_dot_k,ZS,TS] = DCMLDA_GNBP_fully(XtrainSparse,WSZS,ell_dot_k,ZS,TS,cjpj(1),gamma0,eta);
 *      Xtrain is sparse
 *
 *
 * This is a MEX-file for MATLAB.
 * Copyright 2015 Mingyuan Zhou
 *
 *========================================================*/

#include "mex.h"
#include "string.h"

/* //#include <math.h>
 * //#include <stdlib.h>
 * //#include <stdio.h>
 * //#include "matrix.h"
 * #include "cokus.c"
 * #define RAND_MAX_32 4294967295.0 */


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
    double *WSZS,  *ZS, *TS, *ell_dot_k, cjpj, gamma0, *output;
    double  *pr, *prob_cumsum, *prob;
    mwIndex *ir, *jc;
    mwIndex Vsize, Nsize, Ksize;
    
    double cum_sum, probrnd, cum_sum1, temp1,temp2;
    mwIndex k, j, v, token, ji=0, total=0, table, t, Tsize;
    mwIndex starting_row_index, stopping_row_index, current_row_index, jji;
    double * n_vtk, *n_vk, *n_vt, eta, *prob_cumsum_t, n_v;
    void *newptr;
    
    pr = mxGetPr(prhs[0]);
    ir = mxGetIr(prhs[0]);
    jc = mxGetJc(prhs[0]);
    Vsize = mxGetM(prhs[0]);
    Nsize = mxGetN(prhs[0]);
    Ksize = mxGetN(prhs[1]);
    
    plhs[0] = mxDuplicateArray(prhs[1]);
    plhs[1] = mxDuplicateArray(prhs[2]);
    plhs[2] = mxDuplicateArray(prhs[3]);
    plhs[3] = mxDuplicateArray(prhs[4]);

    WSZS = mxGetPr(plhs[0]);
    ell_dot_k = mxGetPr(plhs[1]);
    ZS = mxGetPr(plhs[2]);
    TS = mxGetPr(plhs[3]);

    
    cjpj = mxGetScalar(prhs[5]);
    gamma0 = mxGetScalar(prhs[6]);
    eta = mxGetScalar(prhs[7]);
    
    prob_cumsum = (double *) mxCalloc(Ksize,sizeof(double));
    prob = (double *) mxCalloc(Ksize,sizeof(double));
    
    for (j= 0;j<Nsize;j++) {
        starting_row_index = jc[j];
        stopping_row_index = jc[j+1];
        if (starting_row_index == stopping_row_index)
            continue;
        else {
            for (current_row_index =  starting_row_index; current_row_index<stopping_row_index; current_row_index++) {
                v = ir[current_row_index];
                Tsize = (mwIndex) pr[total];
                n_vtk = (double *) mxCalloc(Tsize*Ksize,sizeof(double));
                n_vk = (double *) mxCalloc(Ksize,sizeof(double));
                n_vt = (double *) mxCalloc(Tsize,sizeof(double));
                prob_cumsum_t = (double *) mxCalloc(Tsize,sizeof(double));
                
                for (token=0, n_v=0;token< Tsize;token++) {
                    k= (mwIndex) ZS[ji+token]-1;
                    t= (mwIndex) TS[ji+token]-1;
                    if ((ZS[ji+token]>0) && (TS[ji+token]>0)){
                        n_vtk[t+k*Tsize]++;
                        n_vk[k]++;
                        n_vt[t]++;
                        /*n_v++;*/
                    }
                }
                token=0;
                for (cum_sum=0;token< Tsize; token++) {
                    k= (mwIndex) ZS[ji+token]-1;
                    t= (mwIndex) TS[ji+token]-1;
                    if ((ZS[ji+token]>0) && (TS[ji+token]>0)){
                        n_vk[k]--;
                        n_vt[t]--;
                        n_vtk[t+k*Tsize]--;
                        /*n_v--;*/
                        if(n_vtk[t+k*Tsize]==0){
                            WSZS[v+k*Vsize]--;
                            ell_dot_k[k]--;
                        }
                    }
                    
                    for (cum_sum =0, k=0; k<Ksize; k++){
                        prob[k] =  (WSZS[v+k*Vsize]+eta)/(ell_dot_k[k]+Vsize*eta)*ell_dot_k[k]/cjpj;
                        cum_sum +=n_vk[k] + prob[k];
                        prob_cumsum[k] = cum_sum;
                    }
                    if ( ((double) rand() / RAND_MAX * (cum_sum +1.0/Vsize * gamma0/cjpj)) < cum_sum){
                        /*K will not increase*/
                        probrnd = (double) rand() / RAND_MAX *cum_sum;
                        k = BinarySearch(probrnd, prob_cumsum, Ksize);
                        if ((double) rand() / RAND_MAX * (n_vk[k] + prob[k]) < n_vk[k]){
                            for (cum_sum =0, t=0; t<Tsize; t++){
                                cum_sum +=n_vtk[t+k*Tsize];
                                prob_cumsum_t[t] = cum_sum;
                            }

                            probrnd = (double) rand() / RAND_MAX * cum_sum;
                            for (t=Tsize-1;t>0;t--){
                                if (n_vtk[t+k*Tsize]>0){
                                    break;
                                }
                            }
                            t = BinarySearch(probrnd, prob_cumsum_t, t+1);
                            /*t = BinarySearch(probrnd, prob_cumsum_t, Tsize);
                             */
                        }
                        else
                        {
                            for (t=0; t<Tsize; t++){
                                if ((mwIndex)n_vt[t]==0)
                                    break;
                            }
                            if (t==Tsize)
                                mexErrMsgTxt("t Tsize wrong");
                        }
                    }
                    else{
                        /*K = K + 1*/
                        t=0;
                        for (k=0; k<Ksize; k++){
                            if ((mwIndex)ell_dot_k[k]==0)
                                break;
                        }
                        if (k==Ksize){
                            Ksize++;
                            
                            newptr = mxRealloc(WSZS,sizeof(*WSZS)*Vsize*Ksize);
                            /*memset (newptr + Vsize*(Ksize-1), 0, Vsize*sizeof(*WSZS)) ;
                             */
                            mxSetPr(plhs[0], newptr);
                            mxSetM(plhs[0], Vsize);
                            mxSetN(plhs[0], Ksize);
                            WSZS = mxGetPr(plhs[0]);
                            memset (WSZS + Vsize*(Ksize-1), 0, Vsize*sizeof(*WSZS)) ;
                            
                            
/*                             for (jji=Vsize*(Ksize-1);jji<Vsize*Ksize;jji++)
//                                 WSZS[jji]=0;*/


                            newptr = mxRealloc(ell_dot_k,sizeof(*ell_dot_k)*Ksize);
                            mxSetPr(plhs[1], newptr);
                            mxSetM(plhs[1], Ksize);
                            mxSetN(plhs[1], 1);
                            ell_dot_k = mxGetPr(plhs[1]);
                            ell_dot_k[Ksize-1]=0;
                            
                            n_vk =  mxRealloc(n_vk,sizeof(*n_vk)*Ksize);
                            n_vk[Ksize-1]=0;
                           
                            n_vtk = mxRealloc(n_vtk,sizeof(*n_vtk)*Tsize*Ksize);
                            memset (n_vtk + Tsize*(Ksize-1), 0, Tsize*sizeof(*n_vtk)) ;
/*                             for (jji=Tsize*(Ksize-1);jji<Tsize*Ksize;jji++)
//                                 n_vtk[jji]=0;
                                //mexPrintf(" %f,",n_vtk[jji]);*/

                            prob_cumsum =  mxRealloc(prob_cumsum,sizeof(*prob_cumsum)*Ksize);
                            prob =  mxRealloc(prob,sizeof(*prob)*Ksize);
                        }
                        
                    }
                     
                    n_vk[k]++;
                    n_vt[t]++;
                    n_vtk[t+k*Tsize]++;
                    /*n_v--;*/
                    ZS[ji+token]=k+1;
                    TS[ji+token]=t+1;
                    
                    if(n_vtk[t+k*Tsize]==1){
                        WSZS[v+k*Vsize]++;
                        ell_dot_k[k]++;
                    }
                    
                }
                
                ji += Tsize;
                
                total++;
                
                mxFree(n_vtk);
                mxFree(n_vk);
                mxFree(n_vt);
                mxFree(prob_cumsum_t);
            }
        }
    }
   
    mxFree(prob_cumsum);
    mxFree(prob);
    
}