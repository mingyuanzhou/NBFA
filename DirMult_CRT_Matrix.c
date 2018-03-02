/*==========================================================
 *
 *
 * The calling syntax is:
 *
 *		[ZSDS,WSZS,ZS] = DirMult_CRT_Matrix(Xtrain,Phi,Theta,ZS);
 *      Xtrain is sparse
 *
 *
 * This is a MEX-file for MATLAB.
 * Copyright 2015 Mingyuan Zhou
 *
 *========================================================*/

#include "mex.h"
/* //#include <math.h>
 * //#include <stdlib.h>
 * //#include <stdio.h>
 * //#include "matrix.h"
 * #include "cokus.c"
 * #define RAND_MAX_32 4294967295.0 */

/* //  The computational routine */


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

void Multrnd_Matrix(double *ZSDS, double *WSZS, double *Phi, double *Theta, mwIndex *ir, mwIndex *jc, double *pr, mwSize Vsize, mwSize Nsize, mwSize Ksize,  double *prob_cumsum, double *prob, double *ZS, double *n_vk)
{
    
    double cum_sum, probrnd;
    mwIndex k, j, v, token, ji=0, total=0, table;
    /*//, ksave;*/
    mwIndex starting_row_index, stopping_row_index, current_row_index;
    
    
    for (j=0;j<Nsize;j++) {
        starting_row_index = jc[j];
        stopping_row_index = jc[j+1];
        if (starting_row_index == stopping_row_index)
            continue;
        else {
            for (current_row_index =  starting_row_index; current_row_index<stopping_row_index; current_row_index++) {
                v = ir[current_row_index];
                for (token=0;token< (mwIndex) pr[total];token++) {
                    k= (mwIndex) ZS[ji+token]-1;
                    n_vk[k]++;
                }
                for (k=0; k<Ksize; k++)
                    prob[k] = Phi[v+ k*Vsize]*Theta[k + Ksize*j];
                
                for (cum_sum=0,token=0;token< (mwIndex) pr[total];token++) {
                    k= (mwIndex) ZS[ji]-1;
                    n_vk[k]--;
                    
                    for (cum_sum=0,k=0; k<Ksize; k++) {
                        cum_sum += n_vk[k] + prob[k];
                        prob_cumsum[k] = cum_sum;
                    }
                    probrnd = (double) rand() / RAND_MAX *cum_sum;
                    k = BinarySearch(probrnd, prob_cumsum, Ksize);
                    n_vk[k]++;
                    ZS[ji]=k+1;
                    ji++;
                }
                for (k=0;k<Ksize;k++)
                {
                    for (token=0, table=0;token< (mwIndex) n_vk[k];token++) {
                        if  (((double) rand() / RAND_MAX) <= (prob[k]/(prob[k]+ token))){
                            ZSDS[k+Ksize*j]++;
                            WSZS[v+k*Vsize]++;
                        }
                    }
                    n_vk[k]=0;
                }
                total++;
            }
        }
    }
    /*// mexPrintf("total=%d, Ji = %d",total,ji);*/
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{
    double *ZSDS, *WSZS, *Phi, *Theta, *RND, *ZS;
    double  *pr, *prob_cumsum, *prob,*n_vk;
    mwIndex *ir, *jc;
    mwIndex Vsize, Nsize, Ksize;
    mxArray *prob_cumsum_Array, *prob_Array, *n_vk_Array;
    
    /*//         mxArray *lhs, *rhs;
//         double *rhsPtr, *lhsPtr;
 
//  mwIndex      row, col, total=0, number_of_columns;
//  mwIndex      starting_row_index, stopping_row_index,
// current_row_index;
// */
    pr = mxGetPr(prhs[0]);
    ir = mxGetIr(prhs[0]);
    jc = mxGetJc(prhs[0]);
    Vsize = mxGetM(prhs[0]);
    Nsize = mxGetN(prhs[0]);
    Ksize = mxGetN(prhs[1]);
    Phi = mxGetPr(prhs[1]);
    Theta = mxGetPr(prhs[2]);
    
    /*// RND = mxGetPr(prhs[3]);*/
    
    plhs[0] = mxCreateDoubleMatrix(Ksize,Nsize,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(Vsize,Ksize,mxREAL);
    plhs[2] = mxDuplicateArray(prhs[3]);
    
    
    ZSDS = mxGetPr(plhs[0]);
    WSZS = mxGetPr(plhs[1]);
    ZS = mxGetPr(plhs[2]);
    
    prob_cumsum_Array = mxCreateDoubleMatrix(Ksize,1,mxREAL);
    prob_cumsum = mxGetPr(prob_cumsum_Array);
    
    prob_Array = mxCreateDoubleMatrix(Ksize,1,mxREAL);
    prob = mxGetPr(prob_Array);
    
    n_vk_Array = mxCreateDoubleMatrix(Ksize,1,mxREAL);
    n_vk = mxGetPr(n_vk_Array);
    
   /* prob_cumsum = (double *) mxCalloc(Ksize,sizeof(double));
    prob = (double *) mxCalloc(Ksize,sizeof(double));
    n_vk = (double *) mxCalloc(Ksize,sizeof(double));*/
    Multrnd_Matrix(ZSDS, WSZS, Phi, Theta, ir, jc, pr,  Vsize, Nsize, Ksize,  prob_cumsum,prob,ZS,n_vk);
    mxDestroyArray(prob_cumsum_Array);
    mxDestroyArray(prob_Array);
    mxDestroyArray(n_vk_Array);
    /*mxFree(prob_cumsum);
    mxFree(prob);
    mxFree(n_vk);*/
    /*//, &lhs, &rhs); */
}