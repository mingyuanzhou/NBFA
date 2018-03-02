function [AveThetaTest,AveThetaTest_Prob] =...
    PFA_MultTM_Test(Xtrain,Phi,r_k,r_star,sampler,burnin,collection)
%% Matlab code for the paper:
% M. Zhou, "Negative binomial factor analysis and Dirichelt-multinomial 
% topic modeling," preprint, 2015.

% First Version: June, 2015
% This Version: September, 2015
%
% Coded by Mingyuan Zhou,
% http://mingyuanzhou.github.io/
% Copyright (C) 2015, Mingyuan Zhou.

if strcmp(sampler,'Beta_NB_collapsed')
    p_k=r_k;
    p_star=r_star;
end
XtrainSparse = sparse(Xtrain);

a0=1e-2; b0=1e-2; e0=1e-2; f0=1e-2;

[V,N] = size(Xtrain);
Theta = r_k(:)*ones(1,N);
K=size(Phi,2);
r_i=ones(1,N);
AveThetaTest=0;
AveThetaTest_Prob=0;
fprintf('\n');
text=[];
for iter=1:burnin+collection
    
    switch sampler
        case 'Beta_NB_collapsed'
            
            [ZSDS,WSZS] = Multrnd_Matrix_mex_fast_v1(XtrainSparse,Phi,Theta);
            L_i = CRT_sum_mex_matrix(sparse(ZSDS),r_i);
            sumlogpk = sum(log(max(1-p_k,realmin)));
            r_i = gamrnd(a0 + L_i, 1./(-sumlogpk + p_star + b0));
            Theta = bsxfun(@times,randg(ZSDS + r_i(ones(1,K),:)), p_k);
            
        %case {'NB_HDP','Gamma_NB','NB_HDP_Kadaptive','Gamma_NB_Kadaptive',...
        %        'Gamma_NB_partially_collapsed', 'HDP_DirectAssignment','Gamma_NB_fully_collapsed'}
        otherwise    
            [ZSDS,WSZS] = Multrnd_Matrix_mex_fast_v1(XtrainSparse,Phi,Theta);
            if strcmp(sampler,'NB_HDP') || strcmp(sampler,'NB_HDP_Kadaptive')
                p_i = 0.5*ones(1,N);
            else
                p_i = betarnd(a0 + sum(Xtrain,1),b0+sum(r_k)+r_star);
            end
            Theta = bsxfun(@times,randg(ZSDS + r_k(:,ones(1,N))), p_i);
    end
    
    if iter>burnin
        AveThetaTest = AveThetaTest + Theta/collection;
        AveThetaTest_Prob = AveThetaTest_Prob + bsxfun(@rdivide,Theta,max(sum(Theta,1),realmin))/collection;
    end
    if mod(iter,10)==0
        fprintf(repmat('\b',1,numel(text)));
        text = sprintf('Test Iter: %d',iter); fprintf(text, iter);
    end
end
AveThetaTest(isnan(AveThetaTest))=0;
AveThetaTest_Prob(isnan(AveThetaTest_Prob))=0;