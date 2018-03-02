function [AveThetaTest,AveThetaTest_Prob] = NBFA_DirMultTM_test(Xtrain,Phi,r_k,r_star,model,sampler,burnin,collection)
% Matlba code for the following papers:

% M. Zhou, "Negative Binomial Factor Anlysis and Dirichlet-Multinomial
% Topic Modeling," preprint, 2015.

% First Version: Sept, 2015
%
% Coded by Mingyuan Zhou,
% http://mingyuanzhou.github.io/
% Copyright (C) 2015, Mingyuan Zhou.
%
XtrainSparse = sparse(Xtrain);

a0=1e-2; b0=1e-2;

[V,N] = size(Xtrain);
Theta = r_k*ones(1,N);
theta_star = r_star*ones(1,N);
K=size(Phi,2);
ZS = randi(K,sum(Xtrain(:)),1);
p_i=ones(1,N)*0.5;
AveThetaTest=0;
AveThetaTest_Prob=0;
c_i=ones(1,N);
fprintf('\n');
text=[];
for iter=1:burnin + collection
    
    if strcmp(model, 'hGNBP_DirMultTM')
        if strcmp(sampler, 'blocked_Gibbs_NB') || strcmp(sampler, 'blocked_Gibbs_NB_Truncated')
            [ZSDS,WSZS] = CRT_Multrnd_Matrix(XtrainSparse,Phi,Theta);
        else
            [ZSDS,WSZS,ZS] = DirMult_CRT_Matrix(XtrainSparse,Phi,Theta,ZS);
        end
        Theta = bsxfun(@rdivide,randg(ZSDS + r_k(:,ones(1,N))), c_i-log(max(1-p_i,realmin)));
        theta_star = randg(r_star*ones(1,N))./(c_i-log(max(1-p_i,realmin)));
        c_i = randg(1+sum(r_k)+r_star)./(1+sum(Theta,1)+theta_star);
        p_i = betarnd(a0 + sum(Xtrain,1),b0+sum(Theta,1)+theta_star);
        ThetaTest = Theta;
    else
        if strcmp(sampler, 'blocked_Gibbs_NB') || strcmp(sampler, 'blocked_Gibbs_NB_Truncated')
            [ZSDS,WSZS] = CRT_Multrnd_Matrix(XtrainSparse,Phi,Theta);
        else
            [ZSDS,WSZS,ZS] = DirMult_CRT_Matrix(XtrainSparse,Phi,Theta,ZS);
        end
        Theta = r_k*ones(1,N);
        ThetaTest = dirrnd(ZSDS + r_k(:,ones(1,N)));
    end
    
    if iter>burnin
        AveThetaTest = AveThetaTest + ThetaTest/collection;
        AveThetaTest_Prob = AveThetaTest_Prob + bsxfun(@rdivide,ThetaTest,max(sum(ThetaTest,1),realmin))/collection;
    end
    if mod(iter,10)==0
        fprintf(repmat('\b',1,numel(text)));
        text = sprintf('Test Iter: %d',iter); fprintf(text, iter);
    end
end
AveThetaTest(isnan(AveThetaTest))=0;
AveThetaTest_Prob(isnan(AveThetaTest_Prob))=0;