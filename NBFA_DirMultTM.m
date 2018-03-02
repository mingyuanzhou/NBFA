function [ave,Phi,Theta,output]=...
    NBFA_DirMultTM(X,model,sampler,eta,percentage,K,state,burnin,collection,CollectionStep,IsSampleEta,IsPlot)
%% Matlab code for the paper:
% M. Zhou, "Negative binomial factor analysis and Dirichelt-multinomial 
% topic modeling," preprint, 2015.

% First Version: June, 2015
% This Version: October, 2015
%
% Coded by Mingyuan Zhou,
% http://mingyuanzhou.github.io/
% Copyright (C) 2015, Mingyuan Zhou.
%%
if ~exist('model','var')
    model = 'hGNBP_DirMultTM';
end
if ~exist('sampler','var')
    sampler = 'blocked_Gibbs_NB';
end
if ~exist('eta','var')
    eta=0.05;
end
if ~exist('percentage','var')
    percentage=101;
end
if ~exist('K','var')
    K=400;
end
if ~exist('state','var')
    state=1;
end
if ~exist('burnin','var')
    burnin=500;
end
if ~exist('collection','var')
    collection=500;
end
if ~exist('CollectionStep','var')
    CollectionStep=5;
end

if ~exist('IsSampleEta','var')
    IsSampleEta =false;
end
if ~exist('IsPlot','var')
    IsPlot = false;
end



K_star=20;
if K==0 && strcmp(sampler, 'collapsed')==0
    K=10;
end


[V,N] = size(X);
P=V;
Phi = rand(V,K);
Phi = bsxfun(@rdivide,Phi,sum(Phi,1));
Theta = zeros(K,N)+1/K;

rng(state,'twister');

%% X is the input term-document count matrix
%% Partition X into training and testing, Xtrain is the training count matrix, WS and DS are the indices for word tokens
[Xtrain,Xtest,WS,DS,WordTrainS,DocTrainS]= PartitionX_v1(X,percentage);
%[Xtrain,Xtest,WS,DS,WordTrainS,DocTrainS]= PartitionX(X,percentage);

WS = WS(WordTrainS);
DS = DS(WordTrainS);

Yflagtrain = Xtrain>0;
Yflagtest = Xtest>0;

loglikeTrain = []; loglike=[];
ave.loglike=[]; ave.K=[]; ave.PhiTheta = sparse(P,N); ave.Count = 0; ave.gamma0=[];
ave.alpha_concentration=[];
ave.PhiThetaSum = zeros(1,N);

ave.eta = zeros(1,burnin + collection);

c_0=1;gamma0=1; r_k= 50/K*ones(K,1);
p_i=0.5*ones(1,N);
c_i=ones(1,N);


disp(model)
disp(sampler)
disp(['state=',num2str(state)])
disp(['eta=',num2str(eta)])
disp(['K_init=',num2str(K)])
disp(['Percentage=',num2str(percentage)])

a0=1e-2; b0=1e-2; e0=1e-2; f0=1e-2; c=1;
loglike=[];
loglikeTrain=[];

XtrainSparse= sparse(Xtrain);

if K>0
    ZS = randi(K,size(WS,1),size(WS,2));
else
    ZS = zeros(size(WS));
end


Xmask=sparse(X);
fprintf('\n');
text=[];
tic
for iter=1:burnin + collection
    
    if strcmp(model, 'hGNBP_DirMultTM')
        switch sampler
            case 'collapsed'
                if iter==1
                    %initilization
                    if K==0;
                        r_k= ones(K,1);
                        ell_dot_k = zeros(K,1);
                        r_star=1;
                        ZSDS=zeros(K,N);
                        WSZS=zeros(P,K);
                        ZS = zeros(size(WS));
                        TS = zeros(size(WS));
                    else
                        TS = ones(size(ZS));
                        r_k= ones(K,1);
                        r_star=1;
                        ZSDS = zeros(K,N);
                        WSZS = zeros(P,K);
                        for v=1:P
                            ZSDS = ZSDS + double(sparse(ZS(WS==v),DS(WS==v),1,K,N)>0);
                        end
                        for j=1:N
                            WSZS = WSZS + double(sparse(WS(DS==j),ZS(DS==j),1,P,K)>0);
                        end
                        ell_dot_k = full(sum(ZSDS,2));
                    end
                    cjpj = c_i-log(max(1-p_i,realmin));
                    sumlogpi = sum(log(max(1-p_i,realmin)));
                end
                DSZS=ZSDS';
                [DSZS,WSZS,ell_dot_k,ZS,TS,r_k,r_star] =...
                    DirMultTM_hGNBP(XtrainSparse,DSZS,WSZS,ell_dot_k,ZS,TS,r_k,r_star,cjpj,gamma0,eta);
                ZSDS=DSZS';
                c_0 = randg(1 + gamma0)/(1+sum(r_k)+r_star);
                if iter==1
                    theta = randg(sum(ZSDS,1)+sum(r_k)+r_star)./(c_i-log(max(1-p_i,realmin)));
                end
                p_i = betarnd(a0 + sum(Xtrain,1),b0+theta);
                p_tilde_i = -log(max(1-p_i,realmin));
                p_tilde_i =  p_tilde_i./(c_i+ p_tilde_i);
                sumlogpi = sum(log(max(1-p_tilde_i,realmin)));
                p_prime = -sumlogpi./(c_0-sumlogpi);
                
                %%Delete unused atoms
                [dexk,kki,kkj] = unique(ZS);
                ZS = kkj;
                ZSDS=ZSDS(dexk,:);
                WSZS=WSZS(:,dexk);
                r_k = r_k(dexk);
                ell_dot_k=ell_dot_k(dexk);
                K = length(r_k);
                gamma0 = gamrnd(a0 + K,1/(b0 - log(max(1-p_prime,realmin))));
                %L_k = CRT_sum_mex_matrix(sparse(ZSDS'),r_k')';
                L_k=zeros(K,1);
                for k=1:K
                    L_k(k) = CRT_sum_mex(ZSDS(k,:),r_k(k));
                end
                r_k = gamrnd(L_k, 1./(-sumlogpi+ c_0));
                r_star=gamrnd(gamma0, 1./(-sumlogpi+ c_0));
                
                if IsSampleEta
                    q_k = betarnd(sum(WSZS,1),eta*V);
                    Lv = CRT_sum_mex(nonzeros(WSZS),eta);
                    eta = gamrnd(0.01+Lv,1./(0.01-V*sum(log(max(1-q_k,realmin)))));
                end
                
                if (iter>burnin && mod(iter,CollectionStep)==0 && percentage<1) || iter== (burnin + collection)  
                    Phi = dirrnd([WSZS,zeros(P,K_star)] + eta);
                    Theta = bsxfun(@rdivide,randg([ZSDS + r_k(:,ones(1,N)) ; r_star/K_star*ones(K_star,N)]), c_i-log(max(1-p_i,realmin)));
                    theta = sum(Theta,1);
                else
                    theta = randg(sum(ZSDS,1)+sum(r_k)+r_star)./(c_i-log(max(1-p_i,realmin)));
                end
                c_i = randg(1+sum(r_k)+r_star)./(1+theta);
                cjpj=c_i-log(max(1-p_i,realmin));
            case {'blocked_Gibbs_NB','blocked_Gibbs_DirMult','blocked_Gibbs_NB_Truncated'}
                if strcmp(sampler, 'blocked_Gibbs_NB') || strcmp(sampler,'blocked_Gibbs_NB_Truncated')
                    %compound Poisson based blocked Gibbs sampler
                    [ZSDS,WSZS] = CRT_Multrnd_Matrix(XtrainSparse,Phi,Theta);
                else
                    %regular blocked Gibbs sampler
                    [ZSDS,WSZS,ZS] = DirMult_CRT_Matrix(XtrainSparse,Phi,Theta,ZS);
                end
                %L_k = CRT_sum_mex_matrix(sparse(ZSDS'),r_k')';
                L_k=zeros(K,1);
                for k=1:K
                    L_k(k) = CRT_sum_mex(ZSDS(k,:),r_k(k));
                end
                c_0 = randg(1 + gamma0)/(1+sum(r_k));
                p_i = betarnd(a0 + sum(Xtrain,1),b0+sum(Theta,1));
                p_tilde_i = -log(max(1-p_i,realmin));
                p_tilde_i =  p_tilde_i./(c_i+ p_tilde_i);
                sumlogpi = sum(log(max(1-p_tilde_i,realmin)));
                p_prime = -sumlogpi./(c_0-sumlogpi);
                
                
                if strcmp(sampler, 'blocked_Gibbs_NB_Truncated')
                    gamma0 = gamrnd(a0 + CRT_sum_mex(L_k,gamma0/K),1/(b0 - log(max(1-p_prime,realmin))));
                    r_k = randg(L_k+gamma0/K)/(-sumlogpi+ c_0);
                else
                    if strcmp(sampler, 'blocked_Gibbs_NB')
                        dexk = find(L_k>0);
                    else
                        [dexk,kki,kkj] = unique(ZS);
                        ZS = kkj;
                    end
                    gamma0 = gamrnd(a0 + length(de xk),1/(b0 - log(max(1-p_prime,realmin))));
                    L_k=L_k(dexk);
                    ZSDS=[ZSDS(dexk,:);zeros(K_star,N)];
                    WSZS=[WSZS(:,dexk),zeros(V,K_star)];
                    r_k = [randg(L_k); randg(gamma0/K_star*ones(K_star,1))]/(-sumlogpi+ c_0);
                    K = length(r_k);
                end
                
                if IsSampleEta
                    q_k = betarnd(sum(WSZS,1),eta*V);
                    Lv = CRT_sum_mex(nonzeros(WSZS),eta);
                    eta = gamrnd(0.01+Lv,1./(0.01-V*sum(log(max(1-q_k,realmin)))));
                end
                
                Theta = bsxfun(@rdivide,randg(ZSDS + r_k(:,ones(1,N))), c_i-log(max(1-p_i,realmin)));
                Phi = dirrnd(bsxfun(@plus,WSZS, eta));
                c_i = randg(1+sum(r_k))./(1+sum(Theta,1));

                if iter==burnin + collection
                    if strcmp(sampler, 'blocked_Gibbs_NB_Truncated')
                        r_star=0;
                    else
                        Phi = Phi(:,1:K-K_star);
                        Theta = Theta(1:K-K_star,:);
                        ZSDS = ZSDS(1:K-K_star,:);
                        WSZS = WSZS(:,1:K-K_star);
                        r_star=sum(r_k(K-K_star+1:end));
                        r_k = r_k(1:K-K_star);
                    end
                end
        end
        
        ave.K(end+1) = nnz(sum(ZSDS,2));
        
    else
        switch sampler
            case 'collapsed'
                
                if iter==1
                    %K=0;
                    if K==0;
                        r_k= ones(K,1);
                        ell_dot_k = zeros(K,1);
                        r_star=1;
                        ZSDS=zeros(K,N);
                        WSZS=zeros(P,K);
                        ZS = zeros(size(WS));
                        TS = zeros(size(WS));
                    else
                        TS = ones(size(ZS));
                        r_k= ones(K,1);
                        r_star=1;
                        ZSDS = zeros(K,N);
                        WSZS = zeros(P,K);
                        for v=1:P
                            ZSDS = ZSDS + double(sparse(ZS(WS==v),DS(WS==v),1,K,N)>0);
                        end
                        for j=1:N
                            WSZS = WSZS + double(sparse(WS(DS==j),ZS(DS==j),1,P,K)>0);
                        end
                        ell_dot_k = full(sum(ZSDS,2));
                    end
                    cjpj = c_i-log(max(1-p_i,realmin));
                    sumlogpi = sum(log(max(1-p_i,realmin)));
                    
                end
                
                [WSZS,ell_dot_k,ZS,TS] = DCMLDA_GNBP_fully(XtrainSparse,WSZS,ell_dot_k,ZS,TS,cjpj(1),gamma0,eta);
                
                L_k=ell_dot_k;
                r_k = gamrnd(L_k, 1./(-sumlogpi+ c_0));
                r_star=gamrnd(gamma0, 1./(-sumlogpi+ c_0));
                p_i = betarnd(a0 + sum(Xtrain,1),b0+sum(r_k)+r_star);
                sumlogpi = sum(log(max(1-p_i,realmin)));
                p_prime = -sumlogpi./(c_0-sumlogpi);
                
                [dexk,kki,kkj] = unique(ZS);
                ZS = kkj;
                WSZS=WSZS(:,dexk);
                L_k = L_k(dexk);
                r_k = r_k(dexk);
                ell_dot_k=ell_dot_k(dexk);
                K = length(r_k);
                gamma0 = gamrnd(a0 + K,1/(b0 - log(max(1-p_prime,realmin))));
                c_0 = randg(1 + gamma0)/(1+sum(r_k)+r_star);
                cjpj=c_0-sumlogpi;
                
                if IsSampleEta
                    q_k = betarnd(sum(WSZS,1),eta*V);
                    Lv = CRT_sum_mex(nonzeros(WSZS),eta);
                    eta = gamrnd(0.01+Lv,1./(0.01-V*sum(log(max(1-q_k,realmin)))));
                end
                
                if (iter>burnin && mod(iter,CollectionStep)==0 && percentage<1) || iter== (burnin + collection)  
                    Theta = r_k*ones(1,N);
                    Phi = dirrnd(WSZS + eta);
                end
                
            case {'blocked_Gibbs_NB','blocked_Gibbs_DirMult'}
                if strcmp(sampler, 'blocked_Gibbs_NB')
                    [ZSDS,WSZS] = CRT_Multrnd_Matrix(XtrainSparse,Phi,Theta);
                else
                    [ZSDS,WSZS,ZS] = DirMult_CRT_Matrix(XtrainSparse,Phi,Theta,ZS);
                end
                L_k=sum(ZSDS,2);
                c_0 = randg(1 + gamma0)/(1+sum(r_k));
                p_i = betarnd(a0 + sum(Xtrain,1),b0+sum(r_k));
                sumlogpi = sum(log(max(1-p_i,realmin)));
                p_prime = -sumlogpi./(c_0-sumlogpi);
                
                
                if strcmp(sampler, 'blocked_Gibbs_NB')
                    dexk = find(L_k>0);
                else
                    [dexk,kki,kkj] = unique(ZS);
                    ZS = kkj;
                end
                gamma0 = gamrnd(a0 + length(dexk),1/(b0 - log(max(1-p_prime,realmin))));
                L_k=L_k(dexk);
                ZSDS=[ZSDS(dexk,:);zeros(K_star,N)];
                WSZS=[WSZS(:,dexk),zeros(V,K_star)];
                r_k = [randg(L_k); randg(gamma0/K_star*ones(K_star,1))]/(-sumlogpi+ c_0);
                K = length(r_k);
                
                if IsSampleEta
                    q_k = betarnd(sum(WSZS,1),eta*V);
                    Lv = CRT_sum_mex(nonzeros(WSZS),eta);
                    eta = gamrnd(0.01+Lv,1./(0.01-V*sum(log(max(1-q_k,realmin)))));
                end
               
                Theta = r_k*ones(1,N);
                Phi = dirrnd(WSZS + eta);
                if iter==burnin + collection
                    Phi = Phi(:,1:K-K_star);
                    Theta = Theta(1:K-K_star,:);
                    ZSDS = ZSDS(1:K-K_star,:);
                    WSZS = WSZS(:,1:K-K_star);
                    r_star=sum(r_k(K-K_star+1:end));
                    r_k = r_k(1:K-K_star);
                end
        end
        ave.K(end+1) = nnz(sum(WSZS,1));
    end
    ave.eta(iter) = eta;
    
    
    if iter>burnin && mod(iter,CollectionStep)==0 && percentage<1
        
        
        %X1 = bsxfun(@times,(Phi*Theta+Xtrain), p_i);
        X1 = Mult_Sparse(Xmask,Phi,Theta);
        X1 = bsxfun(@times,X1+Xtrain,p_i);
        X1sum = (sum(Theta,1)+sum(Xtrain,1)).*p_i;
        ave.PhiTheta = ave.PhiTheta + X1;
        ave.PhiThetaSum = ave.PhiThetaSum + X1sum;
        ave.Count = ave.Count+1;
        X1 = bsxfun(@rdivide, X1,X1sum);
        %loglike(end+1)=sum(X(Yflag).*log(X1(Yflag)));
        %loglike(end)
        %loglike(end+1)=exp(-sum(Xtest(Yflagtest).*log(X1(Yflagtest)))/sum(Xtest(:)));
        loglike(end+1)=sum(Xtest(Yflagtest).*log(X1(Yflagtest)))/sum(Xtest(:));
        loglikeTrain(end+1)=sum(Xtrain(Yflagtrain).*log(X1(Yflagtrain)))/sum(Xtrain(:));
        
        X1 = ave.PhiTheta/ave.Count;
        X1sum = ave.PhiThetaSum/ave.Count;
        X1= bsxfun(@rdivide, X1, X1sum);
        ave.loglike(end+1) = sum(Xtest(Yflagtest).*log(X1(Yflagtest)))/sum(Xtest(:));
        % ave.K(end+1) = nnz(sum(ZSDS,2));
        ave.gamma0(end+1) = gamma0;
        %        ave.alpha_concentration(end+1) = alpha_concentration;
        
        
    end
    
    %     if mod(iter,10)==0
    %         toc
    %         if iter>burnin && percentage<1
    %             disp(full([iter/100,loglikeTrain(end),loglike(end),ave.K(end),ave.loglike(end)]));
    %         else
    %             disp(full(iter/100));
    %         end
    %         tic;
    %     end
    
    if IsPlot && mod(iter,10)==0
        % [temp, Thetadex] = sort(sum(ZSDS,2),'descend');
        [temp, Thetadex] = sort(sum(WSZS,1),'descend');
        subplot(2,2,1);plot((r_k(Thetadex)),'.');
        if ~isempty(loglike)
            subplot(2,2,2);plot(loglikeTrain); hold on; plot(loglike); hold off %p_k(Thetadex));
        end
        subplot(2,2,3);plot(ave.K);
        subplot(2,2,4);plot(p_i);title(num2str(eta))
        drawnow
    end
    if mod(iter,100)==0
        %fprintf(repmat('\b',1,numel(text)));
        text = sprintf('Train Iter: %d',iter); fprintf(text, iter);
    end
end
ave.times=toc;
output.r_k = r_k;
output.r_star = r_star;
output.p_i=p_i;
output.WSZS=WSZS;
output.ZS=ZS;
output.eta=eta;
ave.LogLikeTrain = loglikeTrain;
ave.LogLikeTest = loglike;
if percentage<1
    ave.Perplexity = exp(-ave.loglike(end));
end
