function [ave,Phi,Theta,output]=...
    PFA_MultTM(X,model,sampler,eta,percentage,K,state,burnin,collection,CollectionStep,IsSampleEta,IsPlot)
%% Matlab code for the paper:
% M. Zhou, "Negative binomial factor analysis and Dirichelt-multinomial 
% topic modeling," preprint, 2015.

% First Version: June, 2015
% This Version: October, 2015
%
% Coded by Mingyuan Zhou,
% http://mingyuanzhou.github.io/
% Copyright (C) 2015, Mingyuan Zhou.

% The code is built on: 

%[1] Collapsed Gibbs sampling for the beta-negative binomial process
% multinomial topic model and Poisson factor analysis is described in:
% M. Zhou, "Beta-negative binomial process and exchangeable random partitions
% for mixed-membership modeling," NIPS2014, Montreal, Canada, Dec. 2014.
% Code for that paper: https://github.com/mingyuanzhou/BNBP_collapsed


%[2] Blocked Gibbs sampling with finte truncation is described in:
% M. Zhou and L. Carin, "Negative Binomial Process Count and Mixture Modeling,"
% IEEE Trans. Pattern Analysis and Machine Intelligence, vol. 37, pp. 307-320, Feb. 2015.
% Code for that paper: http://mingyuanzhou.github.io/Softwares/NBP_PFA_v1.zip
%%
if ~exist('model','var')
    model = 'GNBP_PFA';
end
if ~exist('sampler','var')
    sampler = 'Gamma_NB_partially_collapsed';
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

maxIter=burnin + collection;

K_star=20;
if K==0 && nnz(strcmp(sampler, {'HDP_DirectAssignment','Beta_NB_collapsed',...
        'Gamma_NB_partially_collapsed', 'Gamma_NB_fully_collapsed'}))==0
    K=10;
end

[V,N] = size(X);
P=V;
Phi = rand(P,K);
Phi = bsxfun(@rdivide,Phi,sum(Phi,1));
Theta = zeros(K,N)+1/K;

rng(state,'twister');
% rand('state',state)
% randn('state',state)
%% X is the input term-document count matrix
%% Partition X into training and testing, Xtrain is the training count matrix, WS and DS are the indices for word tokens
%[Xtrain,Xtest,WS,DS,WordTrainS,DocTrainS]= PartitionX(X,percentage);
[Xtrain,Xtest,WS,DS,WordTrainS,DocTrainS]= PartitionX_v1(X,percentage);
WS = WS(WordTrainS);
DS = DS(WordTrainS);
ZS = DS-DS;

Yflagtrain = Xtrain>0;
Yflagtest = Xtest>0;
loglikeTrain = []; loglike=[];
ave.loglike=[]; ave.PhiTheta = sparse(V,N); ave.Count = 0;
ave.PhiThetaSum = zeros(1,N);
ave.K=zeros(1,maxIter);  ave.gamma0=zeros(1,maxIter);
ave.eta=zeros(1,maxIter);

%% Intilization
c=1; gamma0=1;
p_i=0.5*ones(1,N);
r_i=0.1*ones(1,N);
r_star = 1;
r_k = ones(K,1);

disp(sampler)
disp(['state=',num2str(state)])
disp(['eta=',num2str(eta)])
disp(['Kstart=',num2str(K)])
disp(['Percentage=',num2str(percentage)])

XtrainSparse= sparse(Xtrain);
Xmask=sparse(X);

if K==0
    DSZS = zeros(N,K);
    WSZS = zeros(V,K);
    TS = ZS-ZS+0;
else
    % K=400;
    ZS = randi(K,length(DS),1);
    DSZS = full(sparse(DS,ZS,1,N,K));
    WSZS = full(sparse(WS,ZS,1,V,K));
    TS=ones(size(ZS));
end
n_dot_k = sum(DSZS,1)';
ell_dot_k=sum(DSZS>0,1)';

a0=1e-2; b0=1e-2; e0=1e-2; f0=1e-2;
loglike=[];
loglikeTrain=[];

fprintf('\n');
text=[];

tic;
for iter=1:maxIter
    
    switch sampler
        case {'NB_HDP','Gamma_NB','NB_HDP_Kadaptive','Gamma_NB_Kadaptive'}
            %[ZSDS,WSZS] = Multrnd_Matrix_mex_fast(XtrainSparse,Phi,Theta);
            [ZSDS,WSZS] = Multrnd_Matrix_mex_fast_v1(XtrainSparse,Phi,Theta);
            if IsSampleEta
                q_k = betarnd(sum(WSZS,1),eta*V);
                Lv = CRT_sum_mex(nonzeros(WSZS),eta);
                eta = gamrnd(0.01+Lv,1./(0.01-V*sum(log(max(1-q_k,realmin)))));
            end
            
            if strcmp(sampler,'NB_HDP') || strcmp(sampler,'NB_HDP_Kadaptive')
                p_i = 0.5*ones(1,N);
            else
                p_i = betarnd(a0 + sum(Xtrain,1),b0+sum(r_k));
            end
            sumlogpi = sum(log(max(1-p_i,realmin)));
            p_prime = -sumlogpi./(c-sumlogpi);
            L_k = CRT_sum_mex_matrix(sparse(ZSDS'),r_k')';
            c = randg(1 + gamma0)/(1+sum(r_k));
            if strcmp(sampler,'NB_HDP') || strcmp(sampler,'Gamma_NB')
                gamma0 = gamrnd(a0 + CRT_sum_mex(L_k,gamma0/K),1/(b0 - log(max(1-p_prime,realmin))));
                r_k = gamrnd(gamma0/K + L_k, 1./(-sumlogpi+ c));
            else
                dexk = find(L_k>0);
                gamma0 = gamrnd(a0 + length(dexk),1/(b0 - log(max(1-p_prime,realmin))));
                L_k=L_k(dexk);
                ZSDS=[ZSDS(dexk,:);zeros(K_star,N)];
                WSZS=[WSZS(:,dexk),zeros(V,K_star)];
                r_k = [randg(L_k); randg(gamma0/K_star*ones(K_star,1))]/(-sumlogpi+ c);
                K = length(r_k);
            end
            Theta = bsxfun(@times,randg(ZSDS + r_k(:,ones(1,N))), p_i);
            Phi = dirrnd(WSZS + eta);
            
            if (iter==burnin + collection)  
                if (strcmp(sampler,'NB_HDP_Kadaptive') || strcmp(sampler,'Gamma_NB_Kadaptive'))
                    Phi = Phi(:,1:K-K_star);
                    Theta = Theta(1:K-K_star,:);
                    ZSDS = ZSDS(1:K-K_star,:);
                    WSZS = WSZS(:,1:K-K_star);
                    r_star=sum(r_k(K-K_star+1:end));
                    r_k = r_k(1:K-K_star);
                else
                    r_star=0;
                end
            end

        case 'NB_collapsed'
            %% Random scan, optional
            dex111=randperm(length(ZS));
            ZS=ZS(dex111); DS=DS(dex111); WS=WS(dex111);
            %% Collapsed inference
            [WSZS,n_dot_k,ZS] = PFA_NBP(WSZS,n_dot_k,ZS,WS,gamma0,eta);
            
            if IsSampleEta
                q_k = betarnd(sum(WSZS,1),eta*V);
                Lv = CRT_sum_mex(nonzeros(WSZS),eta);
                eta = gamrnd(0.01+Lv,1./(0.01-V*sum(log(max(1-q_k,realmin)))));
            end
            
            %% Delete unused atoms
            K=nnz(n_dot_k);
            [kk,kki,kkj] = unique(ZS);
            ZS=kkj;
            WSZS=WSZS(:,kk);
            n_dot_k=n_dot_k(kk);
            %% Sample model parameters
            p = betarnd(a0 + sum(Xtrain(:)),b0+gamma0);
            gamma0 = gamrnd(a0 + K,1/(b0 - log(max(1-p,realmin))));
            r_k = gamrnd(n_dot_k, p/N);
            if (iter>burnin && mod(iter,CollectionStep)==0 && percentage<1) || iter==maxIter
                Theta = [r_k(:,ones(1,N));gamrnd(gamma0/K_star*ones(K_star,1),p/N)*ones(1,N)];
                Phi = dirrnd([WSZS,zeros(V,K_star)] + eta);
            end
            
         case {'Gamma_NB_collapsed_fixK'}
            %% Random scan, optional
            dex111=randperm(length(ZS));
            ZS=ZS(dex111); DS=DS(dex111); WS=WS(dex111);
            %% Partially collapsed inference, the gamma process weights r_k are not marginalized out, K is fixed
            %[WSZS,DSZS,n_dot_k,r_k,ZS] = PFA_GNBP_partial_fixK(WSZS,DSZS,n_dot_k,r_k,ZS,WS,DS,c,gamma0,eta);
            r_star=0;
            [WSZS,DSZS,n_dot_k,r_k,r_star,ZS] = PFA_GNBP_partial(WSZS,DSZS,n_dot_k,r_k,r_star,ZS,WS,DS,c,gamma0,eta);
            if IsSampleEta
                q_k = betarnd(sum(WSZS,1),eta*V);
                Lv = CRT_sum_mex(nonzeros(WSZS),eta);
                eta = gamrnd(0.01+Lv,1./(0.01-V*sum(log(max(1-q_k,realmin)))));
            end
            
            p_i = betarnd(a0 + sum(Xtrain,1),b0+sum(r_k));
            sumlogpi = sum(log(max(1-p_i,realmin)));
            p_prime = -sumlogpi./(c-sumlogpi);
            L_k = CRT_sum_mex_matrix(sparse(DSZS),r_k')';
            c = randg(1 + gamma0)/(1+sum(r_k));
            gamma0 = gamrnd(a0 + CRT_sum_mex(L_k,gamma0/K),1/(b0 - log(max(1-p_prime,realmin))));
            r_k = gamrnd(gamma0/K + L_k, 1./(-sumlogpi+ c));
            
            if (iter>burnin && mod(iter,CollectionStep)==0 && percentage<1) || iter==maxIter
                Phi = dirrnd(WSZS + eta);
                Theta = bsxfun(@times,randg(DSZS' + r_k(:,ones(1,N))), p_i);
            end
            r_star=0;

        case {'Gamma_NB_partially_collapsed', 'HDP_DirectAssignment'}
            %% Random scan, optional
            dex111=randperm(length(ZS));
            ZS=ZS(dex111); DS=DS(dex111); WS=WS(dex111);
            %% Partially collapsed inference, the gamma process weights r_k are not marginalized out
            [WSZS,DSZS,n_dot_k,r_k,r_star,ZS] = PFA_GNBP_partial(WSZS,DSZS,n_dot_k,r_k,r_star,ZS,WS,DS,c,gamma0,eta);
            
            if IsSampleEta
                q_k = betarnd(sum(WSZS,1),eta*V);
                Lv = CRT_sum_mex(nonzeros(WSZS),eta);
                eta = gamrnd(0.01+Lv,1./(0.01-V*sum(log(max(1-q_k,realmin)))));
            end
            
            %% Delete unused atoms
            K=nnz(n_dot_k);
            [kk,kki,kkj] = unique(ZS);
            ZS=kkj;
            DSZS=DSZS(:,kk);
            WSZS=WSZS(:,kk);
            n_dot_k=n_dot_k(kk);
            r_k=r_k(kk);
            %% Sample model parameters for the gamma-negative binomial process
            if strcmp(sampler,'Gamma_NB_partially_collapsed')
                sumlogpi = sum(log(max(1-p_i,realmin)));
                p_prime = -sumlogpi./(c-sumlogpi);
                c = randg(1 + gamma0)/(1+sum(r_k)+r_star);
                gamma0 = gamrnd(a0 + K,1/(b0 - log(max(1-p_prime,realmin))));
                L_k = CRT_sum_mex_matrix(sparse(DSZS),r_k')';
                r_k = gamrnd(L_k, 1./(-sumlogpi+ c));
                r_star = gamrnd(gamma0, 1./(-sumlogpi+ c));
                p_i = betarnd(a0 + sum(DSZS,2)',b0+sum(r_k)+r_star);
                if (iter>burnin && mod(iter,CollectionStep)==0 && percentage<1) || iter==maxIter
                    Phi = dirrnd([WSZS,zeros(V,K_star)] + eta);
                    Theta = bsxfun(@times,randg([DSZS' + r_k(:,ones(1,N)) ; r_star/K_star*ones(K_star,N)]), p_i);
                end
            end
            %% Sample model parameters for the hierachical Dirichlet process
            if strcmp(sampler,'HDP_DirectAssignment')
                if iter==1
                    alpha_concentration = 1;
                end
                L_k = CRT_sum_mex_matrix(sparse(DSZS),r_k')';
                for iii=1:1
                    w0 = betarnd(gamma0+1, sum(L_k));
                    pi0 = (a0+K-1)/((b0+K-1)+sum(L_k)*(f0-log(w0)));
                    %gamma0 = pi0*gamrnd(e0+K0,1/(f0-log(w0))) + (1-pi0)*gamrnd(e0+K0-1,1/(f0-log(w0)));
                    gamma0 = gamrnd(a0+K-(rand(1)>pi0),1/(b0-log(w0)));
                    r_tilde_k = dirrnd([L_k;gamma0]);
                    wj = betarnd(alpha_concentration+1, full(sum(DSZS,2)'));
                    sj = sum(rand(1,N)<sum(DSZS,2)'./(sum(DSZS,2)'+alpha_concentration));
                    alpha_concentration = gamrnd(sum(L_k)+a0 - sj,1/(b0-sum(log(wj))));
                end
                r_k = alpha_concentration*r_tilde_k(1:end-1,:);
                r_star = alpha_concentration*r_tilde_k(end);
                if (iter>burnin && mod(iter,CollectionStep)==0 && percentage<1) || iter==maxIter
                    Phi = dirrnd([WSZS,zeros(V,K_star)] + eta);
                    Theta = dirrnd([DSZS' + r_k(:,ones(1,N)) ; r_star/K_star*ones(K_star,N)]);
                end
            end
            
        case 'Gamma_NB_fully_collapsed'
            %% Fully collapsed inference, the gamma process weights r_k are marginalized out
            sumlogpi = sum(log(max(1-p_i,realmin)));
            c_q_dot = c-sumlogpi;
            [WSZS,DSZS,ell_dot_k,n_dot_k,ZS,TS] = PFA_GNBP_fully(XtrainSparse,WSZS,DSZS,ell_dot_k,n_dot_k,ZS,TS,gamma0,eta,c_q_dot);
            if IsSampleEta
                q_k = betarnd(sum(WSZS,1),eta*V);
                Lv = CRT_sum_mex(nonzeros(WSZS),eta);
                eta = gamrnd(0.01+Lv,1./(0.01-V*sum(log(max(1-q_k,realmin)))));
            end
            %% Delete unsed atoms
            K=nnz(n_dot_k);
            [kk,kki,kkj] = unique(ZS);
            ZS=kkj;
            DSZS=DSZS(:,kk);
            WSZS=WSZS(:,kk);
            n_dot_k=n_dot_k(kk);
            ell_dot_k=ell_dot_k(kk);
            %% Sample model parameters
            sumlogpi = sum(log(max(1-p_i,realmin)));
            p_prime = -sumlogpi./(c-sumlogpi);
            gamma0 = gamrnd(a0 + K,1/(b0 - log(max(1-p_prime,realmin))));
            L_k = ell_dot_k;
            r_k = gamrnd(L_k, 1./(-sumlogpi+ c));
            r_star = randg(gamma0)/(-sumlogpi+ c);
            c = randg(1 + gamma0)/(1+sum(r_k)+r_star);
            p_i = betarnd(a0 + sum(DSZS,2)',b0+sum(r_k)+r_star);
            if (iter>burnin && mod(iter,CollectionStep)==0 && percentage<1) || iter==maxIter
                Phi = dirrnd([WSZS,zeros(V,K_star)] + eta);
                Theta = bsxfun(@times,randg([DSZS' + r_k(:,ones(1,N)) ; r_star/K_star*ones(K_star,N)]), p_i);
            end
        case 'Beta_NB_collapsed'
            %% Random scan, optional
            dex111=randperm(length(ZS));
            ZS=ZS(dex111); DS=DS(dex111); WS=WS(dex111);
            %% Fully collapsed inference, the beta process weights p_k are marginalized out
            [WSZS,DSZS,n_dot_k,ZS] = PFA_BNBP_collapsed(WSZS,DSZS,n_dot_k,ZS,WS,DS,r_i,c,gamma0,eta);
            if IsSampleEta
                q_k = betarnd(sum(WSZS,1),eta*V);
                Lv = CRT_sum_mex(nonzeros(WSZS),eta);
                eta = gamrnd(0.01+Lv,1./(0.01-V*sum(log(max(1-q_k,realmin)))));
            end
            %% Delete unsed atoms
            K=nnz(n_dot_k);
            [kk,kki,kkj] = unique(ZS);
            ZS=kkj;
            DSZS=DSZS(:,kk);
            WSZS=WSZS(:,kk);
            n_dot_k=n_dot_k(kk);
            %% Sample model parameters
            gamma0 = gamrnd(a0+K,1./(b0+psi(c+sum(r_i))-psi(c)));
            p_k = betarnd(n_dot_k,c+sum(r_i));
            p_star = logBeta_rnd(1,gamma0,c+sum(r_i));
            L_i = CRT_sum_mex_matrix(sparse(DSZS'),r_i);
            sumlogpk = sum(log(max(1-p_k,realmin)));
            r_i = gamrnd(a0 + L_i, 1./(-sumlogpk + p_star + b0));
            if 0
                c = sample_c(c,p_k,p_star,r_i,gamma0,DSZS',1,1)
            else
                %% Sample c using griddy-Gibbs
                ccc = 0.01:0.01:0.99;
                c = ccc./(1-ccc);
                % c = 0.001:0.001:1;
                temp = -gamma0*(psi(sum(r_i)+c)-psi(c)) +...
                    K*gammaln(c+sum(r_i)) - sum(gammaln(bsxfun(@plus,c,sum(r_i)+n_dot_k)),1) ;
                temp = exp(temp - max(temp));
                temp(isnan(temp))=0;
                cdf =cumsum(temp);
                c = c(sum(rand(1)*cdf(end)>cdf)+1);
            end
            
            if (iter>burnin && mod(iter,CollectionStep)==0 && percentage<1) || iter==maxIter
                Phi = dirrnd([WSZS,zeros(V,K_star)] + eta);
                Theta = bsxfun(@times,randg([DSZS';zeros(K_star,N)] + r_i(ones(1,K+K_star),:)), [p_k;1-exp(-p_star/K_star)*ones(K_star,1)]);
            end
    end
    
    ave.K(iter) = nnz(sum(WSZS,1));
    ave.gamma0(iter) = gamma0;
    ave.eta(iter) = eta;
    if iter>burnin && mod(iter,CollectionStep)==0 && percentage<1
        X1 = Mult_Sparse(Xmask,Phi,Theta);
        X1sum = sum(Theta,1);
        ave.PhiTheta = ave.PhiTheta + X1;
        ave.PhiThetaSum = ave.PhiThetaSum + X1sum;
        ave.Count = ave.Count+1;
        X1 = bsxfun(@rdivide, X1,X1sum);
        loglike(end+1)=sum(Xtest(Yflagtest).*log(X1(Yflagtest)))/sum(Xtest(:));
        loglikeTrain(end+1)=sum(Xtrain(Yflagtrain).*log(X1(Yflagtrain)))/sum(Xtrain(:));
        
        X1 = ave.PhiTheta/ave.Count;
        X1sum = ave.PhiThetaSum/ave.Count;
        X1= bsxfun(@rdivide, X1,X1sum);
        ave.loglike(end+1) = sum(Xtest(Yflagtest).*log(X1(Yflagtest)))/sum(Xtest(:));
        
        clear X1;
    end
%     
%     if mod(iter,10)==0
%         toc
%         if iter>burnin
%             disp(full([iter/100,loglikeTrain(end),loglike(end),ave.K(end),ave.loglike(end)]));
%         else
%             disp(full(iter/100));
%         end
%         tic;
%     end
    
    if IsPlot && mod(iter,10)==0
        
        [temp, Thetadex] = sort(sum(WSZS,1),'descend');
        %[temp, Thetadex] = sort(n_dot_k,'descend');
        switch sampler
            case 'NB_collapsed'
                subplot(2,2,1);plot((r_k(Thetadex)),'.'); title('r_k')
                subplot(2,2,2);plot(p,'*');title('p')
                subplot(2,2,3);plot(ave.K(1:iter)); title('K')
                subplot(2,2,4);plot(ave.gamma0(1:iter)); %title('gamma0')
            case 'Beta_NB_collapsed'
                subplot(2,2,1);plot((p_k(Thetadex)),'.'); title('p_k')
                subplot(2,2,2);plot(r_i);title('r_i')
                subplot(2,2,3);plot(ave.K(1:iter)); title('K')
                subplot(2,2,4);plot(ave.gamma0(1:iter)); %title('gamma0')
                title(num2str([c,eta]));
            otherwise
                subplot(2,2,1);plot((r_k(Thetadex)),'.');title('r_k')
                subplot(2,2,2);plot(p_i);title('p_i')
                subplot(2,2,3);plot(ave.K(1:iter));title('K')
                subplot(2,2,4);plot(ave.gamma0(1:iter));
                title(num2str([c,eta]));
        end
        
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
if strcmp(sampler,'Beta_NB_collapsed')
    output.r_k = p_k;
    output.r_star = p_star;
end
ave.LogLikeTrain = loglikeTrain;
ave.LogLikeTest = loglike;
if percentage<1
    ave.Perplexity = exp(-ave.loglike(end));
end
