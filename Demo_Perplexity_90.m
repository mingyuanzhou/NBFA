%% Demo
%dataset = 1; %2,3
%eta = 0.05; %[0.005 0.01 0.02 0.05 0.1 0.25 0.5] 
%state = 1 %2, 3, 4, 5
% i = 1; %2 3
% percentage=0.9;  %The ratio of word tokens used for training
%state is the random seed
%eta is the topic Dirichlet smoothing parameter
%%

models = {
    'hGNBP_DirMultTM', %hierachical GNBP Dirichlet-multinomial topic model (NBFA)
    'GNBP_DCMLDA', %GNBP Dirichlet compound multinomial LDA
    'PFA', %Multinomial topic model
    };

model = models{i}

addpath('data')
switch dataset
    case 1
        disp('psychreview')
        load 'bagofwords_psychreview';
        load 'words_psychreview';
        %This dataset is available at
        %http://psiexp.ss.uci.edu/research/programs_data/toolbox.htm
        X = sparse(WS,DS,1,max(WS),max(DS));
        dex = (sum(X>0,2)<5);
        X = X(~dex,:);
        WO = WO(~dex);
        datasetname = 'PsyReview';
    case 2
        disp('JACM')
        %The JACM dataset is available at
        %http://www.cs.princeton.edu/~blei/downloads/
        [X,WO] = InitJACM;
        %load JACM_MZ.mat
        dex=(sum(X>0,2)<=0);
        WO = WO(~dex);
        datasetname = 'JACM';
    case 3
        load nips12raw_str602
        X = counts;
        datasetname = 'NIPS';
    case 4
        IsBinaryClassificaiton=false;
        read20news;
        X=Xtrain;
        datasetname = '20news';
        
        %         IsBinaryClassificaiton=true;
        %         dataset=3;
        %         read20news;
        %         X=Xtrain;
        %         datasetname = '20news3';
        %         size(X)
end



%% Training

Burnin = 2500;
Collection = 2500;
maxIter=Burnin+Collection;



CollectionStep = 5; %Collect one sample per CollectionStep iterations

K_init=400;  %K_init is the initial number of topics

IsPlot=true;
if strcmp(model,'PFA')
    sampler = 'Gamma_NB_partially_collapsed';
    figure
    tic;
    [ave,Phi,Theta,output]=PFA_MultTM(X,model,sampler,eta,percentage,K_init,state,Burnin,Collection,CollectionStep,false,IsPlot);
    ave.TIME = toc;
    ave.PhiTheta=[];
    ave.PhiThetaSum=[];
    save([datasetname,'_Perplexity_', model,'_', sampler,'_K0',num2str(K_init),'_state',num2str(state),'_eta',num2str(eta*1000),'_ratio',num2str(percentage*100),'.mat'],'ave');
    figure;semilogy(ave.K);title([sampler,', K_init=' num2str(K_init)])
else
    sampler = 'blocked_Gibbs_NB';
    figure
    tic;
    [ave,Phi,Theta,output]=NBFA_DirMultTM(X,model,sampler,eta,percentage,K_init,state,Burnin,Collection,CollectionStep,false,IsPlot);
    ave.TIME = toc;
    ave.PhiTheta=[];
    ave.PhiThetaSum=[];
    save([datasetname,'_Perplexity_', model,'_', sampler,'_K0',num2str(K_init),'_state',num2str(state),'_eta',num2str(eta*1000),'_ratio',num2str(percentage*100),'.mat'],'ave');
    figure;semilogy(ave.K);title([sampler,', K_init=' num2str(K_init)])
end
