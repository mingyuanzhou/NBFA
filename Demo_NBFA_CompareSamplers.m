%% Training
clear all
addpath('data')
for dataset=1
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
            
            %                 IsBinaryClassificaiton=true;
            %                 dataset=3;
            %         read20news;
            %         X=Xtrain;
            %         datasetname = '20news3';
            %         size(X)
    end
    
    state = 0;
    
    %% Training
    eta = 0.05;
    Burnin = 0;
    Collection = 100;
    maxIter=Burnin+Collection;
    
    
    
    percentage=1.5; %choose percentage >1.0 to use all words for training
    CollectionStep = 10;
    
    models = {
        'hGNBP_DirMultTM', %hierachical GNBP Dirichlet-multinomial topic model
        'GNBP_DCMLDA', %GNBP Dirichlet compound multinomial LDA
        };
    
    samplers = {
        'blocked_Gibbs_NB',  %blocked Gibbs sampling via the compound Poisson representation of the negative binmomial distritution
        'blocked_Gibbs_DirMult', % blocked Gibbs sampling
        'collapsed'
        };
    
    model = models{2}
    for K_init=0 %[0,500] %[10,400] %[10,400]
        for i=1:3 %1:3
            sampler = samplers{i}
            figure
            [ave,Phi,Theta,output]=NBFA_DirMultTM(X,model,sampler,eta,percentage,K_init,state,Burnin,Collection,CollectionStep,false,true);
           % save([datasetname,'_Compare_', model,'_', sampler,'_',num2str(K_init),'.mat'],'ave');
            figure;semilogy(ave.K);title([sampler,', K_init=' num2str(K_init)])
        end
    end
end
if 0
    datasetname = 'JACM'
    %datasetname = 'PsyReview'
    %datasetname = 'NIPS'
    figure
    count=0;
    colors={'-','--',':'}
    for K_init=[0,500]
        count=count+1;
        subplot(1,2,count)
        count1=0;
        for i=[2,3,1]
            count1=count1+1;
            sampler = samplers{i}
            %[ave,Phi,Theta,r_k,p_i,WSZS]=NBFA_DirMultTM_blocked(X,percentage,model,sampler,eta,state,Burnin,Collection,CollectionStep);
            %load([datasetname,'_Compare_', model,'_', sampler,'_',num2str(K_init),'_',num2str(percentage*100),'.mat']);
            load([datasetname,'_Compare_', model,'_', sampler,'_',num2str(K_init),'.mat']);
            semilogy(ave.K,colors{count1},'LineWidth',1.5); hold on
            %title([sampler,', K_init=' num2str(K_init)])
        end
        xlabel('Gibbs sampling iteration')
        ylabel('Number of active topics K^+')
        h_legend=legend('Blocked Gibbs sampler', 'Collapsed Gibbs sampler', 'Compound Poisson blocked Gibbs sampler')
        
        set(h_legend,'FontSize',11);
    end
        subplot(1,2,1);title('(a)'); % ylim([10,200])
        subplot(1,2,2);title('(b)');%ylim([50,1000])
    
    %     subplot(1,2,1);title('(a) K\_init = 10'); ylim([10,1000])
    %     subplot(1,2,2);title('(b) K\_init = 400');ylim([10,1000])
    if 0
        set(gcf,'papersize',[30 10])
        saveas(gcf,[datasetname,'_Trace.fig'])
        saveas(gcf,[datasetname,'_Trace.pdf'])
    end
end
