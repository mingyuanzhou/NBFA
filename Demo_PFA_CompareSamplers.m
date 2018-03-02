%% Training
clear all
addpath('data')
%dataset=3;
for dataset=1 %1,2]
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
    Collection = 500;
    maxIter=Burnin+Collection;
    
    percentage=1.5; %101;
    CollectionStep = 500;
    
    models = {
        'PFA', %GNBP multinomial topic model
        };
    
    samplers = {
        'Beta_NB_collapsed',
        'HDP_DirectAssignment',
        'Gamma_NB_partially_collapsed', %Partially collapsed (the shared Gamma process is not collapsed)
        'Gamma_NB_Kadaptive', %Blocked Gibbs sampler, adaptive truncating the number of topics
        'Gamma_NB_fully_collapsed', %Fully collapsed (the shared Gamma process is collapsed)
        'Gamma_NB', %Blocked Gibbs sampler
        'Gamma_NB_collapsed_fixK', %Collapsed Sampler, fix K
        };
    
    model = models{1}
    TIME=[];
    for K_init=400 %[0,500] %[10,400] %[10,400]
        for i=7:7%:4
            sampler = samplers{i}
            %[ave,Phi,Theta,r_k,p_i,WSZS]=NBFA_DirMultTM_blocked(X,percentage,model,sampler,eta,state,Burnin,Collection,CollectionStep);
            figure
            tic;
            [ave,Phi,Theta,output]=PFA_MultTM(X,model,sampler,eta,percentage,K_init,state,Burnin,Collection,CollectionStep,false,true);
            TIME(end+1)=toc;
            %save([datasetname,'_Compare_', model,'_', sampler,'_',num2str(K_init),'.mat'],'ave');
            figure;semilogy(ave.K);title([sampler,', K_init=' num2str(K_init)])
        end
    end
    
    if 0
        datasetname = 'JACM'
        datasetname = 'PsyReview'
        datasetname = 'NIPS'
        figure
        count=0;
        colors={'-','--',':','-.'}
        for K_init=[0,500]
            count=count+1;
            subplot(1,2,count)
            count1=0;
            for i= [1,2,3,4]
                count1=count1+1;
                sampler = samplers{i}
                %[ave,Phi,Theta,r_k,p_i,WSZS]=NBFA_DirMultTM_blocked(X,percentage,model,sampler,eta,state,Burnin,Collection,CollectionStep);
                load([datasetname,'_Compare_', model,'_', sampler,'_',num2str(K_init),'.mat']);
                plot(ave.K,colors{count1},'LineWidth',1.5); hold on
                %title([sampler,', K_init=' num2str(K_init)])
            end
            xlabel('Gibbs sampling iteration')
            ylabel('Number of active topics K^+')
            h_legend=legend('BNBP collapsed Gibbs sampler', 'HDP direct assignment', 'GNBP collapsed Gibbs sampler', 'GNBP blocked Gibbs sampler')
            
            set(h_legend,'FontSize',11);
        end
        %     subplot(1,2,1);title('(a) K\_init = 10'); ylim([10,150])
        %     subplot(1,2,2);title('(b) K\_init = 400');ylim([50,1000])
        %
        %     subplot(1,2,1);title('(a) K\_init = 10'); ylim([10,1000])
        %     subplot(1,2,2);title('(b) K\_init = 400');ylim([10,1000])
        if 0
            set(gcf,'papersize',[30 10])
            saveas(gcf,'NIPS1_Trace.pdf')
        end
    end
end