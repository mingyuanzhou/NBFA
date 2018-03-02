%% Demo
% dataset = 0 %1, 2, 3,4
% % dataset=0: 20 newsgroups
% % dataset=1: alt.atheism 1 vs talk.religion.misc 20
% % dataset=2: talk talk.politics.guns 17 vs talk.politics.mideast 18
% % dataset=3: comp comp.sys.ibm.pc.hardware 4 vs comp.sys.mac.hardware 5
% % dataset=4: sci sci.electronics 13 vs sci.med 14

%model = 'PFA'; sampler = 'Gamma_NB'; %K is fixed
%model = 'PFA'; sampler = 'Gamma_NB_collapsed_fixK'; %K is fixed
%model = 'hGNBP_DirMultTM'; sampler = 'blocked_Gibbs_NB_Truncated'; %K is fixed

%model = 'PFA'; sampler = 'Gamma_NB_partially_collapsed'; %K is adaptive
%model = 'hGNBP_DirMultTM'; sampler = 'blocked_Gibbs_NB'; %K is adaptive

%trial = 1 %2, 3, 4, 5, 6, 7, 8....
IsSampleEta = true;
%savepath = 'results/';
savepath = 'results1/';


Burnin = 500*3;
Collection = 500*2;
maxIter=Burnin+Collection;
TestBurnin=250*2;
TestCollection=250*2;

addpath('liblinear-2.1/matlab/') %download libliear package and add its path
addpath('data/')

if dataset>0
    IsBinaryClassificaiton = true;
else
    IsBinaryClassificaiton = false;
end

%read20news
read20news_v1

figure

rng(trial,'twister');
%% Training
X = Xtrain(:,~dexTest);
percentage=101;
burnin = Burnin; collection = Collection;
state = trial; CollectionStep = Collection;

if strcmp(model,'PFA')
    tic;
    [ave,Phi,Theta,output]=PFA_MultTM(X,model,sampler,eta,percentage,K_init,state,Burnin,Collection,CollectionStep,IsSampleEta);
    TIME.Train = toc;
    WSZS=output.WSZS;
    Phi=SamplePhi(WSZS,output.eta,true);
    r_k=output.r_k;
    r_star=output.r_star;
    tic;
    [AveThetaTest,AveThetaTest_Prob] = ...
        PFA_MultTM_Test(Xtrain,Phi,r_k,r_star,sampler,TestBurnin,TestCollection);
    TIME.Test = toc;
else
    tic;
    [ave,Phi,Theta,output]=NBFA_DirMultTM(X,model,sampler,eta,percentage,K_init,state,Burnin,Collection,CollectionStep,IsSampleEta);
    TIME.Train=toc;
    WSZS=output.WSZS;
    Phi=SamplePhi(WSZS,output.eta,true);
    r_k=output.r_k;
    r_star=output.r_star;
    tic;
    [AveThetaTest,AveThetaTest_Prob] = NBFA_DirMultTM_test(Xtrain,Phi,r_k,r_star,model,sampler,TestBurnin,TestCollection);
    TIME.Test=toc;
end


KKK=ave.K;
ETA=ave.eta;

%% Testing
for ii=2:-1:1
    if ii==1
        TrainX=AveThetaTest_Prob(:,~dexTest);
        TestX=AveThetaTest_Prob(:,dexTest);
    else
        AveThetaTest = bsxfun(@rdivide,AveThetaTest,max(sum(AveThetaTest,1),realmin));
        TrainX=AveThetaTest(:,~dexTest);
        TestX=AveThetaTest(:,dexTest);
    end
    TrainY = Xtestlabel(~dexTest);
    TestY = Xtestlabel(dexTest);
    %save temp.mat
    tic
    
    
    option = ['-s 0 -c 1 -q'];
    model_classification = train(double(TrainY), sparse(TrainX'), option);
    [predicted_label, accuracy, prob_estimates] = predict(double(TestY), sparse(TestX'), model_classification, ' -b 1');
    if ii==1
        Accuracies.Default = accuracy(1)
    else
        Accuracies.Default2 = accuracy(1)
    end
    
    if strcmp(sampler,'Gamma_NB') || strcmp(sampler,'blocked_Gibbs_NB_Truncated') || strcmp(sampler,'Gamma_NB_collapsed_fixK')
        save([savepath,'20news_Stampede_ClassfyTruncate_K_',num2str(dataset),'_', model,'_', sampler,'_K0',num2str(K_init),'_trial',num2str(trial),'.mat'],'Accuracies','KKK','ETA','TIME');
    else
        save([savepath,'20news_Stampede_ClassfyLearn_K_eta_',num2str(dataset),'_', model,'_', sampler,'_K0',num2str(K_init),'_trial',num2str(trial),'.mat'],'Accuracies','KKK','ETA','TIME');
    end
    
    if 1
        %cross validate the regulariation parameter for the L2 regularized
        %logistic regression model
        CC=2.^(-10:15);
        ModelOut=zeros(1,length(CC));
        for ij=1:length(CC)
            tic
            ModelOut(ij) = train(double(TrainY), sparse(TrainX'), ['-s 0 -c ', num2str(CC(ij)), ' -v 5 -q ']);
            toc
        end
        [~,maxdex]=max(ModelOut);
        num2str(CC(maxdex))
        option = ['-s 0 -c ', num2str(CC(maxdex)), ' -q'];
        model_classification = train(double(TrainY), sparse(TrainX'), option);
        
        toc
    else
        tic
        c_class = train(double(TrainY), sparse(TrainX'), ['-C -s 0 -c 2^(-10) -v 5 -q ']);
        option = ['-s 0 -c ', num2str(c_class(1)), ' -q'];
        model_classification = train(double(TrainY), sparse(TrainX'), option);
        toc
    end
    [predicted_label, accuracy, prob_estimates] = predict(double(TestY), sparse(TestX'), model_classification, ' -b 1');
    if ii==1
        Accuracies.CrossValidated = accuracy(1);
    else
        Accuracies.CrossValidated2 = accuracy(1);
    end
end

Accuracies
%KKK(trial)=size(Phi,2);


if strcmp(sampler,'Gamma_NB') || strcmp(sampler,'blocked_Gibbs_NB_Truncated') || strcmp(sampler,'Gamma_NB_collapsed_fixK')
    save([savepath,'20news_Stampede_ClassfyTruncate_K_',num2str(dataset),'_', model,'_', sampler,'_K0',num2str(K_init),'_trial',num2str(trial),'.mat'],'Accuracies','KKK','ETA','TIME');
else
    save([savepath,'20news_Stampede_ClassfyLearn_K_eta_',num2str(dataset),'_', model,'_', sampler,'_K0',num2str(K_init),'_trial',num2str(trial),'.mat'],'Accuracies','KKK','ETA','TIME');
end
