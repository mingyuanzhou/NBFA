addpath('/Users/zhoum/Box Sync/Zhou_code_box/NBP_Matrix_code/NBP042014/20news-bydate')
load train.data
load test.data
test(:,1)=max(train(:,1))+test(:,1);
train_test = [train;test];
Xtrain =sparse(train_test(:,2),train_test(:,1),train_test(:,3));
load train.label
load test.label
GroundInd = [train;test];

if IsBinaryClassificaiton
    if dataset==1
        %alt.atheism 1 vs talk.religion.misc 20
        train(train>1&train<20)=[];
        test(test>1&test<20)=[];
        dex=GroundInd>1&GroundInd<20;
        Xtrain(:,dex)=[];
        GroundInd(dex)=[];
    elseif dataset==2    
        %talk talk.politics.guns 17 vs talk.politics.mideast 18
        train(train<17|train>18)=[];
        test(test<17|test>18)=[];
        dex=GroundInd<17|GroundInd>18;
        Xtrain(:,dex)=[];
        GroundInd(dex)=[];
        % GroundInd = GroundInd - 16;
    elseif dataset==3
        % % %comp comp.sys.ibm.pc.hardware 4 vs comp.sys.mac.hardware 5
        train(train<4|train>5)=[];
        test(test<4|test>5)=[];
        dex=GroundInd<4|GroundInd>5;
        Xtrain(:,dex)=[];
        GroundInd(dex)=[];
        % GroundInd = GroundInd - 3;
    elseif dataset==4    
        %sci sci.electronics 13 vs sci.med 14
        train(train<13|train>14)=[];
        test(test<13|test>14)=[];
        dex=GroundInd<13|GroundInd>14;
        Xtrain(:,dex)=[];
        GroundInd(dex)=[];
        % GroundInd = GroundInd - 16;
    end
    
end
if 1
    fid=fopen('stop-word-list.txt');
    stopwords=textscan(fid, '%s');
    stopwords=stopwords{1};
    fclose(fid);
    fid = fopen('vocabulary.txt');
    
    WO = textscan(fid, '%s');
    fclose(fid);
    WO = WO{1};
    dex = true(length(WO),1);
    for i=1:length(WO)
        dex(i)=1-nnz(strcmp(WO(i),stopwords));
    end
    
    WO=WO(dex);
    Xtrain=Xtrain(dex,:);
    
end

%Xtrain=Xtrain(sum(Xtrain,2)>0,:);
WO = WO(sum(Xtrain,2)>=5);
Xtrain=Xtrain(sum(Xtrain,2)>=5,:);


[~,~,GroundInd] = unique(GroundInd,'stable');
NumCategory = GroundInd(end);


X=cell(NumCategory,1);
Xlabel = cell(NumCategory,1);
%Kdex=cell(1,NumCategory);
dexTest=false(size(Xtrain,2),1);
for i=1:NumCategory
    dex = find(GroundInd==i);
    X{i} = Xtrain(:,dex(dex<=length(train)));
    Xlabel{i} = GroundInd(dex(dex<=length(train)));
    % dexTest = [dexTest;dex((round(Len*Percentage/100)+1):length(dex))];
    dexTest(dex(dex>length(train)))=true;
end
DataType = 'Count';


Xtest = Xtrain;
Xtestlabel = GroundInd;

clear train