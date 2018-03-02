function Call_Classify_TruncateK(i1,i2,i3,i4)

dataset = i1

K_init_all = [5,10,25,50,100,200,400,600,800,1000]; %  0.05   0.01 0.005 0.001];
K_init = K_init_all(i2);


switch i3
    case 1
        %model = 'hGNBP_DirMultTM'
        %sampler = 'blocked_Gibbs_NB'
        model = 'hGNBP_DirMultTM'
        sampler = 'blocked_Gibbs_NB_Truncated'
    case 2
        %model = 'PFA'
        %sampler = 'Gamma_NB_partially_collapsed'
        model = 'PFA'
        sampler = 'Gamma_NB'
    case 3
        model = 'PFA'
        sampler = 'Beta_NB_collapsed'
    case 4
        model = 'PFA'
        sampler = 'Gamma_NB_collapsed_fixK'
end

eta = 0.05;

trial = i4;

%maxTrial=8;

IsSampleEta = true;

Demo_FeatureExtract_TruncateK


if 0
    core = 'Classify_TruncateK';
    
    submit=[];
    for i1=[0,1,3,4]
        if i1==0
            RunTime = 24;
        else
            RunTime = 12;
        end
        for i2=1:10
            %for dataset=[0,1,3,4]
            corejob=[core,'_dataset',num2str(i1),'_Kinit',num2str(i2)];
            fid = fopen([corejob,'.q'],'W');
            TaskNum=0;
            %for i1 = dataset
            for i3=[1,2,4]
                for i4 = 1:8
                    TaskNum = TaskNum+1;
                    fprintf(fid,'matlab  -nodisplay -nosplash -nodesktop -r "Call_%s(%d,%d,%d,%d)" -logfile %s_%d_%d_%d_%d.txt\n', core,i1,i2,i3,i4, core,i1,i2,i3,i4);
                end
            end
            fclose(fid);
            filename=jobs_Lonestar(corejob,TaskNum,RunTime);
            submit=[submit,'qsub ',filename,'; '];
        end
        %end
        
        %jobs_Stampede(corejob,TaskNum);
    end
    
    
end




if 0
    serverpath = 'server/';
    addpath(serverpath)
    core = 'Classify_TruncateK';
    
    submit=[];
    for i1=[0,1,3,4]
        if i1==0
            RunTime = 48;
        else
            RunTime = 24;
        end
        for i2=1:10
            %for dataset=[0,1,3,4]
            corejob=[serverpath,core,'_dataset',num2str(i1),'_Kinit',num2str(i2)];
            fid = fopen([corejob,'.q'],'W');
            TaskNum=0;
            %for i1 = dataset
            for i3=[1,2,4]
                for i4 = 1:12
                    TaskNum = TaskNum+1;
                    fprintf(fid,'matlab  -nodisplay -nosplash -nodesktop -r "Call_%s(%d,%d,%d,%d)" -logfile %s_%d_%d_%d_%d.txt\n', core,i1,i2,i3,i4, core,i1,i2,i3,i4);
                end
            end
            fclose(fid);
            %filename=jobs_Lonestar(corejob,TaskNum,RunTime);
            TaksPerNode = 12;
            filename=jobs_Stampede(corejob,TaskNum,RunTime,TaksPerNode);
            submit=[submit,'sbatch ',filename,'; '];
        end
        %end
        
        %
    end

end

