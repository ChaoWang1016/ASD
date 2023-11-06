%%
close all;
clear all;
clc;
warning('off');
addpath(genpath(pwd));
dbstop if error
%% Parameters
K = 180;
P = [0.5,0.5;0.42,0.58;0.58,0.42]; %can with key
NP = size(P,1);
NB = 100;            % integral
NF = 180;            % #features
A=3000; u=70; d=20;  % Gaussian
SZI =[224,224];      % image size
L=1000; T=10000;     % L: the number of training data pairs; T: the total number of training and testing data pairs.
seed=666; rat=50;    % key for (n,m)
%% Get basis functions
BF = cell(NP,2);
for i=1:NP
    [BF_x,BF_y]=KM_BF(SZI,K,P(i,1),P(i,2));
    BF{i,1}=BF_x; BF{i,2}=BF_y;
end
%% Get random map
rng(seed);Rand=randi([1,rat],K+1,K+1);Rand=Rand>1;
%% Get weights
idx=1:NB; W=A*exp(-(idx-u).^2/(2*d^2)); W(W<eps*max(W(:))) = 0; 
sumW = sum(W(:)); if sumW ~= 0, W  = W/sumW; end; pz = W>0.001;
PZ=ones(NP,NB); for i=1:NP, PZ(i,:)=pz; end; PZ=logical(reshape(PZ',1,[]));
%% Training
labOI=zeros(L,1);labAI=ones(L,1);
fetOI=zeros(L,NB*NP);fetAI=zeros(L,NB*NP);
t1=tic;
for i=1:L
    %% Input %% 
    name=num2str(i);
    AI=imread(['image\mnist_attack_PGD\',name,'.png']);
    OI=imread(['image\mnist_original\',name,'.png']);
    %% Resize
    AI=imresize(AI,SZI);
    OI=imresize(OI,SZI);
    %% Color 2 gray
    if size(OI,3)==3; OI = rgb2gray(uint8(OI)); end
    if size(AI,3)==3; AI = rgb2gray(uint8(AI)); end
    %% Feature %%
    fetOI(i,:) = feature_extraction(OI,K,BF,NP,NB,W,Rand);
    fetAI(i,:) = feature_extraction(AI,K,BF,NP,NB,W,Rand);
end
%% SVM Training %%
fetOI=fetOI(:,PZ); fetAI=fetAI(:,PZ); 
figure; imagesc(fetOI); % x-label #feature, y-label #sample
figure; imagesc(fetAI); % x-label #feature, y-label #sample
trainLabels = num2str([labOI;labAI]);
trainData = [fetOI;fetAI];
idx = fscchi2(trainData,trainLabels);
Model = fitcsvm(trainData(:,idx(1:NF)),trainLabels,'OptimizeHyperparameters','all','HyperparameterOptimizationOptions', struct('UseParallel',true) );
trainPredLab = predict(Model,trainData(:,idx(1:NF)));
clc;
TraTime = toc(t1);
TraAcc = sum(trainPredLab==trainLabels)/size(trainLabels,1)*100;

%% Testing
turelabOI=zeros(T-L,1);turelabAI=ones(T-L,1);
predlabOI=zeros(T-L,1);predlabAI=zeros(T-L,1);
t2=tic;
parfor i=L+1:T 
    %% Input %%
    name=num2str(i);
    AI=imread(['image\mnist_attack_PGD\',name,'.png']);
    OI=imread(['image\mnist_original\',name,'.png']);
    %% Resize
    AI=imresize(AI,SZI);
    OI=imresize(OI,SZI);
    %% Color 2 gray
    if size(OI,3)==3; OI = rgb2gray(uint8(OI)); end
    if size(AI,3)==3; AI = rgb2gray(uint8(AI)); end
    %% Feature and SVM Testing %%
    fetperOI=feature_extraction(OI,K,BF,NP,NB,W,Rand);
    fetperAI=feature_extraction(AI,K,BF,NP,NB,W,Rand);
    fetperOI=fetperOI(:,PZ); fetperAI=fetperAI(:,PZ); 
    predlabOI(i-L,:) = str2double(predict(Model,fetperOI(:,idx(1:NF))));
    predlabAI(i-L,:) = str2double(predict(Model,fetperAI(:,idx(1:NF))));
end
TesTime = toc(t2);
predlab = [predlabOI;predlabAI];
turelab = [turelabOI;turelabAI];
TesAcc = sum(predlab==turelab)/size(turelab,1)*100;

TP = sum(predlabAI==turelabAI);
FN = sum(predlabAI~=turelabAI);
FP = sum(predlabOI~=turelabOI);
Recall = (TP/(TP+FN))*100;
Precision = (TP/(TP+FP))*100;
F1 = 2*Precision*Recall/(Precision+Recall);

clc;
disp(table([TraTime;TesTime;TraAcc;TesAcc],'RowNames',{'Train Time';'Test Time';'Train Accuracy';'Test Accuracy'},'VariableNames',{'Value'}));
disp(table([TesAcc;Recall;Precision;F1],'RowNames',{'Accuracy';'Recall';'Precision';'F1'},'VariableNames',{'Value'}));
