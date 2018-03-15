%% Load datase
clear;
% file = dataset2cell(dataset('file','qa.894.raw.train.txt','ReadVarNames',false));
% questions = file(2:2:end,1);
% answers = file(3:2:end,1);
% 
% %% Transform to numerical representation
% words = {};
% for i = 2:size(file,1)
%     words = [words strsplit(erase(char(file(i,1)),','))];
% end
% words = sort(words);
% voc(:,1) = unique(words)';
% voc(:,2) = cellfun(@(x) sum(ismember(words,x)),voc(:,1),'un',0);
% %%
% input = zeros(size(questions,1),31);
% target = zeros(size(answers,1),7);
% for i = 1:size(questions,1)
%     qwords = strsplit(char(questions(i,1)));
%     indices = cellfun(@(x) find(strcmp(voc(:,1),x)),qwords);
%     input(i,1:size(indices,2)) = indices;
% end
% for i = 1:size(answers,1)
%     qwords = strsplit(char(answers(i,1)),', ');
%     indices = cellfun(@(x) find(strcmp(voc(:,1),x)),qwords);
%     target(i,1:size(indices,2)) = indices;
% end
load('traindata.mat');
%% Create and train LSTM Network
clc
inputSize = 1;
outputSize = 10; % Sort of Overfitting parameter. Higher allows for more complex models, but with overfitting
outputMode = 'sequence'; % otherwise 'sequence'/'last'
numClasses = 500;
maxEpochs = 5;
miniBatchSize = 31;
shuffle = 'never';

options = trainingOptions('sgdm', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle', shuffle);

layers = [...
    sequenceInputLayer(inputSize)
    lstmLayer(outputSize,'OutputMode',outputMode)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
X = num2cell(input,2);
Y = num2cell(categorical(target),2);
net = trainNetwork(X,Y,layers,options);

%% Output query representation


%% Output Answer to query