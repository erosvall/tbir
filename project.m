%% Load datase
clear;
file = dataset2cell(dataset('file','qa.894.raw.train.txt','ReadVarNames',false));
questions = file(2:2:end,1);
answers = file(3:2:end,1);

%% Transform to numerical representation
words = {};
for i = 2:size(file,1)
    cell = file(i,1);
    words = [words strsplit(erase(cell{1},','))];
end
words = sort(words);
voc(:,1) = unique(words)';
voc(:,2) = cellfun(@(x) sum(ismember(words,x)),voc(:,1),'un',0);
%%
    cell = questions(i,1);
    qwords = strsplit(cell{1});
    indices = cellfun(@(x) find(strcmp(voc(:,1),x)),qwords);
    input(i,1:size(indices,2)) = indices;
end
%% Create and train LSTM Network
inputSize = 12;
outputSize = 100; % Sort of Overfitting parameter. Higher allows for more complex models, but with overfitting
outputMode = 'last'; % otherwise 'sequence'
numClasses = 37;
maxEpochs = 50;
miniBatchSize = 31;
shuffle = 'never';

options = trainingOptions('sgdm', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle', shuffle);

    sequenceInputLayer(inputSize)
    lstmLayer(outputSize,'OutputMode',outputMode)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
net = trainNetwork(X,Y,layers,options);

%% Output query representation


%% Output Answer to query