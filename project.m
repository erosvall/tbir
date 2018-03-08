%% Load datase
file = dataset2cell(dataset('file','qa.894.raw.train.txt','ReadVarNames',false));
questions = file(2:2:end,1);
answers = file(3:2:end,1);

%% Transform to numerical representation


%% Create and train LSTM Network
inputSize = 'auto';
outputSize = 100; % Sort of Overfitting parameter. Higher allows for more complex models, but with overfitting
outputMode = 'last'; % otherwise 'sequence'
numClasses = 37;
maxEpochs = 50;
miniBatchSize = 27;
shuffle = 'never';

options = trainingOptions('sgdm', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle', shuffle);

layers = [
    sequenceInputLayer(inputSize)
    lstmLayer(outputSize,'OutputMode',outputMode)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

net = trainNetwork(X,Y,layers,options);



%% Output query representation


%% Output Answer to query