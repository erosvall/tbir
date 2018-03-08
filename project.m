%% Load datase
file = dataset2cell(dataset('file','qa.894.raw.train.txt','ReadVarNames',false));
questions = file(3:2:end,1);
answers = file(4:2:end,1);

%% Transform to numerical representation


%% Create and train LSTM Network


%% Output query representation


%% Output Answer to query