
%% Load datase
clear;
file = dataset2cell(dataset('file','qa.894.raw.train.txt','ReadVarNames',false));
file = file(1:10);
questions = file(2:2:end,1);
answers = file(3:2:end,1);

%% Transform to numerical representation
words = {};
for i = 2:size(file,1)
    words = [words strsplit(erase(char(file(i,1)),','))];
end
words = sort(words);
voc(:,1) = unique(words)';
voc(:,2) = cellfun(@(x) sum(ismember(words,x)),voc(:,1),'un',0);

%%
input = zeros(size(questions,1),31);
target = zeros(size(answers,1),7);
for i = 1:size(questions,1)
    qwords = strsplit(char(questions(i,1)));
    indices = cellfun(@(x) find(strcmp(voc(:,1),x)),qwords);
    input(i,1:size(indices,2)) = indices;
end
for i = 1:size(answers,1)
    qwords = strsplit(char(answers(i,1)),', ');
    indices = cellfun(@(x) find(strcmp(voc(:,1),x)),qwords);
    target(i,1:size(indices,2)) = indices;
end
%% 
str = questions(3:5,1);
ind = str2ind(str,voc)
ind2str(ind,voc)