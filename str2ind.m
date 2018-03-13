function [ind,voc] = str2ind(str,voc)
% Convert from sentence to word-index-matrix
% while adding new words to the vocabulary
%
% str - m*1 cell array containing strings
% voc - s*2 cell array
%       first column strings
%       second column numbers
% ind - m*n double array

ind = zeros(size(str));
for i = 1:size(str,1)
    words = strsplit(erase(char(str(i,1)),','));
    for j = 1:size(words,2)
        index = find(strcmp(voc(:,1),words(j)));
        if index > 0
            voc(index,2) = num2cell(cell2mat(voc(index,2))+1);
        else
            index = size(voc,1)+1;
            voc(index,1:2) = [words(j) 1]; 
        end
        ind(i,j) = index;
    end
end
end
