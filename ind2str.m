function str = ind2str(ind,voc)
% Convert from word-index-matrix to sentence
%
% ind - m*n double array
% voc - s*2 cell array
%       first column strings
%       second column numbers
% str - m*1 cell array containing strings

str=cell(size(ind,1),1);
for i = 1:size(ind,1)
    str(i) = voc(ind(i,1),1);
    for j = 2:size(ind,2)
        if ind(i,j) == 0 
            break;end
        str(i) = strcat(str(i),{' '},voc(ind(i,j),1));
    end
end
end

