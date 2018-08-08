function OnInd = get_onInd( dbn, DropOutRate, strbm )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deep Neural Network:                                     %
% Copyright (C) 2013 Masayuki Tanaka. All rights reserved. %
%                    mtanaka@ctrl.titech.ac.jp             %                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if( ~exist('strbm', 'var') || isempty(strbm) )
	strbm = 1;
end

OnInd = cell(numel(dbn.rbm),1);

for n=1:numel(dbn.rbm)
    dimV = size(dbn.rbm{n}.W,1);
    if( n >= strbm )
        OnNum = round(dimV*DropOutRate(n));
        OnInd{n} = sort(randperm(dimV, OnNum));
    else
        OnInd{n} = 1:dimV;
    end
end