function [w,b] = perceptron_ch2_example2_1
%% the training data
trainData = struct('x',[3,3;4,3;1,1],'y',[1;1;-1]);
N = size(trainData.x,2);
L = size(trainData.x,1);
%% initialize w and b
w = zeros(N,1);
b = 0;
modelValue = 0;

%% remove the wrong categorize by make y_i*(w*x_i+b)>0, i = 1,2,...,N
while modelValue<=0
    [ind, modelValue] = checkModel(trainData,w,b);
    if ind<=L
        %% if y_i*(w*x_i+b)<=0, w = w+y_i*x_i
        %  b= b+y_i 
        w = w+trainData.y(ind).*trainData.x(ind,:).';
        b = b+trainData.y(ind);
    else
        break;
    end
end

function [ind, tmp] = checkModel(dataStruct,w,b)
%% calculate the value y_i*(w*x_i+b), and check its sign, if sign is -1
% return to main
L = size(dataStruct.x,1);
ind = L+1;
for ii = 1:L
    tmp = (w'*dataStruct.x(ii,:).'+b)*dataStruct.y(ii);
    if tmp<=0
        ind = ii;
        break;
    end
end