function [alpha,b] = perceptron_ch2_example2_2
%% the training data
trainData = struct('x',[3,3;4,3;1,1],'y',[1;1;-1]);
L = size(trainData.x,1);
%% initialize alpha and b
alpha = zeros(L,1);
b = 0;
eta = 1;
modelValue = 0;
%% remove the wrong categorize by make y_i*[sum(alpha.y.*x_j.*x_i')+b]>0, i = 1,2,...,N
while modelValue<=0
    [ind, modelValue] = checkModel(trainData,alpha,b);
    if ind<=L
        %% if y_i*[sum(alpha.y.*x_j.*x_i')+b]<=0, alpha_i = alpha_i+eta
        %  b= b+y_i*eta
        alpha(ind) = alpha(ind)+eta;
        b = b+eta*trainData.y(ind);
    else
        break;
    end
end


function [ind, tmp] = checkModel(dataStruct,alpha,b)
%% check value y_i*[sum(alpha.y.*x_j.*x_i')+b]
L = size(dataStruct.x,1);
ind = L+1;
mGram = dataStruct.x*dataStruct.x';
for ii = 1:L
    tmp = dataStruct.y(ii)*(b+sum(alpha.*dataStruct.y.*mGram(:,ii)));
    if tmp<=0
        ind = ii;
        break;
    end
end