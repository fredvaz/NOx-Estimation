clear all
clc

%% Export matfile to csv
load('../data/tpcda19_02_dataset.mat');

x = [u1 u2 u3 u4 u5 u6 u7];


%%
csvwrite('tpcda19_02_dataset.csv', [FileData.u1,  FileData.u2,  FileData.u3,...
          FileData.u4,  FileData.u5,  FileData.u6,  FileData.u7,  FileData.y]);

