clear all
clc

%% Export matfile to csv
FileData = load('tpcda19_02_dataset.mat');
csvwrite('tpcda19_02_dataset.csv', [FileData.u1,  FileData.u2,  FileData.u3,...
          FileData.u4,  FileData.u5,  FileData.u6,  FileData.u7,  FileData.y]);

