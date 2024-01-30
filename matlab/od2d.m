sbs_file = load('D:\Unity\dataset\solo\sequence.0\sbs.mat');
sbs_positions = sbs_file.position;

obj_file = load('D:\Unity\dataset\solo\sequence.0\objects.mat');
ue_position = obj_file.position;
ue_los= obj_file.los;

fileNames = {'camera.mat', 'camera_0.mat', 'camera_1.mat', 'camera_2.mat'};

aod = zeros(length(fileNames), size(ue_los, 1), 2) - 1;
for i = 1:length(fileNames)
    data = load(fullfile('D:\Unity\dataset\solo\sequence.0\', fileNames{i}));
    % aod{i} = data.aod;
    aod(i, :, :) = [i, i; i, i; i, i; i, i; i, i;];
end



