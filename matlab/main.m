clear
clc

% num_UE = ;
N1 = 16;
N2 = 16;
N = N1*N2;
num_os = 1;
f_c = 100e9;
ue_height = 1.5;
bs_height = 2.8;
pow_tx = 2;
pow_n = 0.1;
rho = 0.5;
thres = 0.01;

dist_range = [10, 10];

phi_range = [30, 40]*pi/180;
theta_range = [30, 40]*pi/180;

[x_min, y_min, z_max] = xyz_transform(dist_range(1),phi_range(1),theta_range(1));
[x_max, y_max, z_min] = xyz_transform(dist_range(2),phi_range(2),theta_range(2));
x_range = [x_min, x_max]';
y_range = [y_min, y_max]';
z_range = [z_min, z_max]';

% bs_loc = [0, 0, 0];

z_axis = [0, 0, 1];
y_axis = [0, 1, 0];
x_axis = [1, 0, 0];


deta_dist_err = 1.96e-2;
deta_phi_err = 1.08*pi/180;
deta_theta_err = 0.737*pi/180;

effd_dist_err = 3.61e-1;
effd_phi_err = 1.96*pi/180;
effd_theta_err = 1.10*pi/180;


ue_est_deta_loc_save = [];

ues = importdata('ues.mat');
ue_loc = ues.position;
ue_block = ues.blockage;
num_UE = size(ue_loc, 2);

sbss = importdata('sbss.mat');
sbs_loc = sbss.position; 
num_SBS = size(sbs_loc, 2);

for trial=1:100
    trial
    channel = zeros(N,num_UE,num_SBS);
    for bs=1:num_SBS
        % ue_loc =
        channel(:, :, bs) = channel_generation(bs_loc(bs), ue_loc, N1, N2, x_axis, y_axis, z_axis, f_c);
    end
        % 5G NR
    [~, ~, NR, R_NR] = beamforming_NR(channel, N1, N2, num_os, pow_tx, pow_n);

    % [NR, R_NR] = CVMPA(channel, N1, N2, num_os, pow_tx, pow_n);
    
    % plot(r_err,R_save)
end
