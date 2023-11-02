% MISO channel
function [beam_codebook_BS, id_beam_BS, beam_BS, R] = beamforming_NR(channel, num_ant_x_BS, num_ant_y_BS, R_min, V_max, num_os, pow_tx)

N = size(channel, 1);
K = size(channel, 2);
M = size(channel, 3);

F_x_BS = dftmtx(num_os * num_ant_x_BS);
F_y_BS = dftmtx(num_os * num_ant_y_BS);

beam_codebook_BS = zeros(num_ant_y_BS * num_ant_x_BS, num_os*num_ant_x_BS*num_os*num_ant_y_BS);

for i=1:num_os*num_ant_y_BS
    for j=1:num_os*num_ant_x_BS
        beam = kron(F_y_BS(1:num_ant_y_BS, i)', F_x_BS(1:num_ant_x_BS, j));
        beam_codebook_BS(:, (i-1)*num_os*num_ant_x_BS + j) = reshape(beam, [], 1);
    end
end


% beam_UE = 1;

% id_beam_UE = zeros(K, 1);
% gain_save = [];
gain_save = zeros(M, K);  % create a matrix of best gain
for sbs=1:M
    beam_gain = abs(channel' * beam_codebook_BS);  % (num_ue, codebook_length)
    [~, id_beam_BS] = max(beam_gain, [], 2);
    beam_BS = beam_codebook_BS(:, id_beam_BS);  % (N, K)
    gain = diag(abs(channel'*beam_BS).^2);  % (K, 1)
    % gain_save = [gain_save; gain'];
    gain_save(sbs, :) = gain;
end

% cell association based on "gain_save"

gain_save_ = reshape(gain_save,1,[]);  % (M, K) -> (M*K)
R_ineq = zeros(K, M*K);  % matrix of inequalities
V_ineq = zeros(M, M*K);  % matrix of inequalities

% for UE k, the column (in flatten form) is equal to the beamgain
for k=1:K
    R_ineq(k, k:K:M*K) = gain_save_(k:K:M*K);
end

% for SBS m, the row (in flatten form) is equal to 1
for m=1:M
    V_ineq(m, (m-1)*K+1:m*K) = 1; % gain_save_((k-1)*K+1:k*K+1);
end

% The rate inequality indicate that sum beamgain of each UE must be larger
% than a predefined threshold
C = intlinprog(gain_save_, ones(1, M*K), ...  % maximize sum rate, and all variables are integer
    [-R_ineq; V_ineq], [-ones(1, K)*R_min; ones(1, M)*V_max], ...  % K+M inequalities
    0, 0, ...
    0, 1);  % (M*K)
C = reshape(C, M, K);

selected_gain = C.*gain_save;
user_gain = sum(selected_gain, 1);

R = np.log2(1+user_gain./noise);
% beam_BS = sqrt(pow_tx/(num_ant_x_BS*num_ant_y_BS*num_UE)) * beam_BS;
% R = rate(channel, beam_BS, noise);
end