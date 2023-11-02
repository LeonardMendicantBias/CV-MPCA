clear all;
iter = 10000;
K = 4;

N = 8; %# of BS antenna



distances = 1:1:10;

C_LCBM = zeros(length(distances),1);
C_5GNR = zeros(length(distances),1);
C_CVBM = zeros(length(distances),1);
C_Oracle = zeros(length(distances),1);
C_Lidar = zeros(length(distances),1);

error_LCBM = zeros(length(distances),1);
error_CVBM = zeros(length(distances),1);
error_Lidar = zeros(length(distances),1);
Q_b = 64;
n = 0:N-1;
angle_qN = 0: pi/Q_b: pi * (Q_b) / Q_b ;
angle_N = 0:1:Q_b-1;

% codebook = zeros(N^2,Q_b^2);
codebook_a = zeros(N,Q_b);
codebook_b = zeros(N,Q_b);
% 

for p = 1:Q_b
    codebook_a(:,p) = exp(-1i * 2 * pi * n' * angle_N(p)/Q_b);
    codebook_b(:,p) = exp(-1i * 2 * pi * n' * angle_N(p)/Q_b);
end

codebook = kron(codebook_a,codebook_b)/N;


frequency = 24 * 10^9; 
c_light = 3 * 10^8;
lambda = c_light / frequency;
for q = 1:length(distances)
    % frequency = 24 * 10^9; 
    % c_light = 3 * 10^8;
    % lambda = c_light / frequency;
    
    Beta_0_dB = -30; %dB
    Beta_0 = 10^(Beta_0_dB / 10) * 10^3;
    d_0 = 1; %m
    % d1 = 50;
    % d2 = 50;

    
    alpha_H = 2.8; %pathloss coefficient b/w BS and RIS/relay
    % alpha_g = 2.8; %pathloss coefficient b/w RIS/relay and UE 
    
    txPower_dBm = 0; %dBm
    txPower = 10^(txPower_dBm/10) / (10^3);
    
    dist_UE = distances(q);
    % LCBM localization error
    a = 0.0296;
    b = 0.0088;
    error_LCBM = a + b*dist_UE;
    e_LCBM(q) = error_LCBM;
    % CVBM localization error
    a = 7.5 * 10^(-7);
    b = 1.581;
    error_CVBM = a * exp(b*dist_UE) + 0.1;
    e_CVBM(q) = error_CVBM;
    % Lidar localization error
    a = 0.2848;
    b = 0.0044;
    error_Lidar = a + b*dist_UE;
    e_Lidar(q) = error_Lidar;
    % dist_RU = 20;

    PG_H = sqrt(Beta_0 * (dist_UE / d_0)^-alpha_H);
    % PG_g = sqrt(Beta_0 * (dist_RU / d_0)^-alpha_g);

    Var_N_dB = -60; %dB
    Var_N = 10^(Var_N_dB / 10);
    
    NofPath = 1; %# of path
    

    C_Oracle_i = zeros(iter,1);
    C_LCBM_i = zeros(iter,1);
    C_CVBM_i = zeros(iter,1);
    C_Lidar_i = zeros(iter,1);
    C_NR_i = zeros(iter,1);
    
    for p = 1:iter
        if mod(p, 100)==0
            fprintf('x = %d, Iteration: %d \n',q, p);
        end
        a_N = zeros(N^2,K);
        AoD = zeros(K,2);
        w_temp = zeros(N^2,K);
        
        true_posi_cart = zeros(K,3);
        est_posi_cart_LCBM = zeros(K,3);
        for k = 1:K
            AoD(k,:) = rand(1,2) * pi - pi/2; %azi
           
            [true_posi_cart(k,1),true_posi_cart(k,2),true_posi_cart(k,3)] = sph2cart(AoD(k,1),AoD(k,2),dist_UE);
        
            a_Ba = exp(-1i * pi * n' * sin(AoD(k,1))*sin(AoD(k,2)));
            a_Bb = exp(-1i * pi * n' * cos(AoD(k,2)));
            a_B = kron(a_Ba,a_Bb);
            a_N(:,k) = a_B;
        end

        %% Oracle
        h = PG_H * a_N';
        w_O = zeros(N^2,K);
        for k = 1:K
            w_O(:,k) = a_N(:,k) / N / sqrt(K);
        end
        % temp = pinv(h);
        % for k = 1:K
        %     w_O(:,k) = temp(:,k) / norm(temp(:,k));
        % end
        Gain = abs(h * w_O).^2* txPower;
        bf_Gain = diag(Gain);
        inf_plus_noise = sum(Gain,1) - bf_Gain' + Var_N;
        SINR = bf_Gain' ./ inf_plus_noise;
        C_Oracle_i(p) = sum(log2(1+SINR));

        %% LCBM
        AoD_error = rand(K,2) * pi;
        
        diff = zeros(1,3);
        sph_est = zeros(K,3);
        w_LCBM = zeros(N^2,K);
        est_posi_cart = zeros(k,3);
        for k = 1:K
            [diff(1),diff(2),diff(3)] = sph2cart(AoD_error(k,1),AoD_error(k,2),error_LCBM);
            est_posi_cart(k,:) = true_posi_cart(k,:) + diff;
            [sph_est(k,1),sph_est(k,2),sph_est(k,3)] = cart2sph(est_posi_cart(k,1),est_posi_cart(k,2),est_posi_cart(k,3));
            w_Ba = exp(-1i * pi * n' * sin(sph_est(k,1))*sin(sph_est(k,2)));
            w_Bb = exp(-1i * pi * n' * cos(sph_est(k,2)));
            w_B = kron(w_Ba,w_Bb)/N/sqrt(K);
            w_LCBM(:,k) = w_B;
        end
        % for k = 1:K
        %     [diff(1),diff(2),diff(3)] = sph2cart(AoD_error(k,1),AoD_error(k,2),error_LCBM);
        %     est_posi_cart(k,:) = true_posi_cart(k,:) + diff;
        %     [sph_est(k,1),sph_est(k,2),sph_est(k,3)] = cart2sph(est_posi_cart(k,1),est_posi_cart(k,2),est_posi_cart(k,3));
        %     w_Ba = exp(-1i * pi * n' * sin(sph_est(k,1))*sin(sph_est(k,2)));
        %     w_Bb = exp(-1i * pi * n' * cos(sph_est(k,2)));
        %     w_B = kron(w_Ba,w_Bb)/N;
        %     w_temp(:,k) = w_B;
        % end        
        % temp = pinv(w_temp');
        % for k = 1:K
        %     w_LCBM(:,k) = temp(:,k) / norm(temp(:,k));
        % end
        Gain = abs(h * w_LCBM).^2* txPower;
        bf_Gain = diag(Gain);
        inf_plus_noise = sum(Gain,1) - bf_Gain' + Var_N;
        SINR = bf_Gain' ./ inf_plus_noise;
        C_LCBM_i(p) = sum(log2(1+SINR));

        %% CVBM
        AoD_error = rand(K,2) * pi;
        
        diff = zeros(1,3);
        sph_est = zeros(K,3);
        w_CVBM = zeros(N^2,K);
        est_posi_cart = zeros(k,3);
        for k = 1:K
            [diff(1),diff(2),diff(3)] = sph2cart(AoD_error(k,1),AoD_error(k,2),error_CVBM);
            est_posi_cart(k,:) = true_posi_cart(k,:) + diff;
            [sph_est(k,1),sph_est(k,2),sph_est(k,3)] = cart2sph(est_posi_cart(k,1),est_posi_cart(k,2),est_posi_cart(k,3));
            w_Ba = exp(-1i * pi * n' * sin(sph_est(k,1))*sin(sph_est(k,2)));
            w_Bb = exp(-1i * pi * n' * cos(sph_est(k,2)));
            w_B = kron(w_Ba,w_Bb)/N/sqrt(K);
            w_CVBM(:,k) = w_B;
        end
        % for k = 1:K
        %     [diff(1),diff(2),diff(3)] = sph2cart(AoD_error(k,1),AoD_error(k,2),error_CVBM);
        %     est_posi_cart(k,:) = true_posi_cart(k,:) + diff;
        %     [sph_est(k,1),sph_est(k,2),sph_est(k,3)] = cart2sph(est_posi_cart(k,1),est_posi_cart(k,2),est_posi_cart(k,3));
        %     w_Ba = exp(-1i * pi * n' * sin(sph_est(k,1))*sin(sph_est(k,2)));
        %     w_Bb = exp(-1i * pi * n' * cos(sph_est(k,2)));
        %     w_B = kron(w_Ba,w_Bb)/N;
        %     w_temp(:,k) = w_B;
        % end        
        % temp = pinv(w_temp');
        % for k = 1:K
        %     w_CVBM(:,k) = temp(:,k) / norm(temp(:,k));
        % end
        
        Gain = abs(h * w_CVBM).^2* txPower;
        bf_Gain = diag(Gain);
        inf_plus_noise = sum(Gain,1) - bf_Gain' + Var_N;
        SINR = bf_Gain' ./ inf_plus_noise;
        C_CVBM_i(p) = sum(log2(1+SINR));

        %% Lidar
        AoD_error = rand(K,2) * pi;
        
        diff = zeros(1,3);
        sph_est = zeros(K,3);
        w_Lidar = zeros(N^2,K);
        est_posi_cart = zeros(k,3);
        for k = 1:K
            [diff(1),diff(2),diff(3)] = sph2cart(AoD_error(k,1),AoD_error(k,2),error_Lidar);
            est_posi_cart(k,:) = true_posi_cart(k,:) + diff;
            [sph_est(k,1),sph_est(k,2),sph_est(k,3)] = cart2sph(est_posi_cart(k,1),est_posi_cart(k,2),est_posi_cart(k,3));
            w_Ba = exp(-1i * pi * n' * sin(sph_est(k,1))*sin(sph_est(k,2)));
            w_Bb = exp(-1i * pi * n' * cos(sph_est(k,2)));
            w_B = kron(w_Ba,w_Bb)/N/sqrt(K);
            w_Lidar(:,k) = w_B;
        end
        % for k = 1:K
        %     [diff(1),diff(2),diff(3)] = sph2cart(AoD_error(k,1),AoD_error(k,2),error_Lidar);
        %     est_posi_cart(k,:) = true_posi_cart(k,:) + diff;
        %     [sph_est(k,1),sph_est(k,2),sph_est(k,3)] = cart2sph(est_posi_cart(k,1),est_posi_cart(k,2),est_posi_cart(k,3));
        %     w_Ba = exp(-1i * pi * n' * sin(sph_est(k,1))*sin(sph_est(k,2)));
        %     w_Bb = exp(-1i * pi * n' * cos(sph_est(k,2)));
        %     w_B = kron(w_Ba,w_Bb)/N;
        %     w_temp(:,k) = w_B;
        % end        
        % temp = pinv(w_temp');
        % for k = 1:K
        %     w_Lidar(:,k) = temp(:,k) / norm(temp(:,k));
        % end
        Gain = abs(h * w_Lidar).^2* txPower;
        bf_Gain = diag(Gain);
        inf_plus_noise = sum(Gain,1) - bf_Gain' + Var_N;
        SINR = bf_Gain' ./ inf_plus_noise;
        C_Lidar_i(p) = sum(log2(1+SINR));

        %% 5G NR

        w_NR = zeros(N^2,K);
        for k = 1:K
            NR_Beamforming_gain = zeros(N^2,1);
            for pp = 1:N^2
                NR_Beamforming_gain(pp) = norm(h(k,:) * codebook(:,pp))^2 * txPower;
            end
            [tempa, tempb] = max(NR_Beamforming_gain);
            w_NR(:,k) = codebook(:,tempb)/sqrt(K);
        end
        % for k = 1:K
        %     NR_Beamforming_gain = zeros(N^2,1);
        %     for pp = 1:N^2
        %         NR_Beamforming_gain(pp) = norm(h * codebook(:,pp))^2 * txPower;
        %     end
        %     [tempa, tempb] = max(NR_Beamforming_gain);
        %     w_temp(:,k) = codebook(:,tempb);
        % end
        % temp = pinv(w_temp');
        % for k = 1:K
        %     w_NR(:,k) = temp(:,k) / norm(temp(:,k));
        % end
        Gain = abs(h * w_NR).^2* txPower;
        bf_Gain = diag(Gain);
        inf_plus_noise = sum(Gain,1) - bf_Gain' + Var_N;
        SINR = bf_Gain' ./ inf_plus_noise;
        C_NR_i(p) = sum(log2(1+SINR));
      
    end
   

   C_Oracle(q) = mean(C_Oracle_i);
   C_LCBM(q) = mean(C_LCBM_i);
   C_CVBM(q) = mean(C_CVBM_i);
   C_Lidar(q) = mean(C_Lidar_i);
   C_5GNR(q) = mean(C_NR_i);

end

figure;
plot(distances,C_LCBM,'ro-'); hold on;
plot(distances,C_Oracle,'sb-'); hold on;
plot(distances,C_CVBM,'*k-'); hold on;
plot(distances,C_Lidar,'^m-'); hold on;
plot(distances,C_5GNR,'>g-');


width = 5;     % Width in inches
height = 4;    % Height in inches
alw = 1.2;    % AxesLineWidth
lw = 1.5;      % LineWidth
msz = 7;       % MarkerSize

x_axis = distances;

fig_N = figure;
pos = get(gcf, 'Position');
set(gcf, 'Position', [pos(1) pos(2) width*100, height*100]);
plot(x_axis, C_Oracle, '-or', 'LineWidth', lw, 'MarkerSize', msz);
hold on
plot(x_axis, C_LCBM, '-sb', 'LineWidth', lw, 'MarkerSize', msz);
% hold on
%  plot(x_axis, C_CVBM, '-*m', 'LineWidth', lw, 'MarkerSize', msz)
hold on
plot(x_axis, C_Lidar, '-^g', 'LineWidth', lw, 'MarkerSize', msz)
hold on
plot(x_axis, C_5GNR, '-dk', 'LineWidth', lw, 'MarkerSize', msz)


xlim([x_axis(1), x_axis(length(x_axis))]);
xticks(x_axis);
xlabel('Distance (m)', 'interpreter', 'latex')
ylabel('Sum rate (bps/Hz)', 'interpreter', 'latex')
ax = gca;
ax.GridLineStyle = ':';
ax.LineWidth = alw;
grid on    
legend('Ideal', 'LCBM','SMBM','5G-BM','interpreter','latex')