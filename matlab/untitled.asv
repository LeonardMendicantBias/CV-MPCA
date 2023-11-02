clear
% ues = importdata('./ues.mat');
% ue_loc = ues.position;
% ue_block = ues.blockage;

% sbss = importdata('./sbss.mat');
% sbs_loc = sbss.position; 

M = 3;
K = 2;

C = reshape(1:M*K, K, M)';
C
C = reshape(C',1,[]);
C

R_ineq = zeros(K, M*K);
V_ineq = zeros(M, M*K);

for k=1:K
    R_ineq(k,k:K:M*K) = C(k:K:M*K);
end
R_ineq

for m=1:M
    V_ineq(m,(m-1)*K+1:m*K) = 1;
end
V_ineq