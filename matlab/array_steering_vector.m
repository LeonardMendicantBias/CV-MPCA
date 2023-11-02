function a = array_steering_vector(r_11, phi, theta, num_ant_x, num_ant_y, f_c)

K = length(r_11);
lambda = 3e8/f_c;
d = lambda/2;
% r = zeros(num_ant_y*num_ant_x,1);

a1 = (0:1:num_ant_x-1);
a2 = (0:1:num_ant_y-1);

r_K = [];
for k=1:K
    r = -d*((a1*cos(phi(k))*sin(theta(k))) + (a2*sin(phi)*sin(theta(k)))');
    r = reshape(r,[],1);
    r_K = [r_K, r];
end

a = exp(-1j*r_K*2*pi/lambda);


