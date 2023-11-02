function channel = channel_generation(pos_BS, pos_UE, blockage, num_ant_x_BS, num_ant_y_BS, x_axis, y_axis, z_axis, f_c)

num_UE = size(pos_UE,2);
c = 3e8;

% [d, AoD_phi, AoD_theta] = angle_transform(pos_UE, x_axis, y_axis, z_axis);
[d, AoD_phi, AoD_theta] = angle_transform(vecnorm(pos_BS - pos_UE, 2, 1), x_axis, y_axis, z_axis);

LSF = large_scale_parameter(d, f_c);

channel = zeros(num_ant_y_BS*num_ant_x_BS, num_UE);
for k = 1 : num_UE
    channel(:, k) = sqrt(LSF(k)) * exp(-1j*2*pi*f_c/c*d(k)) * array_steering_vector_2(num_ant_x_BS, num_ant_y_BS, f_c, [AoD_phi(k), AoD_theta(k), d(k)]);
    channel(:, k) = channel(:, k) * blockage(k);  % channel=0 if there is blockage
end
