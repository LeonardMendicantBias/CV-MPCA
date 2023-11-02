function R = rate(channel, beam_vec, noise)
num_UE = size(channel, 2);
gain = abs(channel' * beam_vec).^2;
num = diag(gain);
den = noise;
R = log2(1 + num ./ den);

end