function LSF = large_scale_parameter(d, f_c)

c = 3e8;

% PL = 32.4 + 17.3 * log10(d) + 20 * log10(f_c);

G_free = (f_c./(4*pi*d*c)).^2;

LSF = G_free;
