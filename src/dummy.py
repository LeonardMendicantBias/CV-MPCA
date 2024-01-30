def cart2sph(coor):  # coor: [M, K, 3]
    x, y, z = coor[:, :, 0], coor[:, :, 1], coor[:, :, 2]
    XsqPlusYsq = x**2 + y**2

    r = np.sqrt(XsqPlusYsq + z**2)                              # r
    elev = np.arccos(z/r)                                       # phi
    az = np.sign(y) * np.arccos(x/XsqPlusYsq)  + (x==0)*np.pi   # theta
    return r, az, elev

def path_gain(dist, freq):      # dist in meters, freq in GHz
    c = 3e8
    return 10**(-(32.4+17.3*np.log10(dist)+20*np.log10(freq))/10), \
           10**(-(32.4+20*np.log10(freq)+31.9*np.log10(dist))/10), \
           10**(-(31.84+21.50*np.log10(dist)+19*np.log10(freq))/10), \
           (c / (4*np.pi*dist*freq*1e9))**2

def UPA_array_response(theta, phi, N_h, N_v):
    M, K = np.shape(theta)
    a2 = (ULA_array_response_2(theta, phi, N_h)[..., None] * np.expand_dims(ULA_array_response_1(theta, phi, N_v), -2)).reshape(M, K, -1)
    return a2

def ULA_array_response_2(theta, phi, N):
    if len(theta.shape) == 2:
        a = 1/np.sqrt(N) * np.exp(
            -1j*np.pi*np.sin(theta)[..., None]*np.sin(phi)[..., None]*np.reshape(np.arange(N), [1, 1, N])
        )
        
    else:
        a = 1/np.sqrt(N) * np.exp(
            -1j*np.pi*np.sin(theta)*np.sin(phi)*np.arange(N)
        )
    # print(np.linalg.norm(a[0,0]))
    return a

def ULA_array_response_1(theta, phi, N):
    if len(theta.shape) == 2:
        a = 1/np.sqrt(N) * np.exp(-1j*np.pi*np.cos(phi)[..., None]*np.reshape(np.arange(N), [1, 1, N]))
        
    else:
        a = 1/np.sqrt(N) * np.exp(
            -1j*np.pi*np.cos(phi)*np.arange(N)
        )
        
    return a

# def ULA_array_response_2(theta, phi, N):
#     if len(theta.shape) == 2:
#         a = 1/np.sqrt(N) * np.exp(
#             -1j*np.pi*np.cos(theta)[..., None]*np.sin(phi)[..., None]*np.reshape(np.arange(N), [1, 1, N])
#         )
        
#     else:
#         a = 1/np.sqrt(N) * np.exp(
#           -1j*np.pi*np.cos(theta)*np.sin(phi)*np.arange(N)
#         )
#     # print(np.linalg.norm(a[0,0]))
#     return a

# def ULA_array_response_1(theta, phi, N):
#     if len(theta.shape) == 2:
#         a = 1/np.sqrt(N) * np.exp(-1j*np.pi*np.sin(phi)[..., None]*np.sin(theta)[..., None]*np.reshape(np.arange(N), [1, 1, N]))
        
#     else:
#         a = 1/np.sqrt(N) * np.exp(
#         -1j*np.pi*np.sin(phi)*np.sin(theta)*np.arange(N)
#         )
#     return a


def cell_association(required_rate, beam_gain, V_max, noise):
    M, K = beam_gain.shape
    options = []
    for i in range(V_max):
        for ids in CWOR(np.arange(K), i+1):
            temp = np.zeros(K) # _temp.copy()
            temp[list(ids)] = 1
            options.append(temp)

    max_rate = -1
    best_ca = None
    for ca in CWR(options, M):
        ca = np.array(ca)
        # snr = np.sum(beam_gain*ca, axis=0)/noise
        snr = np.abs(np.sum(beam_gain*ca, axis=0))**2/noise
        # print(snr)
        rates = np.log2(1 + snr)
        if (R:=rates.sum()) > max_rate and (rates >= required_rate).all():
            max_rate = R
            best_ca = ca
    return best_ca


# def AoD_to_beamgain(base_loc, user_loc, esti_AoD, esti_distance, power, LoS, cell_asso, N_h, N_v, freq, noise):
def AoD_to_beamgain(base_loc, user_loc, esti_AoD, power, LoS, N_h, N_v, freq):
    M, K = base_loc.shape[0], user_loc.shape[0]
    
    distance, theta, phi = cart2sph(np.reshape(user_loc, [1, K, 3]) - np.reshape(base_loc, [M, 1, 3]))
    true_gain = np.sqrt(path_gain(distance, freq/1e9)[-2]) * LoS
    # print(true_gain)
    true_array_response = UPA_array_response(theta, phi, N_h, N_v)

    esti_theta, esti_phi = esti_AoD
    esti_beamforming = np.sqrt(power) * UPA_array_response(esti_theta, esti_phi, N_h, N_v)
    beam_gain = np.zeros([M, K], dtype=np.complex64)
    
    for m in range(M):
        for k in range(K):
            beam_gain[m, k] = np.vdot(np.sqrt(N_h*N_v) * true_gain[m, k] * true_array_response[m, k], esti_beamforming[m, k])

    return esti_beamforming, beam_gain

# def NR_beamforming_to_beamgain(base_loc, user_loc, power, LoS, cell_asso, N_h, N_v, freq, noise, num_os):
def NR_beamforming_to_beamgain(base_loc, user_loc, power, LoS, N_h, N_v, freq, num_os):
    M, K = base_loc.shape[0], user_loc.shape[0]
    
    distance, theta, phi = cart2sph(np.reshape(user_loc, [1, K, 3]) - np.reshape(base_loc, [M, 1, 3]))
    true_gain = np.sqrt(path_gain(distance, freq/1e9)[-2]) * LoS
    true_array_response = UPA_array_response(theta, phi, N_h, N_v)

    beam_codebook_h = dft(N_h*num_os)[:, :N_h]/np.sqrt(N_h)
    beam_codebook_v = dft(N_v*num_os)[:, :N_v]/np.sqrt(N_v)
    beam_codebook_BS = np.zeros((num_os**2*N_h*N_v, N_h*N_v), dtype=np.complex64)

    for i in range(num_os*N_h):
        for j in range(num_os*N_v):
            beam = np.sqrt(power) * np.kron(beam_codebook_h[i],beam_codebook_v[j])
            beam_codebook_BS[i*num_os*N_v + j] = np.reshape(beam, -1)

    beam_gain = np.zeros([M, K], dtype=np.complex64)
    beamforming_vector = np.zeros([M, K, N_h*N_v], dtype = np.complex64)
    best_beam_index = np.zeros((M, K))
    
    for m in range(M):
        for k in range(K):
            beam_sweeping_gain = np.matmul(np.conjugate(np.sqrt(N_h*N_v) * true_gain[m, k] * true_array_response[m, k]) , np.transpose(beam_codebook_BS))
            abs_beam_sweeping_gain = np.abs(beam_sweeping_gain)
            
            max_index = np.argmax(abs_beam_sweeping_gain)
            best_beam_index[m,k] = max_index
            beamforming_vector[m, k] = beam_codebook_BS[max_index]
            beam_gain[m, k] = beam_sweeping_gain[max_index]

    return beamforming_vector, beam_gain, beam_sweeping_gain

def sumrate_evaluation(beam_gain, cell_asso, noise):
    sum_rate = np.sum(np.log2(1 + np.abs(np.sum(beam_gain*cell_asso, 0))**2/noise))
    # print(np.mean(np.log2(1 + np.abs(np.sum(beam_gain*cell_asso, 0))**2/noise)))

    return sum_rate

def bf2angle(vector, N_h, N_v):
    M, K, N = vector.shape
    vector = vector / np.linalg.norm(vector,axis=-1)[..., None] * np.sqrt(N_h*N_v)
    UPA = vector.reshape((M,K,N_h,N_v))
    ULA_1 = UPA[:,:,0]
    ULA_2 = UPA[:,:,:,0]
    cosphi = np.real(np.log(ULA_1[:,:,1])/(-1j*np.pi))
    sinthetasinphi = np.real(np.log(ULA_2[:,:,1])/(-1j*np.pi))
    # print('sinthetasinphi:',sinthetasinphi)

    phi = np.arccos(cosphi)
    sintheta = (np.sin(phi) == 0) * 0 + (np.sin(phi) != 0) * sinthetasinphi/(np.sin(phi)+0.000001)
    
    theta = (cosphi >= 0) * np.arcsin(np.clip(sintheta,-1,1)) + \
    (cosphi < 0) * (sintheta >= 0) * (np.pi - np.arcsin(np.clip(sintheta,-1,1))) + \
    (cosphi < 0) * (sintheta < 0) * (-np.pi - np.arcsin(np.clip(sintheta,-1,1)))

    return (theta, phi)


            # if (a:=len(logit)) < (b:=len(cat)):  # when there are less predictions than labels
            #     _temp = b-a
            #     predictions = [logit[idx] for idx in pred_ids] + [0]*_temp
            #     labels = [0]*b
            # else:  # when there are more predictions than labels
            #     _temp = a-b
            #     # there are b labels and _temp false positives
            #     _ids = [idx for idx in list(range(len(logit))) if idx not in pred_ids]
            #     predictions = [logit[idx] for idx in pred_ids] + [logit[idx] for idx in _ids]
            #     # predictions = [score[0] for score in phone_scores]
            #     labels = [0]*b + [1]*_temp