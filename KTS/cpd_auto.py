import numpy as np
from cpd_nonlin import cpd_nonlin

def method_KTS_total(h5Name, videoName, total):
    plt.figure("automatic selection of the number of change-points")
    m = 5
    #print("max KTS : ", str(m))

    file = h5py.File(h5Name, 'r')
    #print("\n file : " + h5Name)
    #print("\n vid name : " + videoName)
    data = list(file[videoName + '/features'])
    file.close()
    n_frames = total
    #n_frames = file[videoName + '/n_frames'][...]
    #n_frames = 3067

    #print("\n n_frames : " + str(n_frames))
    #print("\n len(data) : " + str(len(data)))

    X = np.array(data)
    plt.plot(X)
    K = np.dot(X, X.T)
    cps, scores = cpd_auto(K, m, 1)
    print("Estimated: (m=%d)" % len(cps), cps)

    mi = np.min(X)
    ma = np.max(X)
    for cp in cps:
        plt.plot([cp, cp], [mi, ma], 'r')

    list_cps = []
    list_fps = []


    for i in range(0, len(cps) + 1):
        temp = []

        if (i == 0):  # [0, 0번째 요소]
            fir = 0
            last = cps[i]
            fps = (last - fir) + 1

            temp.append(fir)
            temp.append(last)

        elif (i == len(cps)):  # [마지막 요소, frame 수-1]
            fir = cps[i - 1] + 1
            last = n_frames - 1
            fps = (last - fir) + 1

            temp.append(fir)
            temp.append(last)

        else:
            fir = cps[i - 1] + 1
            last = cps[i]
            fps = (last - fir) + 1

            temp.append(fir)
            temp.append(last)

        list_cps.append(temp)
        list_fps.append(fps)


    return list_cps[-1], list_cps[-1][0], list_cps


    #saveLoc = resultLoc + "resultPlt.jpg"
    #plt.savefig(saveLoc)

def cpd_auto(K, ncp, vmax, desc_rate=1, **kwargs):
    """Main interface
    
    Detect change points automatically selecting their number
        K       - kernel between each pair of frames in video
        ncp     - maximum ncp
        vmax    - special parameter
    Optional arguments:
        lmin     - minimum segment length
        lmax     - maximum segment length
        desc_rate - rate of descriptor sampling (vmax always corresponds to 1x)

    Note:
        - cps are always calculated in subsampled coordinates irrespective to
            desc_rate
        - lmin and m should be in agreement
    ---
    Returns: (cps, costs)
        cps   - best selected change-points
        costs - costs for 0,1,2,...,m change-points
        
    Memory requirement: ~ (3*N*N + N*ncp)*4 bytes ~= 16 * N^2 bytes
    That is 1,6 Gb for the N=10000.
    """
    m = ncp
    (_, scores) = cpd_nonlin(K, m, backtrack=False, **kwargs)
    
    N = K.shape[0]
    N2 = N*desc_rate  # length of the video before subsampling
    
    penalties = np.zeros(m+1)
    # Prevent division by zero (in case of 0 changes)
    ncp = np.arange(1, m+1)
    penalties[1:] = (vmax*ncp/(2.0*N2))*(np.log(float(N2)/ncp)+1)
    
    costs = scores/float(N) + penalties
    m_best = np.argmin(costs)
    (cps, scores2) = cpd_nonlin(K, m_best, **kwargs)

    return (cps, costs)
    

# ------------------------------------------------------------------------------
# Extra functions (currently not used)

def estimate_vmax(K_stable):
    """K_stable - kernel between all frames of a stable segment"""
    n = K_stable.shape[0]
    vmax = np.trace(centering(K_stable)/n)
    return vmax


def centering(K):
    """Apply kernel centering"""
    mean_rows = np.mean(K, 1)[:, np.newaxis]
    return K - mean_rows - mean_rows.T + np.mean(mean_rows)


def eval_score(K, cps):
    """ Evaluate unnormalized empirical score
        (sum of kernelized scatters) for the given change-points """
    N = K.shape[0]
    cps = [0] + list(cps) + [N]
    V1 = 0
    V2 = 0
    for i in range(len(cps)-1):
        K_sub = K[cps[i]:cps[i+1], :][:, cps[i]:cps[i+1]]
        V1 += np.sum(np.diag(K_sub))
        V2 += np.sum(K_sub) / float(cps[i+1] - cps[i])
    return (V1 - V2)


def eval_cost(K, cps, score, vmax):
    """ Evaluate cost function for automatic number of change points selection
    K      - kernel between all frames
    cps    - selected change-points
    score  - unnormalized empirical score (sum of kernelized scatters)
    vmax   - vmax parameter"""
    
    N = K.shape[0]
    penalty = (vmax*len(cps)/(2.0*N))*(np.log(float(N)/len(cps))+1)
    return score/float(N) + penalty
