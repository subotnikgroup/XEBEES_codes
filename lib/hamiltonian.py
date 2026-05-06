import xp
from scipy.special import factorial
import numpy

def get_stencil_coefficients(stencil_size, derivative_order):
    if stencil_size % 2 == 0:
        raise ValueError("Stencil size must be odd.")

    half_size = stencil_size // 2
    A = xp.vander(xp.arange(-half_size, half_size + 1.0), increasing=True).T
    b = xp.zeros(stencil_size)
    b[derivative_order] = factorial(derivative_order)
    return xp.linalg.solve(A, b)


def KE(N, dx, mass=None, stencil_size=11, order=2, cyclic=False, bare=False):
    stencil = get_stencil_coefficients(stencil_size, order) / dx**order
    center = stencil_size // 2

    if cyclic:
        fft_size = N
        eye = xp.eye(N)

    else:
        # zero-pad to next power of 2
        fft_size = int(2 ** numpy.ceil(numpy.log2(N + stencil_size - 1)))
        eye = xp.zeros((N, fft_size))
        eye[xp.arange(N), xp.arange(N)] = 1.0

    stencil_k = xp.zeros(fft_size, dtype=xp.complex128)
    stencil_k[:stencil_size] = stencil
    stencil_k = xp.roll(stencil_k, -center)
    stencil_k = xp.fft.fft(stencil_k)

    T = xp.fft.ifft(stencil_k * xp.fft.fft(eye)).real[:, :N]

    if not bare:
        T *= -1 / (2 * mass)

    return T


def KE_FFT(N, P, R):
    Tp = xp.diag(-P**2)
    exp_RP = xp.exp(1j * xp.outer(P, R))
    KE = ((exp_RP.T.conj() @ Tp @ exp_RP) / N)
    print("KE_FFT: throwing out imag from KE_FFT", xp.sum(xp.abs(KE.imag)))
    return KE

def KE_FFT_R(N, P, R):
    Tp = xp.diag(-P**2)
    exp_RP = xp.exp(1j * xp.outer(P, R))
    KE = ((exp_RP.T.conj() @ Tp @ exp_RP) / N)
    print("KE_FFT: throwing out imag from KE_FFT", xp.sum(xp.abs(KE.imag)))
    return KE.real

# for equally spaced points; if unequal, pass J.
# tol specifies maximum mean Hermitian deviation
def KE_Borisov(x, tol=1e-6, mass=None, bare=False, order=2):
    # A. G. Borisov, J. Chem. Phys. 114, 7770–7777 (2001)
    # https://doi.org/10.1063/1.1358867

    N = len(x)
    x_max = x[-1]
    g = xp.gradient(x)
    g = g[0] if isinstance(g, tuple) else g
    J = g * N / x_max


    bound = lambda a, b: xp.arange(a,b+1)
    al = lambda k: xp.where((k == 0) | (k == N), 1/numpy.sqrt(2), 1)

    # Helper function to pre-compute sine and cosine matrices (Asin & Acos above)
    def DTT(N, func):
        k = bound(0, N)
        m = bound(1, N)
        return func(xp.outer(2*m-1,k) * xp.pi/N/2)

    COS = DTT(N, xp.cos)
    SIN = DTT(N, xp.sin)

    Ac = COS.T * al(bound(0,N))[:,xp.newaxis]
    As = (SIN * al(bound(0,N))).T
    Acv = COS * al(bound(0,N))[xp.newaxis, :] * (2/N)
    Asv = SIN * al(bound(0,N))[xp.newaxis, :] * (2/N)

    F = x

    b = 1/xp.sqrt(F * J)
    R = F / J
    k = xp.arange(N+1) * xp.pi / x[-1]

    if order == 2:  # L should be symmetric
        L = -b[:,None] * Acv * k @ As * R @ Asv * k @ Ac * b
        deviation = xp.mean(xp.abs(L-L.T))
        L = (L + L.T)/2
    elif order == 1:  # iL is Hermitian
        L = b[:,None] * (Acv * k @ As - Asv * k @ Ac) * b
        deviation = xp.mean(xp.abs(L+L.T))
        L = (L - L.T)/2
    else:
        raise RuntimeError(f"Borisov derivatives of order {order} not implemented!")


    if deviation > tol:
        raise RuntimeError("Deviation from Hermitian too large:", deviation)

    if not bare:
        L *= -1 / (2 * mass)

    return L, J

def KE_Borisov_3D(x, tol=1e-6, mass=None, bare=False, order=2):
    # A. G. Borisov, J. Chem. Phys. 114, 7770–7777 (2001)
    # https://doi.org/10.1063/1.1358867
    # spherical coordinate case given in equations 20-21

    N = len(x)
    x_max = x[-1]
    g = xp.gradient(x)
    g = g[0] if isinstance(g, tuple) else g
    J = g * N / x_max


    bound = lambda a, b: xp.arange(a,b) ## change 3D
    al = lambda k: xp.where((k == 0) | (k == N), 1/numpy.sqrt(2), 1)

    # Helper function to pre-compute sine and cosine matrices (Asin & Acos above)
    def DTT(N, func):
        k = bound(0, N)
        m = bound(0, N) ## change 3D
        return func(xp.outer(2*m,k) * xp.pi/N/2)

    COS = DTT(N, xp.cos)
    SIN = DTT(N, xp.sin)

    Ac = COS.T * al(bound(0,N))[:,xp.newaxis]
    As = (SIN * al(bound(0,N))).T
    Acv = COS * al(bound(0,N))[xp.newaxis, :] * (2/N)
    Asv = SIN * al(bound(0,N))[xp.newaxis, :] * (2/N)

    F = x

    b = 1/xp.sqrt(J) ## change 3D
    R = F / J
    k = xp.arange(N) * xp.pi / x[-1] ## change 3D

    if order == 2:  # L should be symmetric
        L = -b[:,None] * Asv * k @ Ac * b**2 @ Acv * k @ As * b ## change 3D
        deviation = xp.mean(xp.abs(L-L.T))
        L = (L + L.T)/2

    elif order == 1:  # iL is Hermitian
        L = b[:,None] * (Acv * k @ As - Asv * k @ Ac) * b
        deviation = xp.mean(xp.abs(L+L.T))
        L = (L - L.T)/2
    else:
        raise RuntimeError(f"Borisov derivatives of order {order} not implemented!")


    if deviation > tol:
        raise RuntimeError("Deviation from Hermitian too large:", deviation)

    if not bare:
        L *= -1 / (2 * mass)

    return L, J


def KE_ColbertMiller_zero_inf(N, dx, mass=None, bare=False, order=2):
    # DVR 2 in Appendix A of Colbert and Miller, JCP (1992).
    T = xp.zeros((N, N))
    # since we do not include the 0 point i->i+1; i+j-> i+j+2

    for i in range(N):
        for j in range(N):
            if order==2:
                if i == j:
                    T[i,i] = -1*(xp.pi**2/3 - 1/2/(i+1)**2)/ dx**2
                else:
                    T[i,j] = -1*((-1)**(i-j) * (2/(i-j)**2 - 2/(i+j+2)**2))/ dx**2
            elif order==1:
                if i!=j:
                    T[i,j] = (-1)**(i-j+1)/dx*((i-j)**(-1))

    if order==1 and bare==False: print("Warning, dividing first derivative by 2*mass")
    if not bare:
        T *= 1 / (2 * mass)

    return T 

def inverse_weyl_transform_old(E, NR, R, P):
    """
    Perform the inverse Weyl transform: only works for odd NR!!!
    """
    HPS = xp.zeros((NR, NR), dtype=complex)
    EPP = xp.zeros((NR, NR), dtype=complex)
    EPS_half = xp.zeros((NR + 1, NR), dtype=complex)
    dR = R[1] - R[0]
    R_half = xp.linspace(R[0] - dR/2, R[-1] + dR/2, NR + 1)

    # Build EPP
    for i in range(NR):
        for j in range(NR):
            for k in range(NR):
                EPP[j, i] += xp.exp(-1j * R[k] * P[j]) * E[k, i] / xp.sqrt(NR)

    # Build EPS_half
    for i in range(NR):
        for j in range(NR + 1):
            for k in range(NR):
                EPS_half[j, i] += xp.exp(1j * R_half[j] * P[k]) * EPP[k, i] / xp.sqrt(NR)

    # Build HPS
    for j in range(NR):
        for q1 in range(NR):
            for q2 in range(NR):
                if (q1 - q2) % 2 == 0:
                    HPS[q1, q2] += (xp.exp(-1j * (R[q1] - R[q2]) * P[j])
                                    * E[(q1 + q2) // 2, j] / NR)
                else:
                    idx = (q1 + q2 + 1) // 2
                    HPS[q1, q2] += (xp.exp(-1j * (R[q1] - R[q2]) * P[j])
                                    * EPS_half[idx, j] / NR)

    return HPS


def inverse_weyl_transform(E, NR, R, P):
    """
    Perform the inverse Weyl transform 
    """
    HPS = xp.zeros((NR, NR), dtype=complex)
    HPS_j = xp.zeros((NR, NR, NR), dtype=complex)
    EPP = xp.zeros((NR, NR), dtype=complex)
    EPS_half = xp.zeros((NR + 1, NR), dtype=complex)
    dR = R[1] - R[0]
    R_half = xp.linspace(R[0] - dR/2, R[-1] + dR/2, NR + 1)

    EPP= xp.matmul(xp.exp(-1j * xp.outer(P,R)), E)/xp.sqrt(NR)
    EPS_half = xp.matmul(xp.exp(1j * xp.outer(R_half, P)),EPP)/ xp.sqrt(NR)
    RRgrid = xp.meshgrid(R,R, indexing='ij')
    RRindxgrid = xp.meshgrid(xp.arange(NR),xp.arange(NR), indexing='ij')
    sumindx = RRindxgrid[0]+RRindxgrid[1]

    mask = (RRindxgrid[0]-RRindxgrid[1])%2 ==0
    coeff = xp.exp(-1j*(RRgrid[0]-RRgrid[1])[:,:,None]*P[None,None,:])

    HPS_j[mask,:] = E[sumindx[mask]//2,:]
    HPS_j[~mask,:] = EPS_half[(sumindx[~mask]+1)//2,:]

    HPS = xp.sum(HPS_j*coeff / NR,axis=2)

    # Inverse Weyl of a real symbol is Hermitian; enforce to fix numerical asymmetry
    #HPS = 0.5 * (HPS + HPS.conj().T)
    return HPS