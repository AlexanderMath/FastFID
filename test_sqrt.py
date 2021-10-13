"""
    Compute trace of square root matrix on a case where we know the solution.
    Compare the error of old and new algorithm. 

    Expected output: 


        > python test_sqrt.py 
        | Batch Size  | SciPy Time  | New Time    | Real Tr     | SciPy Tr    | New Tr      | SciPy Error | New Error   |
        | 8           | 4.2649      | 0.3275      | 14337.4844  | 14513.4677  | 14337.4854  | -175.9833   | -0.0010     |
        | 16          | 4.4771      | 0.0022      | 30551.6914  | 30777.3852  | 30551.6895  | -225.6938   | 0.0020      |
        | 32          | 4.4943      | 0.0029      | 64191.7812  | 64496.6673  | 64191.7812  | -304.8860   | 0.0000      |
        | 64          | 5.0165      | 0.0040      | 129579.5312 | 129989.2554 | 129579.5156 | -409.7242   | 0.0156      |
        | 128         | 4.3869      | 0.0072      | 259934.5469 | 260500.5697 | 259934.5625 | -566.0228   | -0.0156     |
        
    Let C1=C2 then we can easily compute 

        sqrt( C1 @ C1.T @ C2 @ C2.T) = sqrt( C1 @ C1.T @ C1 @ C1.T ) 
                                     = sqrt( (C1 @ C1.T)^2 )
                                     = C1 @ C1.T 

"""

import numpy as np
import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from fid.fast_sqrt import * 
from scipy.linalg import sqrtm 
import time

if __name__ == "__main__": 

    """
      Compute trace of matrix square root on example where we know solution,
      to compare the error of scipy.linalg.sqrtm with our algorithm. 
    """

    print("| %-11s | %-11s | %-11s | %-11s | %-11s | %-11s | %-11s | %-11s |" % ("Batch Size", "SciPy Time", "New Time", "Real Tr", "SciPy Tr", "New Tr", "SciPy Error", "New Error") )
    d   = 2048

    for bs in [8, 16, 32, 64, 128]:

        # Simulate computing covariance matrix so it has rank bs-1 as needed. 
        A  = torch.zeros((d, bs), device='cuda').normal_(0, 1)
        C1 = A - torch.mean(A, dim=1).view(d, 1) @ torch.ones((1, bs)) 
        C2 = C1

        C1_ = C1.clone().detach().cpu().numpy().astype(np.float32)
        C2_ = C2.clone().detach().cpu().numpy().astype(np.float32)

        # Since C2=C1 we can compute real tr(sqrt( C1 C1.T C2 C2.T ) ) 
        real        = torch.trace( C1 @ C1.T ).item()

        # Compute using scipy.linalg.sqrtm
        t0          = time.time()
        scipy       = np.real(np.trace( sqrtm( C1_ @ C1_.T @ C2_ @ C2_.T )))
        time_scipy  = time.time() - t0 


        # Compute with our algorithm 
        torch.cuda.synchronize() 
        t0          = time.time()
        fast        = trace_of_matrix_sqrt(C1, C2)
        torch.cuda.synchronize() 
        time_fast   = time.time() - t0 

        print("| %-11.1i | %-11.4f | %-11.4f | %-11.4f | %-11.4f | %-11.4f | %-11.4f | %-11.4f |" % (bs, time_scipy, time_fast, real, scipy, fast, real-scipy, real-fast) )


