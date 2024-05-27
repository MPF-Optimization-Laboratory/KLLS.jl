The labeling matches || A x - b_avg ||_2 + KL(x|mu)

--- rho-meson_testproblem ---
In the rho-meson_testproblem.npz are the following arrays
 A, (the transformation kernel) omega^2  e^{-\tau \omega}
 x, (the input spectrum) \omega^2 \rho
 b_avg, (The average of b's generated via Ax + eta = b) 
        b_avg has been normalized using b_avg[i]/b_avg[0] thus it will produce an x vector which sums to 1.
 b_std, (The standard deviation of b's generated via Ax + eta = b), 
        computed via b_std = np.std(b_samples/b_avg[0], axis=0), so it accounts for the normalization
 mu,    (the prior) this is a flat prior normalized to 1.
 taus,  (The tau discretization in units 1/GeV)
 omegas, (The omega discretization in units GeV)


 --- interacting electron gas testproblem ---
 Note DSF mean S_k(\omega) at k = .2981 k_{D,e}
In the synthetic-UEG_testproblem.npz are the following arrays
 A, (the transformation kernel) e^{- \tau \omega} + e^{- (\beta - \tau) \omega}
 x, (the input spectrum) Atwal and Ashcroft's model of DSF normalized to 1
 b_avg, (The average of b's generated via Ax + eta = b) 
        b_avg has been normalized using b_avg[i]/b_avg[0] thus it will produce an x vector which sums to 1.
 b_std, (The standard deviation of b's generated via Ax + eta = b), 
        computed via b_std = np.std(b_samples/b_avg[0], axis=0), so it accounts for the normalization
 mu,    (the prior) this is the RPA model of the DSF normalized to 1.
 taus,  (The tau discretization in units 1/GeV)
 omegas, (The omega discretization in units GeV)