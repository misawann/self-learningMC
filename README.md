# self-learningMC

Implementation of "Self-learning Monte Carlo method", Liu et al (2017)

## implemented models

- [x] Ising model

  - original model
    $$H = -J \sum_{\langle i, j \rangle} \sigma_i \sigma_j$$
  - effective model
    $$H = - \sum_i h_i \sigma_i $$

- [x] 4 spin interaction model

  - original model
    $$H = -J \sum_{\langle i, j \rangle} \sigma_i \sigma_j - K \sum_{ijkl\in \square} \sigma_i \sigma_j \sigma_k \sigma_l$$
  - effective model  
    $$H_{\mathrm{trial}} = E_0- \tilde J_1 \sum_{\langle i, j \rangle_1} \sigma_i \sigma_j - \tilde J_2\sum_{\langle i, j \rangle_2} \sigma_i\sigma_j - \tilde J_3\sum_{\langle i, j \rangle_3}\sigma_i\sigma_j$$
    <p align="center">
      <img src="https://github.com/misawann/self-learningMC/blob/main/images/4spin_interaction_sorted_energy.png"/>
    </p>

## implemented algorithms

- [x] Metropolis algorithm
- [ ] Wolff algorithm
