# Readme

Hi =)

The code has been refactored, partially written, and documented by ChatGPT, so don't be surprized (I am a mathematician, not a coder).

*The `.mat` files need to be in the `mat_converted_N={NN}`* folder, where `NN` is the resolution (that is, the volumes are $N \times N \times N$)

1. The script `final_cov_matr_centered.py` builds the covariance matrix (three of them: uncentered, centered per volume, centered globally). Then it expands each volume in the eigebasis of this matrix.

2. The script `final_plot_energy_PCA.py` plots the comparison graphs of the energy.