# Mesh dimensions
N_cells_x = 80
N_cells_y = 80
# Number of "halo-data" layers
N_halo_x = 2
N_halo_y = 2
# Grid spacing (uniform mesh)
dx = 1 / N_cells_x
dy = 1 / N_cells_y
# Reynolds number
Re = 1000
nu = 1/Re # kinematic viscosity
# maximum number of inner loop iterations
maxit_inner_u = 4
maxit_inner_v = 4
maxit_inner_p = 8
# error tolerance
error_tolerance = 0.0001
# max number of iterations.
maxit = 15000

# under-relaxation factor for u and v equations
relaxation_factor_u = 0.3
relaxation_factor_v = 0.3
relaxation_factor_p = 0.2
relaxation_factor_c = 0.3

# choose a convection scheme
convection_scheme = "QUICK"
# choose the order of approximation for the diffusive term.
diffusion_order = 4
Hybrid_switch = False

reaction_rate_constant = 0.5
diffusivity = 0.05