import os.path
import time
import numpy as np
from helpers import init_solution_arrays, calculate_cell_centre_values
from solver import SIMPLE
from user_setting import N_cells_x,N_cells_y,N_halo_x,N_halo_y,convection_scheme,diffusion_order
import csv

# create u, v and p numpy arrays
u, v, p = init_solution_arrays(N_cells_x, N_cells_y, N_halo_x, N_halo_y)
c_A = np.ones((N_cells_x + 2 * N_halo_x + 1, N_cells_y + 2 * N_halo_y + 1))
c_B = np.zeros((N_cells_x + 2 * N_halo_x + 1, N_cells_y + 2 * N_halo_y + 1))
sol_arrays_list = [u, v, p , c_A, c_B]
# start recording the time
start_time = time.process_time()
# Solve
sol_arrays_list, res = SIMPLE().solve(sol_arrays_list)
[u, v, p , c_A, c_B] = sol_arrays_list
# Display the process time
print(f"process time in cpus:{time.process_time()-start_time}")

# find the values of u and v at the centre of p-control volumes
ux_centre, uy_centre = calculate_cell_centre_values(u, v)

# create solution directory
if os.path.isdir("./sol") == False:
    os.mkdir("./sol")

# Save the solutions in csv files
solution_names = ["u", "v", "ux_centre", "uy_centre", "p", "c", "c2"]

for name, sol_data in zip(solution_names, [u , v , ux_centre , uy_centre , p , c_A , c_B]):
    np.savetxt(f"./sol/{name}.csv", sol_data, delimiter=",")

# record information about iterations and residuals
with open('./sol/alog.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Iteration", "Residual_u", "Residual_v", "Residual_p", "Residual_c", "Residual_c2"])
    writer.writerows(res)
