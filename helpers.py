import numpy as np
from user_setting import N_cells_x, N_cells_y, N_halo_x, N_halo_y
from numba import njit
@njit
def init_solution_arrays(nx_domain : int, ny_domain : int, nx_halo : int, ny_halo: int) -> (np.ndarray, np.ndarray, np.ndarray):
    '''
    initializes the array in which degrees of freedoms (of mesh boundaries) are stored, which include:
    x-component of velocity (ux), y-component of the velocity (uy) and pressure (p).

    args:
    nx_domain : number of main domain's cells in x direction
    ny_domain : number of main domain's cells in y direction
    nx_halo : number of halo cells on each side of x-axis
    ny_halo : number of halo cells on each side of y-axis

    returns:
    ux : x-component of velocity
    uy: y-component velocity
    p: np.ndarray for p
    '''
    array_shape = (nx_domain + 2 * nx_halo + 1, ny_domain + 2 * ny_halo + 1)
    ux = np.zeros(array_shape)
    uy = np.zeros(array_shape)
    p = np.zeros(array_shape)

    return ux, uy, p
@njit
def init_coef_arrays(nx_domain : int, ny_domain : int, nx_halo : int, ny_halo: int):
    '''
    creates numpy arrays which includes the coefficients for each mesh cell and a numpy array for the residual
    at the centre of the cell.

    args:
    nx_domain : number of main domain's cells in x direction
    ny_domain : number of main domain's cells in y direction
    nx_halo : number of halo cells on each side of x-axis
    ny_halo : number of halo cells on each side of y-axis

    returns:
    AE,AW,AN,AS,AEE,AWW,ANN,ASS,AP, Bu (all np.ndarray)
    '''
    array_shape = (nx_domain + 2 * nx_halo + 1, ny_domain + 2 * ny_halo + 1)
    AE = np.zeros(array_shape)
    AW = np.zeros(array_shape)
    AN = np.zeros(array_shape)
    AS = np.zeros(array_shape)
    AEE = np.zeros(array_shape)
    AWW = np.zeros(array_shape)
    ANN = np.zeros(array_shape)
    ASS = np.zeros(array_shape)
    AP = np.zeros(array_shape)
    res = np.zeros(array_shape)
    return AE,AW,AN,AS,AEE,AWW,ANN,ASS,AP, res

def grid_centre(nx_domain : int, ny_domain : int) -> (np.ndarray, np.ndarray):
    '''
    creates grid which does NOT include the halo cells.
    nx_domain : number of mesh cells in x direction
    ny_domain : number of mesh cells in y direction

    return: np.ndarray for coordinates of mesh cell centres
    x,y
    '''
    # uniform grid in x and y
    dx = 1.0 / nx_domain
    dy = 1.0 / ny_domain

    x = np.zeros((nx_domain , ny_domain))
    y = np.zeros((nx_domain , ny_domain))
    for i in range(nx_domain):
        for j in range(ny_domain):
            x[i] = dx * i + dx/2
            y[j] = dy * j + dx/2
    return x, y

def calculate_cell_centre_values(ux : np.ndarray, uy : np.ndarray) -> (np.ndarray, np.ndarray):
    """
    calculates the velocity values at the centre of p-control volume

    args:
    ux : np.ndarray for x-component of velocity
    uy : np.ndarray for y-component of velocity

    returns:
    ux_centre: numpy array for u-velocity values at the centre of p-control volume
    uy_centre: numpy array for v-velocity values at the centre of p-control volume
    """
    ux_centre = np.zeros((N_cells_x , N_cells_y))
    uy_centre = np.zeros((N_cells_x , N_cells_y))
    for i in range(N_cells_x):
        uy_centre[i , :] = (uy[i + N_halo_x , N_halo_y + 1:N_cells_y + N_halo_y + 1] + uy[i + N_halo_x + 1, N_halo_y + 1:N_cells_y + N_halo_y + 1]) / 2
    for j in range(N_cells_y):
        ux_centre[: , j] = (ux[N_halo_x:N_halo_x + N_cells_x, N_halo_y + j] + ux[N_halo_x:N_halo_x + N_cells_x, N_halo_y + j + 1]) / 2
    return ux_centre, uy_centre