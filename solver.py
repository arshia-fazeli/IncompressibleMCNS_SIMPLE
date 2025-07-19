import numpy as np
from numba import njit
from numba.types import unicode_type, int64, float64
from user_setting import N_cells_x, N_cells_y, N_halo_x, N_halo_y, Re, error_tolerance, Hybrid_switch, dx, dy, \
    maxit_inner_u, maxit_inner_v, \
    maxit_inner_p, reaction_rate_constant, diffusivity, relaxation_factor_u, relaxation_factor_v, relaxation_factor_p, \
    nu, maxit, convection_scheme, diffusion_order, relaxation_factor_c
from helpers import init_coef_arrays
from numba.experimental import jitclass
from numba import float64

# the data types that different classes will use, for jitclass
spec = [('a', float64), ('b', float64), ('convection_scheme', unicode_type), ('diffusion_order', int64)]

@jitclass(spec)
class FVM_coefficients:
    """
    A class to compute the finite volume method (FVM) coefficients for convection and diffusion terms
    based on the selected numerical scheme.
    """

    def __init__(self, convection_scheme : str, diffusion_order : int) -> None:
        """
        Initializes FVM_coefficients with convection scheme and diffusion order.
        """
        self.convection_scheme = convection_scheme
        self.diffusion_order = diffusion_order

    def get_convective_coefficients(self, ux_e : float, ux_w : float, uy_n : float, uy_s : float, i : int, j :int):
        """
        evaluates the convection part of the neighbouring coefficients (A_nb) and A_P at a certain mesh cell using
        different convection schemes.

        args:
        ux_e: ux at the east face
        ux_w: ux at the west face
        uy_n: uy at the north face
        uy_s: uy at the south face
        i : reference index in x-axis position (used to activate UDS on boundaries for quick)
        j : reference index in y-axis position (used to activate UDS on boundaries for quick)
        convection_scheme: "UDS" or "QUICK""

        return:
        [AE,AW,AN,AS,AEE,AWW,ANN,ASS,AP]
        """

        if self.convection_scheme=="UDS":
            AE = dy * max(0, -ux_e)
            AW = dy * max(0, ux_w)
            AN = dx * max(0, -uy_n)
            AS = dx * max(0, uy_s)
            AEE = 0
            AWW = 0
            ANN = 0
            ASS = 0
            AP = AS+AW+AN+AE+ASS+AWW+ANN+AEE

        elif self.convection_scheme =="QUICK":
            # at the boundaries use UDS
            if j == N_halo_y + 1 or j == N_cells_y + N_halo_y or i == N_cells_x + N_halo_x - 1 or i == N_halo_x:
                AE = dy * max(0, -ux_e)
                AW = dy * max(0, ux_w)
                AN = dx * max(0, -uy_n)
                AS = dx * max(0, uy_s)
                AEE = 0
                AWW = 0
                ANN = 0
                ASS = 0
                AP = AS + AW + AN + AE + ASS + AWW + ANN + AEE
            else:
                AE = -dy*(-0.75*max(0, -ux_e) + 0.375*max(0,ux_e) - 0.125*max(0, -ux_w))
                AW = -dy*(-0.125*max(0, ux_e) + 0.375*max(0, -ux_w) - 0.75*max(0, ux_w))
                AN = -dx * (-0.75 * max(0, -uy_n) + 0.375 * max(0, uy_n) - 0.125 * max(0, -uy_s))
                AS = -dx*(-0.125*max(0, uy_n) + 0.375*max(0, -uy_n) - 0.75*max(0, uy_s))
                AEE = -0.125*dy*max(0, -ux_e)
                AWW = -0.125*dy*max(0, ux_w)
                ANN = -0.125*dx*max(0, -uy_n)
                ASS = -0.125*dx*max(0, uy_s)
                AP = AS+AW+AN+AE+ASS+AWW+ANN+AEE
        else:
            raise Exception("Select one of the following convection schemes: UDS or QUICK. "
                            "For Hybrid, change Hybrid in user_setting.py to True ")

        coefs = np.array([AE,AW,AN,AS,AEE,AWW,ANN,ASS,AP])

        return coefs

    def Hybrid(self, ux_e : float, ux_w : float, uy_n : float, uy_s : float, nu : float):
        """
        evaluates the net (convection part + diffusion part)  neighbouring coefficients (A_nb) and A_P using
        hybrid scheme.

        args:
        ux_e: u velocity at the east face
        ux_w: u velocity at the west face
        uy_n: v velocity at the north face
        nu: v velocity at the south face
        convection_scheme: "UDS" or "QUICK""
        diffusion_order: order of the scheme used for closing the diffusive terms

        return:
        [AE,AW,AN,AS,AEE,AWW,ANN,ASS,AP]
        """
        if self.convection_scheme != "UDS":
            raise Exception("Hybrid scheme is only implemented for UDS convection scheme")
        # use the second-order diffusion approximation
        [DE,DW,DN,DS,DEE,DWW,DNN,DSS,DP] = self.get_diffusive_coefficients(nu)
        AE = dy * max(0, -ux_e, DE / dy - ux_e / 2)
        AW = dy * max(0, ux_w, DW / dy + ux_w / 2)
        AN = dx * max(0, -uy_n, DN / dx - uy_n / 2)
        AS = dx * max(0, uy_s, DS / dx + uy_s / 2)
        AEE = 0
        AWW = 0
        ANN = 0
        ASS = 0
        AP = AS+AW+AN+AE+ASS+AWW+ANN+AEE
        return np.array([AE, AW, AN, AS, AEE, AWW, ANN, ASS, AP])

    def get_diffusive_coefficients(self, nu : float, i , j):
        """
        evaluates the diffusion part of the neighbouring coefficients (A_nb).
        args:
        nu: kinematic viscosity
        diffusion_order: order of the scheme used for closing the diffusive terms

        return:
        [AE,AW,AN,AS,AEE,AWW,ANN,ASS]
        """
        # the coefficients for the second order approximation
        if self.diffusion_order == 2:
            AE = dy * nu / dx
            AW = dy * nu / dx
            AN = dx * nu / dy
            AS = dx * nu / dy
            AEE = 0
            AWW = 0
            ANN = 0
            ASS = 0
            AP =  - nu*(-2*dx/dy - 2*dy/dx)
        # the coefficients for the fourth order approximation
        elif self.diffusion_order == 4:
            if j == N_halo_y + 1 or j == N_cells_y + N_halo_y or i == N_cells_x + N_halo_x - 1 or i == N_halo_x:
                AE = dy * nu / dx
                AW = dy * nu / dx
                AN = dx * nu / dy
                AS = dx * nu / dy
                AEE = 0
                AWW = 0
                ANN = 0
                ASS = 0
                AP = - nu * (-2 * dx / dy - 2 * dy / dx)
            else:
                AE = 4/3*dy*nu/dx
                AW = 4/3*dy*nu/dx
                AN = 4/3*dx*nu/dy
                AS = 4/3*dx*nu/dy
                AEE = -1/12*dy*nu/dx
                AWW = -1/12*dy*nu/dx
                ANN = -1/12*dx*nu/dy
                ASS = -1/12*dx*nu/dy
                AP = - nu*(-2.5*dx/dy - 2.5*dy/dx)
        else:
            raise Exception("the implemented diffusion schemes: 2nd order, 4th order")
        coefs = np.array([AE,AW,AN,AS,AEE,AWW,ANN,ASS,AP])

        return coefs
@jitclass(spec)
class SIMPLE:
    """
    A class implementing the SIMPLE (Semi-Implicit Method for Pressure-Linked Equations) algorithm
    to solve coupled velocity, pressure, and scalar transport equations for incompressible flow.
    """
    def __init__(self):
        pass

    def solve(self, sol_arrays_list : list) -> (list, list):
        res = []
        print("=" * 50)
        print("         SIMPLE Solver Progress Log")
        print("=" * 50)
        print("Iteration |   Res_u     |   Res_v     |   Res_p     |   Res_c     |   Res_c2   ")
        print("-" * 50)
        for iter in range(1, maxit + 1):
            [u, v, p , c, c2] = sol_arrays_list
            u, res_u, APU = self.calculate_u([u, v, p, c, c2])
            v, res_v, APV = self.calculate_v([u, v, p, c, c2])
            p, res_p = self.calculate_p([u, v, p, c, c2], APU, APV)
            c, res_c = self.calculate_ca([u, v, p, c, c2])
            c2, res_c2 = self.calculate_cb([u, v, p, c, c2])
            res.append([iter, res_u, res_v, res_p])
            sol_arrays_list = [u , v , p , c , c2]
            # Print iteration progress every 10 iterations, or always for the first iteration
            if iter % 10 == 0 or iter == 1:
                print(iter, "|", res_u, "|", res_v,"|", res_p,"|", res_c,"|", res_c2,"|")

            # define convergence criterion
            if max(res_u, res_v, res_p, res_c, res_c2) <= error_tolerance and iter > 1:
                print("-" * 50)
                print(f"Solver converged in {iter} iterations.")
                print("The solution and iteration logging details can be found in the sol directory.")
                print("=" * 50)
                return sol_arrays_list, res
            elif iter == maxit - 1:
                print("-"*50)
                print("Solver DID NOT converge within maximum iterations.")
                print("The uncoverged solution logging details can be found in the sol directory.")
                print("="*50)
                return sol_arrays_list, res

    def calculate_u(self, sol_arrays_list):
        [u, v, p, c, c2] = sol_arrays_list
        AEU, AWU, ANU, ASU, AEEU, AWWU, ANNU, ASSU, APU, res_cell_centre = init_coef_arrays(N_cells_x, N_cells_y, N_halo_x, N_halo_y)
        res_cell_centre = np.zeros((N_cells_x + 2 * N_halo_x + 1, N_cells_y + 2 * N_halo_y + 1))
        fvm = FVM_coefficients(convection_scheme, diffusion_order)
        # # Swift direction: south to north
        for i in range(N_cells_x + N_halo_x - 1, N_halo_x - 1 , -1):
            # Swift direction: west to east (does not include the most eastern cell)
            for j in range(N_halo_y , N_cells_y + N_halo_y + 1):
                # define coefficients for u-equation
                # linear interpolation to find variables on the CV boundary
                U_1_ue = (u[i, j] + u[i, j + 1]) / 2
                U_1_uw = (u[i, j] + u[i, j - 1]) / 2
                U_1_vn = (v[i, j] + v[i, j + 1]) / 2
                U_1_vs = (v[i + 1, j] + v[i + 1, j + 1]) / 2
                if Hybrid_switch == False:
                    # calculate the coefficients for u_i,j
                    [AEU[i,j], AWU[i,j], ANU[i,j], ASU[i,j], AEEU[i,j], AWWU[i,j], ANNU[i,j], ASSU[i,j], APU[i,j]] = \
                        fvm.get_convective_coefficients(U_1_ue, U_1_uw, U_1_vn, U_1_vs, i, j) \
                        + fvm.get_diffusive_coefficients(nu, i , j)
                # if Hybrid_switch == True, then do hybrid scheme
                else:
                    [AEU[i, j], AWU[i, j], ANU[i, j], ASU[i, j], AEEU[i, j], AWWU[i, j], ANNU[i, j], ASSU[i, j], APU[i, j]]\
                        = fvm.Hybrid(U_1_ue, U_1_uw, U_1_vn, U_1_vs, nu)

        # Apply BCs to the south and north boundaries
        # south
        u[-N_halo_x + 1, N_halo_y + 1:N_cells_y + N_halo_y + 1] = - u[-N_halo_x, N_halo_y + 1:N_cells_y + N_halo_y + 1]
        # north
        u[N_halo_x - 1, N_halo_y + 1:N_cells_y + N_halo_y + 1] = 2 - u[N_halo_x, N_halo_y + 1:N_cells_y + N_halo_y + 1]
        # west
        #u[N_halo_x:N_halo_x + N_cells_x, N_halo_y] = 2 - u[N_halo_x:N_halo_x + N_cells_x, N_halo_y + 1]
        # east
        #u[N_halo_x:N_cells_x + N_halo_x, N_cells_y + N_halo_y] = +u[N_halo_x:N_cells_x + N_halo_x, N_cells_y + N_halo_y - 1]

        # calculate residuals
        res_u = 0.0
        for i in range(N_cells_x + N_halo_x - 1, N_halo_x - 1, -1):
            for j in range(N_halo_y + 1, N_cells_y + N_halo_y):
                res_cell_centre[i,j] = (AEU[i,j] * u[i, j + 1] + AWU[i,j] * u[i, j - 1] + ANU[i,j] * u[i - 1, j] + ASU[i,j] * u[i + 1, j]) \
                       + (AEEU[i,j] * u[i, j + 2] + AWWU[i,j] * u[i, j - 2] + ANNU[i,j] * u[i - 2, j] + ASSU[i,j] * u[i + 2, j]) \
                       + (p[i, j] - p[i, j + 1]) * dy - APU[i,j] * u[i, j]
                res_u += abs(res_cell_centre[i,j])

        # inner iterations for the u equation
        for inner_iter in range(1, maxit_inner_u + 1):
            for i in range(N_cells_x + N_halo_x - 1, N_halo_x - 1, -1):
                for j in range(N_halo_y + 1, N_cells_y + N_halo_y):
                    u[i,j] =  relaxation_factor_u * ((AEU[i,j] * u[i, j + 1] + AWU[i,j] * u[i, j - 1] + ANU[i,j] * u[i - 1, j] + ASU[i,j] * u[i + 1, j]) \
                       + (AEEU[i,j] * u[i, j + 2] + AWWU[i,j] * u[i, j - 2] + ANNU[i,j] * u[i - 2, j] + ASSU[i,j] * u[i + 2, j]) \
                       + (p[i, j] - p[i, j + 1]) * dy )/APU[i,j] + (1 - relaxation_factor_u) * u[i, j]


        return u, res_u, APU


    def calculate_v(self, sol_arrays_list):
        [u, v, p, c, c2] = sol_arrays_list
        AEV, AWV, ANV, ASV, AEEV, AWWV, ANNV, ASSV, APV, res_cell_centre = init_coef_arrays(N_cells_x, N_cells_y, N_halo_x, N_halo_y)
        # # Swift direction: south to north
        for i in range(N_cells_x + N_halo_x + 1, N_halo_x - 1, -1):
            # Swift direction: west to east (does not include the most eastern cell)
            #changed from N_halo_y + 2 to N_halo_y + 1
            for j in range(N_halo_y , N_cells_y + N_halo_y + 2):
                # define coefficients for v-equation
                # linear interpolation to find variables on the CV boundary
                V_1_ue = (u[i - 1, j] + u[i, j]) / 2
                V_1_uw = (u[i, j - 1] + u[i - 1, j - 1]) / 2
                V_1_vn = (v[i, j] + v[i - 1, j]) / 2
                V_1_vs = (v[i, j] + v[i + 1, j]) / 2
                fvm = FVM_coefficients(convection_scheme, diffusion_order)
                # calculate the coefficients for v_i,j
                if Hybrid_switch == False:
                    [AEV[i,j], AWV[i,j], ANV[i,j], ASV[i,j], AEEV[i,j], AWWV[i,j], ANNV[i,j], ASSV[i,j], APV[i,j]] = \
                        fvm.get_convective_coefficients(V_1_ue, V_1_uw, V_1_vn, V_1_vs, i, j) \
                        + fvm.get_diffusive_coefficients(nu , i , j)
                else:
                    [AEV[i,j], AWV[i,j], ANV[i,j], ASV[i,j], AEEV[i,j], AWWV[i,j], ANNV[i,j], ASSV[i,j], APV[i,j]]= \
                        fvm.Hybrid(V_1_ue, V_1_uw, V_1_vn, V_1_vs, nu)


        # Apply the boundary conditions to the west and east faces
        # west
        v[N_halo_x:N_halo_x + N_cells_x, N_halo_y] = -v[N_halo_x:N_halo_x + N_cells_x, N_halo_y + 1]
        # east
        v[N_halo_x:N_cells_x + N_halo_x, N_cells_y + N_halo_y + 1] = -v[N_halo_x:N_cells_x + N_halo_x, N_cells_y + N_halo_y + 1]

        # calculate residuals
        res_v = 0.0
        for i in range(N_cells_x + N_halo_x - 1, N_halo_x, -1):
            for j in range(N_halo_y + 1, N_cells_y + N_halo_y + 1):
                res_cell_centre[i,j] = (AEV[i,j] * v[i, j + 1] + AWV[i,j] * v[i, j - 1] + ANV[i,j] * v[i - 1, j] + ASV[i,j] * v[i + 1, j]) \
                       + (AEEV[i,j] * v[i, j + 2] + AWWV[i,j] * v[i, j - 2] + ANNV[i,j] * v[i - 2, j] + ASSV[i,j] * v[i + 2, j]) \
                       + (p[i, j] - p[i - 1, j]) * dx - APV[i,j] * v[i, j]
                res_v += abs(res_cell_centre[i,j])

        # inner iterations for the v equation
        for inner_iter in range(1, maxit_inner_v + 1):
            for i in range(N_cells_x + N_halo_x - 1, N_halo_x, -1):
                for j in range(N_halo_y + 1, N_cells_y + N_halo_y + 1):
                    v[i,j] =  relaxation_factor_u * ((AEV[i,j] * v[i, j + 1] + AWV[i,j] * v[i, j - 1] + ANV[i,j] * v[i - 1, j] + ASV[i,j] * v[i + 1, j]) \
                       + (AEEV[i,j] * v[i, j + 2] + AWWV[i,j] * v[i, j - 2] + ANNV[i,j] * v[i - 2, j] + ASSV[i,j] * v[i + 2, j]) \
                       + (p[i, j] - p[i-1, j]) * dy )/APV[i,j] + (1 -relaxation_factor_u) * v[i, j]

        return v, res_v, APV

    def calculate_p(self, sol_arrays_list, APU, APV):
        [u, v, p, c, c2] = sol_arrays_list
        AEP, AWP, ANP, ASP, AEEP, AWWP, ANNP, ASSP, APP, res_cell_centre = init_coef_arrays(N_cells_x, N_cells_y, N_halo_x, N_halo_y)
        p_correct = np.zeros((N_cells_x + 2 * N_halo_x + 1, N_cells_y + 2 * N_halo_y + 1))
        # # Swift direction: south to north
        for i in range(N_cells_x + N_halo_x - 1, N_halo_x - 1, -1):
            for j in range(N_halo_y + 1, N_cells_y + N_halo_y + 1):
                AEP[i,j] = dy ** 2 / APU[i,j]
                AWP[i,j] = dy ** 2 / APU[i,j-1]
                ANP[i,j] = dx ** 2 / APV[i,j]
                ASP[i,j] = dx ** 2 / APV[i+1 , j]


        AWP[:, N_halo_y + 1] = 0
        AEP[:, N_cells_y + N_halo_y] = 0
        ASP[N_cells_x + N_halo_x - 1, :] = 0
        ANP[N_halo_x, :] = 0


        # calculate residuals
        res_p = 0.0
        for i in range(N_cells_x + N_halo_x - 1, N_halo_x - 1, -1):
            for j in range(N_halo_y + 1, N_cells_y + N_halo_y + 1):
                APP[i,j] = AEP[i,j] + AWP[i,j] + ANP[i,j] + ASP[i,j]
                res_cell_centre[i,j] = -dx*(-v[i + 1, j] + v[i, j]) - dy*(-u[i , j-1] + u[i, j])
                res_p += abs(res_cell_centre[i,j])

        # inner iterations for p'-equation
        for N in range(1, maxit_inner_p + 1):
            for i in range(N_cells_x + N_halo_x - 1, N_halo_x - 1, -1):
                for j in range(N_halo_y + 1, N_cells_y + N_halo_y + 1):
                    p_correct[i,j] = res_cell_centre[i,j]/APP[i,j]
                    p_correct[i, j] += (AEP[i,j] * p_correct[i,j+1] + AWP[i,j] * p_correct[i,j-1] + ASP[i,j] * p_correct[i+1,j] + ANP[i,j] * p_correct[i-1,j])/APP[i,j]
        # correct u-velocity
        for i in range(N_cells_x + N_halo_x - 1, N_halo_x - 1, -1):
            for j in range(N_halo_y + 1, N_cells_y + N_halo_y):
                u_correct = dy* (p_correct[i,j] - p_correct[i,j+1])/APU[i,j]
                u[i,j] += relaxation_factor_u * u_correct
                pass

        # correct v-velocity
        for i in range(N_cells_x + N_halo_x - 1, N_halo_x, -1):
            for j in range(N_halo_y + 1, N_cells_y + N_halo_y + 1):
                v_correct = dy * (p_correct[i, j] - p_correct[i - 1, j]) / APV[i, j]
                v[i, j] += relaxation_factor_u * v_correct

        # correct p (the pressure)
        for i in range(N_cells_x + N_halo_x - 1, N_halo_x - 1, -1):
            for j in range(N_halo_y + 1, N_cells_y + N_halo_y + 1):
                p[i,j] += relaxation_factor_p * p_correct[i,j]

        return p, res_p

    def calculate_ca(self, sol_arrays_list):
        [u, v, p, c, c2] = sol_arrays_list
        AEC, AWC, ANC, ASC, AEEC, AWWC, ANNC, ASSC, APC, res_cell_centre= init_coef_arrays(N_cells_x, N_cells_y, N_halo_x, N_halo_y)
        fvm = FVM_coefficients(convection_scheme, diffusion_order)
        # # Swift direction: south to north
        for i in range(N_cells_x + N_halo_x - 1, N_halo_x - 1, -1):
            for j in range(N_halo_y + 1, N_cells_y + N_halo_y + 1):
                # define coefficients for u-equation
                # linear interpolation to find variables on the CV boundary
                U_1_ue = u[i,j]
                U_1_uw = u[i,j-1]
                U_1_vn = v[i,j]
                U_1_vs = v[i+1,j]
                if Hybrid_switch == False:
                    # calculate the coefficients for u_i,j
                    [AEC[i, j], AWC[i, j], ANC[i, j], ASC[i, j], AEEC[i, j], AWWC[i, j], ANNC[i, j], ASSC[i, j],
                     APC[i, j]] = \
                        fvm.get_convective_coefficients(U_1_ue, U_1_uw, U_1_vn, U_1_vs, i, j) \
                        + fvm.get_diffusive_coefficients(diffusivity, i , j)
                # if Hybrid_switch == True, then do hybrid scheme
                else:
                    [AEC[i, j], AWC[i, j], ANC[i, j], ASC[i, j], AEEC[i, j], AWWC[i, j], ANNC[i, j], ASSC[i, j], APC[i, j]] \
                        = fvm.Hybrid(U_1_ue, U_1_uw, U_1_vn, U_1_vs, diffusivity)

        # Apply BCs
        # south
        c[-N_halo_x - 1, N_halo_y + 1:N_cells_y + N_halo_y + 1] = + c[-N_halo_x -2, N_halo_y + 1:N_cells_y + N_halo_y + 1]
        # north
        c[N_halo_x - 1, N_halo_y + 1:N_cells_y + N_halo_y + 1] =  2 - c[N_halo_x, N_halo_y + 1:N_cells_y + N_halo_y + 1]
        # west
        c[N_halo_x:N_halo_x + N_cells_x, N_halo_y] = + c[N_halo_x:N_halo_x + N_cells_x, N_halo_y + 1]
        # east
        c[N_halo_x:N_cells_x + N_halo_x, N_cells_y + N_halo_y + 1] =  +c[N_halo_x:N_cells_x + N_halo_x, N_cells_y + N_halo_y]



        # calculate residuals
        res_c = 0.0
        for i in range(N_cells_x + N_halo_x - 1, N_halo_x - 1, -1):
            for j in range(N_halo_y + 1, N_cells_y + N_halo_y + 1):
                S = - c[i, j] * reaction_rate_constant * dx * dy
                res_cell_centre[i, j] = (AEC[i, j] * c[i, j + 1] + AWC[i, j] * c[i, j - 1] + ANC[i, j] * c[i - 1, j] + ASC[i, j] * c[
                    i + 1, j]) \
                           + (AEEC[i, j] * c[i, j + 2] + AWWC[i, j] * c[i, j - 2] + ANNC[i, j] * c[i - 2, j] + ASSC[i, j] *
                              c[i + 2, j]) \
                            - (APC[i, j]) * c[i, j] + S
                res_c += abs(res_cell_centre[i, j])

        # inner iterations for the u equation
        for inner_iter in range(1, maxit_inner_u + 1):
            for i in range(N_cells_x + N_halo_x - 1, N_halo_x - 1, -1):
                for j in range(N_halo_y + 1, N_cells_y + N_halo_y + 1):
                    S = - c[i, j] * reaction_rate_constant * dx * dy
                    c[i, j] = relaxation_factor_c * ((AEC[i, j] * c[i, j + 1] + AWC[i, j] * c[i, j - 1] + ANC[i, j] * c[
                        i - 1, j] + ASC[i, j] * c[i + 1, j]) \
                                                     + (AEEC[i, j] * c[i, j + 2] + AWWC[i, j] * c[i, j - 2] + ANNC[i, j] *
                                                        c[i - 2, j] + ASSC[i, j] * c[i + 2, j]) + S) / (APC[i, j]) + (
                                          1 - relaxation_factor_c) * c[i, j]

        return c, res_c


    def calculate_cb(self, sol_arrays_list):
        [u, v, p, ca, cb] = sol_arrays_list
        AEC, AWC, ANC, ASC, AEEC, AWWC, ANNC, ASSC, APC, res_cell_centre = init_coef_arrays(N_cells_x, N_cells_y, N_halo_x, N_halo_y)
        fvm = FVM_coefficients(convection_scheme, diffusion_order)
        # # Swift direction: south to north
        for i in range(N_cells_x + N_halo_x - 1, N_halo_x - 1, -1):
            for j in range(N_halo_y + 1, N_cells_y + N_halo_y + 1):
                # define coefficients for u-equation
                # linear interpolation to find variables on the CV boundary
                U_1_ue = u[i,j]
                U_1_uw = u[i,j-1]
                U_1_vn = v[i,j]
                U_1_vs = v[i+1,j]
                if Hybrid_switch == False:
                    # calculate the coefficients for u_i,j
                    [AEC[i, j], AWC[i, j], ANC[i, j], ASC[i, j], AEEC[i, j], AWWC[i, j], ANNC[i, j], ASSC[i, j],
                     APC[i, j]] = \
                        fvm.get_convective_coefficients(U_1_ue, U_1_uw, U_1_vn, U_1_vs, i, j) \
                        + fvm.get_diffusive_coefficients(diffusivity, i , j)
                # if Hybrid_switch == True, then do hybrid scheme
                else:
                    [AEC[i, j], AWC[i, j], ANC[i, j], ASC[i, j], AEEC[i, j], AWWC[i, j], ANNC[i, j], ASSC[i, j], APC[i, j]] \
                        = fvm.Hybrid(U_1_ue, U_1_uw, U_1_vn, U_1_vs, diffusivity)

        # Apply BCs
        # south
        cb[-N_halo_x - 1, N_halo_y + 1:N_cells_y + N_halo_y + 1] = + cb[-N_halo_x - 2, N_halo_y + 1:N_cells_y + N_halo_y + 1]
        # north
        cb[N_halo_x - 1, N_halo_y + 1:N_cells_y + N_halo_y + 1] = - cb[N_halo_x, N_halo_y + 1:N_cells_y + N_halo_y + 1]
        # west
        cb[N_halo_x:N_halo_x + N_cells_x, N_halo_y] = + cb[N_halo_x:N_halo_x + N_cells_x, N_halo_y + 1]
        # east
        cb[N_halo_x:N_cells_x + N_halo_x, N_cells_y + N_halo_y + 1] =  + cb[N_halo_x:N_cells_x + N_halo_x, N_cells_y + N_halo_y]

        # calculate residuals
        res_c = 0.0
        for i in range(N_cells_x + N_halo_x - 1, N_halo_x - 1, -1):
            for j in range(N_halo_y + 1, N_cells_y + N_halo_y + 1):
                S = + ca[i, j] * reaction_rate_constant * dx * dy
                res_cell_centre[i, j] = (AEC[i, j] * cb[i, j + 1] + AWC[i, j] * cb[i, j - 1] + ANC[i, j] * cb[i - 1, j] +
                                         ASC[i, j] * cb[
                                             i + 1, j]) \
                                        + (AEEC[i, j] * cb[i, j + 2] + AWWC[i, j] * cb[i, j - 2] + ANNC[i, j] * cb[
                    i - 2, j] + ASSC[i, j] *
                                           cb[i + 2, j]) \
                                        - (APC[i, j]) * cb[i, j] + S
                res_c += abs(res_cell_centre[i, j])

        # inner iterations for the u equation
        for inner_iter in range(1, maxit_inner_u + 1):
            for i in range(N_cells_x + N_halo_x - 1, N_halo_x - 1, -1):
                for j in range(N_halo_y + 1, N_cells_y + N_halo_y + 1):
                    S = + ca[i, j] * reaction_rate_constant * dx * dy
                    cb[i, j] = relaxation_factor_c * ((AEC[i, j] * cb[i, j + 1] + AWC[i, j] * cb[i, j - 1] + ANC[i, j] * cb[
                        i - 1, j] + ASC[i, j] * cb[i + 1, j]) \
                                                     + (AEEC[i, j] * cb[i, j + 2] + AWWC[i, j] * cb[i, j - 2] + ANNC[
                                i, j] *
                                                        cb[i - 2, j] + ASSC[i, j] * cb[i + 2, j]) + S) / (APC[i, j]) + (
                                      1 - relaxation_factor_c) * cb[i, j]

        return cb, res_c
