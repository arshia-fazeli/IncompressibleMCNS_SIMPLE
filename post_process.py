import numpy as np
from numpy import genfromtxt
from helpers import grid_centre
import matplotlib.pyplot as plt
from user_setting import dy, dx, N_cells_x, N_cells_y, reaction_rate_constant

# get u,v and p arrays from the stored solution files
u = genfromtxt('./sol/ux_centre.csv', delimiter=',')
v = genfromtxt('./sol/uy_centre.csv', delimiter=',')
p = genfromtxt('./sol/p.csv', delimiter=',')
c = genfromtxt('./sol/c.csv', delimiter=',')
c2 = genfromtxt('./sol/c2.csv', delimiter=',')

# invert the solution arrays so that it matches the grid
u = u[::-1]
v = v[::-1]
p = p[::-1]
c = c[::-1]
c2 = c2[::-1]

# calculate the velocity magnitude
velocity_magnitude = np.sqrt(u**2 + v**2)

# different grids depending on the simulation
x, y = grid_centre (100 , 100)
Y40, X40 = np.mgrid[dy/2:1-dy/2:40j, dx/2:1-dx/2:40j]
Y80, X80 = np.mgrid[dy/2:1-dy/2:80j, dx/2:1-dx/2:80j]
Y120, X120 = np.mgrid[dy/2:1-dy/2:120j, dx/2:1-dx/2:120j]

# check the user setting for the grid
if N_cells_x == 40:
    Y,X = Y40, X40
elif N_cells_x == 80:
    Y, X = Y80, X80
elif N_cells_x == 120:
    Y, X = Y120, X120

if N_cells_x == 40 and N_cells_y == 80:
    X40, Y80 = np.mgrid[dy / 2:1 - dy / 2:40j,dx / 2:2 - dx / 2:80j]
    X, Y = X40, Y80


# plot the streamlines
plt.streamplot(X,Y,u,v,density=2,linewidth=1,color=velocity_magnitude,cmap='jet')
clb = plt.colorbar()
clb.ax.tick_params(labelsize=8)
clb.ax.set_title('velocity magnitude',fontsize=8)
plt.clim(0,1)
plt.savefig("./sol/streamlines.png")
plt.show()


# plot the pressure profile
plt.pcolormesh(X, Y, p[3:N_cells_x + 3, 3:N_cells_y + 3], cmap='gist_ncar')
clb = plt.colorbar(shrink=0.5)
clb.ax.tick_params(labelsize=8)
clb.ax.set_title('perssure magnitude',fontsize=8)
plt.clim(0,1)
plt.savefig("./sol/pressure_profile.png")
plt.show()

# plot the CA profile
plt.pcolormesh(X, Y, c[3:N_cells_x + 3, 3:N_cells_y + 3], cmap='RdBu')
clb = plt.colorbar(shrink=0.5)
clb.ax.tick_params(labelsize=8)
clb.ax.set_title('cA profile',fontsize=8)
plt.clim(0,1)
plt.savefig("./sol/cA_profile.png")
plt.show()

# plot the CB profile
plt.pcolormesh(X, Y, c2[3:N_cells_x + 3, 3:N_cells_y + 3], cmap='RdBu')
clb = plt.colorbar(shrink=0.5)
clb.ax.tick_params(labelsize=8)
clb.ax.set_title('cB profile',fontsize=8)
plt.clim(0,1)
plt.savefig("./sol/cB_profile.png")
plt.show()


avg_ls_c1 = list()
avg_ls_c2 = list()
for i in range(3, N_cells_y + 4):
    avg_val_c1= np.sum(c[3:N_cells_x + 3, i]) / len(c[3:N_cells_x + 3, i])
    avg_val_c2 = np.sum(c2[3:N_cells_x + 3, i]) / len(c2[3:N_cells_x + 3, i])
    avg_ls_c1.append(avg_val_c1)
    avg_ls_c2.append(avg_val_c2)


#xpoint=np.linspace(0,2,81)
#xpoint_2=np.linspace(0,2,21)
#plt.plot(xpoint,avg_ls_c1,label="c1-SIMPLE")
#plt.plot(xpoint_2,np.exp(-xpoint_2*reaction_rate_constant),"*",label="c1-analytical")
#plt.plot(xpoint,avg_ls_c2,label="c2-SIMPLE")
#plt.plot(xpoint_2,1-np.exp(-xpoint_2*reaction_rate_constant),"*",label="c2-analytical")
#plt.legend()
#plt.xlabel("x (m)")
#plt.ylabel("c (m)")
#plt.savefig("analytical_vs_exact_lowD.png")
#plt.show()




