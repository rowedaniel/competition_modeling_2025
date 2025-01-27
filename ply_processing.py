import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.optimize import curve_fit
from scipy import interpolate
import matplotlib

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)

##
# get and process point data
def read_ply_data(filename, frame, point_to_cm=100):
  plydata = PlyData.read(filename+('0000000'+str(frame))[-7:]+'.ply')
  data = plydata['vertex']

  # recenter 
  data['x'] -= data['x'].min()
  data['y'] -= data['y'].min()

  # convert 
  data['x'] *= point_to_cm
  data['y'] *= point_to_cm
  data['z'] *= point_to_cm
  return data

def sanitize_data(df, x_lim = None, y_lim = None, z_lim = None, swap_x_y = False):
  if x_lim is not None:
    df = df[df['x'] > x_lim[0]]
    df = df[df['x'] < x_lim[1]]
  if y_lim is not None:
    df = df[df['y'] > y_lim[0]]
    df = df[df['y'] < y_lim[1]]
  if z_lim is not None:
    df = df[df['z'] > z_lim[0]]
    df = df[df['z'] < z_lim[1]]
  df['x'] -= df['x'].min()
  df['y'] -= df['y'].min()
  if not swap_x_y:
    return (df['x'], df['y'], df['z'])
  return (df['y'], df['x'], df['z'])






##
# process a particular dataset
dset = "2025-01-25--16-21-25" # data set 1
# dset = "2025-01-25--16-25-07" # data set 2

t = 20
swap_x_y = True
data = read_ply_data(f'./data/{dset}/PLY/', t)
x,y,z = sanitize_data(data, y_lim=[25, 50], z_lim=[-100, -40], swap_x_y=swap_x_y) # use for data set 1
# x,y,z = sanitize_data(data, y_lim=[6, 35], swap_x_y=swap_x_y) # use for data set 2

print('read data')






##
# 3d plot before sanitization
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
if swap_x_y:
  x_raw = data['y'][::100]
  y_raw = data['x'][::100]
  z_raw = data['z'][::100]
else:
  x_raw = data['x'][::100]
  y_raw = data['y'][::100]
  z_raw = data['z'][::100]
ax.scatter(x_raw, y_raw, z_raw, marker='.')
ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")
ax.set_zlabel("z (cm)")
plt.axis('equal')

##
# 3d plot after sanitization
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x[::100], y[::100], z[::100], marker='.')
ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")
ax.set_zlabel("z (cm)")
plt.axis('equal')
plt.show()








##
# find stair plane
def plane(xy, a, b, c):
  x,y = xy
  return a*x + b*y + c

popt, pcov = curve_fit(plane, (x,y), z, p0=(0,0,0))
z_plane = plane((x,y), *popt)
z = z - z_plane


##
# Interpolate to quadrature points
Nx = 100 # number of quadrature points in x
Ny = 100 # number of quadrature points in x
x_reg = np.linspace(x_raw.min(), x_raw.max(), Nx)
y_reg = np.linspace(y_raw.min(), y_raw.max(), Ny)
xx, yy = np.meshgrid(x_reg, y_reg)

# if regular spaced points are needed, use the following.
# spline_interp = interpolate.bisplrep(x, y, z)
# zz_interpolated = interpolate.bisplev(x_reg, y_reg, spline_interp)






##
# 3d plot and stair plane
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x_raw, y_raw, z_raw, marker='.')
ax.scatter(xx[10:-10, 10:-10],
           yy[10:-10, 10:-10],
           plane((xx[10:-10, 10:-10], yy[10:-10, 10:-10]), *popt),
           marker='.')
ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")
ax.set_zlabel("z (cm)")
plt.show()








##
# match to guassians
def gaussian(xy, amplitude, x0, y0, sigma_x, sigma_y):
  x,y = xy
  g = -amplitude**2 * np.exp(
      -(
          (x-x0)**2 / sigma_x ** 2 +
          (y-y0)**2 / sigma_y ** 2
        )
  )
  return g

def two_gaussians(xy,
                  a1, x01, y01, sx1, sy1,
                  a2, x02, y02, sx2, sy2,
                  ):
  return gaussian(xy, a1,x01,y01,sx1,sy1) + gaussian(xy, a2, x02, y02, sx2, sy2)


print("fitting guassians")
popt, pcov = curve_fit(two_gaussians, (x, y), z,
                       p0=(1, x.min(), y.min()/2, (x.max()-x.min())/3, (y.max()-y.min())/3,
                           1, x.max(), y.min()/2, (x.max()-x.min())/3, (y.max()-y.min())/3)
                       )








##
# plot gaussian fit in 3d
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(x[::100],
           y[::100],
           two_gaussians((x[::100],y[::100]), *popt),
           marker='.')

ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")
ax.set_zlabel("z (cm)")



##
# plot gaussian fit using a heatmap
fig = plt.figure()
ax = fig.add_subplot()
c = ax.pcolormesh(xx, yy, two_gaussians((xx, yy), *popt))
ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")
fig.colorbar(c, ax=ax)


plt.show()

