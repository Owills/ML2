import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation, rc
from IPython.display import HTML, Image

rc('animation', html='jshtml')

nsamp = 50
w_true = [-1,-2]
x_data = np.random.uniform(-1,1,nsamp)
y_data = np.polyval(w_true[::-1], x_data) + np.random.normal(0, .1, size=x_data.shape)

plt.plot(x_data, y_data, 'ob', markeredgecolor='black');

#3d model
def Model(x,w):
    return np.polyval(w[::-1],x)

def LossFunc(x_batch,y_batch, w):
    return np.mean((y_batch-Model(x_batch,w))**2)/2

M = 25; N = 25
w0space = np.linspace(-8,8,M)
w1space = np.linspace(-8,8,N)
W0, W1 = np.meshgrid(w0space, w1space, indexing='ij')
J = np.empty((M,N))

for i in range(M):
    for j in range(N):
        J[i,j] = LossFunc(x_data, y_data, np.array([W0[i,j], W1[i,j]]))

fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
ax.plot_surface(W0,W1,J, cmap=plt.get_cmap('Blues'), zorder=1)

J0 = LossFunc(x_data,y_data,w_true)
ax.scatter(w_true[0], w_true[1], J0, c='blue', edgecolors='black', zorder=10)

ax.set_xlabel(r'$w_0$')
ax.set_ylabel(r'$w_1$')
ax.set_zlabel('J')

#UNDERSTANDING THE COST FUNCTION
w = [-1, -8]
fig1 = plt.figure()#fig1
ax = fig1.add_subplot(111, projection='3d')
ax.plot_surface(W0,W1,J, cmap=plt.get_cmap('Blues'), zorder=1)

Jw = LossFunc(x_data,y_data,w)
ax.scatter(w[0], w[1], Jw, s=100, c='red', edgecolors='black')
ax.set_xlabel(r'$w_0$')
ax.set_ylabel(r'$w_1$')
ax.set_zlabel('J')
fig2 = plt.figure()#fig2
plt.plot(x_data, y_data, 'ob', markeredgecolor='black')
plt.plot(x_data, np.polyval(w[::-1], x_data))

fig = plt.figure()
# gradient descent
max_step = 1000
w = np.empty((max_step,2))
Jw = np.empty((max_step,1))

w[0,:] = np.array([8,8])
Jw[0] = LossFunc(x_data, y_data, w[0,:])

alpha = 0.05 #0.05

def GradJ(x,y,w):
    mx  = np.mean(x)
    mxx = np.mean(x**2)
    mxy = np.mean(x*y)
    return np.array([w[0] + w[1]*mx - np.mean(y), w[0]*mx+ w[1]*mxx - mxy])


n = 1
break_in = 0
while(Jw[n-1]>.1):
    if( n>=max_step ):
        break
    w[n,:] = w[n-1,:] - alpha*GradJ(x_data, y_data, w[n-1,:])
    Jw[n]  = LossFunc(x_data, y_data, w[n])
    if( Jw[n] > Jw[n-1]):
        if(break_in<10):
            break_in+=1
        else:
            break
    n += 1

nstep = n
print(nstep)

w = w[:nstep,:]
Jw = Jw[:nstep]
plt.plot(np.arange(nstep), Jw);
fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')

ax.plot_surface(W0,W1,J, cmap=plt.get_cmap('Blues'), zorder=1)

J0 = LossFunc(x_data,y_data,w_true)
ax.plot([w_true[0]], [w_true[1]], [J0], 'ob', zorder=10)

ax.plot(w[:,0].flatten(), w[:,1].flatten(), Jw.flatten(), '-k', lw=2, zorder=10)

ax.azim = -60
ax.elev = 60

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.xaxis.pane.set_edgecolor('white')
ax.yaxis.pane.set_edgecolor('white')
ax.zaxis.pane.set_edgecolor('white')

ax.set_xlabel(r'$w_0$')
ax.set_ylabel(r'$w_1$')
ax.set_zlabel('J')

ax.set_xlim([-8,8])
ax.set_ylim([-8,8])
ax.set_zlim([0,50])


fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')

z = np.zeros((0))
Z = np.zeros((0,0))
objs = [ax.plot_surface(W0, W1, J, cmap=plt.get_cmap('Blues'), zorder=1),
        ax.plot(z, z, z, '-k', zorder=10)[0],
        ax.plot(z, z, z, 'ob', mec='black', zorder=10)[0],
        ax.text2D(0.05, 0.9, '', transform=ax.transAxes)]

ax.azim = -90
ax.elev = 20

ax.grid(True)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.xaxis.pane.set_edgecolor('white')
ax.yaxis.pane.set_edgecolor('white')
ax.zaxis.pane.set_edgecolor('white')

ax.set_xlabel(r'$w_0$')
ax.set_ylabel(r'$w_1$')
ax.set_zlabel('J');
#plt.figure()


def init():
    ax.azim = -90
    ax.elev = 20
    return objs

def animate(i):
    objs[1].set_data_3d(w[:i+1,0].flatten(), w[:i+1,1].flatten(), Jw[:i+1].flatten())
    objs[2].set_data_3d(w[i,0], w[i,1], Jw[i])
    ax.azim = ax.azim - .3
    ax.elev = ax.elev + .2
    objs[-1].set_text('n = %d' % (i+1))
    return objs

mpl.rcParams['animation.embed_limit'] = 120e6
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=nstep,
                               interval=20, blit=True)

plt.show()
#print(anim)