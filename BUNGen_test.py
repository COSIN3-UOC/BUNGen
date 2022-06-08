import sys
import os
import warnings
warnings.simplefilter('always', UserWarning)

from netgen import NetworkGenerator
from netgen.xiConnRelationship import xiConnRelationship

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#%%
'''Script to reproduce the figures from the manuscript:
    BUNGen: Synthetic generator for structured ecological networks (2022).
    M.J. Palazzi, A. Lampo, A. Solé-Ribalta and J. Borge-Holthoefer
'''
    
#modify mainDir to local path of your empirical Networks
mainDir = "./Empirical_Net/"
#if the user wants to save figures change save_figures variable to True
save_figures = False
# <codecell>
# =============================================================================
# Figure 1: 
# =============================================================================
fig = plt.figure(constrained_layout=False)
gs = plt.GridSpec(1, 3, left=0.05, right=0.5, wspace=0.05, hspace=0.05)
clist = ['white', 'green', 'yellow', 'magenta']
cmap = ListedColormap(clist, 'indexed')

gen = NetworkGenerator(
        rows=30, 
        columns=30, 
        block_number=1,
        P=0.0,
        mu=0.0, 
        gamma=0.0,
        min_block_size=5,
        bipartite=True, 
        fixedConn=False, 
        link_density=1.5)

M, Pij, _, _ = gen()

ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

ax1.imshow(M, cmap=cmap)
ax1.set_title('Nested network', fontsize=4)
ax1.get_xaxis().set_ticks([])
ax1.get_yaxis().set_ticks([])
ax1.set_ylabel('Nodes of type A', fontsize=3)

gen = NetworkGenerator(
        rows=30, 
        columns=30, 
        block_number=3,
        P=1.0,
        mu=0.0, 
        gamma=0.0,
        min_block_size=5,
        bipartite=True, 
        fixedConn=False, 
        link_density=0.5)

M, Pij, crows, _ = gen()

crows = np.asarray(crows) + 1
count_vals = np.bincount(crows)

# just for coloring purposes
M[M>0] = len(clist)
sc_cum = 0
for k in range(np.max(crows)):
    sc = count_vals[k+1]
    sc_cum += count_vals[k]
    for i in range(sc_cum, sc_cum+sc):
        for j in range(sc_cum, sc_cum+sc):
            if M[i,j] > 0:
                M[i,j] = k+1


ax2.imshow(M, cmap=cmap)
ax2.set_title('Modular network', fontsize=4)
ax2.get_xaxis().set_ticks([])
ax2.get_yaxis().set_ticks([])
ax2.set_xlabel('Nodes of type B', fontsize=3)

gen = NetworkGenerator(
        rows=30, 
        columns=30, 
        block_number=3,
        P=0.0,
        mu=0.0, 
        gamma=0.0,
        min_block_size=5,
        bipartite=True, 
        fixedConn=False, 
        link_density=1.5)

M, Pij, crows, _ = gen()

crows = np.asarray(crows) + 1
count_vals = np.bincount(crows)

M[M>0] = len(clist)
sc_cum = 0
for k in range(np.max(crows)):
    sc = count_vals[k+1]
    sc_cum += count_vals[k]
    for i in range(sc_cum, sc_cum+sc):
        for j in range(sc_cum, sc_cum+sc):
            if M[i,j] > 0:
                M[i,j] = k+1

clist = ['white', 'green', 'yellow', 'magenta', 'cyan', 'blue', 'lightgrey' ]
cmap = ListedColormap(clist, 'indexed')
ax3.imshow(M, cmap=cmap)
ax3.set_title('In-block nested network', fontsize=4)
ax3.get_xaxis().set_ticks([])
ax3.get_yaxis().set_ticks([])
if save_figures:
    plt.savefig('fig1.jpg', dpi=720, bbox_inches='tight')

# <codecell>
# =============================================================================
# Figure 2: 
# =============================================================================
clistmu = ['white', 'yellow', 'magenta', 'cyan', 'green', 'lightgrey' ]
clistnomu = ['white', 'yellow', 'magenta', 'cyan', 'green' ]
cmapmu = ListedColormap(clistmu, 'indexed')
cmapnomu = ListedColormap(clistnomu, 'indexed')
fi = 3
co = 4
fig, axs = plt.subplots(fi, co, sharex=True, sharey=True)

r= 40
c= 40
Bs = [1,2,4]
ps = [0, 0.1, 0.5, 1.0]
mus = [0, 0.1, 0.5, 1.0]

# upper panels (row 0 of the figure)
xis = [0.5, 1, 2.0, 4.0]
for idx, xi in enumerate(xis):
    gen = NetworkGenerator(
            rows=r, 
            columns=c, 
            block_number=1,
            P=ps[idx],
            mu=0.0, 
            gamma=0.0,
            min_block_size=5,
            bipartite=True, 
            fixedConn=False, 
            link_density=xi)
    M, _, _, _ = gen()
    
    axs[0,idx].imshow(M, cmap=cmapnomu)
    st = r'$\xi = $' + str(xi) + r', $p = $' + str(ps[idx])
    axs[0,idx].set_title(st, fontsize=8)
    axs[0,idx].get_xaxis().set_ticks([])
    axs[0,idx].get_yaxis().set_ticks([])
    if idx==0: 
        st = 'B = 1'
        axs[0,idx].set_ylabel(st, fontsize=8)

# middle and lower panels (rows 1 and 2 of the figure)
counter = 0
for x in np.arange(1,fi): 
    B = Bs[x]
    for y in range(co):
        p = ps[y]
        if x==1: p=1.0
        mu = mus[y]         
        gen = NetworkGenerator(
                rows=r, 
                columns=c, 
                block_number=B,
                P=p,
                mu=mu, 
                gamma=0.0,
                min_block_size=5,
                bipartite=True, 
                fixedConn=False, 
                link_density=1.5)
        
        M, Pij, crows, _ = gen()

        l = np.sum(np.sum(M))
        d = l/(r*c)

        crows = np.asarray(crows) + 1
        count_vals = np.bincount(crows)
        
        if B > 0:
            M[M>0] = len(clistmu)
            sc_cum = 0
            for k in range(np.max(crows)):
                sc = count_vals[k+1]
                sc_cum += count_vals[k]
                for i in range(sc_cum, sc_cum+sc):
                    for j in range(sc_cum, sc_cum+sc):
                        if M[i,j] > 0:
                            M[i,j] = k+1

        fila = x
        columna = counter % 4
        
        axs[fila,columna].imshow(M, cmap=cmapmu)
        st = r'$p = $' + str(p) + r', $\mu = $' + str(mu)
        
        if columna == 0: axs[fila,columna].imshow(M, cmap=cmapnomu)
        
        axs[fila,columna].set_title(st, fontsize=8)
        axs[fila,columna].get_xaxis().set_ticks([])
        axs[fila,columna].get_yaxis().set_ticks([])
        if y==0: 
            st = 'B = ' + str(B)
            axs[fila,columna].set_ylabel(st, fontsize=8)
        
        counter += 1
        
if save_figures:
    # plt.tight_layout()
    plt.savefig('fig2.jpg', dpi=720, bbox_inches='tight')


# <codecell>
# =============================================================================
# Figure Table 2 (top): 
# =============================================================================
gen = NetworkGenerator(
        rows=50, 
        columns=50, 
        block_number=1,
        P=0.0, # perfectly nested structure
        ### irrelevant because $B=1$ ####
        mu=0.0, 
        gamma=0.0,
        min_block_size=5,
        #################################
        bipartite=True, 
        fixedConn=True, 
        link_density=0.25)

M, Pij, _, _ = gen()

fig = plt.figure(constrained_layout=False)
gs = plt.GridSpec(1, 2, left=0.05, right=0.5, wspace=0.05, hspace=0.05)

ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])

cmap = ListedColormap(['white', 'green'], 'indexed')
ax1.imshow(M, cmap=cmap)
ax1.set_title(r'$M$')
ax1.get_xaxis().set_ticks([])
ax1.get_yaxis().set_ticks([])

ax2.imshow(Pij, cmap='coolwarm', vmin=0, vmax=1)
ax2.set_title(r'$P_{ij}$')
ax2.get_xaxis().set_ticks([])
ax2.get_yaxis().set_ticks([])

if save_figures:
    plt.savefig('figT2top.jpg', dpi=720, bbox_inches='tight')

# <codecell>
# =============================================================================
# Figure Table 2 (bottom):
# =============================================================================
gen = NetworkGenerator(
        rows=50, 
        columns=50, 
        block_number=1,
        P=0.25, # noisy nested structure
        ### irrelevant because $B=1$ ####
        mu=0.0, 
        gamma=0.0,
        min_block_size=5,
        #################################
        bipartite=True, 
        fixedConn=True, 
        link_density=0.25)

M, Pij, _, _ = gen()

fig = plt.figure(constrained_layout=False)
gs = plt.GridSpec(1, 2, left=0.05, right=0.5, wspace=0.05, hspace=0.05)

ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])

cmap = ListedColormap(['white', 'green'], 'indexed')
ax1.imshow(M, cmap=cmap)
ax1.set_title(r'$M$')
ax1.get_xaxis().set_ticks([])
ax1.get_yaxis().set_ticks([])

ax2.imshow(Pij, cmap='coolwarm', vmin=0, vmax=1)
ax2.set_title(r'$P_{ij}$')
ax2.get_xaxis().set_ticks([])
ax2.get_yaxis().set_ticks([])

if save_figures:
    plt.savefig('figT2bottom.jpg', dpi=720, bbox_inches='tight')

# <codecell>
# =============================================================================
# Figure Table 3 (top):
# =============================================================================
gen = NetworkGenerator(
        rows=50, 
        columns=50, 
        block_number=3,
        P=1.0, # no intra-block structure
        mu=0.1, # inter-block noise
        gamma=0.0, # equally-sized blocks
        min_block_size=5,
        bipartite=True, 
        fixedConn=True, 
        link_density=0.25)

M, Pij, crows, _ = gen()

crows = np.asarray(crows) + 1
count_vals = np.bincount(crows)

M[M>0] = len(clist)
sc_cum = 0
for k in range(np.max(crows)):
    sc = count_vals[k+1]
    sc_cum += count_vals[k]
    for i in range(sc_cum, sc_cum+sc):
        for j in range(sc_cum, sc_cum+sc):
            if M[i,j] > 0:
                M[i,j] = k+1


fig = plt.figure(constrained_layout=False)
gs = plt.GridSpec(1, 2, left=0.05, right=0.5, wspace=0.05, hspace=0.05)

ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])

clist = ['white', 'green', 'yellow', 'magenta', 'cyan', 'blue', 'lightgrey' ]
cmap = ListedColormap(clist, 'indexed')
ax1.imshow(M, cmap=cmap)
ax1.set_title(r'$M$')
ax1.get_xaxis().set_ticks([])
ax1.get_yaxis().set_ticks([])

ax2.imshow(Pij, cmap='coolwarm', vmin=0, vmax=1)
ax2.set_title(r'$P_{ij}$')
ax2.get_xaxis().set_ticks([])
ax2.get_yaxis().set_ticks([])

if save_figures:
    plt.savefig('figT3top.jpg', dpi=720, bbox_inches='tight')

# <codecell>
# =============================================================================
# Figure Table 3 (bottom): 
# =============================================================================
gen = NetworkGenerator(
        rows=50, 
        columns=50, 
        block_number=5,
        P=1.0, # no intra-block structure
        mu=0.25, # inter-block noise
        gamma=2.5, # size of blocks as $B^{-\gamma}$
        min_block_size=5,
        bipartite=True, 
        fixedConn=True, 
        link_density=0.15)

M, Pij, crows, _ = gen()

crows = np.asarray(crows) + 1
count_vals = np.bincount(crows)

M[M>0] = len(clist)
sc_cum = 0
for k in range(np.max(crows)):
    sc = count_vals[k+1]
    sc_cum += count_vals[k]
    for i in range(sc_cum, sc_cum+sc):
        for j in range(sc_cum, sc_cum+sc):
            if M[i,j] > 0:
                M[i,j] = k+1

fig = plt.figure(constrained_layout=False)
gs = plt.GridSpec(1, 2, left=0.05, right=0.5, wspace=0.05, hspace=0.05)

ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])

clist = ['white', 'green', 'yellow', 'magenta', 'cyan', 'blue', 'lightgrey' ]
cmap = ListedColormap(clist, 'indexed')
ax1.imshow(M, cmap=cmap)
ax1.set_title(r'$M$')
ax1.get_xaxis().set_ticks([])
ax1.get_yaxis().set_ticks([])

ax2.imshow(Pij, cmap='coolwarm', vmin=0, vmax=1)
ax2.set_title(r'$P_{ij}$')
ax2.get_xaxis().set_ticks([])
ax2.get_yaxis().set_ticks([])

if save_figures:
    plt.savefig('figT3bottom.jpg', dpi=720, bbox_inches='tight')


# <codecell>
# =============================================================================
# Figure Table 4 (top):
# =============================================================================
gen = NetworkGenerator(
        rows=50, 
        columns=50, 
        block_number=4,
        P=0.0, # perfect intra-block structure
        mu=0.0, # perfect inter-block structure
        gamma=0.0, # equally-sized blocks
        min_block_size=5,
        bipartite=True, 
        fixedConn=True, 
        link_density=0.1)

M, Pij, crows, _ = gen()

crows = np.asarray(crows) + 1
count_vals = np.bincount(crows)

M[M>0] = len(clist)
sc_cum = 0
for k in range(np.max(crows)):
    sc = count_vals[k+1]
    sc_cum += count_vals[k]
    for i in range(sc_cum, sc_cum+sc):
        for j in range(sc_cum, sc_cum+sc):
            if M[i,j] > 0:
                M[i,j] = k+1

fig = plt.figure(constrained_layout=False)
gs = plt.GridSpec(1, 2, left=0.05, right=0.5, wspace=0.05, hspace=0.05)

ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])

cmap = ListedColormap(['white', 'green', 'yellow', 'magenta', 'cyan'], 'indexed')
ax1.imshow(M, cmap=cmap)
ax1.set_title(r'$M$')
ax1.get_xaxis().set_ticks([])
ax1.get_yaxis().set_ticks([])

ax2.imshow(Pij, cmap='coolwarm', vmin=0, vmax=1)
ax2.set_title(r'$P_{ij}$')
ax2.get_xaxis().set_ticks([])
ax2.get_yaxis().set_ticks([])

if save_figures:
    plt.savefig('figT4top.jpg', dpi=720, bbox_inches='tight')

# <codecell>
# =============================================================================
# Figure Table 4 (middle):
# =============================================================================
gen = NetworkGenerator(
        rows=50, 
        columns=50, 
        block_number=4,
        P=0.25, # intra-block noise
        mu=0.25, # inter-block noise
        gamma=2.5, # size of blocks as $B^{-\gamma}$
        min_block_size=5,
        bipartite=True, 
        fixedConn=True, 
        link_density=0.1)

M, Pij, crows, _ = gen()

crows = np.asarray(crows) + 1
count_vals = np.bincount(crows)

M[M>0] = len(clist)
sc_cum = 0
for k in range(np.max(crows)):
    sc = count_vals[k+1]
    sc_cum += count_vals[k]
    for i in range(sc_cum, sc_cum+sc):
        for j in range(sc_cum, sc_cum+sc):
            if M[i,j] > 0:
                M[i,j] = k+1

fig = plt.figure(constrained_layout=False)
gs = plt.GridSpec(1, 2, left=0.05, right=0.5, wspace=0.05, hspace=0.05)

ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])

clist = ['white', 'green', 'yellow', 'magenta', 'cyan', 'blue', 'lightgrey' ]
cmap = ListedColormap(clist, 'indexed')
ax1.imshow(M, cmap=cmap)
ax1.set_title(r'$M$')
ax1.get_xaxis().set_ticks([])
ax1.get_yaxis().set_ticks([])

ax2.imshow(Pij, cmap='coolwarm', vmin=0, vmax=1)
ax2.set_title(r'$P_{ij}$')
ax2.get_xaxis().set_ticks([])
ax2.get_yaxis().set_ticks([])

if save_figures:
    plt.savefig('figT4middle.jpg', dpi=720, bbox_inches='tight')

# <codecell>
# =============================================================================
# Figure Table 4 (bottom):
# =============================================================================
gen = NetworkGenerator(
        rows=50, 
        columns=50, 
        block_number=4,
        P=1.0, # no intra-block structure
        mu=1.0, # no inter-block structure
        gamma=0.0,
        min_block_size=5,
        bipartite=True, 
        fixedConn=True, 
        link_density=0.1)

M, Pij, crows, _ = gen()

crows = np.asarray(crows) + 1
count_vals = np.bincount(crows)

M[M>0] = len(clist)
sc_cum = 0
for k in range(np.max(crows)):
    sc = count_vals[k+1]
    sc_cum += count_vals[k]
    for i in range(sc_cum, sc_cum+sc):
        for j in range(sc_cum, sc_cum+sc):
            if M[i,j] > 0:
                M[i,j] = k+1

fig = plt.figure(constrained_layout=False)
gs = plt.GridSpec(1, 2, left=0.05, right=0.5, wspace=0.05, hspace=0.05)

ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])

clist = ['white', 'green', 'yellow', 'magenta', 'cyan', 'blue', 'lightgrey' ]
cmap = ListedColormap(clist, 'indexed')
ax1.imshow(M, cmap=cmap)
ax1.set_title(r'$M$')
ax1.get_xaxis().set_ticks([])
ax1.get_yaxis().set_ticks([])

ax2.imshow(Pij, cmap='coolwarm', vmin=0, vmax=1)
ax2.set_title(r'$P_{ij}$')
ax2.get_xaxis().set_ticks([])
ax2.get_yaxis().set_ticks([])

if save_figures:
    plt.savefig('figT4bottom.jpg', dpi=720, bbox_inches='tight')

# <codecell>
# =============================================================================
# Figure 3:  
# =============================================================================
fi = 3
co = 5

fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(fi, co, wspace=0.25, hspace=0.0)

ax = []
ax.append(fig.add_subplot(gs[0,0]))
ax.append(fig.add_subplot(gs[1,0]))
ax.append(fig.add_subplot(gs[2,0]))
ax.append(fig.add_subplot(gs[:,1:]))


# Left panels (matrices)
r= 8
c= 8
Bs = [1,2,4]
ds = [1.0, 0.5, 0.25]
p = 1.0
mu = 0
counter = 0
for x in range(fi): 
    B = Bs[x]
    
    gen = NetworkGenerator(
            rows=r, 
            columns=c, 
            block_number=B,
            P=p,
            mu=mu, 
            gamma=0.0,
            min_block_size=5,
            bipartite=True, 
            fixedConn=True, 
            link_density=1/B)
    
    M, Pij, _, _ = gen()    
    ax[x].imshow(M, cmap='binary', vmin=0, vmax=1)
    ax[x].set(xticks=[], yticks=[])
    ax[x].text(4, 3.5, 'd = ' + str(ds[x]), color='r', va='center', ha='center')


# right panel
num = 500
xs = np.linspace(1, 20, num=num)
ys = 1/xs

# ax.title.set_text(r'$\gamma$ = 2')
ax[3].set_xlabel(r'$B$')
ax[3].set_ylabel(r'$d_{max}$')

commFold = os.listdir(mainDir)
# CLASS [Pollinators, etc.] LOOP
for comm in commFold:
    
    if comm == ".DS_Store": continue

    sys.stderr.write('WORKING ON ' + str(comm) + '\n')
    commDir = mainDir+comm
    matList = os.listdir(commDir)
    
    # MATRIX LOOP
    for zzz, matName in enumerate(matList):
        
        # LOAD MATRIX
        fname = commDir+"/"+matName
        try:
            mat = np.loadtxt(fname, delimiter=',',dtype='float')
        except (ValueError,FileNotFoundError):
            continue
        
        r, c = mat.shape    
    
        # make it binary
        mat = (mat>0).astype('int')
  
        l0 = np.sum(mat)
        d0 = l0/(r*c)
        sys.stderr.write('\trows = ' + str(r) + '\t')
        sys.stderr.write('cols = ' + str(c) + '\t')
        sys.stderr.write('d = ' + str(d0) + '\n')
            
        cte = d0*np.ones(num)
        cte2 = d0*np.ones(num)
        cte[cte>ys] = -10
        cte2[cte2<ys] = -10

        ax[3].scatter(xs, cte, color='g', s=1, alpha=0.1)
        ax[3].scatter(xs, cte2, color='r', s=1, alpha=0.1)

ax[3].plot(xs, ys, '-k')
ax[3].set(xticks=list(np.arange(min(xs+1), max(xs)+0.05, 2)), yticks=list(np.arange(0, 1.1, 0.1)))
ax[3].text(6, 0.22, s=r'$1/B$', fontsize=16)
ax[3].set_xlim([1-0.05, 20+0.05])
ax[3].set_ylim([0, 1])

if save_figures:
    # plt.tight_layout()
    plt.savefig('fig3.jpg', dpi=720, bbox_inches='tight')

# <codecell>
# =============================================================================
# Figure 4 (left panel):  density vs size
# =============================================================================

fig, ax = plt.subplots()
rs = [4, 8, 10, 20, 40]
xis = np.linspace(0.1,4,20)
d = np.zeros((len(xis)))
for rw in rs:
    cl = rw
    for idx, x in enumerate(xis):
        d[idx] = xiConnRelationship(rw, cl, x)
        d[idx] /= rw*cl
           
    ax.scatter(xis, d, s=rw, label=r'$N=M=$' + str(rw))
    
ax.legend()        
plt.xlabel(r'$\xi$')
plt.ylabel(r'$d$')

if save_figures:
    plt.savefig('fig4a.jpg', dpi=720, bbox_inches='tight')


# <codecell>
# =============================================================================
# Figure 4 (right panel):  eccentricity test
# =============================================================================
rs = np.arange(4, 201, 2)
cs = np.flip(rs)
B = [1, 2]
p = 0.0
mu = 0.0
dens = 0.1
reps = 20
fig, ax = plt.subplots()
minabs = 1
maxabs = -1
ecc = rs/cs
ds = np.zeros(len(ecc))
for idx, x in enumerate(rs): 
    sys.stderr.write('Eccentricity = ' + str(ecc[idx]) + '\n')
    r = x
    c = cs[idx]
    sys.stderr.write('\tr = ' + str(r) + '\tc = ' + str(c) + '\n')
    
    gen = NetworkGenerator(
        rows=r, 
        columns=c, 
        block_number=int(np.random.choice(B)),
        P= np.random.uniform(),
        mu= np.random.uniform(), 
        gamma=0.0,
        min_block_size=5,
        bipartite=True, 
        fixedConn=True, 
        link_density=dens)

    _, Pij, _, _ = gen()
    dp = np.zeros(reps)
    eccp = ecc[idx]*np.ones(reps)
    for i in range(reps):
        Mrand = np.array(np.random.uniform(0, 1, size=(r, c)))
        M = (Pij > Mrand).astype(int)
        l = np.sum(M)
        d = l / (r*c)
        if minabs > d: minabs = d
        if maxabs < d: maxabs = d
        ds[idx] += d
        dp[i] = d
        sys.stderr.write('\t\td = ' + str(d) + '\n')
    
        
    ax.scatter(eccp, dp, s=2.5, c='lightgrey')
    ds[idx] /= reps
        
plt.xlabel(r'$N/M$')
plt.ylabel(r'$d$')
plt.plot(ecc, ds, 'og', ms=5)

xs = np.linspace(0.1,10, 100)
ys = dens*np.ones(100)
plt.plot(xs, ys, '-r')
plt.axvline(x=1, color='b', ls='--', lw=1)
plt.axvline(x=0.02, color='r', ls='--', lw=1)
plt.axvline(x=50, color='r', ls='--', lw=1)
plt.xlim([ecc[0]-.01, ecc[-1]+5])
# plt.ylim([minabs-0.001, maxabs+10])
ax.set_xscale('log')

rect_ax = inset_axes(ax, width=2, height=2, bbox_to_anchor=(0.4, .05, .6, .15),
                   bbox_transform=ax.transAxes, loc="lower left", borderpad=5)
rect_ax.axis("off")
rect=plt.Rectangle((0.1,0.05), 0.2, 0.8, transform=rect_ax.transAxes, fill=False, ec="k")
rect_ax.add_patch(rect)
rect_ax.text(0.0, 0.4, s=r'$N$', fontsize=12)
rect_ax.text(0.155, 0.875, s=r'$M$', fontsize=12)

rect_ax2 = inset_axes(ax, width=2, height=2, bbox_to_anchor=(-0.1, .25, .1, .35),
                   bbox_transform=ax.transAxes, loc="lower left", borderpad=5)
rect_ax2.axis("off")
rect=plt.Rectangle((0.1,0.05), 0.8, 0.2, transform=rect_ax2.transAxes, fill=False, ec="k")
rect_ax2.add_patch(rect)
rect_ax2.text(0.0, 0.125, s=r'$N$', fontsize=12)
rect_ax2.text(0.45, 0.265, s=r'$M$', fontsize=12)

if save_figures:
    # plt.tight_layout()
    plt.savefig('fig4b.jpg', dpi=720, bbox_inches='tight')

