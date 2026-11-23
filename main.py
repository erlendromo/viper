import numpy as np, glob, matplotlib.pyplot as plt
# P = [np.loadtxt(f) for f in sorted(glob.glob("data/latest/poses/*.txt"))[:2000]]
# R = np.stack([p[:3,:3] for p in P]); t = np.stack([p[:3,3] for p in P])
# print("det R:", np.linalg.det(R[0]))                      # should be +1
# C_w2c = -np.einsum("nji,nj->ni", R, t)                    # position if world-to-camera
# step = lambda C: np.median(np.linalg.norm(np.diff(C, axis=0), axis=1))
# print("step if world-to-camera:", step(C_w2c))
# print("step if camera-to-world:", step(t))
# print("extent:", np.ptp(C_w2c, axis=0))

P = [np.loadtxt(f) for f in sorted(glob.glob("data/latest/poses/*.txt"))]
C = np.stack([-p[:3,:3].T @ p[:3,3] for p in P])
Cc = C - C.mean(0)
_, _, V = np.linalg.svd(Cc, full_matrices=False)
print("extent along principal axes:", np.ptp(Cc @ V.T, axis=0))
print("ship axis direction:", V[0])

Q = Cc @ V.T                                   # along, across, vertical
print("robust extent (1-99%):", np.percentile(Q, 99, axis=0) - np.percentile(Q, 1, axis=0))

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].scatter(Q[:,0], Q[:,1], s=1, c=range(len(Q))); ax[0].set_aspect("equal"); ax[0].set_title("top-down")
ax[1].hist(C[:,2], bins=200); ax[1].set_title("camera height")
plt.savefig("pose_check.png", dpi=150)