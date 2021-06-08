import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_circles
from sklearn.svm import SVC

X,y = make_circles(90, factor=0.2, noise=0.1)
# noise = standard deviation of Gaussian noise added in data.
# factor = scale factor between the two circles
# plt.scatter(X[:,0],X[:,1], c=y, s=50, cmap='seismic')

svm_model = SVC(kernel='poly') # , C=1., gamma=0.5
classify = svm_model.fit(X, y)
def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    print('initial decision function shape; ', np.shape(Z))
    Z = Z.reshape(xx.shape)
    print('after reshape: ', np.shape(Z))
    out = ax.contourf(xx, yy, Z, **params)
    return out
def make_meshgrid(x, y, h=.1):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))  # ,
    # np.arange(z_min, z_max, h))
    return xx, yy

X0, X1 = X[:,0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

fig, ax = plt.subplots(figsize=(12, 9))
fig.patch.set_facecolor('white')
cdict1 = {0: 'lime', 1: 'deeppink'}

Y_tar_list = y.tolist()
yl1 = [int(target1) for target1 in Y_tar_list]
labels1 = yl1
labl1 = {0: 'Malignant', 1: 'Benign'}
marker1 = {0: '*', 1: 'd'}
alpha1 = {0: .8, 1: 0.5}
for l1 in np.unique(labels1):
    ix1 = np.where(labels1 == l1)
    ax.scatter(X0[ix1], X1[ix1], c=cdict1[l1], label=labl1[l1], s=70, marker=marker1[l1], alpha=alpha1[l1])
ax.scatter(svm_model.support_vectors_[:, 0], svm_model.support_vectors_[:, 1], s=40, facecolors='none',
           edgecolors='navy', label='Support Vectors')
plot_contours(ax, classify, xx, yy, cmap='seismic', alpha=0.4)
plt.legend(fontsize=15)
plt.xlabel("1st Principal Component", fontsize=14)
plt.ylabel("2nd Principal Component", fontsize=14)
plt.show()


sklearn.svm.SVC(kernel="linear")

