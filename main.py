import wknml
import scipy.stats
import numpy as np
mypath = r"C:\Users\Kevin\Downloads\annotation-20230418T0801.079.k\annotation.nml"
with open(mypath, "rb") as f:
    nml = wknml.parse_nml(f)
collectora = []
collect_length = []
for tree in nml.trees:
    if tree.name.startswith('ma28') or tree.name.startswith('ma23')or tree.name.startswith('ma26'):
        continue
    nodes = np.array([node.position for node in tree.nodes])
    assert(nodes[0,0]==np.min(nodes[:,0]))
    diffs = np.diff(nodes, axis=0)
    collectora.append(diffs[:,1]/diffs[:,0])
    collectora.append(diffs[:,2]/diffs[:,0])
    collect_length.append(np.mean(np.sqrt(np.sum(diffs**2,axis=1))))
    print(nodes.shape[0])
print(np.mean(collect_length))
#%%
corr = [[] for _ in range(100)]
for idx, target in enumerate(corr):
    for item in collectora:
        if len(item) > idx:
            if ~np.isinf(item[idx]):
                target.append(item[idx])
                
corr_store=[1]                
corr_store.append(scipy.stats.pearsonr(corr[0],corr[1]).statistic)
corr_store.append(scipy.stats.pearsonr(corr[0],corr[2]).statistic)
corr_store.append(scipy.stats.pearsonr(corr[0],corr[3]).statistic)
corr_store.append(scipy.stats.pearsonr(corr[0],corr[4]).statistic)


package={}
for tree in nml.trees:
    if tree.name.startswith('ma28') or tree.name.startswith('ma23')or tree.name.startswith('ma26'):
        continue
    nodes = np.array([node.position for node in tree.nodes])
    diffs = np.diff(nodes, axis=0)
    tempa = np.array(corr_store)*diffs[:5,1]/diffs[:5,0]
    tempb = np.array(corr_store)*diffs[:5,2]/diffs[:5,0]
    
    package[tree.name[:4]] = [nodes[0,:], np.mean(tempa), np.mean(tempb)]
#%%

mypath = r"C:\Users\Kevin\Downloads\annotation-20230412T1709.151.k\annotation.nml"
with open(mypath, "rb") as f:
    nml = wknml.parse_nml(f)
tree_names = [tree.name.strip() for tree in nml.trees]
package2={}
for tree in nml.trees:
    if tree.name.startswith('ma28') or tree.name.startswith('ma23')or tree.name.startswith('ma26'):
        continue
    nodes = np.array([node.position for node in tree.nodes])
    assert(nodes[0,0]==np.max(nodes[:,0]))
    diffs = np.diff(nodes, axis=0)
    tempa = np.array(corr_store)*diffs[:5,1]/diffs[:5,0]
    tempb = np.array(corr_store)*diffs[:5,2]/diffs[:5,0]
    
    package2[tree.name[:4]] =([nodes[0,:], np.mean(tempa), np.mean(tempb)])
#%%
import scipy.optimize
matched = np.intersect1d(list(package.keys()), list(package2.keys()))

a= np.array([[[mypackage[key][0][idx] for mypackage in [package, package2]] for idx in range(3)] for key in matched])
aa=np.array([[[mypackage[key][idx]    for mypackage in [package, package2]] for idx in [1,2]]    for key in matched])             

#%%
import scipy.interpolate



def measure_tension(x,y):
    tension = 0
    for idx in range(x.shape[0]):
        interp = scipy.interpolate.CloughTocher2DInterpolator(np.delete(x,idx,axis=0),np.delete(y,idx,axis=0))
        if not np.isnan(np.linalg.norm(interp(x[idx,:])-y[idx,:])):
            tension += np.linalg.norm(interp(x[idx,:])-y[idx,:])
    return tension
def abcde(m):
    print(m)
    aaa=(a[:,:2,:]+m*aa)
   
    return measure_tension(aaa[:,:,0],aaa[:,:,1])
for idx in range(20):
    print(idx)
    print(abcde((idx-10)/10))
print(scipy.optimize.minimize(abcde,0.1))

print(sum(corr_store)*2*0.1324*0.6)

