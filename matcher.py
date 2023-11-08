
import numpy as np
from PIL import Image
import sys
import cv2
import scipy.spatial
import pickle
import kimimaro
import matplotlib.pyplot as plt
import scipy.spatial
import scipy.optimize
import wknml
import copy
#%%

data_for_skel=np.zeros([1161,1161,1001],dtype=np.uint32)

for idx in range(1001):
    print(idx)
    data_for_skel[:,:,idx] = np.array(Image.open(r"Z:\data_gregg\ESRF\Funke_LSD\kb_matchacrosskerf\esrf_1k_labels\slice" + str(idx+10_000)[1:] + '.tif'))

#%%

output = kimimaro.skeletonize(data_for_skel)
#%%

with open('filename.pickle', 'wb') as handle:
    pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#%%

with open(r"filename.pickle", "rb") as input_file:
    output = pickle.load(input_file)

#%%

x = [np.array(Image.open('slice0520.tif')), np.array(Image.open('slice0500.tif'))]
unq2 = np.unique(np.array(x))[1:]
collection = [[], []]
collection_start = [[],[]]
collection_id = [[], []]
for dir_idx in range(2):
    for idx in np.unique(x[dir_idx])[1:]:
        if idx not in output:
            continue
        _, binary_image = cv2.threshold((x[dir_idx]==idx).astype(np.uint8)*255, 1, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
        for idx_label in range(num_labels-1):
            dm = scipy.spatial.distance.cdist(output[idx].vertices,
                                              np.array([[centroids[idx_label+1,1],centroids[idx_label+1,0],520]]),
                                              'euclidean')
            mynode = np.argmin(dm)
            for_dijkstra = np.zeros([len(dm),len(dm)]) + np.Inf
            for eidx in range(output[idx].edges.shape[0]):
                thisedge = output[idx].edges[eidx,:]
                for_dijkstra[thisedge[0],thisedge[1]] = np.linalg.norm(output[idx].vertices[thisedge[0]]-output[idx].vertices[thisedge[1]])
            distM = scipy.sparse.csgraph.dijkstra(for_dijkstra, directed=False, indices = mynode)
            goodones = np.logical_and(np.logical_and(distM>140,distM<150), (1-2*dir_idx)*output[idx].vertices[:,2]>(1-2*dir_idx)*output[idx].vertices[mynode,2])
            if np.sum(goodones) > 0:
                vector_calc = np.mean(output[idx].vertices[goodones,:],axis = 0) - output[idx].vertices[mynode,:]
            else:
                goodones = np.where((1-2*dir_idx)*output[idx].vertices[:,2]>(1-2*dir_idx)*output[idx].vertices[mynode,2])[0]
                if np.sum(goodones) == 0:
                    continue
                thisgood = goodones[np.argmax(distM[goodones])]
                vector_calc = output[idx].vertices[thisgood,:]-output[idx].vertices[mynode,:]
            vector_calc /= vector_calc[2]
            if np.linalg.norm(vector_calc) > 10:
                vector_calc = vector_calc * 0
            collection[dir_idx].append(np.array([centroids[idx_label+1,1],centroids[idx_label+1,0]]) - 10 * (1-2*dir_idx)* vector_calc[:2])
            collection_start[dir_idx].append(np.array([centroids[idx_label+1,1],centroids[idx_label+1,0]]))
            collection_id[dir_idx].append(idx)
            
                
#%%

dm2 = scipy.spatial.distance_matrix(collection[0],collection[1])
matching = scipy.optimize.linear_sum_assignment(np.sqrt(dm2))
score = 0
collection_correct = copy.deepcopy(collection)
collection_singleton = copy.deepcopy(collection)

for idx in range(len(matching[0])):
    if collection_id[0][matching[0][idx]] == collection_id[1][matching[1][idx]]:
        score += 1
        collection_correct[0][matching[0][idx]] = True
        collection_correct[1][matching[1][idx]] = True
    else:
        collection_correct[0][matching[0][idx]] = False
        collection_correct[1][matching[1][idx]] = False
        flag = len(np.intersect1d([collection_id[0][matching[0][idx]], collection_id[1][matching[1][idx]]],
                                  np.setxor1d(collection_id[0],collection_id[1]))) > 0
        collection_singleton[0][matching[0][idx]] = flag
        collection_singleton[1][matching[1][idx]] = flag
        
            
        
#%%
dm_fake = np.ones([len(collection_id[0]),len(collection_id[1])])
for id_1,item_1 in enumerate(collection_id[0]):
    for id_2,item_2 in enumerate(collection_id[1]):
        if item_1 == item_2:
            dm_fake[id_1,id_2]=0
matching = scipy.optimize.linear_sum_assignment(np.sqrt(dm_fake))
score_fake = 0
for idx in range(len(matching[0])):
    if collection_id[0][matching[0][idx]] == collection_id[1][matching[1][idx]]:
        score_fake += 1


#%%

trees = []
alpha=255
counter =0

for idx, key in enumerate(output):
    nodes = []
    edges = []
    lookup = dict()
    for idx_row in range(output[key].vertices.shape[0]):
        lookup[idx_row]=counter
        nodes.append(wknml.Node(id=counter,
                                position=[output[key].vertices[idx_row,1],
                                          output[key].vertices[idx_row,0],
                                          output[key].vertices[idx_row,2]]))
        counter += 1
    for idx_row in range(output[key].edges.shape[0]):
        edges.append(wknml.Edge(source=lookup[output[key].edges[idx_row,0]],
                                target=lookup[output[key].edges[idx_row,1]]))

    trees.append(wknml.Tree(id=idx,
                            color=[255,0,0,alpha],
                            name='id'+str(key),
                            nodes=nodes,
                            edges=edges))

nml = wknml.NML(trees=trees,
                branchpoints=[],
                comments=[],
                groups=[],
                parameters=wknml.NMLParameters(name='esrf_xray_1k_raw',
                                               scale=[50,50,50]))
with open("out5.nml", "wb") as f:
    wknml.write_nml(f, nml)


#%%
trees = []
alpha=255
counter =0
pos_start = [520, 500]
for dir_idx in range(2):
    for idx, _ in enumerate(collection[dir_idx]):
        nodes = [wknml.Node(id=counter,
                            position=np.hstack([collection[dir_idx][idx][[1,0]],[np.mean(pos_start)]])),
                 wknml.Node(id=counter+1,
                            position=np.hstack([collection_start[dir_idx][idx][[1,0]],[pos_start[dir_idx]]]))]
        edges = [wknml.Edge(source=counter,target=counter+1)]
        counter += 2
        color = [255,255,0,alpha]
        if collection_correct[dir_idx][idx] is True:
            color = [0,255,0,alpha]
        if collection_correct[dir_idx][idx] is False:
            if collection_singleton[dir_idx][idx]:
                continue
            color = [255,0,0,alpha]
        
        trees.append(wknml.Tree(id=idx*2+dir_idx,
                                color=color,
                                name='id'+str(collection_id[dir_idx][idx]),
                                nodes=nodes,
                                edges=edges))

nml = wknml.NML(trees=trees,
                branchpoints=[],
                comments=[],
                groups=[],
                parameters=wknml.NMLParameters(name='esrf_xray_1k_raw',
                                               scale=[50,50,50]))
with open("out6.nml", "wb") as f:
    wknml.write_nml(f, nml)
