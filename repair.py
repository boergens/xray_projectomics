# -*- coding: utf-8 -*-

#assumes the first bit of matcher ran first
import numpy as np
import sklearn.metrics
collector = []
for key in output:
    unique, counts = np.unique(output[key].edges.flatten(), return_counts=True)
    for idx in range(sum(counts==1)):
        collector.append([key, unique[counts==1][idx]])
node_collector = []
node_collector_memory = []
for item in collector:
    node = output[item[0]].vertices[item[1]] 
    if np.all(node<np.array([1161,1161,1001])-5) and np.all(node>5):
        node_collector.append(node)
        node_collector_memory.append(item)

dm = sklearn.metrics.pairwise_distances(node_collector)
np.fill_diagonal(dm, np.Inf)

dm_meta = 1
trees = []
alpha=255
counter = 0
while(True):
    todos = np.where(dm<=dm_meta)
        
    flag = True
    
    for idx in range(len(todos[0])):
        if todos[0][idx]>todos[1][idx]:
            cosedges = []
            cosstore = []
            for lrid in range(2):
                collectorK = node_collector_memory[todos[lrid][idx]]
                nodeidK = collectorK[1]
                keyK = collectorK[0]
                edgeK = np.where(np.any(output[keyK].edges==nodeidK, axis=1))[0][0]
                follownodeidK = np.setdiff1d(output[keyK].edges[edgeK], [nodeidK])[0]
                for _ in range(10):
                    edgeK = np.setdiff1d(np.where(np.any(output[keyK].edges==follownodeidK, axis=1))[0],edgeK)
                    follownodeidK = np.setdiff1d(output[keyK].edges[edgeK], [follownodeidK])[0]
                    
                cosedges.append(output[keyK].vertices[nodeidK]-output[keyK].vertices[follownodeidK])
                cosstore.append(output[keyK].vertices[follownodeidK])
                
            if np.dot(-cosedges[0],cosedges[1])/np.linalg.norm(cosedges[0])/np.linalg.norm(cosedges[1]) < 1/np.sqrt(2):
                continue
            nodes = [wknml.Node(id=counter*4+0,
                                position=cosstore[0][[1,0,2]]),
                     wknml.Node(id=counter*4+1,
                                position=node_collector[todos[0][idx]][[1,0,2]]),
                     wknml.Node(id=counter*4+2,
                                position=node_collector[todos[1][idx]][[1,0,2]]),
                     wknml.Node(id=counter*4+3,
                                position=cosstore[1][[1,0,2]])]
            
            edges = [wknml.Edge(source=counter*4+0,target=counter*4+1),
                     wknml.Edge(source=counter*4+1,target=counter*4+2),
                     wknml.Edge(source=counter*4+2,target=counter*4+3)]

            color = [255,255,0,alpha]
            trees.append(wknml.Tree(id=counter,
                                    color=color,
                                    name='connectorbit'+str(counter),
                                    nodes=nodes,
                                    edges=edges))
            counter += 1
            dm[todos[0][idx],todos[1][idx]] = np.Inf
            flag = False
            
            break
    if flag:
        dm_meta += 1
        if dm_meta>5:
            break

    
    

nml = wknml.NML(trees=trees,
                branchpoints=[],
                comments=[],
                groups=[],
                parameters=wknml.NMLParameters(name='esrf_xray_1k_raw',
                                               scale=[50,50,50]))
with open("out7.nml", "wb") as f:
    wknml.write_nml(f, nml)
