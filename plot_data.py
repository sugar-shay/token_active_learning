# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 12:07:26 2021

@author: Shadow
"""

import numpy as np
import matplotlib.pyplot as plt


import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from collections import Counter
import pandas as pd

import pickle 
import os
from collections import Counter 


category = 'memc'
directory = 'results/memc'


with open(directory+'/'+'bert_train_stats.pkl', 'rb') as f:
    log = pickle.load(f)
    
with open(directory+'/'+'bert_test_stats.pkl', 'rb') as f:
    test_log = pickle.load(f)

if not os.path.exists(directory+'/plots'):
        os.makedirs(directory+'/plots')


#Creating the Train/Val loss and acc plots
epoch_range = range(1,len(log['train_losses'])+1)
loss_plot1, = plt.plot(epoch_range, log['train_losses'])
loss_plot2, = plt.plot(epoch_range, log['val_losses'][:-1]) #this includes the sanity check but not the last val loss
plt.legend([loss_plot1, loss_plot2],['train', 'val'], loc = 'center right', bbox_to_anchor=(0.5, -0.4))
plt.xlabel('Epochs')
plt.ylabel('Loss')
#title = directory + ' Train/Val Loss'
title = 'MEMC Train/Val Loss'
plt.title(title)
plt.show()

#plt.savefig(directory+'/plots/'+category+'_loss.png')

acc_plot1, = plt.plot(epoch_range, log['train_accs'])
acc_plot2, = plt.plot(epoch_range, log['val_accs'][:-1]) #this includes the sanity check but not the last val loss
plt.legend([acc_plot1, acc_plot2],['train', 'val'], loc = 'center right', bbox_to_anchor=(0.5, -0.4))
plt.xlabel('Epochs')
plt.ylabel('Acc')
#title = directory + ' Train/Val Acc'
title = 'MEMC Train/Val Acc'
plt.title(title)
plt.show()
#plt.savefig(directory+'/plots/'+category+'_acc.png')


#Calculating the variance, confidence, and correctness of the training samples
confidence = np.mean(log['gt_probs'], axis=-1)
variance = np.var(log['gt_probs'], axis=-1)
#correctness = np.mean(log['correctness'], axis = -1)

correctness = []
for idx in range(log['correctness'].shape[0]):
    
    correct_bool = log['correctness'][idx,:]
    correctness.append(np.count_nonzero(correct_bool))

correctness = np.array(correctness)

#Performing our cluster analysis
from sklearn.cluster import SpectralClustering

#this includes correctness
cluster_data_df = pd.DataFrame(data={'variance': variance, 'confidence':confidence, 'correctness':correctness, 'labels': log['train_labels']})

#cluster_data_sample = cluster_data_df.sample(n = 25000, random_state=0)

def get_clusters(cluster_data):
    
    #clustering 3d includes correctness as a feature
    clustering_3d = SpectralClustering(n_clusters=3,
                                       assign_labels='discretize',
                                       random_state=0).fit(cluster_data.iloc[:, :-1])
    
    #clustering 2d does not include correctnesss as a feature
    #this is how the clusters were created in datamaps and mind your outliers 
    clustering_2d = SpectralClustering(n_clusters=3,
                                       assign_labels='discretize',
                                       random_state=0).fit(cluster_data.iloc[:,:-2])
    
    return {'cluster2d_labels':clustering_2d.labels_, 'cluster3d_labels':clustering_3d.labels_}

if os.path.isfile(directory+'/'+'clusters.pkl') == True:
    with open(directory+'/'+'clusters.pkl', 'rb') as f:
        clusters = pickle.load(f)

else:
    clusters = get_clusters(cluster_data_df)
    with open(directory+'/'+'clusters.pkl', 'wb') as f:
        pickle.dump(clusters, f)
    


#2D Data Maps for clusters created with Var/Conf and clusters created from Var/Conf/Corr
plt.scatter(cluster_data_df['variance'], cluster_data_df['confidence'], c = clusters['cluster2d_labels'], cmap='brg')
plt.xlabel('Variance')
plt.ylabel('Confidence')
plt.title('HTTPRS Data Map w/ Var/Conf Clusters')
plt.show()

plt.scatter(cluster_data_df['variance'], cluster_data_df['confidence'], c = clusters['cluster3d_labels'], cmap='brg')
plt.xlabel('Variance')
plt.ylabel('Confidence')
plt.title('HTTPRS Data Map w/ Var/Conf/Corr Clusters')
plt.show()

#3D data map with clusters created from variance and confidence 
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

ax.set_xlabel("Variance")
ax.set_ylabel("Confidence")
ax.set_zlabel("Correctness")
ax.set_title('HTTPRS 3D Data Map w/ Var/Conf Clusters')

ax.scatter(cluster_data_df['variance'], cluster_data_df['confidence'], cluster_data_df['correctness'], c = clusters['cluster2d_labels'], cmap = 'brg')

plt.show()

#3D data map with clusters created from variance/confidence/correctness
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

ax.set_xlabel("Variance")
ax.set_ylabel("Confidence")
ax.set_zlabel("Correctness")
ax.set_title('HTTPRS 3D Data Map w/ Var/Conf/Correct Clusters')

ax.scatter(cluster_data_df['variance'], cluster_data_df['confidence'], cluster_data_df['correctness'], c = clusters['cluster3d_labels'], cmap = 'brg')

plt.show()

#calculating statistics for 2D cluster

def cluster_region_stats(cluster_labels, cluster_data_df):
    
    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = [i for i in range(0, len(cluster_labels))]
    cluster_map['cluster'] = cluster_labels
    
     #Cluster 1
    cluster1_mask = np.array(cluster_map[cluster_map.cluster == 0].data_index)
    #print('cluster mask shape: ', cluster1_mask.shape)
    cluster1 = cluster_data_df.iloc[cluster1_mask,:]
    
    '''
    #2D Datamap
    plt.scatter(cluster1['variance'], cluster1['confidence'])
    plt.xlabel('Variance')
    plt.ylabel('Confidence')
    plt.title('Cluster1 2D')
    plt.show()
    '''
    #3D data map 
    cluster1_fig = plt.figure()
    cluster1_ax = cluster1_fig.add_subplot(111, projection = '3d')
    
    cluster1_ax.set_xlabel("Variance")
    cluster1_ax.set_ylabel("Confidence")
    cluster1_ax.set_zlabel("Correctness")
    cluster1_ax.set_title('Cluster #1 3D')
    
    cluster1_ax.scatter(cluster1['variance'], cluster1['confidence'], cluster1['correctness'])
    
    plt.show()
    tag_count = Counter(cluster1['labels'].to_list())
    print()
    print('The # of "O" tags in Cluster1: ', tag_count['O'])
    print('The # of "SN" tags in Cluster1: ', tag_count['SN'])
    print('The # of "SV" tags in Cluster1: ', tag_count['SV'])
    
    #Cluster 2
    cluster2_mask = np.array(cluster_map[cluster_map.cluster == 1].data_index)
    #print('cluster mask shape: ', cluster1_mask.shape)
    cluster2 = cluster_data_df.iloc[cluster2_mask,:]
    
    '''
    #2D Datamap
    plt.scatter(cluster2['variance'], cluster2['confidence'])
    plt.xlabel('Variance')
    plt.ylabel('Confidence')
    plt.title('Cluster1 2D')
    plt.show()
    '''
    
    
    #3D data map 
    cluster2_fig = plt.figure()
    cluster2_ax = cluster2_fig.add_subplot(111, projection = '3d')
    
    cluster2_ax.set_xlabel("Variance")
    cluster2_ax.set_ylabel("Confidence")
    cluster2_ax.set_zlabel("Correctness")
    cluster2_ax.set_title('Cluster #2')
    
    cluster2_ax.scatter(cluster2['variance'], cluster2['confidence'], cluster2['correctness'])
    
    plt.show()
    
    tag_count = Counter(cluster2['labels'].to_list())
    print()
    print('The # of "O" tags in Cluster2: ', tag_count['O'])
    print('The # of "SN" tags in Cluster2: ', tag_count['SN'])
    print('The # of "SV" tags in Cluster2: ', tag_count['SV'])
    
    #Cluster 3
    cluster3_mask = np.array(cluster_map[cluster_map.cluster == 2].data_index)
    #print('cluster mask shape: ', cluster1_mask.shape)
    cluster3 = cluster_data_df.iloc[cluster3_mask,:]
    
    #3D data map with clusters created from variance/confidence/correctness
    cluster3_fig = plt.figure()
    cluster3_ax = cluster3_fig.add_subplot(111, projection = '3d')
    
    cluster3_ax.set_xlabel("Variance")
    cluster3_ax.set_ylabel("Confidence")
    cluster3_ax.set_zlabel("Correctness")
    cluster3_ax.set_title('Cluster #3')
    
    cluster3_ax.scatter(cluster3['variance'], cluster3['confidence'], cluster3['correctness'])
    
    plt.show()
    
    tag_count = Counter(cluster3['labels'].to_list())
    print()
    print('The # of "O" tags in Cluster3: ', tag_count['O'])
    print('The # of "SN" tags in Cluster3: ', tag_count['SN'])
    print('The # of "SV" tags in Cluster3: ', tag_count['SV'])

cluster_region_stats(clusters['cluster2d_labels'], cluster_data_df)

cluster_region_stats(clusters['cluster3d_labels'], cluster_data_df)




'''
cluster_count2d = Counter(clusters['cluster3d_labels'])
cluster_count3d = Counter(clusters['cluster3d_labels'])

def get_cluster_regions(cluster_labels, cluster_data_df):
    
    
    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = [i for i in range(0, len(cluster_labels))]
    cluster_map['cluster'] = cluster_labels
    
    #print('Cluster maps shape: ', cluster_map.shape)
    
    
    #Cluster 1
    cluster1_mask = np.array(cluster_map[cluster_map.cluster == 0].data_index)
    #print('cluster mask shape: ', cluster1_mask.shape)
    cluster1 = cluster_data_df.iloc[cluster1_mask,:]
    
    #the first dim of cluster_mean is the variance and the second is the confidenc 
    cluster1_mean = cluster1['confidence'].mean()
    
    #Cluster 2
    cluster2_mask = cluster_map[cluster_map.cluster == 1].data_index
    cluster2 = cluster_data_df.iloc[cluster2_mask, :]
    
    cluster2_mean = cluster2['confidence'].mean()
    
    
    #Cluster 3
    cluster3_mask = cluster_map[cluster_map.cluster == 2].data_index
    cluster3 = cluster_data_df.iloc[cluster3_mask, :]
    
    cluster3_mean = cluster3['confidence'].mean()
    
    clus_means = [cluster1_mean, cluster2_mean, cluster3_mean]
    
    if cluster1_mean == np.max(clus_means):
        easy_cluster = cluster1
        
    elif cluster1_mean == np.min(clus_means):
        hard_cluster = cluster1
    
    else:
        ambig_cluster = cluster1
    
    
    if cluster2_mean == np.max(clus_means):
        easy_cluster = cluster2
        
    elif cluster2_mean == np.min(clus_means):
        hard_cluster = cluster2
    
    else:
        ambig_cluster = cluster2
    
    
    if cluster3_mean == np.max(clus_means):
        easy_cluster = cluster3
        
    elif cluster3_mean == np.min(clus_means):
        hard_cluster = cluster3
    
    else:
        ambig_cluster = cluster3
    
    return {'easy':easy_cluster, 'ambig':ambig_cluster, 'hard':hard_cluster}

#analysis of clusters created from Var, Conf
cluster_region2d = get_cluster_regions(clusters['cluster2d_labels'], cluster_data_df)

with open(directory+'/cluster_region2d.pkl', 'wb') as f:
    pickle.dump(cluster_region2d, f)

print('# of samples in easy cluster: ', cluster_region2d['easy'].shape[0])
print('# of samples in ambig cluster: ', cluster_region2d['ambig'].shape[0])
print('# of samples in hard cluster: ', cluster_region2d['hard'].shape[0])

easy_df = cluster_region2d['easy']

easy_cluster_mask = np.array(easy_df[easy_df.correctness == 0.0])
print('The # of <= .4 correctness samples in the easy class: ', len(easy_cluster_mask))

cluster_region3d = get_cluster_regions(clusters['cluster3d_labels'], cluster_data_df)

with open(directory+'/cluster_region3d.pkl', 'wb') as f:
    pickle.dump(cluster_region3d, f)

print('# of samples in easy cluster: ', cluster_region3d['easy'].shape[0])
print('# of samples in ambig cluster: ', cluster_region3d['ambig'].shape[0])
print('# of samples in hard cluster: ', cluster_region3d['hard'].shape[0])
'''
