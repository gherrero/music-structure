import numpy as np
import matplotlib.pyplot as plt
import structure_retrieval as sr
import structure_annotation as sa
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
from scipy import signal
import peakdetect
from sklearn import neighbors
from scipy.interpolate import interp1d

'''visualization stuff'''

gt_path   = sr.gt_path
desc_path = sr.desc_path
cand_list = sr.cand_list
sf_pickle = sr.sf_pickle
sf_path   = sr.sf_path
res_path  = 'annotation_results/chopin-all-n100/'

K = 5

font = {'family' : 'serif',
        'color'  : 'darkred',
        'weight' : 'normal',
        'size'   : 11,
        }

label_color =  {'0.0' : '#D8D8D8', #grey 
				'1.0' : '#7FC37F', #green 
				'2.0' : '#DCE27F', #yellow 
				'3.0' : '#9F7FFF', #blue
				'4.0' : '#FF9AF7', #magenta
				'5.0' : '#C1A883', #red
				'6.0' : '#B2EDEC', #cyan
				'7.0' : '#C6782C', #orange
				'8.0' : '#CCFF00',
				'9.0' : '#FFA9C1',
				}

def annotations(annotations_list):
	newall = []
	fig = plt.figure('Annotations')
	m = 0
	#i couldn't figure out how to get the max without all this shit first
	for annotation in annotations_list:
		new = [0.0]
		for line in annotation:
			new.append(float(line[-2]))
		if m <= max(new):	m = max(new)
	for i, annotation in enumerate(annotations_list):
		new = [0.0]
		for line in annotation:
			new.append(float(line[-2]))
			newall.append(new)
		p = plt.subplot(len(annotations_list),1,i+1)
		plt.title(filename_list[i],fontsize=9)
		plt.vlines(new, 0, 1, colors='k', linestyles='solid', label='')
		plt.vlines(new[-1], 0, 1, colors='r', linestyles='solid', label='') #red when it's the last boundary
		plt.tick_params(axis='y',which='both',right='off',left='off',top='off',labelleft='off')
		xinterval = np.arange(0,int(m),int(5)) 
		p.set_xticks(xinterval)
		p.set_xticklabels(xinterval,fontsize=10)
		plt.axis([0,m,0,1])
	fig.text(0.5, 0.04, 'time (s)', ha='center', va='center')
	plt.show()
 	
def addGaussians(query_fn):
	train_set = sa.getPickle(sf_pickle,cand_list)
	tree = neighbors.KDTree(train_set,leaf_size=100,p=2)

	songs_list   = sr.getNeighbors(query_fn,tree)
	M            = 23000 
	query_ann    = sr.getAnnotationList(gt_path,[query_fn])
	query_labels = [elem[-1] for elem in query_ann[0]]
	query_ann    = np.floor((np.array(sr.getAnnotation(query_ann))*1000)).astype(int)
	length       = query_ann[-1]
	total        = np.zeros(int(np.ceil(length)))
	
	neighbors_annotations = sr.getAnnotationList(gt_path,songs_list)
	neighbors_annotations_rescaled = []
	train_set = sa.getPickle(sf_pickle,cand_list)
	tree = neighbors.KDTree(train_set,leaf_size=100,p=2)
	fig = plt.figure()
	for c, song in enumerate(songs_list):
		print song
		gt_list = sr.getAnnotationList(gt_path,[song])
		ann = np.floor((np.array(sr.getAnnotation(gt_list))*1000)).astype(int) #convert to miliseconds to mantain res
 		neighbor_dur = ann[-1]
 		ann_with_sides = ann
		ann = ann[1:-1] #exclude starting and ending points
		a = np.zeros(int(np.ceil(length)))
		r = float(length)/float(neighbor_dur) #rescale according to query duration
		ann = np.floor(ann*r)

		ann_with_sides = np.floor(ann_with_sides*r) 
		labels = [x[-1] for x in gt_list[0]] # get the labels
		ax = fig.add_subplot(len(songs_list)+2,1,c+1)
		annotation_rescaled = []
		for elem in neighbors_annotations[c]:
			label=elem[-1] #save the label so it doesnt get affected by rescaling
			elem[0]=int(np.floor(float(elem[0])*1000*r)) #rescale the rest
			elem[1]=int(np.floor(float(elem[1])*1000*r))
			annotation_rescaled.append([elem[0],elem[1],label])
		neighbors_annotations_rescaled.append(annotation_rescaled)
		for i, loc in enumerate(ann,start=1):
			section_length=ann_with_sides[i]-ann_with_sides[i-1]
			sigma = 0.1*section_length
			g1 = signal.gaussian(M,std=sigma)
			half1 = int(np.floor(len(g1)/2))
			section_length = ann_with_sides[i+1]-ann_with_sides[i]
			sigma = 0.1*section_length
			g2 = signal.gaussian(M,std=sigma)
			half2 = int(np.floor(len(g2)/2))

			g = np.concatenate((g1[:half1],g2[half2:]))
			currentaxis = plt.gca()
			ax.add_patch(Rectangle((ann_with_sides[i-1], 0), ann_with_sides[i]-ann_with_sides[i-1], 1, facecolor=label_color[labels[i-1]], alpha=1))
			ax.add_patch(Rectangle((ann_with_sides[-2], 0), ann_with_sides[-1]-ann_with_sides[-2], 1, facecolor=label_color[labels[-1]], alpha=1))
		
			if loc < np.floor(M/2):
				a += np.array(np.concatenate((g[int(np.floor(M/2)-loc):],np.zeros(int(length-loc-np.floor(M/2))))))
			elif loc + np.floor(M/2) > length:
				a += np.array(np.concatenate((np.zeros(int(loc-np.floor(M/2))),g[:int(length+np.floor(M/2)-loc)])))
			else:
				a += np.array(np.concatenate((np.zeros(int(loc-np.floor(M/2))),g,np.zeros(int(length-loc-np.floor(M/2))))))
			ax.set_xticklabels([])
			ax.set_yticklabels([])
		ax.set_ylabel('K = %d'%(c+1),rotation='vertical')

		total += a
		plt.vlines(ann,0,1,colors='r')
		plt.plot(a,color='k')
		plt.xlim([0,length])
		plt.ylim([0,1])

	total = total/float(max(total))
	ax = fig.add_subplot(len(songs_list)+2,1,len(songs_list)+1)
	plt.plot(total)
	plt.xlim([0,length])
	plt.ylim([0,1])
	peaks = sr.getPeaks(total,neighbors_annotations)	
	all_songs_segmented = [sr.segmentLabel(elem) for elem in neighbors_annotations_rescaled]
	res_boundaries = sorted(peaks)
	res_boundaries.insert(0,0)
	res_boundaries.append(length) #cause all songs are supposed to be the same length now
	res_labels = sr.mergeLabels(res_boundaries,all_songs_segmented)

	for i in np.arange(len(res_labels)):
		ax.add_patch(Rectangle((res_boundaries[i], 0),res_boundaries[i+1]-res_boundaries[i], 1, facecolor  = label_color[res_labels[i]], alpha=1))
		ax.add_patch(Rectangle((res_boundaries[-2], 0), res_boundaries[-1]-res_boundaries[-2], 1, facecolor = label_color[res_labels[-1]], alpha=1))
	plt.vlines(peaks,0,1,'r','dotted',linewidths=2)
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_ylabel('Result')
	ax = fig.add_subplot(len(songs_list)+2,1,len(songs_list)+2)

	for i in np.arange(len(query_labels)):
		ax.add_patch(Rectangle((query_ann[i], 0),query_ann[i+1]-query_ann[i], 1, facecolor  = label_color[query_labels[i]], alpha=1))

	plt.vlines(query_ann,0,1,'g',linewidths=3)
	ax.set_yticklabels([])
	ax.set_xlabel('time (ms)')
	ax.set_ylabel('Ground Truth')
	plt.xlim([0,length])
	plt.ylim([0,1])
	plt.draw()
	res_annotations=sr.formatAnnotation(res_boundaries,res_labels)
	return res_annotations

def labels(query_fn):
	train_set      = sa.getPickle(sf_pickle,cand_list)
	tree           = neighbors.KDTree(train_set,leaf_size=100,p=2)
	songs_list     = sr.getNeighbors(query_fn,cand_list,tree,K) 
	print songs_list
	res_annotation = sr.labelsMethod(query_fn,cand_list,tree,K) 
	res_boundaries = sr.getAnnotation([res_annotation])
	res_boundaries = np.floor(np.array(res_boundaries)*1000).astype(int)
	res_labels     = [x[-1] for x in res_annotation]
	x=0
	y=0
	new=[]
	# while x<= len(res_boundaries):
	# 	if(abs(res_boundaries[i]-res_labels[j])):
	# 		new.append(res_labels[j])
	# 		i=i+1
	# 		j=j+1
	# 	else:
	# 		new.append(res_boundaries[i])
	# 		i=i+1
	# res_boundaries=new
	fig = plt.figure()
	query_ann    = sr.getAnnotationList(gt_path,[query_fn])
	query_labels = [elem[-1] for elem in query_ann[0]]
	query_ann    = np.floor((np.array(sr.getAnnotation(query_ann))*1000)).astype(int)
	length       = query_ann[-1]
	for c, song in enumerate(songs_list):
		print song
		gt_list = sr.getAnnotationList(gt_path,[song])
		ann = np.floor((np.array(sr.getAnnotation(gt_list))*1000)).astype(int) #convert to miliseconds to mantain res
 		neighbor_dur = ann[-1]
 		ann_with_sides = ann
		ann = ann[1:-1] #exclude starting and ending points
		a = np.zeros(int(np.ceil(length)))
		r = float(length)/float(neighbor_dur) #rescale according to query duration
		ann = np.floor(ann*r)
		ann_with_sides = np.floor(ann_with_sides*r).astype(int)
		labels = [x[-1] for x in gt_list[0]] # get the labels
		ax = fig.add_subplot(len(songs_list)+2,1,c+1)
		annotation_rescaled=[]
		for i, loc in enumerate(ann,start=1):
			currentaxis = plt.gca()
			ax.add_patch(Rectangle((ann_with_sides[i-1], 0), ann_with_sides[i]-ann_with_sides[i-1], 2, facecolor=label_color[labels[i-1]], alpha=1))
			ax.add_patch(Rectangle((ann_with_sides[-2], 0), ann_with_sides[-1]-ann_with_sides[-2], 2, facecolor=label_color[labels[-1]], alpha=1))
			ax.set_yticklabels([])
			ax.set_xticklabels([])

		plt.xlim([0,length])
		plt.ylim([0,1])
		ax.set_ylabel('K=%d'%(c+1))

	ax = fig.add_subplot(len(songs_list)+2,1,len(songs_list)+1)
	for i in np.arange(len(res_labels)):
		ax.add_patch(Rectangle((res_boundaries[i], 0),res_boundaries[i+1]-res_boundaries[i], 1, facecolor  = label_color[res_labels[i]], alpha=1))
		ax.add_patch(Rectangle((res_boundaries[-2], 0), res_boundaries[-1]-res_boundaries[-2], 1, facecolor = label_color[res_labels[-1]], alpha=1))
		ax.set_yticklabels([])
	ax.set_ylabel('Result')
	ax.set_yticklabels([],fontsize=10)
	ax.set_xticklabels([])
	plt.xlim([0,length])
	plt.ylim([0,1])
	ax = fig.add_subplot(len(songs_list)+2,1,len(songs_list)+2)
	for i in np.arange(len(query_labels)):
		ax.add_patch(Rectangle((query_ann[i], 0),query_ann[i+1]-query_ann[i], 1, facecolor  = label_color[query_labels[i]], alpha=1))
	ax.set_ylabel('Ground Truth')
	ax.set_yticklabels([],fontsize=10)
	ax.set_xlabel('time (ms)')
	plt.xlim([0,length])
	plt.ylim([0,1])
	print labels

def earlyFusion(query_fn):
	train_set = sa.getPickle(sf_pickle,cand_list)
	tree = neighbors.KDTree(train_set,leaf_size=100,p=2)
	songs_list = sr.getNeighbors(query_fn,cand_list,tree,K)
	query_ann = sr.getAnnotationList(gt_path,[query_fn])
	query_labels = [elem[-1] for elem in query_ann[0]]
	query_ann = np.floor((np.array(sr.getAnnotation(sr.getAnnotationList(gt_path,[query_fn])))*1000)).astype(int)
	length = query_ann[-1]
	w = (sa.m-1)-sa.tau # to account for delay coord delay its added at the end to locs
	sfList = []
	ratio = length/sa.n #downsampled to miliseconds
	fig = plt.figure()
	neighbors_annotations = sr.getAnnotationList(gt_path,songs_list)
	neighbors_annotations_rescaled = []

	for i, song in enumerate(songs_list):
		gt_list = sr.getAnnotationList(gt_path,[song])
		print song
		sfArray = np.loadtxt(sf_path+song[:-3]+'csv')
		sfMatrix = sa.convertToMatrix(sfArray)
		ax = fig.add_subplot(len(songs_list)+4,1,i+1)
		plt.imshow(sfMatrix, cmap = cm.binary,interpolation='nearest', aspect='auto')
		sfList.append(sfMatrix)
		annotation_rescaled = []
		ann = np.floor((np.array(sr.getAnnotation(gt_list))*1000)).astype(int) #convert to miliseconds to mantain res
 		neighbor_dur = ann[-1]
		r = float(length)/float(neighbor_dur) #rescale according to query duration
		for elem in neighbors_annotations[i]:
			label = elem[-1] #save the label so it doesnt get affected by rescaling
			elem[0] = int(np.floor(float(elem[0])*1000*r)) #rescale the rest
			elem[1] = int(np.floor(float(elem[1])*1000*r))
			annotation_rescaled.append([elem[0],elem[1],label])
		neighbors_annotations_rescaled.append(annotation_rescaled)
	sfQueryMatrix = sa.convertToMatrix(np.loadtxt(sf_path+query_fn))
	sfList.append(sfQueryMatrix)
	sfMean = sr.MeanSF(sfList)
	plt.subplot(len(songs_list)+4,1,len(songs_list)+1)
	plt.imshow(sfMean, cmap = cm.binary,interpolation='nearest', aspect='auto')
	novelty_curve = np.array(sa.novelty(sfMean))
	locs = sa.peakdet(novelty_curve)
	print "locs"
	# print 
	# locs=locs.tolist()
	# print peaks
	# locs = [elem[0] for elem in peaks[0]] #get only the idx
	# maxs = [elem[1] for elem in peaks[0]] #get only the value
	print locs
	ax = fig.add_subplot(len(songs_list)+4,1,len(songs_list)+2)
	plt.vlines(locs,0,1,'r',linewidth=2)
	plt.plot(novelty_curve)
	locs = np.array(locs)*ratio
	locs += (w/2)
	ax = fig.add_subplot(len(songs_list)+4,1,len(songs_list)+3)
	plt.vlines(locs,0,1,'k',linewidth=1)
	plt.xlim([0,length])
	all_songs_segmented = [sr.segmentLabel(elem) for elem in neighbors_annotations_rescaled]
	res_boundaries = locs
	res_boundaries = np.insert(res_boundaries,0,0)
	res_boundaries =	np.append(res_boundaries,length) #cause all songs are supposed to be the same length now
	res_labels = sr.mergeLabels(res_boundaries,all_songs_segmented)
	ax.set_yticklabels([])
	for i in np.arange(len(res_labels)):
		ax.add_patch(Rectangle((res_boundaries[i], 0),res_boundaries[i+1]-res_boundaries[i], 1, facecolor  = label_color[res_labels[i]], alpha=1))
		ax.add_patch(Rectangle((res_boundaries[-2], 0), res_boundaries[-1]-res_boundaries[-2], 1, facecolor = label_color[res_labels[-1]], alpha=1))
		ax.set_yticklabels([])
	ax = fig.add_subplot(len(songs_list)+4,1,len(songs_list)+4)
	plt.vlines(query_ann,0,1,'k',linewidth=1)
	plt.xlim([0,length])
	for i in np.arange(len(query_labels)):
		ax.add_patch(Rectangle((query_ann[i], 0),query_ann[i+1]-query_ann[i], 1, facecolor  = label_color[query_labels[i]], alpha=1))
	print res_labels
	return novelty_curve

def process(query_fn):
	fig      = plt.figure(query_fn[:-4])
	hpcps    = np.loadtxt(desc_path + query_fn, delimiter=',')
	R        = sa.ssm(sa.delayCoord(hpcps))
	P        = sa.gaussianBlur(sa.circShift(R))
	D        = sa.downsample(P)
	C        = sa.novelty(P)
	gt_list  = sr.getAnnotationList(gt_path,[query_fn])
	gt       = sr.getAnnotation(gt_list)
	ann_list = sr.getAnnotationList(res_path,[query_fn])
	ann      = sr.getAnnotation(ann_list)

	plt.rc('xtick', labelsize=10) 
	plt.rc('ytick', labelsize=10) 
	ax1 = plt.subplot2grid((6,2), (0,0), rowspan=2)
	ax1.set_title('HPCPs', fontdict=font)
	ax1.imshow(hpcps.T, interpolation='nearest', aspect='auto')

	ax2 = plt.subplot2grid((6,2), (2,0),rowspan=4)
	ax2.set_title('SSM',fontdict=font)
	ax2.imshow(R, cmap = cm.binary, interpolation='nearest')

	ax3 = plt.subplot2grid((6,2), (0, 1), rowspan=2, colspan=1)
	ax3.set_title('Structure Features',fontdict=font)
	ax3.imshow(P, cmap = cm.Reds,interpolation='nearest')
	ax3.set_aspect('auto')

	ax4 = plt.subplot2grid((6,2), (2, 1),rowspan=2)
	ax4.set_title('Novelty Curve',fontdict=font)
	ax4.set_xlim([0,len(C)])
	ax4.plot(C)

	ax5 = plt.subplot2grid((6,2), (4, 1))
	ax5.set_xticks(gt)
	ax5.tick_params(axis='y', which='both', right='off', left='off', top='off', labelleft='off')
	ax5.set_title('Ground Truth',fontdict=font)
	ax5.vlines(ann, 0, 1, colors='k', linestyles='solid', label='')

	ax6 = plt.subplot2grid((6,2), (5, 1))
	ax6.set_xticks(ann)
	ax6.tick_params(axis='y',which='both',right='off',left='off',top='off',labelleft='off')
	ax6.set_title('Annotation',fontdict=font)
	ax6.vlines(gt, 0, 1, colors='k', linestyles='solid', label='')

	plt.tight_layout()
	plt.draw()

if __name__ == "__main__":

	earlyFusion('Chopin_Op006No1_Ashkenazy-1981_pid9058-01.mp3.csv')
	# labels('Chopin_Op068No1_Sztompka-1959_pid9170b-19.mp3.csv')
	# labels('Chopin_Op006No1_Ashkenazy-1981_pid9058-01.mp3.csv')
	# addGaussians('Beatles_HoldMeTight_Beatles_1963-WithTheBeatles-09.wav.csv')
	plt.show()

