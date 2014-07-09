import numpy as np
import matplotlib.pyplot as plt
import structure_retrieval as sr
import structure_annotation as sa
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
from scipy import signal
import peakdetect
from sklearn import neighbors


'''visualization stuff'''

gt_path   = 'metadata/all/'	
desc_path = 'hpcp_ah6_al5_csv/'
cand_list = 'annotation_results/alldatasets.txt'
sf_pickle = 'pickles/sf-alldatasets-n100.pickle'

res_path  = 'annotation_results/chopin-all-n100/'

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
	m=0
	#i couldn't figure out how to get the max without all this shit
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
	M            = 23000 #VALORES MUY ALTOS FALLA si M/2>loc+length
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
		labels=[x[-1] for x in gt_list[0]] # get the labels
		ax = fig.add_subplot(len(songs_list)+2,1,c+1)
		annotation_rescaled=[]
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
			half1=int(np.floor(len(g1)/2))
			section_length=ann_with_sides[i+1]-ann_with_sides[i]
			sigma=0.1*section_length
			g2=signal.gaussian(M,std=sigma)
			half2=int(np.floor(len(g2)/2))

			g=np.concatenate((g1[:half1],g2[half2:]))
			currentaxis=plt.gca()
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
	total=total/float(max(total))
	ax = fig.add_subplot(len(songs_list)+2,1,len(songs_list)+1)
	plt.plot(total)
	plt.xlim([0,length])
	plt.ylim([0,1])
	peaks = sr.getPeaks(total,neighbors_annotations)
	
	all_songs_segmented = [sr.segmentLabel(elem) for elem in neighbors_annotations_rescaled]
	res_boundaries=sorted(peaks)
	res_boundaries.insert(0,0)
	res_boundaries.append(length) #cause all songs are supposed to be the same length now
	res_labels = sr.mergeLabels(res_boundaries,all_songs_segmented)

	# print sr.formatAnnotation(res_boundaries,res_labels)
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
	train_set = sa.getPickle(sf_pickle,cand_list)
	tree = neighbors.KDTree(train_set,leaf_size=100,p=2)
	songs_list   = sr.getNeighbors(query_fn,tree) 
	res_annotation = sr.labelsMethod(query_fn,tree) 
	res_boundaries=sr.getAnnotation([res_annotation])
	res_boundaries    = np.floor(np.array(res_boundaries)*1000).astype(int)
	res_labels=[x[-1] for x in res_annotation]

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
		labels=[x[-1] for x in gt_list[0]] # get the labels
		ax = fig.add_subplot(len(songs_list)+2,1,c+1)
		annotation_rescaled=[]
		for i, loc in enumerate(ann,start=1):
			currentaxis=plt.gca()
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

	addGaussians('Beatles_TellMeWhatYouSee_Beatles_1965-Help-11.wav.csv')
	# labels('Chopin_Op007No2_Bacha-1997_pid9166c-09.mp3.csv')
	# labels('Beatles_AcrossTheUniverse_Beatles_1970-LetItBe-03.wav.csv')
	plt.show()

