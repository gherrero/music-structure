import numpy as np
import matplotlib.pyplot as plt
import structure_retrieval as sr
import structure_annotation as sa
import matplotlib.cm as cm
from scipy import signal

'''receives a list of annotations an plots them'''

gt_path         = 'metadata/all/'
desc_path       = 'hpcp_ah6_al5_csv/'
res_path        = 'annotation_results/beatlesQMUL-n100/'


font = {'family' : 'serif',
        'color'  : 'darkred',
        'weight' : 'normal',
        'size'   : 11,
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
	# plt.tight_layout()
	plt.show()
 
def gaussians(query_fn):
	gt_list = sr.getAnnotationList(gt_path,[query_fn])
	ann     = getAnnotation(gt_list)
	
	ann    = np.floor((np.array(ann)*1000)).astype(int) #convert to miliseconds to mantain res
	length = np.ceil(ann[-1])
	M      = 10000 #must be even

	ann=ann[1:-1]
	g = signal.gaussian(M,std=1000)
	a=np.zeros(int(np.ceil(length)))

	for loc in ann:
		if loc < np.floor(M/2):
			a+=np.array(np.concatenate((g[int(np.floor(M/2)-loc):],np.zeros(int(length-loc-np.floor(M/2))))))
		elif loc + np.floor(M/2) > length:
			a+=np.array(np.concatenate((np.zeros(int(loc-np.floor(M/2))),g[:int(length+np.floor(M/2)-loc)])))
		else:
			a+=np.array(np.concatenate((np.zeros(int(loc-np.floor(M/2))),g,np.zeros(int(length-loc-np.floor(M/2))))))
	plt.vlines(ann,0,1,colors='r')
	plt.plot(a)
	plt.show()
	

def addGaussians(query_fn):
	songs_list = sr.getNeighbors(query_fn)
	M          = 20000
	delta      = 1000
	g          = signal.gaussian(M,std=delta)
	length     = 0
	lengths    = []
	# this next line is impossible to read. change in the near future
	query_ann    = np.floor((np.array(getAnnotation(sr.getAnnotationList(gt_path,[query_fn])))*1000)).astype(int)
	query_dur  = query_ann[-1]
	print query_dur
	for song in songs_list:
		gt_list    = sr.getAnnotationList(gt_path,[song])
		ann        = getAnnotation(gt_list)
		ann        = np.floor((np.array(ann)*1000)).astype(int)
		lengths.append(ann[-1])
	length=query_dur
	total=np.zeros(int(np.ceil(length)))

	for i, song in enumerate(songs_list):
		print song
		gt_list = sr.getAnnotationList(gt_path,[song])
		ann     = getAnnotation(gt_list)
		ann     = np.floor((np.array(ann)*1000)).astype(int) #convert to miliseconds to mantain res
		neighbor_dur = ann[-1]
		ann=ann[1:-1]
		a=np.zeros(int(np.ceil(length)))
		r=float(query_dur)/float(neighbor_dur) #rescale according to query duration
		ann=np.floor(ann*r)

		for loc in ann:
			if loc < np.floor(M/2):
				a+=np.array(np.concatenate((g[int(np.floor(M/2)-loc):],np.zeros(int(length-loc-np.floor(M/2))))))
			elif loc + np.floor(M/2) > length:
				a+=np.array(np.concatenate((np.zeros(int(loc-np.floor(M/2))),g[:int(length+np.floor(M/2)-loc)])))
			else:
				a+=np.array(np.concatenate((np.zeros(int(loc-np.floor(M/2))),g,np.zeros(int(length-loc-np.floor(M/2))))))
		total+=a
		plt.subplot(len(songs_list)+1,1,i+1)
		plt.vlines(ann,0,1,colors='r')
		plt.plot(a)
		plt.xlim([0,length])
	plt.subplot(len(songs_list)+1,1,len(songs_list)+1)
	plt.plot(total)
	plt.vlines(query_ann,0,1,'g',linewidths=3)
	plt.xlim([0,length])
	plt.show()

def getAnnotation(ann_list):
	new=[0.0]
	for line in ann_list[0]:
		new.append(float(line[-2]))
	return new

def process(query_fn):
	fig      = plt.figure(query_fn[:-4])
	hpcps    = np.loadtxt(desc_path + query_fn, delimiter=',')
	R        = sa.ssm(sa.delayCoord(hpcps))
	P        = sa.gaussianBlur(sa.circShift(R))
	D        = sa.downsample(P)
	C        = sa.novelty(P)
	gt_list  = sr.getAnnotationList(gt_path,[query_fn])
	gt       = getAnnotation(gt_list)
	ann_list = sr.getAnnotationList(res_path,[query_fn])
	ann      = getAnnotation(ann_list)

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

	# ax7 = plt.subplot2grid((8,4), (5, 2),rowspan=2, colspan=2)
	# ax7.text(0.1,0.8,'Query and results information',fontsize=15,fontweight='bold')

	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	# ann_list = sr.getAnnotationList(filename_list)
	# process('Beatles_AllYouNeedIsLove_Beatles_1967-MagicalMysteryTour-11.wav.csv')
	# print getAnnotation(ann_list)
	# annotations(ann_list)
	# gaussians('Beatles_AllYouNeedIsLove_Beatles_1967-MagicalMysteryTour-11.wav.csv')
	addGaussians('Chopin_Op006No1_Magin-1975_pid9138-01.mp3.csv')

