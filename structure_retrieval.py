import numpy as np
import cProfile
from sklearn import neighbors
import os.path
import structure_annotation as sa
from scipy import signal
from scipy import stats
import peakdetect





gt_path   = 'metadata/all/'
desc_path = 'hpcp_ah6_al5_csv/'
cand_list = 'annotation_results/alldatasets.txt'
sf_pickle = 'pickles/sf-alldatasets-n100.pickle'

sf_path   = 'sfs/sf-alldatasets-n100/'
K         = 5
T         = 500
songList  = open(cand_list).readlines()


#------GAUSSIANS METHOD---------#
def addGaussians(query_fn,tree):
	songs_list = getNeighbors(query_fn,tree)
	M          = 20000
	delta      = 1000
	g          = signal.gaussian(M,std=delta)

	query_ann    = getAnnotationList(gt_path,[query_fn])
	query_labels = [elem[-1] for elem in query_ann[0]]
	query_ann    = np.floor((np.array(getAnnotation(query_ann))*1000)).astype(int)
	length       = query_ann[-1]
	total        = np.zeros(int(np.ceil(length)))
	neighbors_annotations_rescaled = []

	neighbors_annotations = getAnnotationList(gt_path,songs_list)

	annotation_rescaled=[]

	for i, song in enumerate(songs_list):
		# print song
		gt_list = getAnnotationList(gt_path,[song])
		ann     = np.floor((np.array(getAnnotation(gt_list))*1000)).astype(int) #convert to miliseconds to mantain res
		neighbor_dur = ann[-1]
		ann = ann[1:-1]
		a = np.zeros(int(np.ceil(length)))
		r = float(length)/float(neighbor_dur) #rescale according to query duration
		ann = np.floor(ann*r)
		labels=[x[-1] for x in gt_list[0]] # get the labels

		for elem in neighbors_annotations[i]:
			label=elem[-1] #save the label so it doesnt get affected by rescaling
			elem[0]=int(np.floor(float(elem[0])*1000*r)) #rescale the rest
			elem[1]=int(np.floor(float(elem[1])*1000*r))
			annotation_rescaled.append([elem[0],elem[1],label])
		neighbors_annotations_rescaled.append(annotation_rescaled)
		for loc in ann:
			if loc < np.floor(M/2):
				a+=np.array(np.concatenate((g[int(np.floor(M/2)-loc):],np.zeros(int(length-loc-np.floor(M/2))))))
			elif loc + np.floor(M/2) > length:
				a+=np.array(np.concatenate((np.zeros(int(loc-np.floor(M/2))),g[:int(length+np.floor(M/2)-loc)])))
			else:
				a+=np.array(np.concatenate((np.zeros(int(loc-np.floor(M/2))),g,np.zeros(int(length-loc-np.floor(M/2))))))
		total+=a
	total=total/float(max(total))

	peaks = getPeaks(total,neighbors_annotations)
	# print "peaks"
	# print peaks
	all_songs_segmented = [segmentLabel(elem) for elem in neighbors_annotations_rescaled]
	res_boundaries=sorted(peaks)
	res_boundaries.insert(0,0)
	res_boundaries.append(length)
	res_labels = mergeLabels(res_boundaries,all_songs_segmented)
	res_annotations = formatAnnotation(res_boundaries,res_labels)
	return res_annotations
# ---------------------------- #

# ------- LABELS METHOD ------ #
def labelsMethod(query_fn,tree):
	songs_list = getNeighbors(query_fn,tree)
	query_ann  = np.floor((np.array(getAnnotation(getAnnotationList(gt_path,[query_fn])))*1000)).astype(int)
	length     = query_ann[-1]
	neighbors_annotations = getAnnotationList(gt_path,songs_list)
	neighbors_annotations_rescaled = []
	for i, song in enumerate(songs_list):
		gt_list = getAnnotationList(gt_path,[song])
		ann     = np.floor((np.array(getAnnotation(gt_list))*1000)).astype(int) #convert to miliseconds to mantain res
		neighbor_dur = ann[-1]
		ann = ann[1:-1]
		r = float(length)/float(neighbor_dur) #rescale according to query duration
		ann = np.floor(ann*r)
		labels=[x[-1] for x in gt_list[0]] # get the labels
		for elem in neighbors_annotations[i]:
			label=elem[-1] #save the label so it doesnt get affected by rescaling
			elem[0]=int(np.floor(float(elem[0])*1000*r)) #rescale the rest
			elem[1]=int(np.floor(float(elem[1])*1000*r))
			annotation_rescaled.append([elem[0],elem[1],label])
		neighbors_annotations_rescaled.append(annotation_rescaled)
	all_songs_segmented = [sr.segmentLabel(elem) for elem in neighbors_annotations_rescaled]
	res_labels  = []
	for i in np.arange(len(all_songs_segmented[0])):
		current_value=[]
		for song in all_songs_segmented:
			current_value.append(song[i])
		res_labels.append(list(stats.mode(current_value)[0])[0])

	return res_labels
# --------------------------- #

def querySong(query,tree,Kneighbors):
	index = tree.query(query,k=Kneighbors,return_distance=False)
	return index

def purgeNeighbors(neighbors_fn_list):
	'''Removes occurrences of same piece in the neighbors list'''
	comp_string = [elem.replace('-','_').split('_')[1] for elem in neighbors_fn_list]
	new_list=[]
	repeated = list(set([x for x in comp_string if comp_string.count(x) > 1]))
	if not repeated:
		new_list=neighbors_fn_list
	else:
		for i,elem in enumerate(neighbors_fn_list):
			if comp_string[i] in elem and not comp_string[i] in repeated:
				new_list.append(elem)
		repeated_complete=[]
		for x in neighbors_fn_list:
			for r in repeated:
				if r in x: repeated_complete.append(x)
		new_list.append(repeated_complete[0])
	return new_list

def getNeighbors(query_fn,tree):
	sf_query = np.loadtxt(sf_path+query_fn)
	songnumb = querySong(sf_query,tree,Kneighbors=K).tolist()[0]
	neighbors_fn_list = [songList[i][:-1] for i in songnumb if not songList[i][:-5]==query_fn[:-4]]
	neighbors_fn_list = purgeNeighbors(neighbors_fn_list)
	c=0
	while len(neighbors_fn_list)!=K:
		c+=1
		sf_query = np.loadtxt(sf_path+query_fn)
		songnumb = querySong(sf_query,tree,K+c).tolist()[0]
		neighbors_fn_list = [songList[i][:-1] for i in songnumb if not songList[i][:-5]==query_fn[:-4]]
		neighbors_fn_list = purgeNeighbors(neighbors_fn_list)
	return neighbors_fn_list

def unpickleAsArray(sf_pickle):
	dataM = []
	data = sa.getPickle(sf_pickle)
	for row in data:
		dataM.append(sa.convertToMatrix(row,50))
	return dataM

def getPeaks(signal, neighbors_annotations):
	# print neighbors_annotations
	num_boundaries_all = [len(x)-1 for x in neighbors_annotations]
	median_num_boundaries = int(np.median(num_boundaries_all))
	peaks = list(peakdetect.peakdetect(signal,lookahead=500,delta=0.2))
	
	locs = [elem[0] for elem in peaks[0]] #get only the idx
	maxs = [elem[1] for elem in peaks[0]] #get only the value
	both_zipped=zip(maxs,locs) #zip them both to sort them
	both_sorted=sorted(both_zipped)
	sort_locs=[elem[1] for elem in both_sorted]
	if len(both_sorted) > median_num_boundaries:
		sort_locs=sort_locs[-median_num_boundaries:]
	return sort_locs

def getAnnotation(ann_list):
	new = [0.0]
	for line in ann_list[0]:
		new.append(float(line[-2]))
	return new

def getAnnotationList(path,fn_list):
	ann_list = []
	for res in fn_list:
		fn = path+res[:-3]+"lab"
		if os.path.isfile(fn):
			f = open(fn,'r')
			lines = f.readlines()
			ann = []
			for i, l in enumerate(lines):
				ann.append(l[:-1].split('\t'))
			ann_list.append(ann)
			f.close()
		else: print fn+" not found"
	return ann_list

def parseAnnotation(unparsed_annotation):
	'''Parses a formatted annotation string into a list of floats with the corresponding label '''
	parsed_annotation=[] 
	for elem in unparsed_annotation:
		new_elem    = elem[:-1].split('\t')
		new_elem[0] = float(new_elem[0])
		new_elem[1] = float(new_elem[1])
		parsed_annotation.append(new_elem)
	return parsed_annotation

# def segmentLabel(annotation):
# 	''' Segments the labels into an array of labels. Resolution: 1 label/ms. This is probably too much '''
# 	annotation_array = getAnnotation([annotation])
# 	labels = [elem[-1] for elem in annotation]
# 	segmented=[]
# 	for i in np.arange(len(labels)):
# 		N = int(np.floor(annotation_array[i+1]-annotation_array[i]))
# 		segmented.extend(N*[labels[i]])
# 	print "_"*100
# 	print segmented[:5000]
# 	return segmented

def segmentLabel(annotation):
	''' Segments the labels into an array of labels. Resolution: 1 label/t ms. '''
	t = T
	annotation_array = getAnnotation([annotation])
	labels = [elem[-1] for elem in annotation]
	segmented=[]
	short = []
	for i in np.arange(len(labels)):
		N = int(np.floor(annotation_array[i+1]-annotation_array[i]))
		segmented.extend(N*[labels[i]])
	short=segmented[0::t]
 	return short	
 

# def mergeLabels(res_boundaries,all_songs_segmented):
# 	''' Combines all songs annotations to get the resulting one'''

# 	res_array  = []
# 	res_labels = []
	
# 	for i in np.arange(len(all_songs_segmented[0])):
# 		current_value=[]
# 		for song in all_songs_segmented:
# 			current_value.append(song[i])
# 		res_array.append(list(stats.mode(current_value)[0])[0])
# 	for i in np.arange(len(res_boundaries)-1):
# 		section=res_array[res_boundaries[i]:res_boundaries[i+1]]
# 		print "section"
# 		print "_"*50
# 		print section
# 		res_labels.append(list(stats.mode(section)[0])[0])
# 	return res_labels

def mergeLabels(res_boundaries,all_songs_segmented):
	''' Combines all songs annotations to get the resulting one'''
	res_array  = []
	res_labels = []
	# print "res_boundaries"
	# print res_boundaries
	# print len(all_songs_segmented[0])
	res_boundaries=np.array(res_boundaries)/T
	# print res_boundaries
	for i in np.arange(len(all_songs_segmented[0])):
		current_value=[]
		for song in all_songs_segmented:
			current_value.append(song[i])
		res_array.append(list(stats.mode(current_value)[0])[0])
	for i in np.arange(len(res_boundaries)-1):
		section=res_array[res_boundaries[i]:res_boundaries[i+1]]
		res_labels.append(list(stats.mode(section)[0])[0])
	return res_labels



def storeResults(list_fn,cand_list,res_path):
	f = open(list_fn,'r')
	filelist = f.readlines()
	train_set = sa.getPickle(sf_pickle,cand_list)
	tree = neighbors.KDTree(train_set,leaf_size=100,p=2)

	for i,line in enumerate(filelist):
		# print "q: "+str(line[:-4])
		if i%50==0: print i
		fn = sf_path+line[:-4]+"csv"
		
		if os.path.isfile(fn):

			# --- USING METHOD I (GAUSSIANS)
			res_annotations=addGaussians(line[:-4]+"csv",tree)
			ann_list=[res_annotations] #this is needed so it doesnt mess up with the rest
			# print ann_list

			# --- USING METHOD 0
			# sf_query = np.loadtxt(fn)
			# songnumb = querySong(sf_query,tree).tolist()[0]
			# songs = [songList[i][:-1] for i in songnumb if not songList[i][:-4]==line[:-4]]
			# ann_query = getAnnotationList(gt_path,[line[:-1]])
			# duration_query = ann_query[0][-1][-2]
			# ann_list = getAnnotationList(gt_path,songs)
			# duration_result = ann_list[0][-1][-2]

			# r = float(duration_query)/float(duration_result)
			resFile = open(res_path+line[:-4]+'lab','w')
			for elem in ann_list[0]:
				# label = elem[-1] 							# save the label so it's not modified
				# elem = [float(el)*r for el in elem[:-1]] 	# rescale the boundaries according to duration
				# elem.append(label) 						 	# append label again
				lin = ''.join([str(elem[j])+'\t' for j in range(len(elem))])[:-1]+'\n'
				resFile.writelines(lin)
			resFile.close()
		else:
			print fn + " not found"
	f.close()



# def formatAnnotationArray(annotation_array,labels_array):
# 	new_ann=[]
# 	new_elem=[0]
# 	new_elem.append(float(annotation_array[0])/1000) #converting to s goes here for now
# 	new_elem.append('0') #random label for now
# 	new_ann.append(new_elem) 
# 	for i in np.arange(1,len(annotation_array)):
# 		new_elem=[]
# 		new_elem.append(float(annotation_array[i-1])/1000)
# 		new_elem.append(float(annotation_array[i])/1000)
# 		new_elem.append('0')
# 		new_ann.append(new_elem)
# 	return new_ann

def formatAnnotation(boundaries_array, labels_array):
	new_ann = []
	for i in np.arange(len(labels_array)):
		new_elem=[]
		new_elem.append(float(boundaries_array[i])/1000)
		new_elem.append(float(boundaries_array[i+1])/1000)
		new_elem.append(labels_array[i])
		new_ann.append(new_elem)
	return new_ann


def printInfo(list_fn,cand_list):
	f = open(list_fn)
	lines = f.readlines()
	train_set = sa.getPickle(sf_pickle,cand_list)
	tree = neighbors.KDTree(train_set,leaf_size=100,p=2.0)
	for i, line in enumerate(lines):
		fn = sf_path+line[:-4]+"csv"
		print "Query: "+line[:-5]
		sf_query = np.loadtxt(fn)
		songnumb = querySong(sf_query,tree,Kneighbors=K).tolist()[0]
		songs = [songList[i][:-1] for i in songnumb if not songList[i][:-4]==line[:-4]]
		for song in songs: print "Result: "+song
		print "\n"
		ann_list = getAnnotationList(gt_path,songs)

def listSelection(case):

	return {
		# (pickle, query, cand, results)
		1: ('pickles/sf-beatles-n100.pickle','annotation_results/beatlesQMUL.txt','annotation_results/beatlesQMUL.txt','annotation_results/beatlesQMUL-n100/'),
		2: ('pickles/sf-alldatasets-n100.pickle','annotation_results/beatlesQMUL.txt','annotation_results/alldatasets.txt','annotation_results/beatlesQMUL-n100/'),
		3: ('pickles/sf-chopin-n100.pickle','annotation_results/chopin.txt','annotation_results/chopin.txt','annotation_results/chopin-n100/'),
		4: ('pickles/sf-alldatasets-n100.pickle','annotation_results/chopin.txt','annotation_results/alldatasets.txt','annotation_results/chopin-n100/'),
		5: ('pickles/sf-rwcP-n100.pickle','annotation_results/rwcP.txt','annotation_results/rwcP.txt','annotation_results/rwcAIST-n100/'),
		6: ('pickles/sf-alldatasets-n100.pickle','annotation_results/rwcP.txt','annotation_results/alldatasets.txt','annotation_results/rwcAIST-n100/'),
	}[case]

if __name__ == "__main__":

	# desc_path  = 'hpcp_ah6_al5_csv/'
	# gt_path    = 'metadata/all/'
	# sf_path    = 'sfs/sf-alldatasets-n100/'

	#	Case 1: beatlesQMUL vs beatlesQMUl
	#	Case 2: beatlesQMUL vs all
	#	Case 3: mazurkas vs mazurkas
	#	Case 4: mazurkas vs all
	#	Case 5: rwcP vs rwcP
	#	Case 6: rwcP vs all
	
	(sf_pickle, query_list,	cand_list, res_path) = listSelection(3)
	# songList = open(cand_list).readlines()

	# query_list = 'annotation_results/test.txt'
	storeResults(query_list,cand_list,res_path)
	# printInfo(query_list,cand_list)
	# n = getNeighbors('Chopin_Op006No1_Ashkenazy-1981_pid9058-01.mp3.csv')
	# print n
	# print len(n)
	# print segmentLabel('Beatles_TheContinuingStoryOfBungalowBill_Beatles_1968-TheBeatlesTheWhiteAlbumDisc1-06.wav.lab')
	# train_set = sa.getPickle(sf_pickle,cand_list)
	# tree = neighbors.KDTree(train_set,leaf_size=100,p=2)
	# n=getNeighbors('Beatles_TheContinuingStoryOfBungalowBill_Beatles_1968-TheBeatlesTheWhiteAlbumDisc1-06.wav.csv',tree)
	# n = getPeaks('Beatles_AnnaGoToHim_Beatles_1963-PleasePleaseMe-03.wav.csv')
	# m=addGaussians('Beatles_TheContinuingStoryOfBungalowBill_Beatles_1968-TheBeatlesTheWhiteAlbumDisc1-06.wav.csv',tree)
	# print formatAnnotationArray(m)
	# fn_list=['Beatles_AnnaGoToHim_Beatles_1963-PleasePleaseMe-03.wav.csv','Beatles_TheContinuingStoryOfBungalowBill_Beatles_1968-TheBeatlesTheWhiteAlbumDisc1-06.wav.csv']
	# ann_list=getAnnotationList(gt_path,fn_list)
	# print "ann_list"
	# print ann_list
	# test(ann_list)	
	# segmentLabel(ann)
	

