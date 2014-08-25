import numpy as np
import cProfile
from sklearn import neighbors
import os.path
import structure_annotation as sa
from scipy import signal
from scipy import stats
from scipy.ndimage import morphology
import peakdetect
from collections import Counter
import operator
import matplotlib.pyplot as plt
import evaluation.evaluate
import pickle
import os
import csv

gt_path   = 'metadata/all/'
desc_path = 'hpcp_ah6_al5_csv/'
cand_list = 'annotation_results/alldatasets.txt'
sf_pickle = 'pickles/sf-alldatasets-n100.pickle'
sf_path = 'sfs/sf-alldatasets-n100/'

K       = 5
T       = 100
n       = 1000

#Peak detection parameters
lambd  = 2
delta  = 0.1
window = 3000 # this is for the labels+gaussians meth combi (in seconds)	

#------METHOD I---------#
def addGaussians(query_fn,cand_list,tree,K):
	songs_list   = getNeighbors(query_fn,cand_list,tree,K)
	M            = 23000
	query_ann    = getAnnotationList(gt_path,[query_fn])
	query_labels = [elem[-1] for elem in query_ann[0]]
	query_ann    = np.floor((np.array(getAnnotation(query_ann))*1000)).astype(int)
	length       = query_ann[-1]
	total        = np.zeros(int(np.ceil(length)))

	neighbors_annotations_rescaled = []
	neighbors_annotations = getAnnotationList(gt_path,songs_list)

	for i, song in enumerate(songs_list):
		gt_list        = getAnnotationList(gt_path,[song])
		ann            = np.floor((np.array(getAnnotation(gt_list))*1000)).astype(int) #convert to miliseconds to mantain res
		neighbor_dur   = ann[-1]
		ann_with_sides = ann
		ann            = ann[1:-1]
		a              = np.zeros(int(np.ceil(length)))
		r              = float(length)/float(neighbor_dur) #rescale according to query duration
		ann            = np.floor(ann*r)
		ann_with_sides = np.floor(ann_with_sides*r) 

		labels = [x[-1] for x in gt_list[0]] # get the labels
		annotation_rescaled = []
		for elem in neighbors_annotations[i]:
			label = elem[-1] #save the label so it doesnt get affected by rescaling
			elem[0] = int(np.floor(float(elem[0])*1000*r)) #rescale the rest
			elem[1] = int(np.floor(float(elem[1])*1000*r))
			annotation_rescaled.append([elem[0],elem[1],label])
		neighbors_annotations_rescaled.append(annotation_rescaled)
		for i, loc in enumerate(ann,1):
			section_length = ann_with_sides[i]-ann_with_sides[i-1]
			sigma = 0.1*section_length
			# M=int(np.floor(0.6*section_length))
			g1 = signal.gaussian(M,std=sigma)
			half1 = int(np.floor(len(g1)/2))
			section_length = ann_with_sides[i+1]-ann_with_sides[i]
			sigma = 0.1*section_length
			g2 = signal.gaussian(M,std=sigma)
			half2 = int(np.floor(len(g2)/2))
			g = np.concatenate((g1[:half1],g2[half2:]))			
			if loc < np.floor(M/2):
				a += np.array(np.concatenate((g[int(np.floor(M/2)-loc):],np.zeros(int(length-loc-np.floor(M/2))))))
			elif loc + np.floor(M/2) > length:
				a += np.array(np.concatenate((np.zeros(int(loc-np.floor(M/2))),g[:int(length+np.floor(M/2)-loc)])))
			else:
				a += np.array(np.concatenate((np.zeros(int(loc-np.floor(M/2))),g,np.zeros(int(length-loc-np.floor(M/2))))))
		total += a
	total = total/float(max(total))
	peaks = getPeaks(total,neighbors_annotations)
	all_songs_segmented = [segmentLabel(elem) for elem in neighbors_annotations_rescaled]
	res_boundaries = sorted(peaks)
	res_boundaries.insert(0,0)
	res_boundaries.append(length)
	res_labels = mergeLabels(res_boundaries,all_songs_segmented)
	res_annotations = formatAnnotation(res_boundaries,res_labels)
	return res_annotations
# ---------------------------- #

# ------- METHOD II --------- #
def labelsMethod(query_fn,cand_list,tree,K):
	print "extracting annotation using labels method"
	songs_list = getNeighbors(query_fn,cand_list,tree,K)
	query_ann = np.floor((np.array(getAnnotation(getAnnotationList(gt_path,[query_fn])))*1000)).astype(int)
	length = query_ann[-1]
	neighbors_annotations = getAnnotationList(gt_path,songs_list)
	neighbors_annotations_rescaled = []
	num_boundaries_all = [len(x)-1 for x in neighbors_annotations]
	median_num_boundaries = int(np.median(num_boundaries_all))

	for i, song in enumerate(songs_list):
		gt_list = getAnnotationList(gt_path,[song])
		ann = np.floor((np.array(getAnnotation(gt_list))*1000)).astype(int) #convert to miliseconds to mantain res
		neighbor_dur = ann[-1]
		ann = ann[1:-1]
		r = float(length)/float(neighbor_dur) #rescale according to query duration
		ann = np.floor(ann*r)
		labels = [x[-1] for x in gt_list[0]] # get the labels
		annotation_rescaled = []
		for elem in neighbors_annotations[i]:
			label = elem[-1] #save the label so it doesnt get affected by rescaling
			elem[0] = int(np.floor(float(elem[0])*1000*r)) #rescale the rest
			elem[1] = int(np.floor(float(elem[1])*1000*r))
			annotation_rescaled.append([elem[0],elem[1],label])
		neighbors_annotations_rescaled.append(annotation_rescaled)
	all_songs_segmented = [segmentLabel(elem) for elem in neighbors_annotations_rescaled]
	res_array  = []
	all_lenghts = [len(x) for x in all_songs_segmented]
	min_len = min(all_lenghts) #chapuza para evitar errores cuando hay 1 muestra de diferencia en la longitud
	for i in np.arange(min_len):
		current_value=[]
		for song in all_songs_segmented:
			current_value.append(song[i])
		res_array.append(list(stats.mode(current_value)[0])[0])
	res_array     = removeGaps(res_array,W=11) #
	boundaries    = [0]
	current_count = 0
	labels        = []
	occurrences   = []
	summ          = 0
	for i in np.arange(len(res_array)-1): 
		current_label = res_array[i]
		if current_label == res_array[i+1]:
			current_count += 1
			if i>=len(res_array)-2:
				labels.append(current_label)
				occurrences.append(current_count+1)
				current_count = 0
		else:
			labels.append(current_label)
			occurrences.append(current_count+1)
			current_count = 0
	for i in np.arange(len(occurrences)):
		summ += occurrences[i]
		boundaries.append(summ)
	boundaries = np.floor(np.array(boundaries)*T).astype(int)
	annotation = formatAnnotation(boundaries, labels)
	return annotation
# --------------------------- #

# ------- METHOD III -------- #
def earlyFusion(query_fn,cand_list,tree,K):
	songs_list = getNeighbors(query_fn,cand_list,tree,K)
	query_ann = getAnnotationList(gt_path,[query_fn])
	query_labels = [elem[-1] for elem in query_ann[0]]
	query_ann = np.floor((np.array(getAnnotation(getAnnotationList(gt_path,[query_fn])))*1000)).astype(int)
	length = query_ann[-1]
	w = (sa.m-1)-sa.tau # to account for delay coord delay its added at the end to locs
	sfList = []
	ratio = length/sa.n #downsampled to miliseconds
	neighbors_annotations = getAnnotationList(gt_path,songs_list)
	neighbors_annotations_rescaled = []

	for i, song in enumerate(songs_list):
		gt_list = getAnnotationList(gt_path,[song])
		sfArray = np.loadtxt(sf_path+song[:-3]+'csv')
		sfMatrix = sa.convertToMatrix(sfArray)
		sfList.append(sfMatrix)
		annotation_rescaled = []
		ann = np.floor((np.array(getAnnotation(gt_list))*1000)).astype(int) #convert to miliseconds to mantain res
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
	sfMean = MeanSF(sfList)
	novelty_curve = np.array(sa.novelty(sfMean))
	locs = sa.peakdet(novelty_curve)
	locs = np.array(locs)*ratio
	locs += (w/2)
	all_songs_segmented = [segmentLabel(elem) for elem in neighbors_annotations_rescaled]
	res_boundaries = locs
	res_boundaries = np.insert(res_boundaries,0,0)
	res_boundaries =	np.append(res_boundaries,length) #cause all songs are supposed to be the same length now
	res_labels = mergeLabels(res_boundaries,all_songs_segmented)
	res_annotations = formatAnnotation(res_boundaries,res_labels)
	return res_annotations

def MeanSF(sfList):
	'''gets a list of all sfs as input and returns a sf thats the mean columnwise of all'''
	sfMean = np.mean(sfList,axis=0)
	return sfMean

def removeGaps(res_array, W):
	'''window must be odd?'''
	hW = int(np.floor(W/2))
	for i in np.arange(hW+1,len(res_array)-hW):
		current_window = res_array[i-hW-1:i+hW]
		if len(set(current_window))>1 and current_window[0]==current_window[-1]:
			res_array[i-hW-1:i+hW] = [current_window[0]]*W
	return res_array

def removeGaps2(res_array,median_num_boundaries):
	marks,occurrences = getMarks(res_array)
	num_boundaries = len(marks)-2
	while num_boundaries>median_num_boundaries:
		min_idx = min(enumerate(occurrences), key=operator.itemgetter(1))[0]
		if 1<min_idx<len(occurrences)-1: # check if it's inside res_array
			if occurrences[min_idx]<occurrences[min_idx-1]:
				res_array[marks[min_idx]] = res_array[marks[min_idx]-1] #extend to the right
			else:
				res_array[marks[min_idx]-1] = res_array[marks[min_idx]] #extend to the left
		marks,occurrences = getMarks(res_array)
		num_boundaries = len(marks)-2
	return res_array

def getMarks(res_array):
	summ = 0
	marks = [0]
	occurrences = []
	for i in np.arange(len(res_array)-1):
		summ += 1
		if res_array[i]!=res_array[i+1]:
			marks.append(i+1)
			occurrences.append(summ)
			summ = 0
	marks.append(i+1)
	occurrences.append(marks[-1]-marks[-2]+1) #CHAPUZA
	return marks, occurrences

def querySong(query,tree,Kneighbors):
	index = tree.query(query,k=Kneighbors,return_distance=False)
	return index

def purgeNeighbors(neighbors_fn_list):
	'''Removes occurrences of same piece in the neighbors list'''
	comp_string = [elem.replace('-','_').split('_')[1] for elem in neighbors_fn_list]
	new_list = []
	repeated = list(set([x for x in comp_string if comp_string.count(x) > 1]))
	if not repeated:
		new_list = neighbors_fn_list
	else:
		for i,elem in enumerate(neighbors_fn_list):
			if comp_string[i] in elem and not comp_string[i] in repeated:
				new_list.append(elem)
		repeated_complete = []
		for x in neighbors_fn_list:
			for r in repeated:
				if r in x: repeated_complete.append(x)
		new_list.append(repeated_complete[0])
	return new_list

def getNeighbors(query_fn,cand_list,tree,K):
	sf_query = np.loadtxt(sf_path+query_fn)
	song_list = open(cand_list).readlines()
	songnumb = querySong(sf_query,tree,Kneighbors=K+1).tolist()[0] #K+1 neighbors cause later we remove the query
	n_list=[song_list[i] for i in songnumb]
	neighbors_fn_list = [song_list[i][:-1] for i in songnumb if not song_list[i][:-5]==query_fn[:-4]]
	while len(neighbors_fn_list) > K: #if the number of results after removing the query is still higher than K remove the last one.
		neighbors_fn_list=neighbors_fn_list[:-1]
	# while len(neighbors_fn_list)!=K:
	# 	c+=1
	# 	sf_query = np.loadtxt(sf_path+query_fn)
	# 	songnumb = querySong(sf_query,tree,K+c).tolist()[0]
	# 	neighbors_fn_list = [songList[i][:-1] for i in songnumb if not songList[i][:-5]==query_fn[:-4]]
	# 	neighbors_fn_list = purgeNeighbors(neighbors_fn_list)
	return neighbors_fn_list

def unpickleAsArray(sf_pickle):
	dataM = []
	data = sa.getPickle(sf_pickle)
	for row in data:
		dataM.append(sa.convertToMatrix(row,50))
	return dataM

def getPeaks(signal, neighbors_annotations):
	num_boundaries_all = [len(x)-1 for x in neighbors_annotations]
	median_num_boundaries = int(np.median(num_boundaries_all))
	peaks = list(peakdetect.peakdetect(signal,lookahead=500,delta=0.2))
	locs = [elem[0] for elem in peaks[0]] #get only the idx
	maxs = [elem[1] for elem in peaks[0]] #get only the value
	both_zipped = zip(maxs,locs) #zip them both to sort them
	both_sorted = sorted(both_zipped)
	sort_locs = [elem[1] for elem in both_sorted]
	if len(both_sorted) > median_num_boundaries:
		sort_locs = sort_locs[-median_num_boundaries:]
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
	parsed_annotation = [] 
	for elem in unparsed_annotation:
		new_elem    = elem[:-1].split('\t')
		new_elem[0] = float(new_elem[0])
		new_elem[1] = float(new_elem[1])
		parsed_annotation.append(new_elem)
	return parsed_annotation

def segmentLabel(annotation):
	''' Segments the labels into an array of labels. Resolution: 1 label/t ms. '''
	t = T
	annotation_array = getAnnotation([annotation])
	labels = [elem[-1] for elem in annotation]
	segmented = []
	short = []
	for i in np.arange(len(labels)):
		N = int(np.floor(annotation_array[i+1]-annotation_array[i]))
		segmented.extend(N*[labels[i]])
	short = segmented[0::t]
 	return short	

def mergeLabels(res_boundaries,all_songs_segmented):
	''' Combines all songs annotations to get the resulting one'''
	res_array  = []
	res_labels = []
	min_len = min([len(x) for x in all_songs_segmented]) #In case the segmentation returns different lengths for each song get the min of all. the error is t
	res_boundaries = np.array(res_boundaries)/T
	for i in np.arange(len(res_boundaries)-1): #dirty way of avoiding errors if sections are too short
		if res_boundaries[i]==res_boundaries[i+1]: 
			res_boundaries[i+1]=res_boundaries[i]+1
	for i in np.arange(min_len):
		current_value = []
		for c, song in enumerate(all_songs_segmented):
			current_value.append(song[i])
		res_array.append(list(stats.mode(current_value)[0])[0])
	for i in np.arange(len(res_boundaries)-1):
		section = res_array[res_boundaries[i]:res_boundaries[i+1]]
		res_labels.append(list(stats.mode(section)[0])[0])
	return res_labels

def storeResults(list_fn,cand_list,res_path,sf_pickle,K):
	f = open(list_fn,'r')
	filelist = f.readlines()
	train_set = sa.getPickle(sf_pickle,cand_list) #uncomment to use static candidates lists (using stored pickles)
	# train_set = sa.listToData(cand_list) #uncomment to use dynamic candidate lists

	tree = neighbors.KDTree(train_set,leaf_size=100,p=2)
	for i,line in enumerate(filelist):
		if i%50==0: print i
		print "q: "+str(line[:-4])
		fn = sf_path+line[:-4]+"csv"
		if os.path.isfile(fn):
			# --- USING METHOD 0 ---- #
			# sf_query = np.loadtxt(fn)
			# songnumb = querySong(sf_query,tree,K).tolist()[0]
			# songs = [songList[i][:-1] for i in songnumb if not songList[i][:-4]==line[:-4]]
			# ann_query = getAnnotationList(gt_path,[line[:-1]])
			# duration_query = ann_query[0][-1][-2]
			# ann_list = getAnnotationList(gt_path,songs)
			# duration_result = ann_list[0][-1][-2]
			# r = float(duration_query)/float(duration_result)

			# --- USING METHOD I (GAUSSIANS) --- #
			# res_annotations = addGaussians(line[:-4]+"csv",cand_list,tree,K)
			# ann_list = [res_annotations]

			# --- USING METHOD II (LABELS) ----- #
			# res_annotations = labelsMethod(line[:-4]+"csv",cand_list,tree,K)
			# ann_list = [res_annotations] 	 #this is needed so it doesnt mess up with the rest

			# --- USING METHOD III (EARLY FUSION) ----- #
			res_annotations=earlyFusion(line[:-4]+"csv",cand_list, tree,K)
			ann_list=[res_annotations] 	 #this is needed so it doesnt mess up with the rest

			resFile = open(res_path+line[:-4]+'lab', 'w')
			for elem in ann_list[0]:
				# -- for METHOD 0 ONLY ---- #
				# label = elem[-1] 							# save the label so it's not modified
				# elem = [float(el)*r for el in elem[:-1]] 	# rescale the boundaries according to duration
				# elem.append(label) 						 	# append label again
				# ------------------------- #
				lin = ''.join([str(elem[j])+'\t' for j in range(len(elem))])[:-1]+'\n'
				resFile.writelines(lin)
			resFile.close()
		else:
			print fn + " not found"
	f.close()

def storeResultsMazurkas(query_fn,cand_list,res_path,K):
	fn = sf_path+query_fn[:-4]+'csv'
	if os.path.isfile(fn):
		train_set = sa.listToData(cand_list)
		tree = neighbors.KDTree(train_set,leaf_size=100,p=2)
		res_annotations = labelsMethod(query_fn[:-4]+'csv',cand_list,tree,K)
		ann_list = [res_annotations] 	 #this is needed so it doesnt mess up with the rest
		resFile = open(res_path+query_fn[:-4]+'lab','w')
		for elem in ann_list[0]:
			lin = ''.join([str(elem[j])+'\t' for j in range(len(elem))])[:-1]+'\n'
			resFile.writelines(lin)
		resFile.close()
	else:
		print fn + " not found"

def formatAnnotation(boundaries_array, labels_array):
	new_ann = []
	for i in np.arange(len(labels_array)):
		new_elem = []
		new_elem.append(float(boundaries_array[i])/1000)
		new_elem.append(float(boundaries_array[i+1])/1000)
		new_elem.append(labels_array[i])
		new_ann.append(new_elem)
	return new_ann

def printInfo(list_fn,cand_list):
	f = open(list_fn)
	lines = f.readlines()
	train_set = sa.getPickle(sf_pickle,cand_list)
	song_list = open(cand_list).readlines()

	tree = neighbors.KDTree(train_set,leaf_size=100,p=2.0)
	for i, line in enumerate(lines):
		fn = sf_path+line[:-4]+"csv"
		print "Query: "+line[:-5]
		sf_query = np.loadtxt(fn)
		songnumb = querySong(sf_query,tree,Kneighbors=K).tolist()[0]
		songs = [song_list[i][:-1] for i in songnumb if not song_list[i][:-4]==line[:-4]]
		for song in songs: print "Result: "+song
		print "\n"
		ann_list = getAnnotationList(gt_path,songs)

def getTrainSetFromFolder(sf_path, cand_list):
	f = open(cand_list,'r')
	pick = open(sf_pickle,'w')
	lines = f.readlines()
	data = []
	for i, line in enumerate(lines):
		print i
		current = np.loadtxt(sf_path+line[:-4]+'csv')
		data.append(current)
		pickle.dump(current,pick)
	f.close()
	pick.close()
	return data

# ------ TESTING SCRIPTS ----- #
def createCandList(query_fn, num_dups):
	''' creates a candidates list with num_dups duplicates for each song.'''
	''' At the moment it removes songs consecutively from the list. Probably should be randomized.'''
	# open the original candidates list with all candidates
	query_op = query_fn.split('_')[1]
	f = open('annotation_results/chopin.txt','r')
	path = 'candlists/%s/'%num_dups
	fn = 'temp_%s.txt'%query_fn[:-4]
	if not os.path.exists(path):
		os.makedirs(path)
	reslist = open(path+fn,'w')
	filelist = f.readlines()
	count = 0
	for i, fil in enumerate(filelist):
		comp_op = fil.split('_')[1]
		if query_op==comp_op:
			count += 1
			if count<=num_dups:
				reslist.writelines(fil)
		else:
			reslist.writelines(fil)

def mazurkaTest():
	query_list = 'annotation_results/chopin.txt'
	res_path   = 'annotation_results/testmazurkas/'
	sf_pickle  = 'pickles/sf-chopin-n100.pickle'
	gt_path    = 'metadata/mazurkas/mylabfilesMPI_tab/'
	f          = open(query_list,'r')
	lines      = f.readlines()
	neigh      = [1,3,5,10]

	with open("testmazurkas2.csv",'a') as fres:
		writer = csv.writer(fres)
		writer.writerow(["#","K","Num Dups","FB","FL"])
		for j in np.arange(4,10): # do it 10 times to get the mean
			rand = np.random.randint(len(lines))
			query_fn = lines[rand][:-4]+'.txt' #get a random song every time
			for k in neigh: # do it for several neighbors
				for i in np.arange(0,10): # 
					print "---------------"
					print "Iteration: "+str(j)
					print "Neighbors: "+str(k)
					print "Number of dups: "+str(i)	
					cand_list = 'candlists/%d/temp_%s'%(i,query_fn)
					storeResultsMazurkas(query_fn,cand_list,res_path,k)
					fP,fR,fF,cP,cR,cF,mediangtt,medianttg = evaluation.evaluate.boundaries_singlefile(res_path+query_fn[:-4]+'lab',gt_path+query_fn[:-4]+'lab')
					P,R,F,So,Su = evaluation.evaluate.labels_singlefile(res_path+query_fn[:-4]+'lab',gt_path+query_fn[:-4]+'lab')
					writer = csv.writer(fres)
					writer.writerow([j,k,i,cF,F,])
					print "Boundaries F:	"+str(cF)
					print "Labels F:	"+str(F)

def ceilingTest():
	query_list = 'annotation_results/chopin.txt'
	gt_path = 'metadata/mazurkas/mylabfilesMPI_tab/'
	maxFB = 0
	maxFL = 0
	FB = []
	FL = []
	f = open(query_list,'r')
	lines = f.readlines()
	for line in lines:
		print line
		for gtline in lines:
			if line!=gtline:
				fP,fR,fF,cP,cR,cF,mediangtt,medianttg = evaluation.evaluate.boundaries_singlefile(gt_path+line[:-1],gt_path+gtline[:-1])
				P,R,F,So,Su = evaluation.evaluate.labels_singlefile(gt_path+line[:-1],gt_path+gtline[:-1])
				if cF>maxFB: maxFB = cF
				if F>maxFL: maxFL = F
		print "maxFB: "+str(maxFB)
		print "maxFL: "+str(maxFL)
		FB.append(maxFB)
		FL.append(maxFL)
		maxFL = 0
		maxFB = 0
	return FB, FL, np.mean(FB),np.std(FB),np.mean(FL),np.std(FL)

def neighborsTest():
	neigh = [1,2,3,5,10,15,20]
	for i in np.arange(1,11):
		(sf_pickle,list_fn,cand_list,res_path) = listSelection(i)
		for k in neigh:
			storeResults(list_fn,cand_list,res_path,sf_pickle,k)
			evaluation.evaluate.process(i,k)

def standardTest():
	for i in np.arange(1,11):
		(sf_pickle,list_fn,cand_list,res_path) = listSelection(i)
		storeResults(list_fn,cand_list,res_path,sf_pickle,5)
		evaluation.evaluate.process(i,5)

def listSelection(case):
	return {
		# (pickle, query, cand, results)
		1: ('pickles/sf-beatles-n100.pickle','annotation_results/beatlesQMUL.txt','annotation_results/beatlesQMUL.txt','annotation_results/beatlesQMUL-n100/'),
		2: ('pickles/sf-alldatasets-n100.pickle','annotation_results/beatlesQMUL.txt','annotation_results/alldatasets.txt','annotation_results/beatlesQMUL-all-n100/'),
		3: ('pickles/sf-chopin-n100.pickle','annotation_results/chopin.txt','annotation_results/chopin.txt','annotation_results/chopin-n100/'),
		4: ('pickles/sf-alldatasets-n100.pickle','annotation_results/chopin.txt','annotation_results/alldatasets.txt','annotation_results/chopin-all-n100/'),
		5: ('pickles/sf-rwcP-n100.pickle','annotation_results/rwcP.txt','annotation_results/rwcP.txt','annotation_results/rwcAIST-n100/'),
		6: ('pickles/sf-alldatasets-n100.pickle','annotation_results/rwcP.txt','annotation_results/alldatasets.txt','annotation_results/rwcAIST-all-n100/'),
		7: ('pickles/sf-beatles-n100.pickle','annotation_results/beatlesTUT.txt','annotation_results/beatlesTUT.txt','annotation_results/beatlesTUT-n100/'),
		8: ('pickles/sf-alldatasets-n100.pickle','annotation_results/beatlesTUT.txt','annotation_results/alldatasets.txt','annotation_results/beatlesTUT-all-n100/'),
		9: ('pickles/sf-rwcP-n100.pickle','annotation_results/rwcP.txt','annotation_results/rwcP.txt','annotation_results/rwcIRISA-n100/'),
		10: ('pickles/sf-alldatasets-n100.pickle','annotation_results/rwcP.txt','annotation_results/alldatasets.txt','annotation_results/rwcIRISA-all-n100/'),
		
	}[case]

if __name__ == "__main__":

	# desc_path  = 'hpcp_ah6_al5_csv/'
	# gt_path    = 'metad ata/all/'
	# sf_path    = 'sfs/sf-alldatasets-n100/'

	# --------------------------- #
	# cand_list='candlists/20/temp_Chopin_Op017No4_Casadesus-1930_pid9180-05.mp3..txt'
	# query_fn='Chopin_Op041No4_Smith-1975_pid9054-29.mp3.csv'
	# train_set = sa.listToData(cand_list)
	# tree = neighbors.KDTree(train_set,leaf_size=100,p=2)
	# print getNeighbors(query_fn,cand_list,tree,50)

	# ------ TESTS ------ #
	# neighborsTest()
	standardTest()


