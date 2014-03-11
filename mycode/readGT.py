import numpy as np
import os
import csvio


folder= os.listdir("../metadata/beatles/mylabfilesQMUL_tab")
songs=[]

for f in folder:
	res=csvio.load('../metadata/beatles/mylabfilesQMUL_tab/'+f,'\t')
	songs.append(res)
# res=csvio.load('../metadata/beatles/mylabfilesQMUL_tab/Beatles_AcrossTheUniverse_Beatles_1970-LetItBe-03.wav.lab','\t')


# folder= os.listdir("../metadata/beatles/annotationSegmentRenamedQueenMary")
# songs=[]
# for aux in folder:
# 	s = {}
# 	time = []
# 	labels = []
# 	f = open("../metadata/beatles/annotationSegmentRenamedQueenMary/%s" % aux, 'r')
# 	lines=f.readlines()
# 	for line in lines:
# 		line=line.replace('\t\t'," ").replace('\t'," ").replace('\n', "").split(" ")
# 		t=map(float,line[0:1])[0]
# 		t2=map(float,line[1:2])[0]
# 		# mat=np.matrix([t],[t2])
# 		# mat=np.matrix([map(float,line[0:1])],[map(float,line[1:2])])
# 		# time.append.mat
# 		print type(t)
		 
# 	s['time'] = time
# 	s['labels'] = labels
# 	songs.append(s)	
# print lines
# print line
# print time
# print songs