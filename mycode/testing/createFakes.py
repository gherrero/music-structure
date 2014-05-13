import os
import csvio
import random


# FAKE BOUNDARIES SYSTEMS

#Dataset-dependent
SYSTEM = 1 # every LAG seconds
SYSTEM = 2 # avg song boundary length averaged over entire dataset
SYSTEM = 3 # avg boundary length of entire dataset
SYSTEM = 4 # NUMBOUND number of boundaries randomly placed
SYSTEM = 5 # average # of boundaries randomly placed
# SYSTEM = 6 # combination of avg # and length? Is this possible?

NUMBOUND = 10
LAG      = 3
#Song-dependent

fullfn='fakeboundaries/'


# READ ANNOTATIONS FILES AND CREATE FAKE RANDOM BOUNDARIES
folder= os.listdir("../metadata/rwc/mylabfilesIRISA_tab")
songs=[]
avgSongLen=0
sumBoundaries=0
sumBoundaryLen=0

filelist=open('filelist.txt','w')

for f in folder:

	res=csvio.load('../metadata/rwc/mylabfilesIRISA_tab/'+f,'\t')
	
	lengthAudio=res[-1][1]
	fFakes=open(fullfn+f,'w')
	filelist.write(f+'\n')

	if SYSTEM==1:
		t=LAG
		# print fFakes
		while t<=lengthAudio:
			fFakes.write(str(t-LAG)+'\t'+str(t)+'\t0.0\n')
			t+=LAG
		fFakes.write(str(t-LAG)+'\t'+str(lengthAudio)+'\t0.0\n')
	if SYSTEM==4: # I guess it doesn't matter if the boundaries are not in order.
		b=range(3,int(lengthAudio)-3)
		random.shuffle(b)
		pre=0.0
		for t in range(0,NUMBOUND):
			fFakes.write(str(float(pre))+'\t'+str(float(b[t]))+'\t0.0\n')
			pre=b[t]
		fFakes.write(str(float(pre))+'\t'+str(lengthAudio)+'\t0.0\n')
		fFakes.close()

	songs.append(res)
	sumBoundaryLen+=lengthAudio # for system 3
	avgSongLen+=lengthAudio/(len(res)+1) # averaging for every song
	sumBoundaries+=len(res)+1 # for system 3

filelist.close()


if SYSTEM==2:
	LAG=avgSongLen/len(songs)
	for f in folder:
		fFakes=open(fullfn+f,'w')
		t=LAG
		# print fFakes
		while t<=lengthAudio:
			fFakes.write(str(t-LAG)+'\t'+str(t)+'\t0.0\n')
			t+=LAG
		fFakes.write(str(t-LAG)+'\t'+str(lengthAudio)+'\t0.0\n')
		fFakes.close()

if SYSTEM==3:
	LAG=sumBoundaryLen/sumBoundaries

	for f in folder:
		fFakes=open(fullfn+f,'w')
		t=LAG
		# print fFakes
		while t<=lengthAudio:
			fFakes.write(str(t-LAG)+'\t'+str(t)+'\t0.0\n')
			t+=LAG
		fFakes.write(str(t-LAG)+'\t'+str(lengthAudio)+'\t0.0\n')

		fFakes.close()
if SYSTEM==5:
	NUMBOUND=sumBoundaries/len(songs)
	
	b=range(3,int(lengthAudio)-3)
	for f in folder:
		fFakes=open(fullfn+f,'w')
		random.shuffle(b)
		pre=0.0
		for t in range(0,NUMBOUND):
			fFakes.write(str(float(pre))+'\t'+str(float(b[t]))+'\t0.0\n')
			pre=b[t]
		fFakes.write(str(float(pre))+'\t'+str(lengthAudio)+'\t0.0\n')
		fFakes.close()

# print songs[0]

