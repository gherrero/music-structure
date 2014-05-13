import sys
from numpy import *
import csvio

if len(sys.argv)!=3:
	print '\nUSAGE: python '+sys.argv[0]+' <filelist_name_mp3/wav.txt> <path_to_my_annotations>\n'
	sys.exit(1)

fn_list=sys.argv[1]
path=sys.argv[2]

EXTENSION='.lab'

f=open(fn_list,'r')
lines=f.readlines()
f.close()

numsongs=0
lengths=[]
ibi=[]
numbounds=[]

for line in lines:
	fields=line[0:len(line)-1].split('/')
	fullfn=path+fields[len(fields)-1]+EXTENSION
	annotation=csvio.load(fullfn,'\t')
	boundaries=[0.0]
	for i in range(0,annotation.shape[0]): boundaries.append(annotation[i,1])
	numbounds.append(len(boundaries)-2)
	for i in range(0,len(boundaries)-1): ibi.append(boundaries[i+1]-boundaries[i])
	lengths.append(annotation[annotation.shape[0]-1,1])
	numsongs+=1

print '-------------------------------------'
print 'Number of songs = '+str(numsongs)
print 'Length = '+str(mean(lengths))+' ('+str(std(lengths))+')'
print 'Num of boundaries = '+str(mean(numbounds))+' ('+str(std(numbounds))+')'
print 'IBI = '+str(mean(ibi))+' ('+str(std(ibi))+')'
print '-------------------------------------'

