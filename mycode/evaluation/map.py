import sys
import csvio
from numpy import *

if len(sys.argv)!=3:
	print '\nUSAGE: python map.py <filenames_list.txt> <result_distance_matrix.csv>\n'
	sys.exit(1)

fn_list=sys.argv[1]
fn_result=sys.argv[2]

print 'Loading data...'

# load filenames
f=open(fn_list,'r')
lines=f.readlines()
f.close()

# get labels
label=[]
for line in lines:
	fields=line.split('/')
	label.append(fields[len(fields)-2])

# load results
distmat=csvio.load(fn_result,',')

print 'Evaluating...'

# evaluation
averagePrecision=[]
for i in range(0,distmat.shape[0]):
	sortedIds=argsort(distmat[i,:])
	relevantDocuments=0
	retrievedDocuments=0
	precision=[]
	for j in sortedIds:
		if i==j: continue
		retrievedDocuments+=1
		if label[i]==label[j]:
			relevantDocuments+=1
			precision.append(float(relevantDocuments)/float(retrievedDocuments))
	if len(precision)>0: averagePrecision.append(mean(precision))
if len(averagePrecision)>0:
	MAP=mean(averagePrecision)
	MAPstd=std(averagePrecision)
else:
	MAP=-1
	MAPstd=0

# result
print '-------------------'
print 'MAP = %.3f'%round(MAP,3)+' (%.3f)'%round(MAPstd,3)
print '-------------------'

sys.exit(0)

print 'Error analysis...'

# errors
while True:
	querynum=int(raw_input('\t> Query number? '))
	if querynum<0: break
	print '\t'+lines[querynum][0:len(lines[querynum])-1]
	sortedIds=argsort(distmat[querynum,:])
	l='\t'
	for i in sortedIds[0:100]:
		l+=str(i)
		if label[querynum]==label[i]: l+=', '
		else: l+=' (X), '
	print l

