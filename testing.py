import matplotlib.pyplot as plt
import numpy as np
import structure_annotation as sa
from sklearn import neighbors

def chunks(list, chunksize):
    for i in xrange(0, len(list), chunksize):
        yield list[i:i+chunksize]

def testDuplicates():
	b_baselines = 0.592
	l_baselines = 0.502
	data = np.loadtxt('testduplicates.csv',delimiter=',')
	FB = data[:,0::2]
	FL = data[:,1::2]	
	meanFB = []
	meanFL = []
	stdFB = []
	stdFL = []
	n = [1,3,5,10]
	x = np.arange(10)
	r=0.03

	for i, el in enumerate(FB):
		meanFB.append(np.mean(FB[i]))
		meanFL.append(np.mean(FL[i]))
		stdFB.append(np.std(FB[i]))
		stdFL.append(np.std(FL[i]))

	meanFB = list(chunks(meanFB,10))
	meanFL = list(chunks(meanFL,10))
	stdFB = list(chunks(stdFB,10))
	stdFL = list(chunks(stdFL,10))

	plt.figure('Labels F-Measure',figsize=(10, 8), dpi=80)
	for i, el in enumerate(n):
		x=x+r

		plt.errorbar(x,meanFB[i],yerr=stdFB[i],fmt='-o',label='$K=%d$'%el,linewidth=2)
		plt.axis([-1,10,0,1.1])
	plt.hlines(l_baselines,-1,10,colors='k', linewidth=2, label='$Baseline$')

	legend = plt.legend(loc='center left', bbox_to_anchor=(0.75,0,1,0.3))
	plt.xlabel('$Number\, of\, versions\, of\, the\, same\, piece$')
	plt.ylabel('$F$')
	plt.grid()
	plt.show()

	return meanFB,meanFL,stdFB,stdFL

def testNeighbors():
	#bqmul chopin rwca btut rwci
	b_baselines = [0.530,0.592, 0.578,0.516,0.577]
	b_refs      = [0.774, 0.699, 0.785,0.753,0.797]
	b_humans    = [0.911, 0, 0.899,0.911, 0.899]
	b_ceilings  = [0.76298, 1, 0.73095, 0.78409, 0.71031]

	l_baselines = [0.491, 0.502, 0.447,0.514, 0]
	l_refs      = [0.711, 0.691, 0.719,0.691, 0.707,0]
	l_humans    = [0.876,0,0,0.876,0]
	l_ceilings  = [0.66481, 1, 0.64695, 0.66628, 0.71144]

	data = np.loadtxt('testneighbors.csv',delimiter=',')

	#tweaking plots to avoid overlap
	r=0.04
	r2=-0.04

	#hardcoded all datasets combinations
	alldata={
		1: data[:7,:],
		2: data[7:14,:],
		3: data[14:21,:],
		4: data[21:28,:],
		5: data[28:35,:]
	}
	for j in np.arange(1,6):
		dataset=list(chunks(alldata[j].T,4))
		data=[elem.T for elem in dataset]
		m1=data[0]
		m2=data[1]
		m3=data[2]
		meanFB= []
		plt.figure('Boundaries F-Measure', figsize=(12, 10), dpi=80)
		plt.subplot(3,2,j)
		
		# max_mean=max([max(elem[:,0]) for elem in data])
		# max_std=max([max(elem[:,1]) for elem in data])
		# min_mean=min([min(elem[:,0]) for elem in data])

		# maxv=(max_mean+max_std)
		# minv=(min_mean-max_std)

		# maxv=maxv+maxv*0.1
		# minv=minv-minv*0.1
		nn    = np.array([1,2,3,5,10,15,20])
		nn2    = np.array([1,2,3,5,10,15,20])
		for i, elem in enumerate(data):
			nn=nn+r
			nn2=nn2+r2
			plt.errorbar(nn,elem[:,0],yerr=elem[:,1],fmt='-o',label='$F_B \, Method \,%d$'%(i+1),color=label_color[i],linewidth=1)
			plt.errorbar(nn2,elem[:,2],yerr=elem[:,3],fmt='--s',label='$F_L \,Method \,%d$'%(i+1),color=label_color[i],linewidth=1)
			plt.axis([0,21,0.2,1.1])
		
		plt.hlines(b_baselines[j-1],0,21,colors='k')
		plt.hlines(b_humans[j-1],0,21,colors='k')
		plt.hlines(b_ceilings[j-1],0,21,colors='g',linewidth=1)
		plt.hlines(l_baselines[j-1],0,21,colors='k',linestyles='dashed')
		plt.hlines(l_humans[j-1],0,21,colors='k',linestyles='dashed')
		plt.hlines(l_ceilings[j-1],0,21,colors='g',linestyles='dashed',linewidth=1)
		plt.hlines(0,0,21,colors='k',linestyles='dashed',linewidth=1, label='$Labels\, references$')
		plt.hlines(0,0,21,colors='k',linestyles='solid',linewidth=1, label='$Boundaries\, references$')
		plt.xticks(n)
		plt.xlabel('$k$')
		plt.ylabel('$F$')
	legend = plt.legend(loc=2, bbox_to_anchor=(1.3,1))
	plt.show()

	
	return data

if __name__=='__main__':
	data=testNeighbors()
	# testDuplicates()