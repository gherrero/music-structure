import numpy as np
import matplotlib.pyplot as plt
import structure_retrieval as sr

'''receives a list of annotations an plots them'''

ann_path= 'metadata/all/'
# filename_list=['Beatles_AcrossTheUniverse_Beatles_1970-LetItBe-03.wav.lab','Beatles_BabyYoureARichMan_Beatles_1967-MagicalMysteryTour-10.wav.lab','Beatles_DontPassMeBy_Beatles_1968-TheBeatlesTheWhiteAlbumDisc1-14.wav.lab','Beatles_Chains_Beatles_1963-PleasePleaseMe-04.wav.csv','RM-P013.wav.csv']
filename_list=['Chopin_Op006No1_Rangell-2001_pid9094-01.mp3.lab','Chopin_Op006No1_Ashkenazy-1981_pid9058-01.mp3.lab','Chopin_Op006No1_Luisada-1990_pid9055-01.mp3.lab']

def visualize(annotations):
	newall = []
	fig = plt.figure('Annotations')
	m=0
	#i couldn't figure out how to get the max without all this shit
	for annotation in annotations:
		new = [0.0]
		for line in annotation:
			new.append(float(line[-2]))
			newall.append(new)
		if m <= max(new):	m = max(new)
	for i, annotation in enumerate(annotations):
		new = [0.0]
		for line in annotation:
			new.append(float(line[-2]))
			newall.append(new)
		p = plt.subplot(len(annotations),1,i+1)
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


if __name__ == "__main__":
	ann_list = sr.getAnnotationList(filename_list)
	visualize(ann_list)

