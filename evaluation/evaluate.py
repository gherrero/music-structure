from numpy import *
import csvio
import csv
import time
import os.path
import sys

THRESHOLD_FINE      = 0.5
THRESHOLD_COARSE    = 3
TEMPORAL_RESOLUTION = 0.3

def boundaries_filelist(filenameList,pathResults,pathGroundTruth):
	f=open(filenameList,'r')
	lines=f.readlines()
	f.close()
	finePrecision   = []
	fineRecall      = []
	fineFmeasure    = []
	coarsePrecision = []
	coarseRecall    = []
	coarseFmeasure  = []
	guess2True      = []
	true2Guess      = []
	for line in lines:
		fields=line[0:len(line)-1].split('/')
		fP,fR,fF,cP,cR,cF,G2T,T2G=boundaries_singlefile(pathResults+fields[len(fields)-1],pathGroundTruth+fields[len(fields)-1][:-4]+'.lab')
		finePrecision.append(fP)
		fineRecall.append(fR)
		fineFmeasure.append(fF)
		coarsePrecision.append(cP)
		coarseRecall.append(cR)
		coarseFmeasure.append(cF)
		guess2True.append(G2T)
		true2Guess.append(T2G)
	return finePrecision,fineRecall,fineFmeasure,coarsePrecision,coarseRecall,coarseFmeasure,guess2True,true2Guess

def labels_filelist(filenameList,pathResults,pathGroundTruth):
	f=open(filenameList,'r')
	lines=f.readlines()
	f.close()
	Precision=[]
	Recall=[]
	Fmeasure=[]
	OverSeg=[]
	UnderSeg=[]
	for i,line in enumerate(lines):
		print i
		fields=line[0:len(line)-1].split('/')
		P,R,F,So,Su=labels_singlefile(pathResults+fields[len(fields)-1],pathGroundTruth+fields[len(fields)-1][:-4]+'.lab')
		Precision.append(P)
		Recall.append(R)
		Fmeasure.append(F)
		OverSeg.append(So)
		UnderSeg.append(Su)
	return Precision,Recall,Fmeasure,OverSeg,UnderSeg

def boundaries_singlefile(fnResult,fnGT):
	# Load files
	res=csvio.load(fnResult,'\t')
	# print res
	# print fnGT
	gt=csvio.load(fnGT,'\t')
	# Get boundaries
	resBoundaries=[]
	for i in range(0,res.shape[0]): resBoundaries.append(res[i,0])
	resBoundaries.append(res[res.shape[0]-1,1])
	gtBoundaries=[]
	for i in range(0,gt.shape[0]): gtBoundaries.append(gt[i,0])
	gtBoundaries.append(gt[gt.shape[0]-1,1])
	# Evaluate boundaries (Precision)
	fineMatches=0
	coarseMatches=0
	for i in range(0,len(resBoundaries)):
		for j in range(0,len(gtBoundaries)):
			if abs(gtBoundaries[j]-resBoundaries[i])<THRESHOLD_FINE:
				fineMatches+=1
				break
	for i in range(0,len(resBoundaries)):
		for j in range(0,len(gtBoundaries)):
			if abs(gtBoundaries[j]-resBoundaries[i])<THRESHOLD_COARSE:
				coarseMatches+=1
				break
	fP=float(fineMatches)/float(len(resBoundaries))
	cP=float(coarseMatches)/float(len(resBoundaries))
	# Evaluate boundaries (guess to true)
	gtt=[]
	for i in range(0,len(resBoundaries)):
		minTime=10000000
		for j in range(0,len(gtBoundaries)):
			dif=abs(gtBoundaries[j]-resBoundaries[i])
			if dif<minTime: minTime=dif
		gtt.append(minTime)
	# Evaluate boundaries (Recall)
	fineMatches=0
	coarseMatches=0
	for i in range(0,len(gtBoundaries)):
		for j in range(0,len(resBoundaries)):
			if abs(gtBoundaries[i]-resBoundaries[j])<THRESHOLD_FINE:
				fineMatches+=1
				break
	for i in range(0,len(gtBoundaries)):
		for j in range(0,len(resBoundaries)):
			if abs(gtBoundaries[i]-resBoundaries[j])<THRESHOLD_COARSE:
				coarseMatches+=1
				break
	fR=float(fineMatches)/float(len(gtBoundaries))
	cR=float(coarseMatches)/float(len(gtBoundaries))
	# Evaluate boundaries (true to guess)
	ttg=[]
	for i in range(0,len(gtBoundaries)):
		minTime=10000000
		for j in range(0,len(resBoundaries)):
			dif=abs(gtBoundaries[i]-resBoundaries[j])
			if dif<minTime: minTime=dif
		ttg.append(minTime)
	# Evaluate boundaries (F-measure)
	if fP>0 or fR>0: fF=2.0*fP*fR/(fP+fR)
	else: fF=0.0
	if cP>0 or cR>0: cF=2.0*cP*cR/(cP+cR)
	else: fF=0.0
	return fP,fR,fF,cP,cR,cF,median(gtt),median(ttg)

def labels_singlefile(fnResult,fnGT):
	# Load files
	res=csvio.load(fnResult,'\t')
	gt=csvio.load(fnGT,'\t')
	# Get boundaries
	resBoundaries=[]
	for i in range(0,res.shape[0]): resBoundaries.append(res[i,0])
	resBoundaries.append(res[res.shape[0]-1,1])
	gtBoundaries=[]
	for i in range(0,gt.shape[0]): gtBoundaries.append(gt[i,0])
	gtBoundaries.append(gt[gt.shape[0]-1,1])
	# Get labels
	resLabels=[]
	for i in range(0,res.shape[0]): resLabels.append(res[i,2])
	gtLabels=[]
	for i in range(0,gt.shape[0]): gtLabels.append(gt[i,2])
	# Generate frames
	resFrames=[]
	ind=0.0
	while ind<resBoundaries[len(resBoundaries)-1]:
		for i in range(1,len(resBoundaries)):
			if ind<resBoundaries[i]:
				break
		resFrames.append(resLabels[i-1])
		ind+=TEMPORAL_RESOLUTION
	gtFrames=[]
	ind=0.0
	while ind<gtBoundaries[len(gtBoundaries)-1]:
		for i in range(1,len(gtBoundaries)):
			if ind<gtBoundaries[i]:
				break
		gtFrames.append(gtLabels[i-1])
		ind+=TEMPORAL_RESOLUTION
	# Pm, Ph
	Pm=[]
	for i in range(0,len(resFrames)):
		for j in range(i+1,len(resFrames)):
			if resFrames[i]==resFrames[j]:
				Pm.append(str(i)+'-'+str(j))
	Ph=[]
	for i in range(0,len(gtFrames)):
		for j in range(i+1,len(gtFrames)):
			if gtFrames[i]==gtFrames[j]:
				Ph.append(str(i)+'-'+str(j))
	intersect=list(set(Pm)&set(Ph))
	# Over and under segmentation
	Le=unique(resLabels)
	La=unique(gtLabels)
	C=0.001*ones((len(La),len(Le)))
	for n in range(0,min(len(gtFrames),len(resFrames))):
		for i in range(0,len(La)):
			if gtFrames[n]==La[i]:
				break
		for j in range(0,len(Le)):
			if resFrames[n]==Le[j]:
				break
		C[i,j]+=1.0
	p=C/sum(C)
	pa=zeros((len(La),1))
	pe=zeros((len(Le),1))
	for i in range(0,len(La)):
		for j in range(0,len(Le)):
			pa[i,0]+=p[i,j]
			pe[j,0]+=p[i,j]
	pae=zeros((len(La),len(Le)))
	pea=zeros((len(La),len(Le)))
	for i in range(0,len(La)):
		for j in range(0,len(Le)):
			pae[i,j]=C[i,j]/sum(C[:,j])
			pea[i,j]=C[i,j]/sum(C[i,:])
	Hea=0.0
	for i in range(0,len(La)):
		tmp=0.0
		for j in range(0,len(Le)):
			tmp+=pea[i,j]*log(pea[i,j])/log(2)
		Hea-=pa[i]*tmp
	Hae=0.0
	for j in range(0,len(Le)):
		tmp=0.0
		for i in range(0,len(La)):
			tmp+=pae[i,j]*log(pae[i,j])/log(2)
		Hae-=pe[j]*tmp
	# Measures
	if len(Pm)>0:
		P=float(len(intersect))/float(len(Pm))
	else:
		P=0.0
	if len(Ph)>0:
		R=float(len(intersect))/float(len(Ph))
	else:
		R=0.0
	if P+R>0:
		F=2*P*R/(P+R)
	else:
		F=0.0
	if len(Le)<=0: So=1.0
	else: So=1.0-Hea/(log(len(Le))/log(2))
	if len(La)<=0: Su=1.0
	else: Su=1.0-Hae/(log(len(La))/log(2))
	return P,R,F,So,Su

def print_boundaries(fP,fR,fF,cP,cR,cF,tG2T,tT2G):
	print '============================================='
	print 'BOUNDARIES:'
	print '============================================='
	print 'Precision ('+str(THRESHOLD_FINE)+' sec) = %.3f'%mean(fP)+' (%.4f)'%std(fP)
	print 'Recall ('+str(THRESHOLD_FINE)+' sec) = %.3f'%mean(fR)+' (%.4f)'%std(fR)
	print 'F-measure ('+str(THRESHOLD_FINE)+' sec) = %.3f'%mean(fF)+' (%.4f)'%std(fF)
	print '---------------------------------------------'
	print 'Precision ('+str(THRESHOLD_COARSE)+' sec) = %.3f'%mean(cP)+' (%.4f)'%std(cP)
	print 'Recall ('+str(THRESHOLD_COARSE)+' sec) = %.3f'%mean(cR)+' (%.4f)'%std(cR)
	print 'F-measure ('+str(THRESHOLD_COARSE)+' sec) = %.3f'%mean(cF)+' (%.4f)'%std(cF)
	print '[2*P*R/(P+R) = %.3f'%(2*mean(cP)*mean(cR)/(mean(cP)+mean(cR)))+']'
	print '---------------------------------------------'
	print 'Median deviation guess-to-true = %.2f'%mean(tG2T)+' (%.3f)'%std(tG2T)
	print 'Median deviation true-to-guess = %.2f'%mean(tT2G)+' (%.3f)'%std(tT2G)
	print '============================================='
	return True

def print_labels(P,R,F,So,Su):
	print '============================================='
	print 'LABELS:'
	print '============================================='
	print 'Precision ('+str(TEMPORAL_RESOLUTION)+' sec) = %.3f'%mean(P)+' (%.4f)'%std(P)
	print 'Recall ('+str(TEMPORAL_RESOLUTION)+' sec) = %.3f'%mean(R)+' (%.4f)'%std(R)
	print 'F-measure ('+str(TEMPORAL_RESOLUTION)+' sec) = %.3f'%mean(F)+' (%.4f)'%std(F)
	print '[2*P*R/(P+R) = %.3f'%(2*mean(P)*mean(R)/(mean(P)+mean(R)))+']'
	print '---------------------------------------------'
	print 'Over-segmentation = %.3f'%mean(So)+' (%.4f)'%std(So)
	print 'Under-segmentation = %.3f'%mean(Su)+' (%.4f)'%std(Su)
	print '============================================='
	return True

def print_overall(fF,F):
	print ''
	print '============================================='
	print 'OVERALL = %.3f'%mean([mean(fF), mean(F)])
	print '============================================='
	print ''
	return True

def selection(case):

	return {
		# (filenameList, pathResults, pathGroundTruth)
		1: ('annotation_results/beatlesQMUL.txt','annotation_results/beatlesQMUL-n100/','metadata/beatles/mylabfilesQMUL_tab/'),
		2: ('annotation_results/beatlesQMUL.txt','annotation_results/beatlesQMUL-all-n100/','metadata/beatles/mylabfilesQMUL_tab/'),		
		3: ('annotation_results/chopin.txt','annotation_results/chopin-n100/','metadata/mazurkas/mylabfilesMPI_tab/'),		
		4: ('annotation_results/chopin.txt','annotation_results/chopin-all-n100/','metadata/mazurkas/mylabfilesMPI_tab/'),		
		5: ('annotation_results/rwcP.txt','annotation_results/rwcAIST-n100/','metadata/rwc/mylabfilesAIST_tab/'),		
		6: ('annotation_results/rwcP.txt','annotation_results/rwcAIST-all-n100/','metadata/rwc/mylabfilesAIST_tab/'),		
		7: ('annotation_results/beatlesTUT.txt','annotation_results/beatlesTUT-n100/','metadata/beatles/mylabfilesTUT_tab/'),		
		8: ('annotation_results/beatlesTUT.txt','annotation_results/beatlesTUT-all-n100/','metadata/beatles/mylabfilesTUT_tab/'),		
		9: ('annotation_results/rwcP.txt','annotation_results/rwcIRISA-n100/','metadata/rwc/mylabfilesIRISA_tab/'),
		10: ('annotation_results/rwcP.txt','annotation_results/rwcIRISA-all-n100/','metadata/rwc/mylabfilesIRISA_tab/'),
	}[case]

def process(i):
	(filenameList,pathResults,pathGroundTruth)=selection(i)

	fP,fR,fF,cP,cR,cF,G2T,T2G=boundaries_filelist(filenameList,pathResults,pathGroundTruth)
	print_boundaries(fP,fR,fF,cP,cR,cF,G2T,T2G)

	f = csv.writer(open("results.csv", "a"))
	
	f.writerow([time.strftime("%H:%M %d/%m"),"Threshold","Precision","Recall","F-Score"])
	f.writerow(["",THRESHOLD_FINE, "%.3f"%mean(fP) +" (" +"%.3f"%std(fP)+")", "%.3f"%mean(fR) +" (" +"%.3f"%std(fR)+")", "%.3f"%mean(fF) +" (" +"%.3f"%std(fF)+")"])
	f.writerow(["",THRESHOLD_COARSE, "%.3f"%mean(cP) +" (" +"%.3f"%std(cP)+")", "%.3f"%mean(cR) +" (" +"%.3f"%std(cR)+")", "%.3f"%mean(cF) +" (" +"%.3f"%std(cF)+")"])

	P,R,F,So,Su=labels_filelist(filenameList,pathResults,pathGroundTruth)
	print_labels(P,R,F,So,Su)

if __name__=="__main__":
	# case=sys.argv[1]
	
	filenameList    = '../annotation_results/chopin.txt'
	pathResults     = '../fakeboundaries/'
	pathGroundTruth = '../metadata/mazurkas/mylabfilesMPI_tab/'

	# (filenameList,pathResults,pathGroundTruth)=selection()

	fP,fR,fF,cP,cR,cF,G2T,T2G=boundaries_filelist(filenameList,pathResults,pathGroundTruth)
	print_boundaries(fP,fR,fF,cP,cR,cF,G2T,T2G)

	f = csv.writer(open("results.csv", "a"))
	
	f.writerow([time.strftime("%H:%M %d/%m"),"Threshold","Precision","Recall","F-Score"])
	f.writerow(["",THRESHOLD_FINE, "%.3f"%mean(fP) +" (" +"%.3f"%std(fP)+")", "%.3f"%mean(fR) +" (" +"%.3f"%std(fR)+")", "%.3f"%mean(fF) +" (" +"%.3f"%std(fF)+")"])
	f.writerow(["",THRESHOLD_COARSE, "%.3f"%mean(cP) +" (" +"%.3f"%std(cP)+")", "%.3f"%mean(cR) +" (" +"%.3f"%std(cR)+")", "%.3f"%mean(cF) +" (" +"%.3f"%std(cF)+")"])

	P,R,F,So,Su=labels_filelist(filenameList,pathResults,pathGroundTruth)
	print_labels(P,R,F,So,Su)
