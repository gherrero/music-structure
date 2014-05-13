from numpy import *

def load(filename,separator):
	f=open(filename,'r')
	lines=f.readlines()
	f.close()
	fields=lines[1].split(separator)
	NROWS=len(lines)
	NCOLS=len(fields)
	data=zeros((NROWS,NCOLS))
	for i in range(0,NROWS):
		fields=lines[i].split(separator)
		for j in range(0,NCOLS):
			data[i,j]=float(fields[j])
	return data

def save(data,filename,separator):
	NROWS=data.shape[0]
	NCOLS=data.shape[1]
	f=open(filename,'w')
	for i in range(0,NROWS):
		f.write(str(data[i,0]))
		for j in range(1,NCOLS):
			f.write(separator+str(data[i,j]))
		f.write('\n')
	f.close()

def append(data,filename,separator):
	NROWS=data.shape[0]
	NCOLS=data.shape[1]
	f=open(filename,'a')
	for i in range(0,NROWS):
		f.write(str(data[i,0]))
		for j in range(1,NCOLS):
			f.write(separator+str(data[i,j]))
		f.write('\n')
	f.close()

