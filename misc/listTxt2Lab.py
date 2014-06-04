import numpy as np

f=open('../sfs/alldatasets-gt.txt','r')
fo=open('alldatasets-labs.txt','w')
lines=f.readlines()
f.close()

for line in lines:
	fo.write(line[:-4]+"lab"+'\n')
fo.close()