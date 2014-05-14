import numpy as np

f=open('../annotation_results/ann-rwcIRISA-n100.txt','r')
fo=open('sf-rwcIRISA-n100.txt','w')
lines=f.readlines()
f.close()

for line in lines:
	fo.write(line[:-4]+"csv"+'\n')
fo.close()