import numpy as np

f=open('sf-chopin-n100.txt','r')
fo=open('ann-chopin-n100lab.txt','w')
lines=f.readlines()
f.close()

for line in lines:
	fo.write(line[:-4]+"lab"+'\n')
fo.close()

