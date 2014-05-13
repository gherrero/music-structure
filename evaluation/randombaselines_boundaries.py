import sys
import wave
import mad
import random

if len(sys.argv)!=3:
	print '\nUSAGE: python '+sys.argv[0]+' <list_all_files.txt> <path_result_annotations>\n'
	sys.exit(1)

fn_list=sys.argv[1]
path_out=sys.argv[2]

# -------------------------------------------------------------
#TYPE_OF_NULL_MODEL=1	# Only boundaries at beginning and end
#TYPE_OF_NULL_MODEL=2	# Boundaries every LAG seconds
TYPE_OF_NULL_MODEL=3	# NUMBOUND randomly distributed boundaries (but with beginning and end (not counted))
LAG=3
NUMBOUND=10
# -------------------------------------------------------------

f=open(fn_list,'r')
lines=f.readlines()
f.close()

for aux in lines:
	line=aux[0:len(aux)-1]
	print line
	try:
		f=wave.open(line,'r')
		lengthAudio=float(f.getnframes())/float(f.getframerate())
		f.close()
	except:
		f=mad.MadFile(line)
		lengthAudio=float(f.total_time())/float(1000)
	print '\tTotal time =',lengthAudio
	fields=line.split('/')
	fn=fields[len(fields)-1]+'.lab'
	fullfn=path_out+'/'+fn
	print '\t'+fullfn

	if TYPE_OF_NULL_MODEL==1:
		f=open(fullfn,'w')
		f.write('0.0\t'+str(lengthAudio)+'\t0.0\n')
		f.close()
	elif TYPE_OF_NULL_MODEL==2:
		f=open(fullfn,'w')
		t=LAG
		while t<=lengthAudio:
			f.write(str(t-LAG)+'\t'+str(t)+'\t0.0\n')
			t+=LAG
		f.close()
	elif TYPE_OF_NULL_MODEL==3:
		b=range(3,int(lengthAudio)-3)
		random.shuffle(b)
		f=open(fullfn,'w')
		pre=0.0
		for t in range(0,NUMBOUND):
			f.write(str(float(pre))+'\t'+str(float(b[t]))+'\t0.0\n')
			pre=b[t]
		f.write(str(float(pre))+'\t'+str(lengthAudio)+'\t0.0\n')
		f.close()

