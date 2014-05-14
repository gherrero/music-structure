import os
import csvio


folder = os.listdir("../metadata/rwc/mylabfilesIRISA_tab")
filelist = open('../annotation_results/ann-rwcIRISA-n100.txt','w')

if __name__ == "__main__":


	for f in folder:
		filelist.write(f+"\n")