import os
import csvio


folder = os.listdir("../metadata/mazurkas/mylabfilesMPI_tab")
filelist = open('../annotation_results/chopinGT.txt','w')

if __name__ == "__main__":


	for f in folder:
		filelist.write(f+"\n")