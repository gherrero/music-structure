import os
import csvio


folder = os.listdir("../metadata/beatles/mylabfilesTUT_tab")
filelist = open('ann-beatlesTUT-n100.txt','w')

if __name__ == "__main__":


	for f in folder:
		filelist.write(f+"\n")