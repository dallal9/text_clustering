from k_means import KMeans
import numpy
import pickle
dic= pickle.load( open( "dat2.dat", "rb" ) )

#pickle.dump(dic, open("dat2.dat","wb"), protocol=2)

lst=[]
for each in dic["vectors"]:
	temp=[]
	for lol in each["vectors"]:
		if len(lol) != 300:
			pass
		else:
			temp.append(lol)
	s = numpy.sum(temp,axis=0)
	try:
		if len(s) == 300:
			lst.append(s.tolist())
	except:
		pass		

word_vectors = lst
num_clusters = 2 
k_means = KMeans(num_clusters, word_vectors)
k_means.train()
print(k_means.get_cluster(lst[0]))