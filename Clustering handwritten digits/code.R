rm(list=ls())
library(mvtnorm)
k =10
niter = 20

# Read handwritten digits data
myData=read.csv("semeion.csv",header=FALSE, sep=" ")

# Build data matrix with pixel and label data
myLabel=apply(myData[,257:266],1,function(xx){
  return(which(xx=="1")-1)
})
myX=data.matrix(myData[,1:256])
d = dim(myX)[2]
N = dim(myX)[1]

#initliaze values using kmeans
hwd_cluster = kmeans(myX, k, nstart = 30)
init_means = hwd_cluster$centers
cluster = hwd_cluster$cluster
init_cmp <- sapply(c(1:k), function(kk){
  return(dim(myX[cluster==kk,])[1]/N)
})
init_gammaik = matrix(0, N, k)
for (obs in c(1:length(cluster))) {
  init_gammaik[obs, cluster[obs]] = 1
}
Nk_init = colSums(init_gammaik)
pik_init = Nk_init/N

#function to compute covariance matrix
computeVariance <- function(kk,q) {
vc2 = matrix(0,d,d)
for (obs in c(1:N)) {
     vc1 = ((myX[obs,]-means[kk,])%*%t(myX[obs,]-means[kk,]))*gammaik[obs,kk]
     vc2 = vc2 + vc1
   }
   varcov = vc2/Nk[kk]
   if(q==0){
     myEig = eigen(varcov, symmetric = TRUE, only.values = TRUE)
     sigma_sq = sum(myEig$values[q+1:d], na.rm = TRUE)/(d-q)
     return(sigma_sq*diag(d))
   }
   myEig = eigen(varcov, symmetric = TRUE)
   Vq = myEig$vectors[,1:q]
   sigma_sq = sum(myEig$values[q+1:d], na.rm = TRUE)/(d-q)
   Wq = Vq%*%diag(sqrt(myEig$values[1:q]-sigma_sq))
   varcovk = Wq%*%t(Wq) + sigma_sq*diag(d)
   return(varcovk)
}

aics <- vector()
PCs = c(0,2,4,6)
obs_dll = matrix(0, length(PCs), niter)
labels = matrix(0, length(PCs), N)
qmeans = array(dim = c(length(PCs), k, d))

# compute for each q
for (q in PCs) {  
  dll <- vector()
  means = init_means
  cmp = init_cmp
  gammaik = init_gammaik
  Nk = Nk_init
  pik = pik_init
  #iterate till convergence
  for (iter in c(1:niter)) { 
  #Estep - compute current class membership probabilities
  prob_density = sapply(c(1:k), function(kk){
    return(pik[kk]*dmvnorm(myX, means[kk,], computeVariance(kk,q)))
  })
  gammaik = prob_density/rowSums(prob_density)
  
  #record data log likelihood per iteration
  dll <- append(dll,sum(log(rowSums(prob_density)))) 
  
  #Mstep - given the current class membership distribution, compute parameter estimates
  Nk = colSums(gammaik)
  pik = Nk/N
  for(i in c(1:k)){
    means[i,] = colSums(sweep(myX, gammaik[,i], MARGIN = 1, '*'))/Nk[i]
  }
  
}
  idx = which(PCs==q, arr.ind = TRUE)
  obs_dll[idx,] = dll
  qmeans[idx,,] = means
  # record AIC for each q
  AIC = -2*tail(dll,1) + 2*(d*q + 1 - q*(q-1)/2)
  aics = append(aics, AIC)
  labels[idx,] = apply(gammaik, 1, function(xx){
    return(which.max(xx)-1)
  })
  plot(dll, type = "l")
}

#plot of data likelihood vs. iteration number
dev.new(width=6,height=6)
par(mar=c(0,0,0,0), mfrow=c(2,2))
for (i in c(1:dim(obs_dll)[1])){
  plot(obs_dll[i,], type="l")
}

#visualization
qchoice = PCs[which.min(aics)] #best q -> min AIC
idx = which(PCs==qchoice, arr.ind = TRUE)
newLabels = labels[idx,]
means = qmeans[idx,,]

dev.new(width=10,height=6)
par(mar=c(0,0,0,0), mfrow=c(10,6))
for(kk in c(1:k)) {
  image(t(matrix(means[kk,],byrow=TRUE,16,16)[16:1,]),col=gray(0:1),axes=FALSE)
  for(i in c(1:5)){
    img = rmvnorm(1, means[kk,], computeVariance(kk, qchoice))
    image(t(matrix(img,byrow=TRUE,16,16)[16:1,]),col=gray(0:1),axes=FALSE)
  }
  }

#calculating the miscategorization rate

groups = split(myLabel, newLabels)
miscategorizeRate = sapply(groups, function(grp){
  mcd = as.numeric(names(which.max(table(grp))))
  return(c(mcd, 1-(length(grp[grp==mcd])/length(grp))))
})
library(knitr)
kable(miscategorizeRate, col.names = c("Most common digit", "Miscategorization Rate"))

total_correct = sum(sapply(groups, function(grp){
  mcd = as.numeric(names(which.max(table(grp))))
  return(length(grp[grp==mcd]))
}))
overallMiscategorizeRate = 1-(total_correct)/N 
print(overallMiscategorizeRate)

