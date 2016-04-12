from attentionModel import attentionModel

engArray = [[0,1,0],[0,1,0]]
chinArray=  [[0,1,0],[0,0,1]]

model = attentionModel(4,len(chinArray[0]),len(engArray[0]),4)
finalH,rsi,rci,tbar = model.train(engArray,chinArray)
print "finalH"
print finalH
#print "rsi"
#print rsi
print "r1"
print rsi
print "r2"
print rci
print "tbar"
print tbar