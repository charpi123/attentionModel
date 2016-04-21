from attentionModel import attentionModel

engArray = [[0,1,0],[0,1,0]]
chinArray=  [[0,1,0],[0,0,1]]

model = attentionModel(10,len(chinArray[0]),len(engArray[0]),10)
finalH,rsi,tbar,ti_temp,ti = model.train(engArray,chinArray)
print "finalH"
print finalH
#print "rsi"
#print rsi
print "r1"
print rsi
print "tbar"
print tbar
print "ti_temp"
print ti_temp
print "ti"
print ti