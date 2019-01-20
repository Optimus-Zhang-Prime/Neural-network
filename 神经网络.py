import numpy#矩阵
import scipy.spatial#激活函数1/1+(e^-x)
from matplotlib import pyplot
import scipy.misc
class neutralnetwork:#神经网络类
    def __init__(self,inputnodes,hiddennodes,outputnodes,learngrates):
        self.inputnode=inputnodes#节点数
        self.hiddennode=hiddennodes
        self.outputnode=outputnodes
        self.weighih=numpy.random.normal(0.0,pow(self.hiddennode,-0.5),(self.hiddennode,self.inputnode))
        #生成权重input—hidden的正态分布矩阵
        self.weighho=numpy.random.normal(0.0,pow(self.outputnode,-0.5),(self.outputnode,self.hiddennode))
        #生成权重hidden—output的正态分布矩阵
        self.learnrate=learngrates
        self.active_fun=lambda x:scipy.special.expit(x)#激活函数
    def connect(self,inputs):#连接神经网络
        inputinput=numpy.array(inputs,ndmin=2).T#矩阵转置
        hiddeninput=numpy.dot(self.weighih,inputinput)#隐藏层的输入矩阵  dot为点乘
        hiddenoutput=self.active_fun(hiddeninput)#隐藏层的输出矩阵
        outputinput=numpy.dot(self.weighho,hiddenoutput)#输出层的输入矩阵
        outputoutput=self.active_fun(outputinput)#输出层的输出矩阵
        return outputoutput
    def train(self,inputs,targets):#训练神经网络
        input=numpy.array(inputs,ndmin=2).T
        target=numpy.array(targets,ndmin=2).T
        hiddeninput=numpy.dot(self.weighih,input)#隐藏层的输入矩阵
        hiddenoutput=self.active_fun(hiddeninput)#隐藏层的输出矩阵
        outputinput=numpy.dot(self.weighho,hiddenoutput)#输出层的输入矩阵
        outputoutput=self.active_fun(outputinput)#输出层的输出矩阵
        outputerror=target-outputoutput#输出的误差
        hiddenerror=numpy.dot(self.weighho.T,outputerror)
        #Wj,k的改变量=学习率*k的误差*sigmoid（Ok）*（1-sigmoid（Ok))点乘Oj的转置
        self.weighho += self.learnrate*numpy.dot((outputerror*outputoutput*(1-outputoutput)),numpy.transpose(hiddenoutput))
        self.weighih += self.learnrate*numpy.dot((hiddenerror*hiddenoutput*(1-hiddenoutput)),numpy.transpose(input))
        #调节权重 其中transpose为转置
#训练
input_nodes=784#784个像素节点
hidden_nodes=200
output_nodes=10
learn_rate=0.1
n=neutralnetwork(input_nodes,hidden_nodes,output_nodes,learn_rate)
datafile=open(r"C:\Users\14531\Desktop\多媒体资料\手写数字\mnist_train.csv","r")
datalist=datafile.readlines()
datafile.close()
for record in datalist:
    allvalues=record.split(',')
    inputarr=(numpy.asfarray(allvalues[1:])/225.0*0.99)+0.01
    targets=numpy.zeros(10)+0.01
    targets[int(allvalues[0])]=0.99
    n.train(inputarr,targets)
#测试成功率
# testdatafile=open(r"C:\Users\14531\Desktop\多媒体资料\手写数字\mnist_test.csv","r")
# testdatalist=testdatafile.readlines()
# testdatafile.close()
# scorescard=[]
# for record in testdatalist:
#     allvalues=record.split(',')
#     correctnum=int(allvalues[0])
#     inputarr=(numpy.asfarray(allvalues[1:])/225.0*0.99)+0.01
#     outputarr=n.connect(inputarr)
#     label=numpy.argmax(outputarr)
#     if label==correctnum:
#         scorescard.append(1)
#     else:
#         scorescard.append(0)
# scorescardarr=numpy.asarray(scorescard)
# print("成功率=",scorescardarr.sum()/scorescardarr.size)


#自己写的数字
'''
img_array=scipy.misc.imread(r"C:\Users\14531\Desktop\手写数字2.png",flatten=True)
img_data=255.0-img_array.reshape(784)
img_data=(img_data/255.0*0.99)+0.01
outputarr=n.connect(img_data)
label=numpy.argmax(outputarr)
print(label)
img_array=scipy.misc.imread(r"C:\Users\14531\Desktop\手写数字3.png",flatten=True)
img_data=255.0-img_array.reshape(784)
img_data=(img_data/255.0*0.99)+0.01
outputarr=n.connect(img_data)
label=numpy.argmax(outputarr)
print(label)
'''
