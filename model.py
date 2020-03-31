import cv2 as cv
import numpy as np
import os
import json
from abc import *
import math


class Singleton:
    __instance = None

    @classmethod
    def __getInstance(cls):
        return cls.__instance

    @classmethod
    def instance(cls, *args, **kargs):
        cls.__instance = cls(*args, **kargs)
        cls.instance = cls.__getInstance
        return cls.__instance


class PathClass(Singleton):
    rawFolder = "C:/Users/eotlr/data/malwareTest/"
    jsonFolder = "C:/Users/eotlr/data/malwareJson/"
    imageFolder = "C:/Users/eotlr/data/malwareImage/"
    correlation = "C:/Users/eotlr/data/testcor/cor.txt"

    def __init__(self):
        pass

    def getRawFolder(self):
        return self.rawFolder

    def getJsonFolder(self):
        return self.jsonFolder

    def getImageFolder(self):
        return self.imageFolder

    def getCorrelation(self):
        return self.correlation

class RawImage:
    savePath = PathClass.instance().getJsonFolder()

    def __init__(self, child):
        self.path = PathClass.instance().getRawFolder() + child
        self.name = child
        self.size = os.path.getsize(self.path)
        self.data = ""
        with open(self.path,encoding="UTF-8",errors="ignore") as f:
            self.data = f.read()
            self.data = str.encode(self.data)
            f.close()

    def getSize(self):
        return self.size

    def saveJson(self):
        self.name = self.name.split(".")[0]+".txt"
        with open(self.savePath+self.name,'w',encoding="UTF-8") as f:
            f.write(json.dumps(list(self.data)))
            f.close()


class TotalSize(Singleton):
    def __init__(self):
        self.max = 0
        self.minimum = 100000
        self.average = None

    def getMaxSize(self):
        return self.max

    def getMinimumSize(self):
        return self.minimum

    def setMaxSize(self, size):
        self.max = size

    def setMinimumSize(self, size):
        self.minimum = size

    def calculatingAverageSize(self):
        self.average = int((self.max+self.minimum)/2)
        return self.average

    def getAverageSize(self):
        return self.average

class CorrelationClass(Singleton):
    def __init__(self):
        with open(PathClass.instance().getCorrelation(),encoding="UTF-8") as f:
            self.__correlationHash = json.loads(f.read())
            f.close()

    def getCorrelation(self, i):
        return self.__correlationHash[i]

class AbstractResizerClass(metaclass=ABCMeta):

    def __init__(self, child):
        self.parentPath = PathClass.instance().getJsonFolder()
        self.savePath = PathClass.instance().getImageFolder()
        self.targetSize = TotalSize.instance().getAverageSize()
        self._child = child
        self._data = None
        with open(self.parentPath+self._child,encoding="UTF-8") as f:
            self._data = json.loads(f.read())
            f.close()
        self._size = len(self._data)
        self._result = None

    @abstractmethod
    def resizing(self):
        pass

    def operating(self):
        self.resizing()
        self.save()

    def chunking(self, filter):
        return (self._data[i:i+filter] for i in range(0,self._size,filter))

    @abstractmethod
    def save(self):
        pass

    def appendPadding(self, originalList):
        for i in range(0,self.targetSize-self._size):
            originalList.append(0)
        return originalList

    def mappingCorrelationWithByte(self):
        result = []
        for i in self._data:
            result.append(CorrelationClass.instance().getCorrelation(str(i)))
        return result

class RiskAverageResizerClass(AbstractResizerClass):

    def __init__(self,child):
        AbstractResizerClass.__init__(self=self, child=child)
        self.__filterSize = int(self._size / self.targetSize)+1

    def operating(self):
        super().operating()

    def resizing(self):
        result = []
        if self.targetSize == self._size:
            result = super().mappingCorrelationWithByte()
        elif self.targetSize < self._size:
            chunkingData = super().chunking(self.__filterSize)
            for chunk in chunkingData:
                risk = 0
                for i in chunk:
                    risk += CorrelationClass.instance().getCorrelation(str(i))
                result.append(risk / self.__filterSize)
            if len(result) < self.targetSize:
                self._size = len(result)
                result = self.appendPadding(result)
        else:
            result = super().appendPadding(self.mappingCorrelationWithByte())
        self._result = result

    def save(self):
        with open(self.savePath + self._child,"w", encoding="UTF-8") as f:
            f.write(json.dumps(self._result))
            f.close()


class ConcatImage(AbstractResizerClass):
    def __init__(self, child):
        AbstractResizerClass.__init__(self=self, child=child)
        self._filterSize = int(self._size/self.targetSize)+1

    def operating(self):
        super().operating()

    def resizing(self):
        if self.targetSize == self._size:
            self._result = self._data
        elif self.targetSize < self._size:
            self._result = self.__concatingImage()
        else:
            self._result = self.appendPadding(self._data)

    def __concatingImage(self):
        result = None
        chungks = self.chunking(self._filterSize)
        for chungk in chungks:
            if not len(chungk)%3 == 0:
                for i in range(0, 3 - (len(chungk)%3)):
                    chungk.append(0)
            if result:
                result = self.blendingPoint(chungk)
            else:
                result =  np.concatenate(result,self.blendingPoint(chungk))
        return result

    def blendingPoint(self, chungk):
        points = self.chunking(3)
        prevRisk = 0
        prevCanvas = None
        for point in points:
            risk = 0
            currentCanvas = np.zeros((1,1,3),np.uint32)
            for byte, j in zip(point, range(0,3)):
                risk += CorrelationClass.instance().getCorrelation(str(byte))
            currentCanvas[0, 0] = point
            if not prevRisk == 0:
                riskProb = round(risk/(risk+prevRisk),2)
                result = cv.addWeighted(prevCanvas, 1-riskProb, currentCanvas, riskProb, 0)
                prevCanvas = result
            else:
                prevCanvas = currentCanvas
            prevRisk = risk
        return prevCanvas

    def save(self):
        pass

class RelationshipImage(AbstractResizerClass):
    def __init__(self,  child):
        AbstractResizerClass.__init__(self=self,child=child)
        self._filter = int(self._size/self.targetSize)
        self.__relationHash = [[0] * 256 for i in range(0, 256)]
        self.__byteHash = [0] * 256

    def operating(self):
        super().operating()

    def resizing(self):
        for i in range(0,self._size-1):
            self.__byteHash[self._data[i]] += 1
            self.__relationHash[self._data[i]][self._data[i+1]] += 1
        self.__byteHash[self._data[-1]] += 1
        self._result = self._makingImageMap()

    def _makingImageMap(self):
        rgbList = [[0] * 256 for i in range(0, 256)]
        for i in range(0,256):
            divider = self.__byteHash[i]
            if divider != 0:
                for j in range(0, 256):
                    rgbList[i][j] = int((self.__relationHash[i][j] / divider) * 100)
        result = np.zeros((256,256,3))
        for i in range(0,256):
            for j in range(0,256):
                # result.itemset((i, j , 0), i)
                # result.itemset((i, j , 1) , j)
                result.itemset((i, j, 2), rgbList[i][j])
        cv.imshow(mat=result,winname="original")
        cv.waitKey(0)
        return result

    def save(self):
        name = self._child.split(".")[0] + ".jpg"
        cv.imwrite(PathClass.instance().getImageFolder()+name, self._result)
        cv.imshow(mat=self._result,winname="wtf")
        cv.waitKey(0)

class ReverseByteNet(AbstractResizerClass):

    def __init__(self, child):
        ReverseByteNet.__init__(self = self, child=child)
        self._relation_prob = json.loads()
        self._chunks = child
        self._encoding_list = []
        self._encoding_activation_function = self.__select_activation_function("sigmoid")

    def __select_activation_function(self, activation_value):
        if activation_value == "sigmoid":
            def sigmoid(x):
                return 1 / (1 + np.exp(-x))
            return sigmoid

    def save(self):
        pass

    def resizing(self):
        pass

    def operating(self):
        self._encoding()
        self._decoding()

    def _encoding(self):
        self. _chunks = self.resizing()
        calculation_list =[]
        for chunk in range(self._chunks):
            calculation_list.append(self.__calculate_likelyhood(chunk))
        self._encoding_list = self.__making_encoding_list(calculation_list)

    def __maing_encoding_list(self, calculation_list):
        size_of_calculation_list = len(calculation_list)
        result = [0 for i in range(0,size_of_calculation_list)]
        for i in range(0, len(result)):
            for j in range(0, i):
                result[j] += calculation_list[i] * (1-self._encoding_activation_function(j-i))
            result[i] += calculation_list[i]
            for k in range(i+1, len(result)):
                result[k] += calculation_list[i] * (1-self._encoding_activation_function(i-k))
        return result

    def _decoding(self):
        self.



    def __calculate_likelyhood(self,chunk):
        prev = -1
        result = 0
        for b in range(chunk):
            if prev == -1:
                pass
            else:
                result += math.log(self._relation_prob[prev][b])
            prev = b
        return result