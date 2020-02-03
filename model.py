import cv2 as cv
import numpy as np
import os
import json
from abc import *
from PIL import Image
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
    rawFolder = "C:/Users/eotlr/data/safe/safeTest/"
    jsonFolder = "C:/Users/eotlr/data/json/safe/training/"
    imageFolder = "C:/Users/eotlr/data/blending/training/safe/"
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
        result = list(self._data[i:i+filter] for i in range(0,self._size,filter))
        return result

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
        self._result = self._makingImageMap()

    def _makingImageMap(self):
        rgbList = [[0] * 256 for i in range(0, 256)]
        result = np.zeros((256, 256, 3))
        for i in range(0,256):
            divider = self.__byteHash[i]
            if divider != 0:
                for j in range(0, 256):
                    if self.__relationHash[i][j] != 0:
                        amount = self.__relationHash[i][j]/divider *256 * 256* 256
                        result.itemset((i,j,0), int(amount%256))
                        amount = int(amount/256)
                        result.itemset((i,j,1), int(amount%256))
                        amount = int(amount/256)
                        result.itemset((i,j,2),amount%256)
                        result.itemset

        for i in range(0,256):
            for j in range(0,256):
                # result.itemset((i, j , 0), i)
                # result.itemset((i, j , 1) , j)
                result.itemset((i, j, 2), rgbList[i][j])
        return result

    def save(self):
        name = self._child.split(".")[0] + ".png"
        # im = Image.fromarray(np.uint8(self._result))
        # im.show()
        # im.save(PathClass.instance().getImageFolder()+name)
        cv.imwrite(PathClass.instance().getImageFolder()+name, self._result)
        # cv.imshow(mat=self._result,winname="wtf")
        # cv.waitKey(0)

    def get_image(self):
        return self._result

class RawByteImage(AbstractResizerClass):
    def __init__(self,child):
        AbstractResizerClass.__init__(self=self,child=child)
        self.targetSize = int(math.sqrt(int(self._size/3)))+1
    def save(self):
        name = self._child.split(".")[0]+".png"
        cv.imwrite(PathClass.instance().getImageFolder()+name, self._result)

    def operating(self):
        super().operating()

    def resizing(self):
        goalSize = self.targetSize * self.targetSize * 3
        if goalSize > self._size:
            for i in range(0, goalSize - self._size):
                self._data.append(0)
        self._making_image()

    def _making_image(self):
        self._data = super().chunking(filter=3)
        self._result = self._making_row(self._data)

    def _making_row(self,original):
        result = np.zeros((self.targetSize,self.targetSize,3))
        original = super().chunking(filter=self.targetSize)
        for i in range(0,self.targetSize):
            for j in range(0, self.targetSize):
                try:
                    if original[i][j]:
                        result[i][j] = np.array(original[i][j])
                except Exception as ex:
                    pass
            # result[i] = np.array(original[i*self.targetSize:(i+1)*self.targetSize])
        return result

    def get_image(self):
        return self._result


class ImageBlender:
    def __init__(self, child):
        self._child = child
        self._relationship = RelationshipImage(child=child)
        self._rawByteImage = RawByteImage(child=child)

    def operating(self):
        self._relationship.resizing()
        self._rawByteImage.resizing()
        relation_image = self._relationship.get_image()
        raw_byte_image = self._rawByteImage.get_image()
        raw_byte_image = cv.resize(raw_byte_image, dsize=(256, 256), interpolation=cv.INTER_AREA)
        raw_byte_rate = 0.5
        relation_rate = 1 - raw_byte_rate
        dst = cv.addWeighted(raw_byte_image, raw_byte_rate, relation_image, relation_rate, 0)
        name = self._child.split(".")[0]+".png"
        cv.imwrite(PathClass.instance().getImageFolder()+name, dst)
