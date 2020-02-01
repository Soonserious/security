from model import *
import os

parentPath = PathClass.instance().getRawFolder()

children = os.listdir(parentPath)
#
# for child  in children:
#     rawImage = RawImage(child)
#     if rawImage.getSize() > TotalSize.instance().getMaxSize():
#         TotalSize.instance().setMaxSize(rawImage.getSize())
#     if rawImage.getSize() < TotalSize.instance().getMinimumSize():
#         TotalSize.instance().setMinimumSize(rawImage.getSize())
#     rawImage.saveJson()

TotalSize.instance().calculatingAverageSize()
parentPath = PathClass.instance().getJsonFolder()
children = os.listdir(parentPath)


for child in children:
    resizerClass = RelationshipImage(child=child)
    resizerClass.operating()
    resizerClass.save()
