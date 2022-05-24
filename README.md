# TimeLine of Implement of Mask RCNN
## Author: JC 
## Time: 2022/05/02 ~ ...

### Some details: 
    1. Slide window -> A LOT of computation! --> Use R-CNN(regional based works)
    2. Many bounding boxes for same object
    3. Still slow, and complicated 2 step process
    4. yolo 

### IOU(intersection over Union) _20220502
       IOU = Area of Intersection / Area Union  
       IoU > 0.5 -> "decent"
       IoU > 0.7 -> "pretty good"
       IoU > 0.9 -> "almost perfect"

     Box1 = [x1_min, y1_min, x1_max, y1_max]
     Box2 = [x2_min, y2_min, x2_max, y2_max]

     intersection box [ max(Box1[0], Box2[0]), max(Box1[1], Box2[1]), 
                        min(Box1[2], Box2[2]), min(Box1[3], Box2[3])]

      
