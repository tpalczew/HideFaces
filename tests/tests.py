import os
import sys
import unittest

try:
   base_dir = os.environ["BASE_DIR"]
except KeyError:
   print("Not exist environment variable %s" % "try sourcing build/environment.sh")

pathToFolder = str(base_dir) + "/src/preprocess/"
sys.path.append(pathToFolder)
import utils_yolo_v2
import load_data

try:
   base_dir = os.environ["BASE_DIR"]
except KeyError:
   print("Not exist environment variable %s" % "try sourcing build/environment.sh")

example_img = str(base_dir) + '/data/raw/face/test/18_Concerts_Concerts_18_1009.jpg'

class Testing(unittest.TestCase):
    """ Tests for HideFaces.AI"""

    """ Intersection over Union tests"""
    def test_intersection_over_union_equal_for_same_boxes(self):
        bbox1 = utils_yolo_v2.BoundBox(1,2,3,4)
        bbox2 = utils_yolo_v2.BoundBox(1,2,3,4)
        IoU = utils_yolo_v2.bbox_iou(bbox1, bbox2)
        self.assertEqual(IoU, 1)

    def test_intersection_over_union_notequal_for_dif_boxes(self):
        bbox1 = utils_yolo_v2.BoundBox(2,5,3,4)
        bbox2 = utils_yolo_v2.BoundBox(1,2,3,4)
        IoU = utils_yolo_v2.bbox_iou(bbox1, bbox2)
        self.assertNotEqual(IoU, 1)

    def test_interval_overlap(self):
        inter = utils_yolo_v2._interval_overlap([1, 2], [1, 2])
        self.assertEqual(inter, 1)

    def test_interval_overlap_for_dif_intervals(self):
        inter = utils_yolo_v2._interval_overlap([1, 2], [2, 5])
        self.assertNotEqual(inter, 1)

    def test_interval_overlap_for_dif_intervals2(self):
        inter = utils_yolo_v2._interval_overlap([1, 4], [2, 8])
        self.assertEqual(inter, 2)

    """ Test Sigmoid """
    def test_sigmoid_function(self):
        xexp = 0.5
        x = utils_yolo_v2._sigmoid(0)
        self.assertEqual(x, xexp)

    """ Test Softmax """
    def test_softmax_function(self):
        z = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
        sofmax_exp = [0.0236405, 0.0642617, 0.174681, 0.474833, 0.0236405, 0.0642617, 0.174681]
        func = lambda x: round(x,4)
        sofmax = utils_yolo_v2._softmax(z).tolist()
        newsoftmax = list(map(func, sofmax))
        newsoftmax_exp = list(map(func, sofmax_exp))
        self.assertListEqual(newsoftmax_exp,newsoftmax)

    """ Test load data """
    def test_load_data_haar_shape(self):
        img = example_img
        image, img_arr, gray, img_shape = load_data.load_pictures_haar(img)
        expected_img_shape = (224,224,3)
        self.assertEqual(img_shape, expected_img_shape)

    def test_load_data_yolo_shape(self):
        img = example_img
        image, input_image = load_data.load_pictures_yolo(img)
        img_shape = input_image.shape
        expected_img_shape = (1,416,416,3)
        self.assertEqual(img_shape, expected_img_shape)



if __name__ == '__main__':
    unittest.main()
