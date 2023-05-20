import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from cv_bridge import CvBridge

from super_gradients.common.object_names import Models
from super_gradients.training import models
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import random

class Status:
    def __init__(self):
        self.STATUS_OK = 1
        self.STATUS_ERR = 2
        
        self.status = self.STATUS_OK
        self.msg = ''

    def set_msg(self, message):
        self.msg = message
    def is_ok(self):
        return self.status == self.STATUS_OK
    
class Painter:
    def __init__(self):
        self._MIN_COEFF = 0
        self._MAX_COEFF = 255
        self._MIN_SUM = 200
        self._colours_list = {}

    def _colour_isok(self, colour):
        l = list(colour)
        sum = l[0] + l[1] + l[2]
        
        if (sum >= self._MIN_SUM):
            return colour not in self._colours_list
        else:
            return False
            

    def get_colour(self, id):
        if id in list(self._colours_list):
            return self._colours_list.get(id)
        
        colour_ok = False
        colour = (None, None, None)
        while (not colour_ok):
            # Generate random colour
            b = random.randint(self._MIN_COEFF, self._MAX_COEFF)
            g = random.randint(self._MIN_COEFF, self._MAX_COEFF)
            r = random.randint(self._MIN_COEFF, self._MAX_COEFF)
            colour = (b, g, r)
            colour_ok = self._colour_isok(colour)
        self._colours_list[id] = colour

        return colour
        

class SuperGradientsNode(Node):
    def __init__(self, node_name):
        super().__init__(node_name)

        self._status = Status()
        self._painter = Painter()

        # Get params from config file

        self._get_params()

    def _get_params(self):
        self.declare_parameters(
            namespace= '',
            parameters= [
                ('node_rate', 10),
                ('camera_topic', '/camera_topic'),
                ('min_prob', 0.2),
                ('bboxes_topic', '/sg_ros2/bboxes'),
                ('publish_image', True),
                ('image_topic', '/sg_ros2/image')
            ]
        )
        self._node_rate = self.get_parameter('node_rate').get_parameter_value().double_value
        self._camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self._min_prob = self.get_parameter('min_prob').get_parameter_value().double_value
        self._bboxes_topic = self.get_parameter('bboxes_topic').get_parameter_value().string_value
        self._publish_image = self.get_parameter('publish_image').get_parameter_value().bool_value
        self._image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        if ((self._min_prob < 0.0) or (self._min_prob > 1.0)):
            self._status.status = self._status.STATUS_ERR
            self._status.set_msg("Minimum Probability must be between 0.0 and 1.0")

    def _init_params(self):
        self._cv_bridge = CvBridge()
        self._image_received = False
        self._figure_setted = False
        self._last_img_msg = None
        self._model = models.get('yolox_n', pretrained_weights="coco")
        self._model = self._model.to("cuda" if torch.cuda.is_available() else "cpu")
        self._model.eval()
        qos = QoSProfile(
            reliability = QoSReliabilityPolicy.BEST_EFFORT,
            history = QoSHistoryPolicy.KEEP_LAST,
            depth = 1
        )

        self._camera_sub = self.create_subscription(
            Image, self._camera_topic, self._camera_cb, qos_profile=qos)
        
        self._bboxes_pub = self.create_publisher(
            Detection2DArray, self._bboxes_topic, qos)

        self._image_pub = self.create_publisher(
            Image, self._image_topic, qos)
    
    def _show_image(self, image: np.ndarray) -> None:
        """Show an image using matplotlib.
        :param image: Image to show in (H, W, C), RGB.
        """
        if (not self._figure_setted):
          plt.figure(figsize=(image.shape[1] / 100.0, image.shape[0] / 100.0), dpi=100)
          self._figure_setted = True
        plt.imshow(image, interpolation="nearest")
        plt.axis("off")
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.05)

    def _camera_cb(self, msg):
        self._last_img_msg = msg

        if (not self._image_received):
            self._image_received = True
    
    def _draw_img(self, img_cv2, bbox, id, score):
        if (not self._publish_image):
            return

        bbox_colour = self._painter.get_colour(id)
        cv2.rectangle(img_cv2,
            (int(bbox[0].item()), int(bbox[1].item())),
            (int(bbox[2].item()), int(bbox[3].item())), bbox_colour, 3)
        cv2.putText(img_cv2, id + " " + str(round(score, 2)),
            (int(bbox[0].item()), int(bbox[1].item())),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    
    def _publish_bboxes(self, predictions, raw_img, img_cv2):
        detection_arr = Detection2DArray()

        # Set the Header
        detection_arr.header.stamp = self.get_clock().now().to_msg()

        for pred in predictions:
            class_names = pred.class_names
            labels = pred.prediction.labels
            confidence = pred.prediction.confidence
            bboxes = pred.prediction.bboxes_xyxy
            for i, (label, conf, bbox) in enumerate(zip(labels, confidence, bboxes)):
                detection = Detection2D()
                result = ObjectHypothesisWithPose()
                result.id = class_names[int(label)]
                result.score = conf.item()
                detection.results.append(result)

                detection.bbox.center.x = float(int((bbox[2].item() + bbox[0].item()) /  2))
                detection.bbox.center.y = float(int((bbox[3].item() + bbox[1].item()) / 2))
                detection.bbox.size_x = float(int(bbox[2].item()) - int(bbox[0].item()))
                detection.bbox.size_y = float(int(bbox[3].item()) - int(bbox[1].item()))

                detection.source_img = raw_img
                detection.is_tracking = False

                self._draw_img(img_cv2, bbox, result.id, result.score)

                detection_arr.detections.append(detection)

        self._bboxes_pub.publish(detection_arr)
        if (self._publish_image):
            img = Image()
            img = self._cv_bridge.cv2_to_imgmsg(img_cv2)
            img.header.frame_id = 'camera_link'
            self._image_pub.publish(img)

    def _step(self):
        if (self._status.is_ok()):
            if (self._image_received):
                raw_img = self._last_img_msg

                img_cv2 = self._cv_bridge.imgmsg_to_cv2(self._last_img_msg,
                                                        self._last_img_msg.encoding)
                img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)

                predictions = self._model.predict(img_cv2, conf=self._min_prob)

                self._publish_bboxes(predictions, raw_img, img_cv2)
        else:
            self.get_logger().error(self._status.msg)

    def run(self):
        # Init params
        
        self._init_params()

        self.create_timer(1.0 / self._node_rate, self._step)

def main(args=None):
    rclpy.init(args=args, context=rclpy.get_default_context())

    super_gradients_node = SuperGradientsNode("super_gradients_ros2_node")
    super_gradients_node.run()
    rclpy.spin(super_gradients_node)

if __name__ == "__main__":
    main()