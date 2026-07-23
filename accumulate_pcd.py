#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np

class LidarAccumulator(Node):
    def __init__(self):
        super().__init__('lidar_accumulator')
        
        # Update topic name if your driver publishes to something else
        self.topic_name = '/livox/lidar'
        self.target_frames = 150  # ~15 seconds at 10Hz
        
        self.subscription = self.create_subscription(
            PointCloud2,
            self.topic_name,
            self.listener_callback,
            10)
        
        self.frame_count = 0
        self.all_points = []
        self.get_logger().info(f"Subscribed to {self.topic_name}. Accumulating {self.target_frames} frames (~15s)... Stay stationary!")

    def listener_callback(self, msg):
        pts = list(pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True))
        if len(pts) > 0:
            pts_np = np.array(pts, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            self.all_points.append(pts_np)
            self.frame_count += 1
            self.get_logger().info(f"Captured frame {self.frame_count}/{self.target_frames} ({len(pts)} pts)")

        if self.frame_count >= self.target_frames:
            self.save_pcd()
            rclpy.shutdown()

    def save_pcd(self):
        all_pts = np.concatenate(self.all_points)
        num_points = len(all_pts)
        pcd_path = 'scene.pcd'
        
        self.get_logger().info(f"Saving {num_points} points to {pcd_path}...")
        
        with open(pcd_path, 'w') as f:
            f.write("# .PCD v0.7 - Point Cloud Data file format\n")
            f.write("VERSION 0.7\n")
            f.write("FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n")
            f.write(f"WIDTH {num_points}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n")
            f.write(f"POINTS {num_points}\nDATA ascii\n")
            for p in all_pts:
                f.write(f"{p['x']:.4f} {p['y']:.4f} {p['z']:.4f}\n")
                
        self.get_logger().info(f"Successfully generated dense PCD with {num_points} points!")

def main():
    rclpy.init()
    node = LidarAccumulator()
    rclpy.spin(node)

if __name__ == '__main__':
    main()