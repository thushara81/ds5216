import torch
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Install required packages:
# pip install torch torchvision opencv-python ultralytics

from ultralytics import YOLO

class PlayerDetectionSystem:
    """
    Complete system for player detection and keypoint detection in sports videos.
    Uses YOLOv8 for object detection and pose estimation.
    """
    
    def __init__(self, detection_model='yolov8n.pt', pose_model='yolov8n-pose.pt'):
        """
        Initialize the detection system.
        
        Args:
            detection_model: Path to YOLO detection model (or model name to download)
            pose_model: Path to YOLO pose estimation model (or model name to download)
        """
        print("Loading models...")
        self.detector = YOLO(detection_model)
        self.pose_estimator = YOLO(pose_model)
        print("Models loaded successfully!")
        
    def detect_players(self, frame, conf_threshold=0.3):
        """
        Detect players in a single frame.
        
        Args:
            frame: Input frame (numpy array)
            conf_threshold: Confidence threshold for detection
            
        Returns:
            List of detections with bounding boxes and confidence scores
        """
        results = self.detector(frame, conf=conf_threshold, classes=[0])  # class 0 is 'person'
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf)
                })
        
        return detections
    
    def detect_keypoints(self, frame, conf_threshold=0.3):
        """
        Detect player keypoints (pose estimation).
        
        Args:
            frame: Input frame (numpy array)
            conf_threshold: Confidence threshold for detection
            
        Returns:
            List of keypoint detections
        """
        results = self.pose_estimator(frame, conf=conf_threshold)
        
        keypoint_data = []
        for result in results:
            if result.keypoints is not None:
                for i, keypoints in enumerate(result.keypoints):
                    kpts = keypoints.xy[0].cpu().numpy()  # Get keypoints
                    conf = keypoints.conf[0].cpu().numpy() if keypoints.conf is not None else None
                    
                    # Get bounding box for this person
                    bbox = result.boxes[i].xyxy[0].cpu().numpy() if result.boxes else None
                    
                    keypoint_data.append({
                        'keypoints': kpts.tolist(),
                        'keypoint_confidence': conf.tolist() if conf is not None else None,
                        'bbox': bbox.tolist() if bbox is not None else None
                    })
        
        return keypoint_data
    
    def process_video(self, video_path, output_path=None, visualize=True, 
                     save_annotations=True, detect_poses=True):
        """
        Process a complete video for player detection and optional keypoint detection.
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            visualize: Whether to draw detections on frames
            save_annotations: Whether to save annotations to JSON
            detect_poses: Whether to perform pose estimation (bonus task)
            
        Returns:
            Dictionary containing all detections and processing info
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        # Setup video writer if output path provided
        writer = None
        if output_path and visualize:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Storage for annotations
        all_detections = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_data = {
                'frame': frame_idx,
                'timestamp': frame_idx / fps
            }
            
            # Player detection
            detections = self.detect_players(frame)
            frame_data['player_detections'] = detections
            
            # Keypoint detection (bonus)
            if detect_poses:
                keypoints = self.detect_keypoints(frame)
                frame_data['keypoints'] = keypoints
            
            all_detections.append(frame_data)
            
            # Visualization
            if visualize:
                vis_frame = frame.copy()
                
                # Draw player bounding boxes
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    conf = det['confidence']
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis_frame, f'Player {conf:.2f}', (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw keypoints
                if detect_poses:
                    for kpt_data in frame_data.get('keypoints', []):
                        keypoints = np.array(kpt_data['keypoints'])
                        # Draw keypoints
                        for kpt in keypoints:
                            if kpt[0] > 0 and kpt[1] > 0:  # Valid keypoint
                                cv2.circle(vis_frame, (int(kpt[0]), int(kpt[1])), 
                                         3, (0, 0, 255), -1)
                        
                        # Draw skeleton connections
                        self._draw_skeleton(vis_frame, keypoints)
                
                if writer:
                    writer.write(vis_frame)
            
            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames...")
        
        cap.release()
        if writer:
            writer.release()
        
        print(f"Processing complete! Processed {frame_idx} frames.")
        
        # Save annotations
        if save_annotations:
            json_path = Path(video_path).stem + '_annotations.json'
            with open(json_path, 'w') as f:
                json.dump({
                    'video_info': {
                        'path': str(video_path),
                        'fps': fps,
                        'resolution': [width, height],
                        'total_frames': frame_idx
                    },
                    'detections': all_detections
                }, f, indent=2)
            print(f"Annotations saved to: {json_path}")
        
        return {
            'video_info': {'fps': fps, 'resolution': [width, height], 'frames': frame_idx},
            'detections': all_detections
        }
    
    def _draw_skeleton(self, frame, keypoints):
        """Draw skeleton connections between keypoints."""
        # COCO keypoint connections (for YOLOv8 pose)
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        for connection in connections:
            pt1_idx, pt2_idx = connection
            if pt1_idx < len(keypoints) and pt2_idx < len(keypoints):
                pt1 = keypoints[pt1_idx]
                pt2 = keypoints[pt2_idx]
                if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                    cv2.line(frame, (int(pt1[0]), int(pt1[1])),
                            (int(pt2[0]), int(pt2[1])), (255, 0, 0), 2)
    
    def process_multiple_videos(self, video_paths, output_dir='outputs'):
        """
        Process multiple videos.
        
        Args:
            video_paths: List of paths to video files
            output_dir: Directory to save outputs
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        results = {}
        for i, video_path in enumerate(video_paths, 1):
            print(f"\n{'='*60}")
            print(f"Processing video {i}/{len(video_paths)}: {video_path}")
            print(f"{'='*60}\n")
            
            video_path = Path(video_path)
            output_video = output_dir / f"{video_path.stem}_processed.mp4"
            
            try:
                result = self.process_video(
                    video_path,
                    output_path=output_video,
                    visualize=True,
                    save_annotations=True,
                    detect_poses=True
                )
                results[str(video_path)] = result
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                results[str(video_path)] = {'error': str(e)}
        
        # Save summary
        summary_path = output_dir / 'processing_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"All videos processed! Summary saved to: {summary_path}")
        print(f"{'='*60}\n")
        
        return results


# Example usage
if __name__ == "__main__":
    # Initialize the system
    system = PlayerDetectionSystem()
    
    # Process a single video
    # result = system.process_video(
    #     'path/to/your/video.mp4',
    #     output_path='output_video.mp4',
    #     visualize=True,
    #     detect_poses=True
    # )
    
    # Process multiple videos
    video_paths = [
        'video1.mp4',
        'video2.mp4',
        'video3.mp4',
        'video4.mp4',
        'video5.mp4',
        'video6.mp4',
        'video7.mp4',
        'video8.mp4'
    ]
    
    results = system.process_multiple_videos(video_paths, output_dir='outputs')
    
    print("\nProcessing Summary:")
    for video, result in results.items():
        if 'error' not in result:
            frames = result['video_info']['frames']
            print(f"{video}: {frames} frames processed")
        else:
            print(f"{video}: ERROR - {result['error']}")