import torch
import cv2
import numpy as np
from pathlib import Path
import json
import os
import subprocess
from datetime import datetime

# Install required packages:
# pip install torch torchvision opencv-python ultralytics gitpython

from ultralytics import YOLO
import git

class PlayerDetectionSystem:
    """
    Complete system for player detection and keypoint detection in sports videos.
    Uses YOLOv8 for object detection and pose estimation.
    Fetches videos from the 'video' branch of the GitHub repository.
    """
    
    def __init__(self, detection_model='yolov8n.pt', pose_model='yolov8n-pose.pt', 
                 repo_path='.', video_branch='videos'):
        """
        Initialize the detection system.
        
        Args:
            detection_model: Path to YOLO detection model (or model name to download)
            pose_model: Path to YOLO pose estimation model (or model name to download)
            repo_path: Path to the git repository (default: current directory)
            video_branch: Name of the branch containing videos (default: 'videos')
        """
        print("Loading models...")
        self.detector = YOLO(detection_model)
        self.pose_estimator = YOLO(pose_model)
        print("Models loaded successfully!")
        
        self.repo_path = Path(repo_path)
        self.video_branch = video_branch
        self.video_dir = Path('videos_temp')
        
    def fetch_videos_from_branch(self, fallback_to_local=True, local_video_dir='videos'):
        """
        Fetch video files from the video branch without switching branches.
        Uses git checkout to extract files from specific branch.
        Falls back to local directory if branch doesn't exist.
        
        Args:
            fallback_to_local: If True, look for videos in local directory if branch fails
            local_video_dir: Local directory to check for videos if branch fetch fails
        """
        try:
            # First, try to fetch from remote
            print(f"Fetching latest from remote...")
            subprocess.run(
                ['git', 'fetch', '--all'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            # Create temporary directory for videos
            self.video_dir.mkdir(exist_ok=True)
            
            print(f"Fetching videos from '{self.video_branch}' branch...")
            
            # Check if branch exists (check both local and remote)
            branch_check = subprocess.run(
                ['git', 'rev-parse', '--verify', f'origin/{self.video_branch}'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            if branch_check.returncode != 0:
                # Try without origin prefix
                branch_check = subprocess.run(
                    ['git', 'rev-parse', '--verify', self.video_branch],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True
                )
                branch_name = self.video_branch
            else:
                branch_name = f'origin/{self.video_branch}'
            
            if branch_check.returncode != 0:
                print(f"⚠ Branch '{self.video_branch}' not found!")
                if fallback_to_local:
                    print(f"Trying to load videos from local directory: {local_video_dir}")
                    return self._load_local_videos(local_video_dir)
                return []
            
            # Get list of video files in the video branch
            result = subprocess.run(
                ['git', 'ls-tree', '-r', '--name-only', branch_name],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            all_files = result.stdout.strip().split('\n')
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
            video_files = [f for f in all_files if any(f.lower().endswith(ext) for ext in video_extensions)]
            
            print(f"Found {len(video_files)} video files in '{branch_name}' branch")
            
            if len(video_files) == 0 and fallback_to_local:
                print(f"No videos found in branch. Trying local directory: {local_video_dir}")
                return self._load_local_videos(local_video_dir)
            
            # Checkout each video file from the video branch
            downloaded_videos = []
            for video_file in video_files:
                try:
                    output_path = self.video_dir / Path(video_file).name
                    
                    # Use git show to get file content from specific branch
                    subprocess.run(
                        ['git', 'show', f'{branch_name}:{video_file}'],
                        cwd=self.repo_path,
                        stdout=open(output_path, 'wb'),
                        check=True
                    )
                    
                    downloaded_videos.append(output_path)
                    print(f"  ✓ Downloaded: {video_file}")
                    
                except subprocess.CalledProcessError as e:
                    print(f"  ✗ Failed to download {video_file}: {e}")
            
            print(f"\nSuccessfully downloaded {len(downloaded_videos)} videos to '{self.video_dir}'")
            return downloaded_videos
            
        except subprocess.CalledProcessError as e:
            print(f"Error accessing git repository: {e}")
            if fallback_to_local:
                print(f"Trying to load videos from local directory: {local_video_dir}")
                return self._load_local_videos(local_video_dir)
            return []
        except Exception as e:
            print(f"Unexpected error: {e}")
            if fallback_to_local:
                print(f"Trying to load videos from local directory: {local_video_dir}")
                return self._load_local_videos(local_video_dir)
            return []
    
    def _load_local_videos(self, video_dir):
        """
        Load videos from a local directory as fallback.
        
        Args:
            video_dir: Directory containing video files
        
        Returns:
            List of video file paths
        """
        video_dir = Path(video_dir)
        
        if not video_dir.exists():
            print(f"✗ Local directory '{video_dir}' does not exist")
            print("\nPlease either:")
            print("1. Create a 'videos' directory and place your 8 video files there, OR")
            print("2. Make sure the 'videos' branch exists in your repository")
            return []
        
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(list(video_dir.glob(f'*{ext}')))
            video_files.extend(list(video_dir.glob(f'*{ext.upper()}')))
        
        if video_files:
            print(f"✓ Found {len(video_files)} video files in '{video_dir}':")
            for vf in video_files:
                print(f"  - {vf.name}")
        else:
            print(f"✗ No video files found in '{video_dir}'")
        
        return video_files
        
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
    
    def process_all_videos_from_branch(self, output_dir='outputs'):
        """
        Fetch videos from the video branch and process all of them.
        
        Args:
            output_dir: Directory to save outputs
        
        Returns:
            Dictionary containing processing results for all videos
        """
        # Fetch videos from the video branch
        video_paths = self.fetch_videos_from_branch()
        
        if not video_paths:
            print("No videos found or downloaded from the video branch!")
            return {}
        
        # Process all downloaded videos
        return self.process_multiple_videos(video_paths, output_dir)
    
    def cleanup_temp_videos(self):
        """
        Clean up temporary video directory.
        """
        import shutil
        if self.video_dir.exists():
            shutil.rmtree(self.video_dir)
            print(f"Cleaned up temporary directory: {self.video_dir}")
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
    print("="*60)
    print("Player Detection and Keypoint Detection System")
    print("="*60 + "\n")
    
    system = PlayerDetectionSystem(
        detection_model='yolov8n.pt',
        pose_model='yolov8n-pose.pt',
        repo_path='.',  # Current directory (assumes you're in the repo root)
        video_branch='videos'  # Branch containing the videos
    )
    
    # Option 1: Automatically fetch and process all videos from the 'videos' branch
    print("\nFetching and processing videos from 'videos' branch...\n")
    results = system.process_all_videos_from_branch(output_dir='outputs')
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    for video, result in results.items():
        if 'error' not in result:
            frames = result['video_info']['frames']
            fps = result['video_info']['fps']
            resolution = result['video_info']['resolution']
            print(f"\n✓ {Path(video).name}")
            print(f"  - Frames: {frames}")
            print(f"  - FPS: {fps}")
            print(f"  - Resolution: {resolution[0]}x{resolution[1]}")
        else:
            print(f"\n✗ {Path(video).name}")
            print(f"  - ERROR: {result['error']}")
    
    print("\n" + "="*60)
    print(f"All outputs saved to: outputs/")
    print("="*60 + "\n")
    
    # Optional: Clean up temporary video files
    # Uncomment the line below if you want to delete the downloaded videos after processing
    # system.cleanup_temp_videos()
    
    # Option 2: Process specific videos manually
    # video_paths = [
    #     'path/to/video1.mp4',
    #     'path/to/video2.mp4',
    # ]
    # results = system.process_multiple_videos(video_paths, output_dir='outputs')
