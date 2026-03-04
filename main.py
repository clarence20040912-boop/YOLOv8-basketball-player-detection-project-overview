"""
Basketball Commentary System - Command-line entry point
Supports image and video analysis
"""

import argparse
import cv2
from src.pipeline import BasketballCommentaryPipeline


def main():
    parser = argparse.ArgumentParser(
        description="🏀 YOLOv8 Basketball Player Action Detection and Commentary Generation System"
    )
    parser.add_argument("input", type=str, 
                       help="Input file path (image or video)")
    parser.add_argument("--detector", type=str, default="yolov8n.pt",
                       help="Detection model path")
    parser.add_argument("--pose", type=str, default="yolov8n-pose.pt",
                       help="Pose model path")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file path")
    parser.add_argument("--language", type=str, default="en",
                       choices=["en"], help="Commentary language")
    parser.add_argument("--use-llm", action="store_true",
                       help="Use LLM for commentary generation")
    parser.add_argument("--api-key", type=str, default=None,
                       help="OpenAI API Key")
    parser.add_argument("--interval", type=float, default=1.0,
                       help="Video keyframe extraction interval (seconds)")
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="Detection confidence threshold")
    parser.add_argument("--show", action="store_true",
                       help="Show result window")
    
    args = parser.parse_args()
    
    # initialize pipeline
    pipeline = BasketballCommentaryPipeline(
        detector_model=args.detector,
        pose_model=args.pose,
        use_llm=args.use_llm,
        llm_api_key=args.api_key,
        confidence_threshold=args.confidence,
        language=args.language
    )
    
    # determine input type
    input_path = args.input
    is_video = input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv'))
    
    if is_video:
        # video analysis
        output_path = args.output or input_path.replace('.', '_output.')
        results = pipeline.analyze_video(
            video_path=input_path,
            output_path=output_path,
            keyframe_interval=args.interval
        )
        
        # print all commentary
        print("\n" + "=" * 60)
        print("📝 Full Commentary:")
        print("=" * 60)
        for result in results:
            for comm in result.commentaries:
                print(f"  [{result.timestamp:.1f}s] {comm.text}")
    
    else:
        # image analysis
        image = cv2.imread(input_path)
        if image is None:
            print(f"❌ Cannot read image: {input_path}")
            return
        
        result = pipeline.analyze_image(image)
        
        # print results
        print("\n" + "=" * 60)
        print("📝 Commentary:")
        print("=" * 60)
        for comm in result.commentaries:
            print(f"  🎙️ {comm.text}")
        
        print(f"\n📊 Detected {len(result.detections)} objects, "
              f"{len(result.actions)} actions")
        
        # save/show results
        if args.output and result.annotated_frame is not None:
            cv2.imwrite(args.output, result.annotated_frame)
            print(f"✅ Result saved: {args.output}")
        
        if args.show and result.annotated_frame is not None:
            cv2.imshow("Basketball Commentary System", result.annotated_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
