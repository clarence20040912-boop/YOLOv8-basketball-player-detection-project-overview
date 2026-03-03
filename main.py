"""
篮球解说系统 - 命令行入口
支持图片和视频分析
"""

import argparse
import cv2
from src.pipeline import BasketballCommentaryPipeline


def main():
    parser = argparse.ArgumentParser(
        description="🏀 YOLOv8篮球运动员动作检测与解说生成系统"
    )
    parser.add_argument("input", type=str, 
                       help="输入文件路径 (图片或视频)")
    parser.add_argument("--detector", type=str, default="yolov8n.pt",
                       help="检测模型路径")
    parser.add_argument("--pose", type=str, default="yolov8n-pose.pt",
                       help="姿态模型路径")
    parser.add_argument("--output", type=str, default=None,
                       help="输出文件路径")
    parser.add_argument("--language", type=str, default="cn",
                       choices=["cn", "en"], help="解说语言")
    parser.add_argument("--use-llm", action="store_true",
                       help="使用LLM生成解说")
    parser.add_argument("--api-key", type=str, default=None,
                       help="OpenAI API Key")
    parser.add_argument("--interval", type=float, default=1.0,
                       help="视频关键帧提取间隔（秒）")
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="检测置信度阈值")
    parser.add_argument("--show", action="store_true",
                       help="显示结果窗口")
    
    args = parser.parse_args()
    
    # 初始化流水线
    pipeline = BasketballCommentaryPipeline(
        detector_model=args.detector,
        pose_model=args.pose,
        use_llm=args.use_llm,
        llm_api_key=args.api_key,
        confidence_threshold=args.confidence,
        language=args.language
    )
    
    # 判断输入类型
    input_path = args.input
    is_video = input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv'))
    
    if is_video:
        # 视频分析
        output_path = args.output or input_path.replace('.', '_output.')
        results = pipeline.analyze_video(
            video_path=input_path,
            output_path=output_path,
            keyframe_interval=args.interval
        )
        
        # 输出所有解说
        print("\n" + "=" * 60)
        print("📝 完整解说文案:")
        print("=" * 60)
        for result in results:
            for comm in result.commentaries:
                print(f"  [{result.timestamp:.1f}s] {comm.text}")
    
    else:
        # 图片分析
        image = cv2.imread(input_path)
        if image is None:
            print(f"❌ 无法读取图片: {input_path}")
            return
        
        result = pipeline.analyze_image(image)
        
        # 输出结果
        print("\n" + "=" * 60)
        print("📝 解说文案:")
        print("=" * 60)
        for comm in result.commentaries:
            print(f"  🎙️ {comm.text}")
        
        print(f"\n📊 检测到 {len(result.detections)} 个目标, "
              f"{len(result.actions)} 个动作")
        
        # 保存/显示结果
        if args.output and result.annotated_frame is not None:
            cv2.imwrite(args.output, result.annotated_frame)
            print(f"✅ 结果已保存: {args.output}")
        
        if args.show and result.annotated_frame is not None:
            cv2.imshow("Basketball Commentary System", result.annotated_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
