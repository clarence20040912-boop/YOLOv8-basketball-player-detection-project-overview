"""
篮球解说系统 - Gradio Web界面
提供图片/视频上传和实时解说功能
"""

import cv2
import numpy as np
import gradio as gr
from PIL import Image
from src.pipeline import BasketballCommentaryPipeline


# 全局流水线实例
pipeline = None


def init_pipeline(detector_model="yolov8n.pt", pose_model="yolov8n-pose.pt"):
    """初始化流水线"""
    global pipeline
    pipeline = BasketballCommentaryPipeline(
        detector_model=detector_model,
        pose_model=pose_model,
        use_llm=False,
        language="cn"
    )


def analyze_image(input_image, language="cn", confidence=0.5):
    """分析上传的图片"""
    global pipeline
    
    if pipeline is None:
        init_pipeline()
    
    pipeline.language = language
    pipeline.detector.confidence_threshold = confidence
    
    # PIL Image → numpy
    if isinstance(input_image, Image.Image):
        image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    else:
        image = input_image
    
    # 分析
    result = pipeline.analyze_image(image, draw=True)
    
    # 收集解说文案
    commentary_text = ""
    for i, comm in enumerate(result.commentaries):
        commentary_text += f"🎙️ {comm.text}\n\n"
    
    if not commentary_text:
        commentary_text = "未检测到球员动作，请尝试上传包含篮球运动员的图片。"
    
    # 统计信息
    stats = f"""📊 **检测统计:**
- 检测到目标: {len(result.detections)} 个
- 识别到球员: {len([d for d in result.detections if d.class_name == 'player'])} 人
- 识别到篮球: {'✅' if any(d.class_name == 'basketball' for d in result.detections) else '❌'}
- 动作识别: {len(result.actions)} 个
"""
    
    # 动作详情
    for i, action in enumerate(result.actions):
        stats += f"\n**球员 #{i+1}:** {action.action_cn} (置信度: {action.confidence:.0%})"
    
    # numpy → PIL
    if result.annotated_frame is not None:
        output_image = Image.fromarray(
            cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB)
        )
    else:
        output_image = input_image
    
    return output_image, commentary_text, stats


def analyze_video(input_video, language="cn", confidence=0.5, interval=1.0):
    """分析上传的视频"""
    global pipeline
    
    if pipeline is None:
        init_pipeline()
    
    pipeline.language = language
    pipeline.detector.confidence_threshold = confidence
    
    if input_video is None:
        return None, "请上传视频文件", ""
    
    output_path = "output_video.mp4"
    results = pipeline.analyze_video(
        video_path=input_video,
        output_path=output_path,
        keyframe_interval=interval,
        show_progress=True
    )
    
    # 收集所有解说
    all_commentary = "📝 **完整比赛解说:**\n\n"
    for result in results:
        for comm in result.commentaries:
            time_str = f"{result.timestamp:.1f}s"
            all_commentary += f"**[{time_str}]** {comm.text}\n\n"
    
    stats = f"📊 分析了 {len(results)} 个关键帧，检测到 {sum(len(r.actions) for r in results)} 个动作"
    
    return output_path, all_commentary, stats


# 创建Gradio界面
def create_app():
    """创建Gradio应用"""
    
    with gr.Blocks(
        title="🏀 篮球解说AI系统",
        theme=gr.themes.Soft()
    ) as app:
        
        gr.Markdown("""
        # 🏀 YOLOv8 篮球运动员动作检测与解说生成系统
        
        上传篮球比赛的**图片**或**视频**，AI将自动检测球员动作并生成专业解说文案！
        
        **支持识别的动作:** 投篮 | 运球 | 传球 | 扣篮 | 盖帽 | 抢篮板 | 跑动 | 站立
        """)
        
        with gr.Tabs():
            # 图片分析标签页
            with gr.Tab("📷 图片分析"):
                with gr.Row():
                    with gr.Column():
                        img_input = gr.Image(type="pil", label="上传篮球图片")
                        with gr.Row():
                            img_lang = gr.Dropdown(
                                ["cn", "en"], value="cn", 
                                label="解说语言"
                            )
                            img_conf = gr.Slider(
                                0.1, 0.9, value=0.5, step=0.05,
                                label="检测置信度"
                            )
                        img_btn = gr.Button("🔍 开始分析", variant="primary")
                    
                    with gr.Column():
                        img_output = gr.Image(type="pil", label="检测结果")
                        img_commentary = gr.Textbox(
                            label="🎙️ 解说文案", lines=5
                        )
                        img_stats = gr.Markdown(label="📊 统计信息")
                
                img_btn.click(
                    analyze_image,
                    inputs=[img_input, img_lang, img_conf],
                    outputs=[img_output, img_commentary, img_stats]
                )
            
            # 视频分析标签页
            with gr.Tab("🎬 视频分析"):
                with gr.Row():
                    with gr.Column():
                        vid_input = gr.Video(label="上传篮球视频")
                        with gr.Row():
                            vid_lang = gr.Dropdown(
                                ["cn", "en"], value="cn",
                                label="解说语言"
                            )
                            vid_conf = gr.Slider(
                                0.1, 0.9, value=0.5, step=0.05,
                                label="检测置信度"
                            )
                            vid_interval = gr.Slider(
                                0.5, 5.0, value=1.0, step=0.5,
                                label="关键帧间隔(秒)"
                            )
                        vid_btn = gr.Button("🔍 开始分析", variant="primary")
                    
                    with gr.Column():
                        vid_output = gr.Video(label="分析结果")
                        vid_commentary = gr.Markdown(label="🎙️ 解说文案")
                        vid_stats = gr.Markdown(label="📊 统计信息")
                
                vid_btn.click(
                    analyze_video,
                    inputs=[vid_input, vid_lang, vid_conf, vid_interval],
                    outputs=[vid_output, vid_commentary, vid_stats]
                )
        
        gr.Markdown("""
        ---
        **技术栈:** YOLOv8 + YOLOv8-Pose + 动作分类 + 解说生成  
        **毕业设计项目** | Powered by Ultralytics & Gradio
        """)
    
    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
