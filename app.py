"""
Basketball Commentary System - Gradio Web Interface
Provides image/video upload and real-time commentary generation
"""

import cv2
import numpy as np
import gradio as gr
from PIL import Image
from src.pipeline import BasketballCommentaryPipeline


# Global pipeline instance
pipeline = None


def init_pipeline(detector_model="yolov8n.pt", pose_model="yolov8n-pose.pt"):
    """Initialize the pipeline"""
    global pipeline
    pipeline = BasketballCommentaryPipeline(
        detector_model=detector_model,
        pose_model=pose_model,
        use_llm=False,
        language="en"
    )


def analyze_image(input_image, confidence=0.5):
    """Analyze an uploaded image"""
    global pipeline
    
    if pipeline is None:
        init_pipeline()
    
    pipeline.language = "en"
    pipeline.detector.confidence_threshold = confidence
    
    # PIL Image → numpy
    if isinstance(input_image, Image.Image):
        image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    else:
        image = input_image
    
    # analyze
    result = pipeline.analyze_image(image, draw=True)
    
    # collect commentary text
    commentary_text = ""
    for i, comm in enumerate(result.commentaries):
        commentary_text += f"🎙️ {comm.text}\n\n"
    
    if not commentary_text:
        commentary_text = "No player actions detected. Try uploading an image that contains basketball players."
    
    # statistics
    stats = f"""📊 **Detection Stats:**
- Objects detected: {len(result.detections)}
- Players recognized: {len([d for d in result.detections if d.class_name == 'player'])}
- Basketball detected: {'✅' if any(d.class_name == 'basketball' for d in result.detections) else '❌'}
- Actions recognized: {len(result.actions)}
"""
    
    # action details
    for i, action in enumerate(result.actions):
        stats += f"\n**Player #{i+1}:** {action.action_en} (confidence: {action.confidence:.0%})"
    
    # numpy → PIL
    if result.annotated_frame is not None:
        output_image = Image.fromarray(
            cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB)
        )
    else:
        output_image = input_image
    
    return output_image, commentary_text, stats


def analyze_video(input_video, confidence=0.5, interval=1.0):
    """Analyze an uploaded video"""
    global pipeline
    
    if pipeline is None:
        init_pipeline()
    
    pipeline.language = "en"
    pipeline.detector.confidence_threshold = confidence
    
    if input_video is None:
        return None, "Please upload a video file.", ""
    
    output_path = "output_video.mp4"
    results = pipeline.analyze_video(
        video_path=input_video,
        output_path=output_path,
        keyframe_interval=interval,
        show_progress=True
    )
    
    # collect all commentary
    all_commentary = "📝 **Full Game Commentary:**\n\n"
    for result in results:
        for comm in result.commentaries:
            time_str = f"{result.timestamp:.1f}s"
            all_commentary += f"**[{time_str}]** {comm.text}\n\n"
    
    stats = f"📊 Analyzed {len(results)} keyframes, detected {sum(len(r.actions) for r in results)} actions"
    
    return output_path, all_commentary, stats


# Create Gradio interface
def create_app():
    """Create the Gradio app"""
    
    with gr.Blocks(
        title="🏀 Basketball Commentary AI System",
        theme=gr.themes.Soft()
    ) as app:
        
        gr.Markdown("""
        # 🏀 YOLOv8 Basketball Player Action Detection & Commentary Generation System
        
        Upload a basketball game **image** or **video** and the AI will automatically detect player actions and generate professional commentary!
        
        **Supported actions:** Shooting | Dribbling | Passing | Dunking | Blocking | Rebounding | Running | Standing
        """)
        
        with gr.Tabs():
            # Image analysis tab
            with gr.Tab("📷 Image Analysis"):
                with gr.Row():
                    with gr.Column():
                        img_input = gr.Image(type="pil", label="Upload Basketball Image")
                        with gr.Row():
                            img_conf = gr.Slider(
                                0.1, 0.9, value=0.5, step=0.05,
                                label="Detection Confidence"
                            )
                        img_btn = gr.Button("🔍 Start Analysis", variant="primary")
                    
                    with gr.Column():
                        img_output = gr.Image(type="pil", label="Detection Result")
                        img_commentary = gr.Textbox(
                            label="🎙️ Commentary", lines=5
                        )
                        img_stats = gr.Markdown(label="📊 Statistics")
                
                img_btn.click(
                    analyze_image,
                    inputs=[img_input, img_conf],
                    outputs=[img_output, img_commentary, img_stats]
                )
            
            # Video analysis tab
            with gr.Tab("🎬 Video Analysis"):
                with gr.Row():
                    with gr.Column():
                        vid_input = gr.Video(label="Upload Basketball Video")
                        with gr.Row():
                            vid_conf = gr.Slider(
                                0.1, 0.9, value=0.5, step=0.05,
                                label="Detection Confidence"
                            )
                            vid_interval = gr.Slider(
                                0.5, 5.0, value=1.0, step=0.5,
                                label="Keyframe Interval (seconds)"
                            )
                        vid_btn = gr.Button("🔍 Start Analysis", variant="primary")
                    
                    with gr.Column():
                        vid_output = gr.Video(label="Analysis Result")
                        vid_commentary = gr.Markdown(label="🎙️ Commentary")
                        vid_stats = gr.Markdown(label="📊 Statistics")
                
                vid_btn.click(
                    analyze_video,
                    inputs=[vid_input, vid_conf, vid_interval],
                    outputs=[vid_output, vid_commentary, vid_stats]
                )
        
        gr.Markdown("---")
    
    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
