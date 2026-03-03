"""
篮球解说文案生成模块
根据检测到的动作生成自然语言解说
支持模板生成和LLM生成两种模式
"""

import random
from typing import List, Dict, Optional
from dataclasses import dataclass

from src.action_recognizer import ActionType, ActionResult


@dataclass
class CommentaryResult:
    """解说结果"""
    text: str              # 解说文案
    action: ActionType     # 对应动作
    style: str             # 解说风格
    language: str          # 语言


class TemplateCommentaryGenerator:
    """
    模板解说生成器
    使用预定义模板，根据动作生成多样化解说
    """
    
    # 中文解说模板
    TEMPLATES_CN = {
        ActionType.SHOOTING: [
            "精彩！{player}举起双手，瞄准篮筐，出手投篮！球在空中划出一道美丽的弧线！",
            "{player}在{position}位置拿球，调整姿势后果断出手，这个投篮动作非常标准！",
            "漂亮！{player}接球后立刻拔起跳投，这个出手速度快如闪电！",
            "{player}三分线外持球，一个干拔跳投！让我们看看这球能不能进！",
            "注意看{player}的投篮姿势，手肘对准篮筐，手腕柔和下压，教科书般的出手！",
            "{player}接到队友传球，转身跳投，这个中距离投篮非常有把握！",
        ],
        ActionType.DRIBBLING: [
            "{player}带球推进，运球节奏非常好，左右手交替运球寻找突破机会！",
            "看{player}这个运球！低重心、快速变向，防守球员完全跟不上节奏！",
            "{player}在外线持球组织进攻，运球过半场，观察队友的跑位情况。",
            "精彩的运球过人！{player}一个crossover变向，轻松过掉了防守球员！",
            "{player}控球推进，节奏掌控得非常好，这就是一个优秀控卫的素养！",
            "{player}持球突破，运球速度非常快，直插篮下！",
        ],
        ActionType.PASSING: [
            "{player}一个漂亮的传球！找到了空位的队友，这个视野太出色了！",
            "妙传！{player}胸前传球，球精准地送到了队友手中！",
            "{player}快速出球，击地传球送到内线，这个传球时机恰到好处！",
            "看这个传球！{player}背后传球骗过了防守，队友获得了绝佳的进攻机会！",
            "{player}组织进攻，一个精妙的传球穿过了防守，助攻到位！",
            "{player}长传找到了快下的队友，这个传球视野令人赞叹！",
        ],
        ActionType.DUNKING: [
            "天哪！{player}起飞了！一个势大力沉的扣篮！全场沸腾了！",
            "暴扣！{player}飞身扣篮！这个力量和弹跳简直不可思议！",
            "{player}接到空中接力，单手劈扣！这个扣篮太震撼了！",
            "看{player}的弹跳！高高跃起，双手暴扣！篮筐都在颤抖！",
            "势不可挡！{player}突破到篮下，一个战斧式劈扣！防守球员只能目送！",
        ],
        ActionType.BLOCKING: [
            "{player}防守到位！张开双臂封堵传球路线，这个防守意识非常好！",
            "大帽！{player}起跳封盖！把对手的投篮扇飞了！",
            "{player}卡住防守位置，展开双臂施压，给进攻球员制造了很大的困难！",
            "出色的防守！{player}脚步移动迅速，始终保持在防守位置上！",
            "{player}张开双臂，积极干扰对手投篮，这个防守强度很高！",
        ],
        ActionType.REBOUNDING: [
            "{player}高高跃起，双手摘下篮板球！这个篮板意识太强了！",
            "抢到了！{player}在人群中拼抢到了这个关键篮板！",
            "{player}卡位抢板，稳稳地将球控制在手中，保护好了这次防守！",
            "又是{player}的篮板！在内线的统治力令人印象深刻！",
        ],
        ActionType.RUNNING: [
            "{player}快速跑动，积极地在场上寻找机会！",
            "{player}全速奔跑，进行快攻推进！速度非常快！",
            "看{player}的无球跑动，这个跑位非常聪明，创造出了空间！",
        ],
        ActionType.STANDING: [
            "{player}在场上观察局势，等待进攻机会。",
            "{player}站在{position}位置，准备接应队友的传球。",
            "{player}在罚球线附近准备，随时准备参与进攻。",
        ],
        ActionType.UNKNOWN: [
            "{player}正在场上积极移动，让我们继续关注比赛的发展！",
            "镜头捕捉到{player}的动作，比赛正在激烈进行中！",
        ],
    }
    
    # 英文解说模板
    TEMPLATES_EN = {
        ActionType.SHOOTING: [
            "What a shot! {player} pulls up and fires away! The ball arcs beautifully towards the basket!",
            "{player} catches the ball at {position}, sets up and takes the shot! Textbook shooting form!",
            "{player} with a quick release jumper! That shot came out lightning fast!",
        ],
        ActionType.DRIBBLING: [
            "{player} brings the ball up the court, handles looking smooth and controlled!",
            "Watch {player}'s handles! Low dribble, quick crossover — the defender can't keep up!",
            "{player} pushing the pace, dribbling through traffic!",
        ],
        ActionType.PASSING: [
            "Beautiful pass by {player}! Found the open teammate with perfect vision!",
            "{player} with the chest pass — right on the money to the teammate!",
            "What a dime from {player}! Threading the needle through the defense!",
        ],
        ActionType.DUNKING: [
            "OH MY! {player} takes flight and THROWS IT DOWN! The crowd goes wild!",
            "SLAM DUNK by {player}! What incredible power and athleticism!",
            "{player} soars through the air for a monster dunk! Unbelievable!",
        ],
        ActionType.BLOCKING: [
            "{player} with the block! Swatted that shot right out of the air!",
            "Great defense by {player}! Arms spread wide, cutting off all passing lanes!",
            "GET THAT OUT OF HERE! {player} rejects the shot attempt!",
        ],
        ActionType.REBOUNDING: [
            "{player} leaps high and grabs the rebound! Great board work!",
            "Rebound {player}! Snatched it right out of the air in traffic!",
        ],
        ActionType.RUNNING: [
            "{player} sprinting up the court on the fast break!",
            "{player} making a smart cut, creating space for the offense!",
        ],
        ActionType.STANDING: [
            "{player} setting up at {position}, reading the defense.",
            "{player} waiting at the perimeter, ready for the catch and shoot.",
        ],
        ActionType.UNKNOWN: [
            "{player} is active on the court, let's see what develops!",
        ],
    }
    
    POSITIONS = [
        "三分线外", "罚球线", "底角", "侧翼", "弧顶", 
        "低位", "高位", "肘区", "半场"
    ]
    POSITIONS_EN = [
        "beyond the arc", "the free-throw line", "the corner",
        "the wing", "the top of the key", "the low post", "the high post"
    ]
    
    def __init__(self):
        print("✅ 模板解说生成器已初始化")
    
    def generate(self, action_result: ActionResult,
                 player_name: str = "这位球员",
                 language: str = "cn",
                 style: str = "excited") -> CommentaryResult:
        """
        生成解说文案
        
        Args:
            action_result: 动作识别结果
            player_name: 球员名称
            language: 语言 ("cn" 中文 / "en" 英文)
            style: 解说风格
            
        Returns:
            解说结果
        """
        templates = self.TEMPLATES_CN if language == "cn" else self.TEMPLATES_EN
        positions = self.POSITIONS if language == "cn" else self.POSITIONS_EN
        
        action_templates = templates.get(action_result.action, templates[ActionType.UNKNOWN])
        template = random.choice(action_templates)
        
        text = template.format(
            player=player_name,
            position=random.choice(positions)
        )
        
        return CommentaryResult(
            text=text,
            action=action_result.action,
            style=style,
            language=language
        )
    
    def generate_multi(self, action_results: List[ActionResult],
                       language: str = "cn") -> str:
        """
        为多个球员生成综合解说
        
        Args:
            action_results: 多个动作识别结果
            language: 语言
            
        Returns:
            综合解说文案
        """
        commentaries = []
        for i, result in enumerate(action_results):
            player_name = f"{'球员' if language == 'cn' else 'Player'} #{i+1}"
            commentary = self.generate(result, player_name, language)
            commentaries.append(commentary.text)
        
        return "\n".join(commentaries)


class LLMCommentaryGenerator:
    """
    LLM解说生成器
    使用大语言模型生成更自然、多样的解说文案
    支持OpenAI API或本地模型
    """
    
    def __init__(self, api_key: Optional[str] = None, 
                 model: str = "gpt-3.5-turbo",
                 use_local: bool = False):
        """
        初始化LLM生成器
        
        Args:
            api_key: OpenAI API密钥
            model: 模型名称
            use_local: 是否使用本地模型
        """
        self.use_local = use_local
        self.model = model
        
        if use_local:
            self._init_local_model()
        else:
            self._init_openai(api_key)
    
    def _init_openai(self, api_key: Optional[str]):
        """初始化OpenAI客户端"""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            print(f"✅ OpenAI LLM已初始化: {self.model}")
        except Exception as e:
            print(f"⚠️ OpenAI初始化失败: {e}")
            print("  将回退到模板生成模式")
            self.client = None
    
    def _init_local_model(self):
        """初始化本地模型 (Transformers)"""
        try:
            from transformers import pipeline
            self.pipe = pipeline("text-generation", 
                               model="Qwen/Qwen2-1.5B-Instruct",
                               max_new_tokens=200)
            print("✅ 本地LLM已初始化")
        except Exception as e:
            print(f"⚠️ 本地模型初始化失败: {e}")
            self.pipe = None
    
    def generate(self, action_result: ActionResult,
                 player_name: str = "这位球员",
                 language: str = "cn") -> CommentaryResult:
        """使用LLM生成解说"""
        
        action_cn = action_result.action_cn
        confidence = action_result.confidence
        
        prompt = f"""你是一位专业的篮球比赛解说员，请根据以下信息生成一段生动、激情的解说词：

球员：{player_name}
检测到的动作：{action_cn}
置信度：{confidence:.1%}

要求：
1. 解说词要生动、有激情，像真正的篮球解说员一样
2. 语言要{'中文' if language == 'cn' else '英文'}
3. 一到两句话即可
4. 加入适当的感叹和语气词

请直接输出解说词，不要加额外说明："""

        text = self._call_llm(prompt)
        
        if not text:
            # 回退到模板
            fallback = TemplateCommentaryGenerator()
            return fallback.generate(action_result, player_name, language)
        
        return CommentaryResult(
            text=text.strip(),
            action=action_result.action,
            style="llm",
            language=language
        )
    
    def _call_llm(self, prompt: str) -> Optional[str]:
        """调用LLM"""
        if self.use_local and self.pipe:
            try:
                result = self.pipe(prompt)
                return result[0]["generated_text"].replace(prompt, "").strip()
            except Exception as e:
                print(f"本地模型调用失败: {e}")
                return None
        
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一位专业的篮球比赛解说员。"},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200,
                    temperature=0.8
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"OpenAI调用失败: {e}")
                return None
        
        return None
