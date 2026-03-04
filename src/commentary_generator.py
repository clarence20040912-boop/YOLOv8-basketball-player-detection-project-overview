"""
Basketball Commentary Generation Module
Generates natural language commentary from detected actions.
Supports template-based and LLM-based generation.
"""

import random
from typing import List, Dict, Optional
from dataclasses import dataclass

from src.action_recognizer import ActionType, ActionResult


@dataclass
class CommentaryResult:
    """Commentary result"""
    text: str              # commentary text
    action: ActionType     # corresponding action
    style: str             # commentary style
    language: str          # language


class TemplateCommentaryGenerator:
    """
    Template-based commentary generator.
    Uses predefined templates to produce varied commentary from detected actions.
    """
    
    # English commentary templates
    TEMPLATES_EN = {
        ActionType.SHOOTING: [
            "BANG! {player} rises up from {position} and lets it fly! That ball is pure silk through the net!",
            "Splash! {player} drains a deep three from {position} — absolutely AUTOMATIC!",
            "{player} with the pull-up jumper from {position}! Textbook form — elbow in, wrist snap — BEAUTIFUL!",
            "Fadeaway from {player} at {position}! Leaning back, no way to contest that — IT'S GOOD!",
            "Catch and shoot! {player} gets the rock at {position} and FIRES in one motion — MONEY!",
            "{player} beats the shot clock! Stepback from {position} — off the glass and IN! The crowd ERUPTS!",
            "Oh, the arc on that one! {player} launches from {position} — it hangs up there forever and DROPS THROUGH!",
            "{player} draws the defense, steps to {position} and releases the smooth mid-range jumper — swish, nothing but net!",
        ],
        ActionType.DRIBBLING: [
            "{player} with the silky crossover at {position}! Left, right, GONE — the defender is absolutely flat-footed!",
            "Behind the back! {player} dribbles through traffic near {position} like it's nothing — pure floor general!",
            "Watch {player} work at {position} — hesitation dribble, one step, another — the defense has NO idea what's coming!",
            "{player} with the between-the-legs handles near {position}! Low dribble, explosive first step — ANKLE BREAKER!",
            "Speed dribble! {player} pushes the pace up the court from {position} — nobody is catching this player tonight!",
            "{player} is the maestro at {position}, controlling the tempo with effortless ball-handling, reading the defense every step of the way!",
            "Shimmy! {player} rattles the defense with a quick stutter-step at {position}, then accelerates past like a missile!",
            "{player} going coast-to-coast on the dribble — weaving through defenders from {position} all the way to the rim!",
        ],
        ActionType.PASSING: [
            "NO-LOOK PASS by {player}! Eyes went one way, ball went the other — the defense had no chance!",
            "Alley-oop feed from {player}! Lofted perfectly from {position} to the cutter — what a vision play!",
            "{player} fires a bullet pass from {position} through THREE defenders — right on the money to the open teammate!",
            "Skip pass! {player} swings it cross-court from {position} — reversing the defense and finding the shooter wide open!",
            "Behind the back! {player} delivers a behind-the-back dime at {position} — the assist of the night, no doubt!",
            "{player} with the court vision of an eagle at {position}! Sees the cutter three passes ahead and delivers a perfect lob!",
            "Bounce pass! {player} threads the needle from {position} through two big men — a perfectly weighted delivery!",
            "{player} whips a snap pass from {position} that catches the entire defense napping — that's textbook ball movement!",
        ],
        ActionType.DUNKING: [
            "OH MY GOODNESS! {player} POSTERIZES the defender — a thunderous two-handed TOMAHAWK that shakes the building!",
            "WINDMILL DUNK! {player} soars from {position}, winds up, and SLAMS it home — the crowd is absolutely LOSING IT!",
            "ALLEY-OOP FINISH! {player} catches the lob above the rim and hammers it DOWN with authority — UNBELIEVABLE athleticism!",
            "RIM-RATTLING DUNK by {player}! The backboard is shaking, the arena is erupting — that was VICIOUS!",
            "{player} takes flight from {position} — one hand, windmill style — THROWN DOWN! You cannot teach that kind of explosion!",
            "CHASE-DOWN DUNK! {player} runs down the fast break and THROWS IT DOWN in traffic — nobody stopping that train!",
            "{player} with a ferocious ONE-HANDED JAM! Rising above everyone from {position} — a flat-out highlight reel play!",
            "The defense tried to stop it but {player} DUNKED RIGHT OVER THEM! Crowd on their feet — pure POWER and WILL!",
        ],
        ActionType.BLOCKING: [
            "REJECTED! {player} comes out of nowhere for the CHASE-DOWN BLOCK — sending it INTO THE STANDS!",
            "GET THAT OUTTA HERE! {player} rises up at {position} and SWATS the shot into another zip code!",
            "{player} with the help-side block! Sneaking in from the weak side — absolute defensive ANCHOR in the paint!",
            "Intimidation factor: {player} challenges every shot at {position}, arms outstretched — nobody wants to drive the lane tonight!",
            "MASSIVE BLOCK by {player}! Timing was PERFECT — meeting the ball right at its apex and denying the basket!",
            "{player} says NO! Seals off the lane from {position} and swats the attempt clean — that's elite rim protection!",
            "Two-handed block by {player} at {position}! The defender has planted the flag — the paint belongs to {player} tonight!",
            "{player} with the deflection at {position} — fantastic read of the play, disrupting the offense and sparking a fast break!",
        ],
        ActionType.REBOUNDING: [
            "{player} BOXES OUT perfectly and snatches the board — crashing the glass with desire and grit!",
            "OFFENSIVE REBOUND by {player}! Second-chance points are coming — {player} outworks everyone for the ball!",
            "{player} times the jump perfectly, leaps HIGH above the crowd near {position} and GRABS that board with two hands!",
            "Dominant paint presence from {player} at {position}! Outmuscling everyone for the rebound — a force of nature inside!",
            "{player} with another board — crashing the glass relentlessly! The glass-eater is feasting tonight at {position}!",
            "AND the rebound goes to {player}! Pure hustle, pure determination — nobody wanted it more in that moment!",
        ],
        ActionType.RUNNING: [
            "{player} is GONE! Sprinting coast-to-coast on the fast break — nobody in the league is catching this player!",
            "Off-ball brilliance from {player}! A perfectly-timed backdoor cut near {position} — creating a lane to the basket!",
            "{player} pushes the pace in transition — full speed from {position}, setting the tempo for the entire offense!",
            "Watch {player} work off the ball! Smart diagonal cut near {position}, reading the play and creating space for teammates!",
            "{player} flies down the wing in transition — blazing speed, attacking the rim before the defense can get set!",
            "Hustle play by {player}! Sprinting from {position} to fill the lane — the effort is RELENTLESS!",
        ],
        ActionType.STANDING: [
            "{player} surveys the court at {position} — reading the defense like a chess grandmaster, calling plays and directing traffic!",
            "Patience from {player} at {position}! Letting the play develop, waiting for the defense to commit before making the move!",
            "{player} sets up at {position}, hands ready — poised for the catch-and-shoot or the drive, keeping the defense guessing!",
            "{player} commanding the offense from {position} — pointing, signaling, orchestrating the attack with calm authority!",
            "{player} holds the ball at {position}, pulling the defense in and creating timing for teammates to cut and relocate!",
        ],
        ActionType.UNKNOWN: [
            "{player} is making things happen at {position} — the intensity out there is ELECTRIC tonight!",
            "The action is FAST AND FURIOUS around {player} near {position} — this is championship-level basketball!",
            "{player} fully locked in and engaged — every second on the court matters in a game like this!",
            "Keep your eyes on {player} near {position} — something special could happen at any moment!",
        ],
    }

    POSITIONS_EN = [
        "beyond the arc", "the free-throw line", "the corner",
        "the wing", "the top of the key", "the low post", "the high post",
        "the elbow", "the mid-range", "halfcourt"
    ]
    
    def __init__(self):
        print("✅ Template commentary generator initialized")
    
    def generate(self, action_result: ActionResult,
                 player_name: str = "This player",
                 language: str = "en",
                 style: str = "excited") -> CommentaryResult:
        """
        Generate commentary text.

        Args:
            action_result: action recognition result
            player_name: player name
            language: language ("en" English)
            style: commentary style

        Returns:
            commentary result
        """
        templates = self.TEMPLATES_EN
        positions = self.POSITIONS_EN
        
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
                       language: str = "en") -> str:
        """
        Generate combined commentary for multiple players.

        Args:
            action_results: list of action recognition results
            language: language

        Returns:
            combined commentary text
        """
        commentaries = []
        for i, result in enumerate(action_results):
            player_name = f"Player #{i+1}"
            commentary = self.generate(result, player_name, language)
            commentaries.append(commentary.text)
        
        return "\n".join(commentaries)


class LLMCommentaryGenerator:
    """
    LLM-based commentary generator.
    Uses a large language model to produce more natural, varied commentary.
    Supports OpenAI API or a local model.
    """
    
    def __init__(self, api_key: Optional[str] = None, 
                 model: str = "gpt-3.5-turbo",
                 use_local: bool = False):
        """
        Initialize the LLM generator.

        Args:
            api_key: OpenAI API key
            model: model name
            use_local: whether to use a local model
        """
        self.use_local = use_local
        self.model = model
        
        if use_local:
            self._init_local_model()
        else:
            self._init_openai(api_key)
    
    def _init_openai(self, api_key: Optional[str]):
        """Initialize the OpenAI client"""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            print(f"✅ OpenAI LLM initialized: {self.model}")
        except Exception as e:
            print(f"⚠️ OpenAI initialization failed: {e}")
            print("  Falling back to template generation mode")
            self.client = None
    
    def _init_local_model(self):
        """Initialize a local model (Transformers)"""
        try:
            from transformers import pipeline
            self.pipe = pipeline("text-generation", 
                               model="Qwen/Qwen2-1.5B-Instruct",
                               max_new_tokens=200)
            print("✅ Local LLM initialized")
        except Exception as e:
            print(f"⚠️ Local model initialization failed: {e}")
            self.pipe = None
    
    def generate(self, action_result: ActionResult,
                 player_name: str = "This player",
                 language: str = "en") -> CommentaryResult:
        """Generate commentary using the LLM"""
        
        action_en = action_result.action_en
        confidence = action_result.confidence
        
        prompt = f"""You are a professional NBA-style basketball commentator. Generate one or two vivid, exciting commentary sentences based on the following information:

Player: {player_name}
Detected action: {action_en}
Confidence: {confidence:.1%}

Requirements:
1. Sound like a real, energetic professional basketball broadcaster
2. Use English
3. One to two sentences only
4. Include appropriate exclamations and energy

Output the commentary directly with no additional explanation:"""

        text = self._call_llm(prompt)
        
        if not text:
            # fall back to templates
            fallback = TemplateCommentaryGenerator()
            return fallback.generate(action_result, player_name, language)
        
        return CommentaryResult(
            text=text.strip(),
            action=action_result.action,
            style="llm",
            language=language
        )
    
    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call the LLM"""
        if self.use_local and self.pipe:
            try:
                result = self.pipe(prompt)
                return result[0]["generated_text"].replace(prompt, "").strip()
            except Exception as e:
                print(f"Local model call failed: {e}")
                return None
        
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a professional NBA-style basketball commentator."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200,
                    temperature=0.8
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"OpenAI call failed: {e}")
                return None
        
        return None
