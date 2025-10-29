#!/usr/bin/env python3
"""
ULTRA Diverse Question Selector - Fixes Type Question Spam
This version properly limits type questions to max 2-3 per game
"""

import math
import random
from typing import Dict, List, Tuple, Optional, Set

class UltraDiverseQuestionSelector:
    """
    Ultra-diverse question selector that HEAVILY penalizes repetitive question types
    """
    
    def __init__(self, matcher, trait_questions):
        self.matcher = matcher
        self.trait_questions = trait_questions
        self.asked_traits = set()
        self.entropy_threshold = 0.01
        
        # Track specific question types to limit spam
        self.type_questions_asked = 0
        self.habitat_questions_asked = 0
        self.color_questions_asked = 0
        self.stat_questions_asked = 0
        
        # STRICT LIMITS
        self.MAX_TYPE_QUESTIONS = 3      # Only 3 type questions per game!
        self.MAX_HABITAT_QUESTIONS = 2   # Only 2 habitat questions
        self.MAX_COLOR_QUESTIONS = 2     # Only 2 color questions
        self.MAX_STAT_QUESTIONS = 4      # Only 4 stat questions
        
        # Question priorities (higher = better)
        self.question_priorities = {
            # ESSENTIAL VARIETY (ask these first)
            'starter_pokemon': 10.0,
            'final_evolution': 9.0,
            'is_legendary': 8.0,
            'is_mythical': 7.0,
            'iconic_pokemon': 6.0,
            
            # PHYSICAL (interesting and diverse)
            'size_small': 5.0,
            'size_medium': 5.0, 
            'size_large': 5.0,
            'weight_light': 5.0,
            'weight_medium': 5.0,
            'weight_heavy': 5.0,
            
            # TYPES (important but limited)
            'type_fire': 4.0,
            'type_water': 4.0,
            'type_grass': 4.0,
            'type_electric': 4.0,
            'type_psychic': 4.0,
            'type_poison': 4.0,
            'type_flying': 4.0,
            'type_ground': 4.0,
            'type_rock': 4.0,
            'type_bug': 4.0,
            'type_ghost': 4.0,
            'type_dragon': 4.0,
            'type_ice': 4.0,
            'type_fighting': 4.0,
            'type_normal': 4.0,
            'type_steel': 4.0,
            'type_fairy': 4.0,
            
            # STATS (moderately interesting)
            'high_hp': 3.0,
            'high_attack': 3.0,
            'high_defense': 3.0,
            'high_special_attack': 3.0,
            'high_special_defense': 3.0,
            'high_speed': 3.0,
            
            # VISUAL (limited use)
            'color_red': 2.0,
            'color_blue': 2.0,
            'color_green': 2.0,
            'color_yellow': 2.0,
            'color_brown': 2.0,
            'color_purple': 2.0,
            'color_pink': 2.0,
            'color_black': 2.0,
            'color_white': 2.0,
            'color_gray': 2.0,
            
            # HABITAT (very limited)
            'habitat_grassland': 1.0,
            'habitat_forest': 1.0,
            'habitat_mountain': 1.0,
            'habitat_cave': 1.0,
            'habitat_sea': 1.0,
            'habitat_urban': 1.0,
            'habitat_rare': 1.0,
            'habitat_rough-terrain': 1.0,
            'habitat_waters-edge': 1.0,
        }
    
    def get_question_type(self, trait: str) -> str:
        """Categorize question type for spam prevention"""
        if trait.startswith('type_'):
            return 'type'
        elif trait.startswith('habitat_'):
            return 'habitat'
        elif trait.startswith('color_'):
            return 'color'
        elif trait.startswith('high_'):
            return 'stat'
        elif trait.startswith('size_') or trait.startswith('weight_'):
            return 'physical'
        else:
            return 'other'
    
    def is_question_type_maxed_out(self, trait: str) -> bool:
        """Check if we've asked too many questions of this type"""
        question_type = self.get_question_type(trait)
        
        if question_type == 'type' and self.type_questions_asked >= self.MAX_TYPE_QUESTIONS:
            return True
        elif question_type == 'habitat' and self.habitat_questions_asked >= self.MAX_HABITAT_QUESTIONS:
            return True
        elif question_type == 'color' and self.color_questions_asked >= self.MAX_COLOR_QUESTIONS:
            return True
        elif question_type == 'stat' and self.stat_questions_asked >= self.MAX_STAT_QUESTIONS:
            return True
        
        return False
    
    def calculate_expected_entropy_reduction(self, trait: str, candidates: List) -> float:
        """Calculate information gain (simplified version)"""
        if not candidates:
            return 0.0
            
        # Current entropy
        current_entropy = 0.0
        total_confidence = sum(conf for _, conf in candidates)
        
        if total_confidence > 0:
            for _, confidence in candidates:
                p = confidence / total_confidence
                if p > 0:
                    current_entropy -= p * math.log2(p)
        
        # Estimate entropy reduction based on trait diversity
        trait_diversity = len(set(
            self.matcher.trait_matrix.get(pokemon, {}).get(trait, 0)
            for pokemon, _ in candidates
        ))
        
        estimated_reduction = current_entropy * (trait_diversity / len(candidates))
        return min(estimated_reduction, current_entropy)
    
    def calculate_final_score(self, trait: str, candidates: List) -> float:
        """Calculate final score with ULTRA diversity enforcement"""
        
        # HARD BLOCK: If question type is maxed out, return 0
        if self.is_question_type_maxed_out(trait):
            return 0.0
        
        # Base information gain
        info_gain = self.calculate_expected_entropy_reduction(trait, candidates)
        
        # Priority bonus
        priority = self.question_priorities.get(trait, 1.0)
        
        # Question type penalty (diminishing returns)
        question_type = self.get_question_type(trait)
        type_penalty = 1.0
        
        if question_type == 'type':
            type_penalty = max(0.1, 1.0 - (self.type_questions_asked * 0.3))
        elif question_type == 'habitat':
            type_penalty = max(0.1, 1.0 - (self.habitat_questions_asked * 0.4))
        elif question_type == 'color':
            type_penalty = max(0.1, 1.0 - (self.color_questions_asked * 0.4))
        elif question_type == 'stat':
            type_penalty = max(0.2, 1.0 - (self.stat_questions_asked * 0.2))
        
        final_score = info_gain * priority * type_penalty
        return final_score
    
    def select_next_question(self) -> Optional[Tuple[str, str]]:
        """Select next question with ULTRA diversity"""
        candidates = self.matcher.get_ranked_candidates()
        
        if len(candidates) <= 1:
            return None
        
        available_traits = [
            trait for trait in self.matcher.trait_names 
            if trait not in self.asked_traits and trait in self.trait_questions
        ]
        
        if not available_traits:
            return None
        
        # Score all available traits
        trait_scores = []
        for trait in available_traits:
            score = self.calculate_final_score(trait, candidates)
            trait_scores.append((trait, score))
        
        # Sort by score
        trait_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select best non-zero scoring trait
        for trait, score in trait_scores:
            if score > 0:
                question_text = self.trait_questions[trait]
                self.asked_traits.add(trait)
                
                # Update counters
                question_type = self.get_question_type(trait)
                if question_type == 'type':
                    self.type_questions_asked += 1
                elif question_type == 'habitat':
                    self.habitat_questions_asked += 1
                elif question_type == 'color':
                    self.color_questions_asked += 1
                elif question_type == 'stat':
                    self.stat_questions_asked += 1
                
                # Debug output
                print(f"🎯 Selected: {trait} (type: {question_type}, score: {score:.3f})")
                print(f"   Type counts: T:{self.type_questions_asked}/{self.MAX_TYPE_QUESTIONS} H:{self.habitat_questions_asked}/{self.MAX_HABITAT_QUESTIONS} C:{self.color_questions_asked}/{self.MAX_COLOR_QUESTIONS}")
                
                return trait, question_text
        
        return None
    
    def reset(self):
        """Reset selector state"""
        self.asked_traits.clear()
        self.type_questions_asked = 0
        self.habitat_questions_asked = 0
        self.color_questions_asked = 0
        self.stat_questions_asked = 0

# Monkey patch function
def patch_engine_for_ultra_diversity():
    """Replace question selector with ultra-diverse version"""
    import engine
    
    original_create = engine.create_optimized_identification_engine
    
    def create_ultra_diverse_engine():
        engine = original_create()
        if engine:
            ultra_selector = UltraDiverseQuestionSelector(
                engine.matcher, 
                engine.question_selector.trait_questions
            )
            engine.question_selector = ultra_selector
            print("🚀 Patched engine with ULTRA diverse question selector!")
            print("📊 Limits: 3 types, 2 habitats, 2 colors, 4 stats per game")
        return engine
    
    engine.create_optimized_identification_engine = create_ultra_diverse_engine
    return create_ultra_diverse_engine

if __name__ == "__main__":
    print("🚀 Ultra Diverse Question Selector")
    print("Fixes type question spam with hard limits")
