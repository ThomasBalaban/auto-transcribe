"""
Gaming Optimization Module for Onomatopoeia Detection.
Handles gaming-specific optimizations and effect filtering.
"""

from typing import List, Dict


class GamingOptimizer:
    """
    Handles gaming-specific optimizations for onomatopoeia effects.
    """
    
    def __init__(self, max_effects_per_minute: int = 12, min_effect_spacing: float = 0.8, log_func=None):
        self.max_effects_per_minute = max_effects_per_minute
        self.min_effect_spacing = min_effect_spacing
        self.log_func = log_func or print
    
    def apply_gaming_optimizations(self, effects: List[Dict]) -> List[Dict]:
        """Apply gaming-specific optimizations to effect list"""
        if not effects:
            return effects
        
        self.log_func(f"🎮 Applying gaming optimizations to {len(effects)} effects...")
        
        # Sort by time
        effects.sort(key=lambda x: x['start_time'])
        
        # Apply density management
        optimized_effects = self._manage_effect_density(effects)
        
        # Apply minimum spacing
        spaced_effects = self._apply_minimum_spacing(optimized_effects)
        
        # Final priority filtering
        final_effects = self._priority_filtering(spaced_effects)
        
        self.log_func(f"✅ Gaming optimization complete: {len(final_effects)} final effects")
        return final_effects
    
    def _manage_effect_density(self, effects: List[Dict]) -> List[Dict]:
        """Manage effect density - max effects per time window"""
        if not effects:
            return effects
        
        # Group effects into 1-minute buckets
        buckets = {}
        for effect in effects:
            bucket = int(effect['start_time'] // 60)
            if bucket not in buckets:
                buckets[bucket] = []
            buckets[bucket].append(effect)
        
        # Keep top N effects per bucket based on confidence
        filtered_effects = []
        for bucket_effects in buckets.values():
            # Sort by confidence
            bucket_effects.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            # Keep top N
            filtered_effects.extend(bucket_effects[:self.max_effects_per_minute])
        
        return sorted(filtered_effects, key=lambda x: x['start_time'])
    
    def _apply_minimum_spacing(self, effects: List[Dict]) -> List[Dict]:
        """Ensure minimum spacing between effects"""
        if not effects:
            return effects
        
        spaced_effects = [effects[0]]  # Keep first effect
        
        for effect in effects[1:]:
            last_effect = spaced_effects[-1]
            time_diff = effect['start_time'] - last_effect['start_time']
            
            if time_diff >= self.min_effect_spacing:
                spaced_effects.append(effect)
            else:
                # Too close - keep higher confidence effect
                if effect.get('confidence', 0) > last_effect.get('confidence', 0):
                    spaced_effects[-1] = effect
        
        return spaced_effects
    
    def _priority_filtering(self, effects: List[Dict]) -> List[Dict]:
        """Final priority filtering based on gaming content importance"""
        priority_effects = []
        
        for effect in effects:
            context = effect.get('context', '')
            confidence = effect.get('confidence', 0)
            
            # Boost certain contexts
            if any(key in context.lower() for key in ['explosion', 'attack', 'damage', 'monster']):
                confidence *= 1.2
            elif 'quiet' in context.lower():
                confidence *= 0.5
            
            # Apply boosted confidence
            effect['final_confidence'] = min(confidence, 1.0)
            
            # Keep effects above threshold
            if effect['final_confidence'] > 0.4:
                priority_effects.append(effect)
        
        return priority_effects