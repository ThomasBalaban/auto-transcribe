# gaming_optimizer.py

from typing import List, Dict

class GamingOptimizer:
    """
    Handles gaming-specific optimizations for onomatopoeia effects with
    recalibrated, more forgiving settings.
    """

    def __init__(self, max_effects_per_minute: int = 12, min_effect_spacing: float = 1.0, log_func=None):
        # === CHANGE: Tuned settings to reduce "spam" ===
        # Reduced max effects and increased spacing to prevent overwhelming the viewer.
        self.max_effects_per_minute = max_effects_per_minute
        self.min_effect_spacing = min_effect_spacing
        self.log_func = log_func or print

    def apply_gaming_optimizations(self, effects: List[Dict]) -> List[Dict]:
        """Apply gaming-specific optimizations to the list of effects."""
        if not effects:
            return effects

        self.log_func(f"🎮 Applying gaming optimizations to {len(effects)} effects...")
        effects.sort(key=lambda x: x['start_time'])

        # Apply density and spacing filters
        optimized_effects = self._manage_effect_density(effects)
        spaced_effects = self._apply_minimum_spacing(optimized_effects)

        self.log_func(f"✅ Gaming optimization complete: {len(spaced_effects)} final effects")
        return spaced_effects

    def _manage_effect_density(self, effects: List[Dict]) -> List[Dict]:
        """Manages the maximum number of effects allowed per minute."""
        if not effects:
            return effects
        
        buckets = {}
        for effect in effects:
            bucket_key = int(effect['start_time'] // 60)
            if bucket_key not in buckets:
                buckets[bucket_key] = []
            buckets[bucket_key].append(effect)

        filtered_effects = []
        for bucket_key in sorted(buckets.keys()):
            bucket_effects = buckets[bucket_key]
            # Sort by confidence to keep the most relevant effects
            bucket_effects.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            filtered_effects.extend(bucket_effects[:self.max_effects_per_minute])

        return sorted(filtered_effects, key=lambda x: x['start_time'])

    def _apply_minimum_spacing(self, effects: List[Dict]) -> List[Dict]:
        """Ensures a minimum time gap between consecutive onomatopoeia effects."""
        if not effects:
            return effects

        final_effects = [effects[0]]
        for current_effect in effects[1:]:
            last_effect = final_effects[-1]
            if current_effect['start_time'] - last_effect['start_time'] >= self.min_effect_spacing:
                final_effects.append(current_effect)
            else:
                # If effects are too close, keep the one with higher confidence
                if current_effect.get('confidence', 0) > last_effect.get('confidence', 0):
                    final_effects[-1] = current_effect

        return final_effects