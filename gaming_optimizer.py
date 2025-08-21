# gaming_optimizer.py - ENHANCED VERSION

from typing import List, Dict
import difflib

class GamingOptimizer:
    """
    Enhanced gaming optimizer with better deduplication and spacing.
    """

    def __init__(self, max_effects_per_minute: int = 10, min_effect_spacing: float = 1.5, log_func=None):
        # More conservative settings to reduce spam
        self.max_effects_per_minute = max_effects_per_minute
        self.min_effect_spacing = min_effect_spacing
        self.log_func = log_func or print
        
        # New parameters for enhanced filtering
        self.similarity_threshold = 0.7  # For detecting similar sounds
        self.repetitive_sound_cooldown = 3.0  # Longer cooldown for repetitive sounds
        self.max_similar_in_window = 2  # Max similar sounds in a time window

    def apply_gaming_optimizations(self, effects: List[Dict]) -> List[Dict]:
        """Apply enhanced gaming-specific optimizations."""
        if not effects:
            return effects

        self.log_func(f"🎮 Applying enhanced gaming optimizations to {len(effects)} effects...")
        effects.sort(key=lambda x: x['start_time'])

        # Step 1: Remove duplicate/similar effects
        deduplicated_effects = self._remove_similar_effects(effects)
        
        # Step 2: Apply density management
        density_managed = self._manage_effect_density(deduplicated_effects)
        
        # Step 3: Apply intelligent spacing based on sound type
        spaced_effects = self._apply_intelligent_spacing(density_managed)
        
        # Step 4: Final quality check
        final_effects = self._final_quality_filter(spaced_effects)

        self.log_func(f"✅ Enhanced gaming optimization complete: {len(final_effects)} final effects")
        return final_effects

    def _remove_similar_effects(self, effects: List[Dict]) -> List[Dict]:
        """
        Removes repetitive and similar effects with context awareness.
        """
        if not effects:
            return effects

        filtered_effects = []
        last_context_time = {}
        repetitive_contexts = ["ladder", "climbing", "footsteps", "walking"]

        for effect in effects:
            word = effect.get('word', '').upper()
            context = effect.get('context', '').lower()
            time = effect['start_time']
            
            is_repetitive = False
            for rep_ctx in repetitive_contexts:
                if rep_ctx in context:
                    if time - last_context_time.get(rep_ctx, -999) < 10.0: # 10-second window
                        self.log_func(f"-> FILTER (Repetitive): '{word}' due to recent '{rep_ctx}' context.")
                        is_repetitive = True
                        break
                    else:
                        last_context_time[rep_ctx] = time

            if not is_repetitive:
                filtered_effects.append(effect)

        self.log_func(f"Context-aware filtering complete: {len(effects)} -> {len(filtered_effects)} effects")
        return filtered_effects

    def _manage_effect_density(self, effects: List[Dict]) -> List[Dict]:
        """Manage density with priority-based selection."""
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
            
            # Sort by priority: energy * confidence * tier_weight
            def calculate_priority(effect):
                energy = effect.get('energy', 0.5)
                confidence = effect.get('confidence', 0.5)
                tier = effect.get('tier', 'medium')
                
                tier_weights = {'major': 1.5, 'medium': 1.0, 'quick': 0.7}
                tier_weight = tier_weights.get(tier, 1.0)
                
                return energy * confidence * tier_weight
            
            bucket_effects.sort(key=calculate_priority, reverse=True)
            
            # Take top effects up to limit
            selected = bucket_effects[:self.max_effects_per_minute]
            filtered_effects.extend(selected)
            
            if len(bucket_effects) > self.max_effects_per_minute:
                self.log_func(f"📊 Density limit applied to minute {bucket_key}: "
                            f"{len(bucket_effects)} → {len(selected)} effects")

        return sorted(filtered_effects, key=lambda x: x['start_time'])

    def _apply_intelligent_spacing(self, effects: List[Dict]) -> List[Dict]:
        """Apply spacing with different rules for different sound types."""
        if not effects:
            return effects

        final_effects = []
        
        for current_effect in effects:
            should_keep = True
            current_time = current_effect['start_time']
            current_tier = current_effect.get('tier', 'medium')
            current_word = current_effect.get('word', '')
            
            # Calculate required spacing based on sound characteristics
            base_spacing = self.min_effect_spacing
            
            # High-impact sounds need more space
            if current_tier == 'major' or any(word in current_word.upper() 
                                            for word in ['BOOM', 'CRASH', 'SLAM', 'BANG']):
                required_spacing = base_spacing * 1.5
            # Repetitive sounds need even more space
            elif any(word in current_word.upper() 
                    for word in ['CLANK', 'STEP', 'TICK', 'TAP']):
                required_spacing = base_spacing * 2.0
            else:
                required_spacing = base_spacing
            
            # Check spacing against recent effects
            for recent_effect in final_effects[-3:]:  # Check last 3 effects
                time_diff = current_time - recent_effect['start_time']
                
                if time_diff < required_spacing:
                    # Decide which effect to keep based on priority
                    current_priority = self._calculate_effect_priority(current_effect)
                    recent_priority = self._calculate_effect_priority(recent_effect)
                    
                    if current_priority > recent_priority * 1.2:  # 20% bonus for new effect
                        # Remove the recent effect and keep current
                        final_effects = [e for e in final_effects if e != recent_effect]
                        self.log_func(f"🔄 Replaced lower priority effect: "
                                    f"'{recent_effect.get('word')}' → '{current_word}'")
                    else:
                        # Keep the recent effect, skip current
                        self.log_func(f"🔇 Skipped due to spacing: '{current_word}' at {current_time:.2f}s "
                                    f"(too close to '{recent_effect.get('word')}')")
                        should_keep = False
                        break
            
            if should_keep:
                final_effects.append(current_effect)

        return final_effects

    def _calculate_effect_priority(self, effect: Dict) -> float:
        """Calculate priority score for an effect."""
        energy = effect.get('energy', 0.5)
        confidence = effect.get('confidence', 0.5) 
        tier = effect.get('tier', 'medium')
        
        tier_weights = {'major': 2.0, 'medium': 1.0, 'quick': 0.5}
        tier_weight = tier_weights.get(tier, 1.0)
        
        # Boost priority for high-impact words
        word = effect.get('word', '').upper()
        impact_boost = 1.0
        if any(high_impact in word for high_impact in ['BOOM', 'CRASH', 'SLAM', 'BANG', 'KABOOM']):
            impact_boost = 1.5
        elif any(low_impact in word for low_impact in ['TICK', 'TAP', 'CLICK']):
            impact_boost = 0.7
        
        return energy * confidence * tier_weight * impact_boost

    def _final_quality_filter(self, effects: List[Dict]) -> List[Dict]:
        """Final pass to ensure quality and remove edge cases."""
        filtered_effects = []
        
        for effect in effects:
            word = effect.get('word', '').strip()
            confidence = effect.get('confidence', 0)
            energy = effect.get('energy', 0)
            
            # Skip if word is too short or confidence too low
            if len(word) < 2:
                self.log_func(f"🔇 Filtered short word: '{word}'")
                continue
                
            if confidence < 0.3:
                self.log_func(f"🔇 Filtered low confidence: '{word}' ({confidence:.2f})")
                continue
                
            if energy < 0.02:
                self.log_func(f"🔇 Filtered low energy: '{word}' ({energy:.3f})")
                continue
            
            filtered_effects.append(effect)
        
        return filtered_effects