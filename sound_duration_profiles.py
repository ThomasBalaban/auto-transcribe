sound_duration_profiles = {
    # Impact sounds - brief and sharp
    'explosion': {'base_duration': 0.8, 'variability': 0.3, 'decay_type': 'sharp'},
    'crash': {'base_duration': 0.6, 'variability': 0.2, 'decay_type': 'sharp'},
    'glass': {'base_duration': 0.4, 'variability': 0.2, 'decay_type': 'sharp'},
    'gunshot': {'base_duration': 0.3, 'variability': 0.1, 'decay_type': 'instant'},
    'slam': {'base_duration': 0.5, 'variability': 0.2, 'decay_type': 'sharp'},
    'thud': {'base_duration': 0.6, 'variability': 0.3, 'decay_type': 'medium'},
    
    # Enhanced thud variants
    'heavy_thud': {'base_duration': 0.8, 'variability': 0.2, 'decay_type': 'medium'},
    'soft_thud': {'base_duration': 0.4, 'variability': 0.2, 'decay_type': 'soft'},
    
    # Crunch sounds - variable duration
    'crunch': {'base_duration': 0.7, 'variability': 0.4, 'decay_type': 'medium'},
    'crackle': {'base_duration': 1.2, 'variability': 0.5, 'decay_type': 'gradual'},
    'break': {'base_duration': 0.5, 'variability': 0.2, 'decay_type': 'sharp'},
    
    # Combat/Fighting sounds - quick and punchy
    'punch': {'base_duration': 0.3, 'variability': 0.1, 'decay_type': 'sharp'},
    'hit': {'base_duration': 0.4, 'variability': 0.1, 'decay_type': 'sharp'},
    'kick': {'base_duration': 0.4, 'variability': 0.1, 'decay_type': 'sharp'},
    'sword': {'base_duration': 0.6, 'variability': 0.2, 'decay_type': 'medium'},
    
    # Gun/Weapon loading sounds - mechanical precision
    'gun_load': {'base_duration': 0.3, 'variability': 0.1, 'decay_type': 'instant'},
    'reload': {'base_duration': 0.4, 'variability': 0.1, 'decay_type': 'instant'},
    'magazine': {'base_duration': 0.2, 'variability': 0.1, 'decay_type': 'instant'},
    'bolt': {'base_duration': 0.3, 'variability': 0.1, 'decay_type': 'instant'},
    
    # Electronic/Mechanical - varies by type
    'bell': {'base_duration': 1.5, 'variability': 0.5, 'decay_type': 'gradual'},
    'buzz': {'base_duration': 1.0, 'variability': 0.3, 'decay_type': 'sustained'},
    'beep': {'base_duration': 0.2, 'variability': 0.1, 'decay_type': 'instant'},
    'click': {'base_duration': 0.1, 'variability': 0.05, 'decay_type': 'instant'},
    'alarm': {'base_duration': 2.0, 'variability': 0.8, 'decay_type': 'sustained'},
    
    # Movement/Air sounds - flowing
    'whoosh': {'base_duration': 1.0, 'variability': 0.4, 'decay_type': 'gradual'},
    'pop': {'base_duration': 0.2, 'variability': 0.1, 'decay_type': 'instant'},
    'whistle': {'base_duration': 1.5, 'variability': 0.6, 'decay_type': 'gradual'},
    'siren': {'base_duration': 3.0, 'variability': 1.0, 'decay_type': 'sustained'},
    'wind': {'base_duration': 2.0, 'variability': 1.0, 'decay_type': 'sustained'},
    
    # Water/Liquid sounds - varies by intensity
    'splash': {'base_duration': 0.8, 'variability': 0.4, 'decay_type': 'medium'},
    'drip': {'base_duration': 0.3, 'variability': 0.1, 'decay_type': 'soft'},
    'pour': {'base_duration': 2.0, 'variability': 1.0, 'decay_type': 'sustained'},
    'bubble': {'base_duration': 1.0, 'variability': 0.5, 'decay_type': 'gradual'},
    
    # Nature sounds - environmental
    'thunder': {'base_duration': 2.5, 'variability': 1.0, 'decay_type': 'gradual'},
    'rain': {'base_duration': 3.0, 'variability': 1.5, 'decay_type': 'sustained'},
    'fire': {'base_duration': 2.0, 'variability': 0.8, 'decay_type': 'sustained'},
    
    # Animal sounds - characteristic durations
    'dog': {'base_duration': 0.8, 'variability': 0.3, 'decay_type': 'sharp'},
    'cat': {'base_duration': 1.0, 'variability': 0.4, 'decay_type': 'medium'},
    'bird': {'base_duration': 0.5, 'variability': 0.3, 'decay_type': 'medium'},
    'horse': {'base_duration': 1.2, 'variability': 0.4, 'decay_type': 'medium'},
    'cow': {'base_duration': 1.5, 'variability': 0.5, 'decay_type': 'gradual'},
    'pig': {'base_duration': 0.8, 'variability': 0.3, 'decay_type': 'medium'},
    'sheep': {'base_duration': 1.0, 'variability': 0.3, 'decay_type': 'medium'},
    'lion': {'base_duration': 2.0, 'variability': 0.8, 'decay_type': 'gradual'},
    'bear': {'base_duration': 1.5, 'variability': 0.6, 'decay_type': 'gradual'},
    'wolf': {'base_duration': 2.5, 'variability': 1.0, 'decay_type': 'gradual'},
    'snake': {'base_duration': 1.0, 'variability': 0.4, 'decay_type': 'sustained'},
    'insect': {'base_duration': 2.0, 'variability': 1.0, 'decay_type': 'sustained'},
    'frog': {'base_duration': 0.6, 'variability': 0.2, 'decay_type': 'medium'},
    
    # Human sounds
    'applause': {'base_duration': 3.0, 'variability': 1.5, 'decay_type': 'sustained'},
    'footsteps': {'base_duration': 0.4, 'variability': 0.2, 'decay_type': 'sharp'},
    'knock': {'base_duration': 0.3, 'variability': 0.1, 'decay_type': 'sharp'},
    'sneeze': {'base_duration': 0.5, 'variability': 0.2, 'decay_type': 'sharp'},
    'cough': {'base_duration': 0.4, 'variability': 0.2, 'decay_type': 'medium'},
    'laugh': {'base_duration': 2.0, 'variability': 1.0, 'decay_type': 'sustained'},
    'gasp': {'base_duration': 0.6, 'variability': 0.2, 'decay_type': 'medium'},
    'whisper': {'base_duration': 1.5, 'variability': 0.8, 'decay_type': 'soft'},
    
    # Vehicle sounds
    'car_horn': {'base_duration': 1.0, 'variability': 0.5, 'decay_type': 'sustained'},
    'engine': {'base_duration': 2.0, 'variability': 1.0, 'decay_type': 'sustained'},
    'brakes': {'base_duration': 1.5, 'variability': 0.6, 'decay_type': 'gradual'},
    'tire': {'base_duration': 1.2, 'variability': 0.5, 'decay_type': 'gradual'},
    'motorcycle': {'base_duration': 1.8, 'variability': 0.8, 'decay_type': 'sustained'},
    'truck': {'base_duration': 2.5, 'variability': 1.0, 'decay_type': 'sustained'},
    
    # Food/Eating sounds
    'chew': {'base_duration': 1.0, 'variability': 0.5, 'decay_type': 'sustained'},
    'bite': {'base_duration': 0.3, 'variability': 0.1, 'decay_type': 'sharp'},
    'slurp': {'base_duration': 0.8, 'variability': 0.3, 'decay_type': 'gradual'},
    'sizzle': {'base_duration': 2.0, 'variability': 1.0, 'decay_type': 'sustained'},
    'boil': {'base_duration': 3.0, 'variability': 1.5, 'decay_type': 'sustained'},
    
    # Technology sounds
    'computer': {'base_duration': 0.3, 'variability': 0.1, 'decay_type': 'instant'},
    'phone': {'base_duration': 1.0, 'variability': 0.4, 'decay_type': 'sustained'},
    'camera': {'base_duration': 0.2, 'variability': 0.1, 'decay_type': 'instant'},
    'printer': {'base_duration': 2.0, 'variability': 1.0, 'decay_type': 'sustained'},
    
    # Miscellaneous
    'zipper': {'base_duration': 0.6, 'variability': 0.3, 'decay_type': 'gradual'},
    'paper': {'base_duration': 0.5, 'variability': 0.3, 'decay_type': 'soft'},
    'fabric': {'base_duration': 0.8, 'variability': 0.4, 'decay_type': 'soft'},
    'door': {'base_duration': 1.0, 'variability': 0.4, 'decay_type': 'gradual'},
    'spring': {'base_duration': 0.6, 'variability': 0.2, 'decay_type': 'medium'},
    'rubber': {'base_duration': 0.4, 'variability': 0.2, 'decay_type': 'medium'}
}