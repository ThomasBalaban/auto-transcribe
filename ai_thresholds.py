"""
AI decision thresholds for onomatopoeia detection.
Lower values = more permissive (detects more sounds)
Higher values = more selective (detects fewer sounds)

Organized by sound type for easy tuning.
"""

AI_THRESHOLDS = {
    # Impact sounds - Usually very distinctive, lower thresholds
    'explosion': 0.10,
    'crash': 0.12,
    'glass': 0.15,
    'gunshot': 0.10,
    'slam': 0.18,
    'thud': 0.20,
    'heavy_thud': 0.18,
    'soft_thud': 0.22,
    'break': 0.15,
    'smash': 0.12,
    'shatter': 0.15,
    'crack': 0.16,
    'snap': 0.14,
    
    # Combat/Fighting sounds - Sharp and distinctive
    'punch': 0.12,
    'hit': 0.14,
    'kick': 0.14,
    'sword': 0.16,
    'slap': 0.15,
    'whack': 0.13,
    
    # Gun/Weapon sounds - Very distinctive
    'gun_load': 0.12,
    'reload': 0.14,
    'magazine': 0.16,
    'bolt': 0.14,
    
    # Crunch sounds - Usually clear
    'crunch': 0.16,
    'crackle': 0.18,
    'chop': 0.15,
    
    # Electronic/Mechanical - Often very clear
    'bell': 0.20,
    'buzz': 0.15,
    'beep': 0.08,
    'click': 0.10,
    'alarm': 0.25,
    'ding': 0.12,
    'ring': 0.15,
    'tick': 0.18,
    'clack': 0.14,
    'hum': 0.20,
    
    # Movement/Air sounds - Can be subtle
    'whoosh': 0.18,
    'pop': 0.12,
    'whistle': 0.20,
    'siren': 0.22,
    'wind': 0.25,
    'swoosh': 0.18,
    'swish': 0.20,
    'zoom': 0.16,
    'zing': 0.14,
    
    # Water/Liquid sounds - Variable quality
    'splash': 0.18,
    'drip': 0.22,
    'pour': 0.25,
    'bubble': 0.24,
    'gurgle': 0.20,
    'fizz': 0.18,
    
    # Nature sounds - Often mixed with background
    'thunder': 0.15,  # Usually very clear
    'rain': 0.28,     # Can be subtle
    'fire': 0.22,
    'wind_strong': 0.25,
    'storm': 0.18,
    'earthquake': 0.20,
    'volcano': 0.18,
    'waterfall': 0.25,
    'ocean_waves': 0.22,
    
    # Animal sounds - Dogs
    'dog': 0.18,
    'woof': 0.16,
    'bark': 0.16,
    'growl': 0.20,
    
    # Animal sounds - Cats  
    'cat': 0.20,
    'meow': 0.18,
    'purr': 0.25,
    'hiss': 0.18,
    
    # Animal sounds - Birds
    'bird': 0.22,
    'chirp': 0.20,
    'tweet': 0.22,
    'squawk': 0.18,
    'caw': 0.16,
    'crow': 0.18,
    'owl': 0.20,
    'hawk': 0.20,
    'seagull': 0.22,
    
    # Animal sounds - Farm animals
    'cow': 0.16,
    'moo': 0.15,
    'pig': 0.18,
    'oink': 0.16,
    'horse': 0.18,
    'sheep': 0.20,
    'goat': 0.18,
    'chicken': 0.20,
    'duck': 0.18,
    'turkey': 0.20,
    'goose': 0.18,
    
    # Animal sounds - Wild animals
    'lion': 0.15,
    'roar': 0.14,
    'bear': 0.18,
    'wolf': 0.16,
    'howl': 0.16,
    'snake': 0.20,
    'elephant': 0.16,
    'monkey': 0.20,
    
    # Animal sounds - Small animals/insects
    'mouse': 0.25,
    'rat': 0.25,
    'bat': 0.28,
    'insect': 0.25,
    'bee': 0.22,
    'fly': 0.28,
    'cricket': 0.22,
    'frog': 0.20,
    'whale': 0.18,
    'dolphin': 0.20,
    'seal': 0.18,
    
    # Human sounds - Speech
    'speech': 0.25,
    'talk': 0.25,
    'shout': 0.18,
    'scream': 0.16,
    'yell': 0.18,
    'whisper': 0.28,
    'singing': 0.22,
    'humming': 0.25,
    'chatter': 0.25,
    
    # Human sounds - Body functions
    'cough': 0.18,
    'sneeze': 0.16,
    'laugh': 0.20,
    'giggle': 0.22,
    'gasp': 0.20,
    'sigh': 0.24,
    'burp': 0.16,
    'hiccup': 0.18,
    'yawn': 0.22,
    'sob': 0.20,
    'cry': 0.18,
    'groan': 0.20,
    'snort': 0.18,
    'pant': 0.22,
    'wheeze': 0.24,
    'fart': 0.16,
    'ahem': 0.20,
    
    # Human sounds - Actions
    'applause': 0.20,
    'clap': 0.16,
    'footsteps': 0.20,
    'stomp': 0.18,
    'walking': 0.22,
    'running': 0.20,
    'marching': 0.18,
    'knock': 0.14,
    'cheer': 0.20,
    'boo': 0.18,
    
    # Vehicle sounds - Cars
    'car_horn': 0.15,
    'honk': 0.14,
    'engine': 0.20,
    'brakes': 0.18,
    'tire': 0.20,
    'screech': 0.16,
    'squeal': 0.18,
    'idle': 0.25,
    'start': 0.18,
    
    # Vehicle sounds - Other vehicles
    'motorcycle': 0.18,
    'truck': 0.20,
    'bus': 0.22,
    'airplane': 0.18,
    'helicopter': 0.16,
    'train': 0.16,
    'ship': 0.20,
    'boat': 0.22,
    'bicycle': 0.25,
    'skateboard': 0.22,
    'subway': 0.20,
    
    # Vehicle sounds - Emergency
    'emergency_siren': 0.18,
    'ambulance': 0.16,
    'fire_truck': 0.16,
    'police_siren': 0.16,
    
    # Tools & Machinery - Power tools
    'drill': 0.15,
    'hammer': 0.12,
    'saw': 0.16,
    'chainsaw': 0.14,
    'jackhammer': 0.14,
    'grinder': 0.16,
    'sandpaper': 0.20,
    'filing': 0.22,
    
    # Tools & Machinery - Household
    'vacuum': 0.18,
    'sewing_machine': 0.20,
    'lawnmower': 0.18,
    'blender': 0.16,
    'mixer': 0.18,
    'grind': 0.16,
    'whir': 0.18,
    
    # Household & Kitchen appliances
    'microwave': 0.16,
    'dishwasher': 0.22,
    'washing_machine': 0.20,
    'blender_kitchen': 0.16,
    'kettle': 0.18,
    'toaster': 0.14,
    'timer': 0.12,
    'doorbell': 0.12,
    'lock': 0.16,
    'drawer': 0.20,
    'cabinet': 0.18,
    
    # Musical Instruments - Strings
    'guitar': 0.18,
    'piano': 0.16,
    'violin': 0.20,
    'banjo': 0.18,
    'harp': 0.20,
    
    # Musical Instruments - Wind
    'trumpet': 0.16,
    'flute': 0.20,
    'saxophone': 0.18,
    'clarinet': 0.20,
    'harmonica': 0.18,
    'bagpipes': 0.16,
    'tuba': 0.16,
    
    # Musical Instruments - Percussion
    'drums': 0.14,
    'cymbals': 0.16,
    'xylophone': 0.18,
    'tambourine': 0.18,
    'organ': 0.20,
    'accordion': 0.20,
    
    # Food/Eating sounds
    'chew': 0.22,
    'bite': 0.18,
    'slurp': 0.20,
    'sizzle': 0.18,
    'boil': 0.22,
    
    # Technology sounds
    'computer': 0.18,
    'phone': 0.16,
    'camera': 0.14,
    'printer': 0.20,
    'notification': 0.12,
    'text_message': 0.14,
    'email': 0.16,
    'video_game': 0.18,
    'radio': 0.22,
    'tv_static': 0.25,
    'radio_static': 0.25,
    'dial_tone': 0.15,
    'busy_signal': 0.15,
    'modem': 0.20,
    'fax': 0.20,
    'walkie_talkie': 0.18,
    
    # Weather & Environment specifics
    'hail': 0.20,
    'snow': 0.28,
    'avalanche': 0.18,
    'stream': 0.25,
    'river': 0.26,
    'chime': 0.16,
    'rustle': 0.24,
    
    # Outdoor & Sports
    'whistle_sport': 0.16,
    'crowd_cheer': 0.20,
    'crowd_boo': 0.20,
    'ball_bounce': 0.18,
    'ball_hit': 0.16,
    'skateboard_trick': 0.20,
    'tennis': 0.18,
    'golf': 0.16,
    'bowling': 0.16,
    'swimming': 0.22,
    'diving': 0.20,
    
    # Miscellaneous
    'zipper': 0.20,
    'paper': 0.24,
    'fabric': 0.26,
    'door': 0.20,
    'spring': 0.18,
    'rubber': 0.20,
    'scratch': 0.22,
    
    # Magic & Fantasy (for special effects)
    'magic': 0.25,
    'sparkle': 0.28,
    'teleport': 0.25,
    'laser': 0.18,
    'energy': 0.20,
    'force_field': 0.22,
    'portal': 0.25,
    'transformation': 0.28,
    
    # Additional base forms
    'ping': 0.12,
    'plop': 0.18,
    'bloop': 0.20,
    'toot': 0.16,
}