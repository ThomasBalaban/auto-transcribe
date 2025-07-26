"""
Sound class to onomatopoeia mappings for comic book-style sound effects.
Each sound category maps to exactly one onomatopoeia word for easy debugging.
"""

SOUND_MAPPINGS = {
    # Impact sounds
    'explosion': ['BOOM'],
    'crash': ['CRASH'],
    'glass': ['SHATTER'],
    'gunshot': ['BANG'],
    'slam': ['SLAM'],
    'thud': ['THUD'],
    
    # Enhanced thud variants
    'heavy_thud': ['THUNK'],
    'soft_thud': ['PLOP'],
    
    # Crunch sounds
    'crunch': ['CRUNCH'],
    'crackle': ['CRACKLE'],
    'break': ['CRACK'],
    
    # Combat/Fighting sounds
    'punch': ['POW'],
    'hit': ['WHACK'],
    'kick': ['THWACK'],
    'sword': ['CLANG'],
    
    # Gun/Weapon loading sounds
    'gun_load': ['CLICK'],
    'reload': ['CLACK'],
    'magazine': ['SNAP'],
    'bolt': ['RACK'],
    
    # Electronic/Mechanical
    'bell': ['DING'],
    'buzz': ['BUZZ'],
    'beep': ['BEEP'],
    'click': ['CLICK'],
    'alarm': ['BLARE'],
    
    # Movement/Air sounds
    'whoosh': ['WHOOSH'],
    'pop': ['POP'],
    'whistle': ['TWEET'],
    'siren': ['WAIL'],
    'wind': ['HOWL'],
    
    # Water/Liquid sounds
    'splash': ['SPLASH'],
    'drip': ['DRIP'],
    'pour': ['GLUG'],
    'bubble': ['BLUB'],
    
    # Nature sounds
    'thunder': ['RUMBLE'],
    'rain': ['PATTER'],
    'fire': ['CRACKLE'],
    
    # Animal sounds
    'dog': ['WOOF'],
    'cat': ['MEOW'],
    'bird': ['CHIRP'],
    'horse': ['NEIGH'],
    'cow': ['MOO'],
    'pig': ['OINK'],
    'sheep': ['BAA'],
    'lion': ['ROAR'],
    'bear': ['GROWL'],
    'wolf': ['HOWL'],
    'snake': ['HISS'],
    'insect': ['BUZZ'],
    'frog': ['CROAK'],
    
    # Human sounds
    'applause': ['CLAP'],
    'footsteps': ['STOMP'],
    'knock': ['KNOCK'],
    'sneeze': ['ACHOO'],
    'cough': ['COUGH'],
    'laugh': ['HAHA'],
    'gasp': ['GASP'],
    'whisper': ['PSST'],
    
    # Vehicle sounds
    'car_horn': ['HONK'],
    'engine': ['VROOM'],
    'brakes': ['SCREECH'],
    'tire': ['SQUEAL'],
    'motorcycle': ['VROOM'],
    'truck': ['RUMBLE'],
    
    # Food/Eating sounds
    'chew': ['MUNCH'],
    'bite': ['CHOMP'],
    'slurp': ['SLURP'],
    'sizzle': ['SIZZLE'],
    'boil': ['BUBBLE'],
    
    # Technology sounds
    'computer': ['BEEP'],
    'phone': ['RING'],
    'camera': ['CLICK'],
    'printer': ['WHIR'],
    
    # Miscellaneous
    'zipper': ['ZIP'],
    'paper': ['RUSTLE'],
    'fabric': ['SWISH'],
    'door': ['CREAK'],
    'spring': ['BOING'],
    'rubber': ['SQUEAK'],
    
    # Musical Instruments
    'guitar': ['STRUM'],
    'piano': ['PLINK'],
    'drums': ['BOOM'],
    'violin': ['SCREECH'],
    'trumpet': ['BLARE'],
    'flute': ['TOOT'],
    'harmonica': ['WHEEZE'],
    'banjo': ['TWANG'],
    'saxophone': ['HONK'],
    'organ': ['DRONE'],
    'harp': ['PLUCK'],
    'xylophone': ['PING'],
    'accordion': ['WHEEZE'],
    'bagpipes': ['SKIRL'],
    'clarinet': ['SQUEAL'],
    'tuba': ['OOMPH'],
    'cymbals': ['CLASH'],
    'tambourine': ['JINGLE'],
    
    # Speech & Human Vocal
    'shout': ['SHOUT'],
    'scream': ['SCREAM'],
    'yell': ['YELL'],
    'whistle_human': ['WHISTLE'],
    'singing': ['LALALA'],
    'humming': ['HMM'],
    'burp': ['BURP'],
    'hiccup': ['HIC'],
    'yawn': ['YAWN'],
    'sniffle': ['SNIFF'],
    'sob': ['BOOHOO'],
    'groan': ['UGH'],
    'sigh': ['SIGH'],
    'giggle': ['HEE'],
    'cheer': ['HOORAY'],
    'boo': ['BOO'],
    'chatter': ['JABBER'],
    
    # Transportation Extended
    'airplane': ['ROAR'],
    'helicopter': ['WHOP'],
    'train': ['CHOO'],
    'ship': ['TOOT'],
    'boat': ['PUTT'],
    'bicycle': ['ZING'],
    'skateboard': ['ROLL'],
    'subway': ['SCREECH'],
    'bus': ['HISS'],
    'emergency_siren': ['WAIL'],
    'ambulance': ['WEE'],
    'fire_truck': ['CLANG'],
    'police_siren': ['WHOOP'],
    
    # Tools & Machinery
    'drill': ['BZZT'],
    'hammer': ['BANG'],
    'saw': ['ZZZT'],
    'chainsaw': ['BRRRR'],
    'sewing_machine': ['WHIRR'],
    'vacuum': ['VROOOM'],
    'blender': ['WHIZZ'],
    'lawnmower': ['BRRRR'],
    'jackhammer': ['RATTA'],
    'grinder': ['SCREEE'],
    'sandpaper': ['SCRITCH'],
    'filing': ['RASP'],
    
    # Household & Kitchen
    'microwave': ['DING'],
    'dishwasher': ['SLOSH'],
    'washing_machine': ['CHURN'],
    'garbage_disposal': ['GRIND'],
    'blender_kitchen': ['WHIRR'],
    'mixer': ['WHIP'],
    'kettle': ['WHISTLE'],
    'toaster': ['POP'],
    'timer': ['DING'],
    'doorbell': ['DING'],
    'lock': ['CLICK'],
    'drawer': ['SLIDE'],
    'cabinet': ['THUNK'],
    
    # Weather & Environment
    'wind_strong': ['HOWL'],
    'storm': ['CRASH'],
    'hail': ['RATTLE'],
    'snow': ['WHIFF'],
    'earthquake': ['RUMBLE'],
    'volcano': ['ROAR'],
    'avalanche': ['ROAR'],
    'waterfall': ['RUSH'],
    'stream': ['BABBLE'],
    'ocean_waves': ['CRASH'],
    'river': ['FLOW'],
    
    # More Animals
    'elephant': ['TRUMPET'],
    'monkey': ['OOK'],
    'goat': ['BLEAT'],
    'duck': ['QUACK'],
    'chicken': ['CLUCK'],
    'turkey': ['GOBBLE'],
    'goose': ['HONK'],
    'cricket': ['CHIRP'],
    'bee': ['BUZZ'],
    'fly': ['BUZZ'],
    'owl': ['HOOT'],
    'hawk': ['SCREECH'],
    'crow': ['CAW'],
    'seagull': ['SCREECH'],
    'whale': ['SONG'],
    'dolphin': ['CLICK'],
    'seal': ['BARK'],
    'mouse': ['SQUEAK'],
    'rat': ['SQUEAK'],
    'bat': ['SCREECH'],
    
    # Outdoor & Sports
    'whistle_sport': ['TWEET'],
    'crowd_cheer': ['ROAR'],
    'crowd_boo': ['BOO'],
    'ball_bounce': ['BOING'],
    'ball_hit': ['THWACK'],
    'skateboard_trick': ['WHACK'],
    'tennis': ['POCK'],
    'golf': ['WHACK'],
    'bowling': ['CRASH'],
    'swimming': ['SPLASH'],
    'diving': ['SPLASH'],
    'running': ['THUD'],
    'walking': ['STEP'],
    'marching': ['STOMP'],
    
    # Electronic Devices
    'notification': ['PING'],
    'text_message': ['DING'],
    'email': ['CHIME'],
    'video_game': ['BEEP'],
    'radio_static': ['STATIC'],
    'tv_static': ['FUZZ'],
    'dial_tone': ['BEEP'],
    'busy_signal': ['BEEP'],
    'modem': ['SCREECH'],
    'fax': ['SCREECH'],
    'walkie_talkie': ['CRACKLE'],
    'radio': ['STATIC'],
    
    # Magic & Fantasy (for special effects)
    'magic': ['POOF'],
    'sparkle': ['TWINKLE'],
    'teleport': ['ZAP'],
    'laser': ['ZAP'],
    'energy': ['BZZT'],
    'force_field': ['HUM'],
    'portal': ['WOOSH'],
    'transformation': ['SHIMMER'],
    
    # Additional Human Sounds
    'cry': ['WAAH'],
    'ahem': ['AHEM'],
    'snort': ['SNORT'],
    'pant': ['PANT'],
    'wheeze': ['WHEEZE'],
    'fart': ['FART'],
    'rumble': ['GRUMBLE'],
    
    # More Specific Animal Sounds  
    'squawk': ['SQUAWK'],
    'caw': ['CAW'],
    'moo': ['MOO'],
    'oink': ['OINK'],
    'bleat': ['BLEAT'],
    'rattle': ['RATTLE'],
    
    # Mechanical/Electronic Specifics
    'tick': ['TICK'],
    'whir': ['WHIR'],
    'hum': ['HUM'],
    'clack': ['CLACK'],
    'idle': ['IDLE'],
    'start': ['VROOM'],
    
    # More Impact Sounds
    'chop': ['CHOP'],
    'smash': ['SMASH'],
    'slap': ['SLAP'],
    
    # Transportation Details
    'horn': ['HONK'],
    'swish': ['SWISH'],
    
    # Nature/Weather Specifics
    'chime': ['CHIME'],
    'rustle': ['RUSTLE'],
    
    # Food/Kitchen Specifics
    'gurgle': ['GURGLE'],
    'grind': ['GRIND'],
    'fizz': ['FIZZ'],
    
    # Additional Onomatopoeia Sounds
    'swoosh': ['SWOOSH'],
    'zoom': ['ZOOM'],
    'zing': ['ZING'],
    'ping': ['PING'],
    'plop': ['PLOP'],
    'bloop': ['BLOOP'],
    
    # Base forms for commonly referenced sounds
    'scratch': ['SCRATCH'],
    'snap': ['SNAP'], 
    'clap': ['CLAP'],
    'hiss': ['HISS'],
    'crack': ['CRACK'],
    'shatter': ['SHATTER'],
    'whistle': ['WHISTLE'],
    'chirp': ['CHIRP'],
    'honk': ['HONK'],
    'toot': ['TOOT'],
    'screech': ['SCREECH'],
    'squeal': ['SQUEAL'],
    'roar': ['ROAR']
}