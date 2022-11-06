
from transformers import GPT2Tokenizer, GPT2Model
import torch
import numpy as np
# device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token
model = GPT2Model.from_pretrained('gpt2-medium')
# text = "Replace me by any text you'd like."
datasets_hub = ['dtd','aircraft','caltech101','cars','cifar10','cifar100','flowers','food','pets','sun397','voc2007']

for dataset in datasets_hub:
    if dataset == 'dtd':
        text = ['banded', 'blotchy', 'braided', 'bubbly', 'bumpy', 'chequered', 'cobwebbed', 'cracked', 'crosshatched', 'crystalline', 'dotted', 'fibrous', 'flecked', 'freckled', 'frilly', 'gauzy', 'grid', 'grooved', 'honeycombed', 'interlaced', 'knitted', 'lacelike', 'lined', 'marbled', 'matted', 'meshed', 'paisley', 'perforated', 'pitted', 'pleated', 'polka-dotted', 'porous', 'potholed', 'scaly', 'smeared', 'spiralled', 'sprinkled', 'stained', 'stratified', 'striped', 'studded', 'swirly', 'veined', 'waffled', 'woven', 'wrinkled', 'zigzagged']
    elif dataset == 'aircraft':
        text = ['707-320', '727-200', '737-200', '737-300', '737-400', '737-500', '737-600', '737-700', '737-800', '737-900', '747-100', '747-200', '747-300', '747-400', '757-200', '757-300', '767-200', '767-300', '767-400', '777-200', '777-300', 'A300B4', 'A310', 'A318', 'A319', 'A320', 'A321', 'A330-200', 'A330-300', 'A340-200', 'A340-300', 'A340-500', 'A340-600', 'A380', 'ATR-42', 'ATR-72', 'An-12', 'BAE 146-200', 'BAE 146-300', 'BAE-125', 'Beechcraft 1900', 'Boeing 717', 'C-130', 'C-47', 'CRJ-200', 'CRJ-700', 'CRJ-900', 'Cessna 172', 'Cessna 208', 'Cessna 525', 'Cessna 560', 'Challenger 600', 'DC-10', 'DC-3', 'DC-6', 'DC-8', 'DC-9-30', 'DH-82', 'DHC-1', 'DHC-6', 'DHC-8-100', 'DHC-8-300', 'DR-400', 'Dornier 328', 'E-170', 'E-190', 'E-195', 'EMB-120', 'ERJ 135', 'ERJ 145', 'Embraer Legacy 600', 'Eurofighter Typhoon', 'F-16A/B', 'F/A-18', 'Falcon 2000', 'Falcon 900', 'Fokker 100', 'Fokker 50', 'Fokker 70', 'Global Express', 'Gulfstream IV', 'Gulfstream V', 'Hawk T1', 'Il-76', 'L-1011', 'MD-11', 'MD-80', 'MD-87', 'MD-90', 'Metroliner', 'Model B200', 'PA-28', 'SR-20', 'Saab 2000', 'Saab 340', 'Spitfire', 'Tornado', 'Tu-134', 'Tu-154', 'Yak-42']
    elif dataset == 'caltech101':
        text = ['BACKGROUND_Google', 'Faces', 'Faces_easy', 'Leopards', 'Motorbikes', 'accordion', 'airplanes', 'anchor', 'ant', 'barrel', 'bass', 'beaver', 'binocular', 'bonsai', 'brain', 'brontosaurus', 'buddha', 'butterfly', 'camera', 'cannon', 'car_side', 'ceiling_fan', 'cellphone', 'chair', 'chandelier', 'cougar_body', 'cougar_face', 'crab', 'crayfish', 'crocodile', 'crocodile_head', 'cup', 'dalmatian', 'dollar_bill', 'dolphin', 'dragonfly', 'electric_guitar', 'elephant', 'emu', 'euphonium', 'ewer', 'ferry', 'flamingo', 'flamingo_head', 'garfield', 'gerenuk', 'gramophone', 'grand_piano', 'hawksbill', 'headphone', 'hedgehog', 'helicopter', 'ibis', 'inline_skate', 'joshua_tree', 'kangaroo', 'ketch', 'lamp', 'laptop', 'llama', 'lobster', 'lotus', 'mandolin', 'mayfly', 'menorah', 'metronome', 'minaret', 'nautilus', 'octopus', 'okapi', 'pagoda', 'panda', 'pigeon', 'pizza', 'platypus', 'pyramid', 'revolver', 'rhino', 'rooster', 'saxophone', 'schooner', 'scissors', 'scorpion', 'sea_horse', 'snoopy', 'soccer_ball', 'stapler', 'starfish', 'stegosaurus', 'stop_sign', 'strawberry', 'sunflower', 'tick', 'trilobite', 'umbrella', 'watch', 'water_lilly', 'wheelchair', 'wild_cat', 'windsor_chair', 'wrench', 'yin_yang']
    elif dataset == 'cars':
        text = ['AM_General_Hummer_SUV_2000', 'Acura_RL_Sedan_2012', 'Acura_TL_Sedan_2012', 'Acura_TL_Type-S_2008', 'Acura_TSX_Sedan_2012', 'Acura_Integra_Type_R_2001', 'Acura_ZDX_Hatchback_2012', 'Aston_Martin_V8_Vantage_Convertible_2012', 'Aston_Martin_V8_Vantage_Coupe_2012', 'Aston_Martin_Virage_Convertible_2012', 'Aston_Martin_Virage_Coupe_2012', 'Audi_RS_4_Convertible_2008', 'Audi_A5_Coupe_2012', 'Audi_TTS_Coupe_2012', 'Audi_R8_Coupe_2012', 'Audi_V8_Sedan_1994', 'Audi_100_Sedan_1994', 'Audi_100_Wagon_1994', 'Audi_TT_Hatchback_2011', 'Audi_S6_Sedan_2011', 'Audi_S5_Convertible_2012', 'Audi_S5_Coupe_2012', 'Audi_S4_Sedan_2012', 'Audi_S4_Sedan_2007', 'Audi_TT_RS_Coupe_2012', 'BMW_ActiveHybrid_5_Sedan_2012', 'BMW_1_Series_Convertible_2012', 'BMW_1_Series_Coupe_2012', 'BMW_3_Series_Sedan_2012', 'BMW_3_Series_Wagon_2012', 'BMW_6_Series_Convertible_2007', 'BMW_X5_SUV_2007', 'BMW_X6_SUV_2012', 'BMW_M3_Coupe_2012', 'BMW_M5_Sedan_2010', 'BMW_M6_Convertible_2010', 'BMW_X3_SUV_2012', 'BMW_Z4_Convertible_2012', 'Bentley_Continental_Supersports_Conv._Convertible_2012', 'Bentley_Arnage_Sedan_2009', 'Bentley_Mulsanne_Sedan_2011', 'Bentley_Continental_GT_Coupe_2012', 'Bentley_Continental_GT_Coupe_2007', 'Bentley_Continental_Flying_Spur_Sedan_2007', 'Bugatti_Veyron_16.4_Convertible_2009', 'Bugatti_Veyron_16.4_Coupe_2009', 'Buick_Regal_GS_2012', 'Buick_Rainier_SUV_2007', 'Buick_Verano_Sedan_2012', 'Buick_Enclave_SUV_2012', 'Cadillac_CTS-V_Sedan_2012', 'Cadillac_SRX_SUV_2012', 'Cadillac_Escalade_EXT_Crew_Cab_2007', 'Chevrolet_Silverado_1500_Hybrid_Crew_Cab_2012', 'Chevrolet_Corvette_Convertible_2012', 'Chevrolet_Corvette_ZR1_2012', 'Chevrolet_Corvette_Ron_Fellows_Edition_Z06_2007', 'Chevrolet_Traverse_SUV_2012', 'Chevrolet_Camaro_Convertible_2012', 'Chevrolet_HHR_SS_2010', 'Chevrolet_Impala_Sedan_2007', 'Chevrolet_Tahoe_Hybrid_SUV_2012', 'Chevrolet_Sonic_Sedan_2012', 'Chevrolet_Express_Cargo_Van_2007', 'Chevrolet_Avalanche_Crew_Cab_2012', 'Chevrolet_Cobalt_SS_2010', 'Chevrolet_Malibu_Hybrid_Sedan_2010', 'Chevrolet_TrailBlazer_SS_2009', 'Chevrolet_Silverado_2500HD_Regular_Cab_2012', 'Chevrolet_Silverado_1500_Classic_Extended_Cab_2007', 'Chevrolet_Express_Van_2007', 'Chevrolet_Monte_Carlo_Coupe_2007', 'Chevrolet_Malibu_Sedan_2007', 'Chevrolet_Silverado_1500_Extended_Cab_2012', 'Chevrolet_Silverado_1500_Regular_Cab_2012', 'Chrysler_Aspen_SUV_2009', 'Chrysler_Sebring_Convertible_2010', 'Chrysler_Town_and_Country_Minivan_2012', 'Chrysler_300_SRT-8_2010', 'Chrysler_Crossfire_Convertible_2008', 'Chrysler_PT_Cruiser_Convertible_2008', 'Daewoo_Nubira_Wagon_2002', 'Dodge_Caliber_Wagon_2012', 'Dodge_Caliber_Wagon_2007', 'Dodge_Caravan_Minivan_1997', 'Dodge_Ram_Pickup_3500_Crew_Cab_2010', 'Dodge_Ram_Pickup_3500_Quad_Cab_2009', 'Dodge_Sprinter_Cargo_Van_2009', 'Dodge_Journey_SUV_2012', 'Dodge_Dakota_Crew_Cab_2010', 'Dodge_Dakota_Club_Cab_2007', 'Dodge_Magnum_Wagon_2008', 'Dodge_Challenger_SRT8_2011', 'Dodge_Durango_SUV_2012', 'Dodge_Durango_SUV_2007', 'Dodge_Charger_Sedan_2012', 'Dodge_Charger_SRT-8_2009', 'Eagle_Talon_Hatchback_1998', 'FIAT_500_Abarth_2012', 'FIAT_500_Convertible_2012', 'Ferrari_FF_Coupe_2012', 'Ferrari_California_Convertible_2012', 'Ferrari_458_Italia_Convertible_2012', 'Ferrari_458_Italia_Coupe_2012', 'Fisker_Karma_Sedan_2012', 'Ford_F-450_Super_Duty_Crew_Cab_2012', 'Ford_Mustang_Convertible_2007', 'Ford_Freestar_Minivan_2007', 'Ford_Expedition_EL_SUV_2009', 'Ford_Edge_SUV_2012', 'Ford_Ranger_SuperCab_2011', 'Ford_GT_Coupe_2006', 'Ford_F-150_Regular_Cab_2012', 'Ford_F-150_Regular_Cab_2007', 'Ford_Focus_Sedan_2007', 'Ford_E-Series_Wagon_Van_2012', 'Ford_Fiesta_Sedan_2012', 'GMC_Terrain_SUV_2012', 'GMC_Savana_Van_2012', 'GMC_Yukon_Hybrid_SUV_2012', 'GMC_Acadia_SUV_2012', 'GMC_Canyon_Extended_Cab_2012', 'Geo_Metro_Convertible_1993', 'HUMMER_H3T_Crew_Cab_2010', 'HUMMER_H2_SUT_Crew_Cab_2009', 'Honda_Odyssey_Minivan_2012', 'Honda_Odyssey_Minivan_2007', 'Honda_Accord_Coupe_2012', 'Honda_Accord_Sedan_2012', 'Hyundai_Veloster_Hatchback_2012', 'Hyundai_Santa_Fe_SUV_2012', 'Hyundai_Tucson_SUV_2012', 'Hyundai_Veracruz_SUV_2012', 'Hyundai_Sonata_Hybrid_Sedan_2012', 'Hyundai_Elantra_Sedan_2007', 'Hyundai_Accent_Sedan_2012', 'Hyundai_Genesis_Sedan_2012', 'Hyundai_Sonata_Sedan_2012', 'Hyundai_Elantra_Touring_Hatchback_2012', 'Hyundai_Azera_Sedan_2012', 'Infiniti_G_Coupe_IPL_2012', 'Infiniti_QX56_SUV_2011', 'Isuzu_Ascender_SUV_2008', 'Jaguar_XK_XKR_2012', 'Jeep_Patriot_SUV_2012', 'Jeep_Wrangler_SUV_2012', 'Jeep_Liberty_SUV_2012', 'Jeep_Grand_Cherokee_SUV_2012', 'Jeep_Compass_SUV_2012', 'Lamborghini_Reventon_Coupe_2008', 'Lamborghini_Aventador_Coupe_2012', 'Lamborghini_Gallardo_LP_570-4_Superleggera_2012', 'Lamborghini_Diablo_Coupe_2001', 'Land_Rover_Range_Rover_SUV_2012', 'Land_Rover_LR2_SUV_2012', 'Lincoln_Town_Car_Sedan_2011', 'MINI_Cooper_Roadster_Convertible_2012', 'Maybach_Landaulet_Convertible_2012', 'Mazda_Tribute_SUV_2011', 'McLaren_MP4-12C_Coupe_2012', 'Mercedes-Benz_300-Class_Convertible_1993', 'Mercedes-Benz_C-Class_Sedan_2012', 'Mercedes-Benz_SL-Class_Coupe_2009', 'Mercedes-Benz_E-Class_Sedan_2012', 'Mercedes-Benz_S-Class_Sedan_2012', 'Mercedes-Benz_Sprinter_Van_2012', 'Mitsubishi_Lancer_Sedan_2012', 'Nissan_Leaf_Hatchback_2012', 'Nissan_NV_Passenger_Van_2012', 'Nissan_Juke_Hatchback_2012', 'Nissan_240SX_Coupe_1998', 'Plymouth_Neon_Coupe_1999', 'Porsche_Panamera_Sedan_2012', 'Ram_C_V_Cargo_Van_Minivan_2012', 'Rolls-Royce_Phantom_Drophead_Coupe_Convertible_2012', 'Rolls-Royce_Ghost_Sedan_2012', 'Rolls-Royce_Phantom_Sedan_2012', 'Scion_xD_Hatchback_2012', 'Spyker_C8_Convertible_2009', 'Spyker_C8_Coupe_2009', 'Suzuki_Aerio_Sedan_2007', 'Suzuki_Kizashi_Sedan_2012', 'Suzuki_SX4_Hatchback_2012', 'Suzuki_SX4_Sedan_2012', 'Tesla_Model_S_Sedan_2012', 'Toyota_Sequoia_SUV_2012', 'Toyota_Camry_Sedan_2012', 'Toyota_Corolla_Sedan_2012', 'Toyota_4Runner_SUV_2012', 'Volkswagen_Golf_Hatchback_2012', 'Volkswagen_Golf_Hatchback_1991', 'Volkswagen_Beetle_Hatchback_2012', 'Volvo_C30_Hatchback_2012', 'Volvo_240_Sedan_1993', 'Volvo_XC90_SUV_2007', 'smart_fortwo_Convertible_2012']
    elif dataset == 'cifar10':
        text = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif dataset == 'cifar100':
        text = ['apple', 'aquarium_fih', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
    elif dataset == 'flowers':
        text = ['pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 'english marigold', 'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood', 'globe thistle', 'snapdragon', "colt's foot", 'king protea', 'spear thistle', 'yellow iris', 'globe-flower', 'purple coneflower', 'peruvian lily', 'balloon flower', 'giant white arum lily', 'fire lily', 'pincushion flower', 'fritillary', 'red ginger', 'grape hyacinth', 'corn poppy', 'prince of wales feathers', 'stemless gentian', 'artichoke', 'sweet william', 'carnation', 'garden phlox', 'love in the mist', 'mexican aster', 'alpine sea holly', 'ruby-lipped cattleya', 'cape flower', 'great masterwort', 'siam tulip', 'lenten rose', 'barbeton daisy', 'daffodil', 'sword lily', 'poinsettia', 'bolero deep blue', 'wallflower', 'marigold', 'buttercup', 'oxeye daisy', 'common dandelion', 'petunia', 'wild pansy', 'primula', 'sunflower', 'pelargonium', 'bishop of llandaff', 'gaura', 'geranium', 'orange dahlia', 'pink-yellow dahlia?', 'cautleya spicata', 'japanese anemone', 'black-eyed susan', 'silverbush', 'californian poppy', 'osteospermum', 'spring crocus', 'bearded iris', 'windflower', 'tree poppy', 'gazania', 'azalea', 'water lily', 'rose', 'thorn apple', 'morning glory', 'passion flower', 'lotus', 'toad lily', 'anthurium', 'frangipani', 'clematis', 'hibiscus', 'columbine', 'desert-rose', 'tree mallow', 'magnolia', 'cyclamen ', 'watercress', 'canna lily', 'hippeastrum ', 'bee balm', 'ball moss', 'foxglove', 'bougainvillea', 'camellia', 'mallow', 'mexican petunia', 'bromelia', 'blanket flower', 'trumpet creeper', 'blackberry lily']
    elif dataset == 'food':
        text = ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake', 'ceviche', 'cheese_plate', 'cheesecake', 'chicken_curry', 'chicken_quesadilla', 'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles']
    elif dataset == 'pets':
        text = ['Abyssinian', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 'Bengal', 'Birman', 'Bombay', 'boxer', 'British_Shorthair', 'chihuahua', 'Egyptian_Mau', 'english_cocker_spaniel', 'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger', 'Maine_Coon', 'miniature_pinscher', 'newfoundland', 'Persian', 'pomeranian', 'pug', 'Ragdoll', 'Russian_Blue', 'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu', 'Siamese', 'Sphynx', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier']
    elif dataset == 'sun397':
        text = ['abbey', 'airplane_cabin', 'airport_terminal', 'alley', 'amphitheater', 'amusement_arcade', 'amusement_park', 'anechoic_chamber', 'apartment_building/outdoor', 'apse/indoor', 'aquarium', 'aqueduct', 'arch', 'archive', 'arrival_gate/outdoor', 'art_gallery', 'art_school', 'art_studio', 'assembly_line', 'athletic_field/outdoor', 'atrium/public', 'attic', 'auditorium', 'auto_factory', 'badlands', 'badminton_court/indoor', 'baggage_claim', 'bakery/shop', 'balcony/exterior', 'balcony/interior', 'ball_pit', 'ballroom', 'bamboo_forest', 'banquet_hall', 'bar', 'barn', 'barndoor', 'baseball_field', 'basement', 'basilica', 'basketball_court/outdoor', 'bathroom', 'batters_box', 'bayou', 'bazaar/indoor', 'bazaar/outdoor', 'beach', 'beauty_salon', 'bedroom', 'berth', 'biology_laboratory', 'bistro/indoor', 'boardwalk', 'boat_deck', 'boathouse', 'bookstore', 'booth/indoor', 'botanical_garden', 'bow_window/indoor', 'bow_window/outdoor', 'bowling_alley', 'boxing_ring', 'brewery/indoor', 'bridge', 'building_facade', 'bullring', 'burial_chamber', 'bus_interior', 'butchers_shop', 'butte', 'cabin/outdoor', 'cafeteria', 'campsite', 'campus', 'canal/natural', 'canal/urban', 'candy_store', 'canyon', 'car_interior/backseat', 'car_interior/frontseat', 'carrousel', 'casino/indoor', 'castle', 'catacomb', 'cathedral/indoor', 'cathedral/outdoor', 'cavern/indoor', 'cemetery', 'chalet', 'cheese_factory', 'chemistry_lab', 'chicken_coop/indoor', 'chicken_coop/outdoor', 'childs_room', 'church/indoor', 'church/outdoor', 'classroom', 'clean_room', 'cliff', 'cloister/indoor', 'closet', 'clothing_store', 'coast', 'cockpit', 'coffee_shop', 'computer_room', 'conference_center', 'conference_room', 'construction_site', 'control_room', 'control_tower/outdoor', 'corn_field', 'corral', 'corridor', 'cottage_garden', 'courthouse', 'courtroom', 'courtyard', 'covered_bridge/exterior', 'creek', 'crevasse', 'crosswalk', 'cubicle/office', 'dam', 'delicatessen', 'dentists_office', 'desert/sand', 'desert/vegetation', 'diner/indoor', 'diner/outdoor', 'dinette/home', 'dinette/vehicle', 'dining_car', 'dining_room', 'discotheque', 'dock', 'doorway/outdoor', 'dorm_room', 'driveway', 'driving_range/outdoor', 'drugstore', 'electrical_substation', 'elevator/door', 'elevator/interior', 'elevator_shaft', 'engine_room', 'escalator/indoor', 'excavation', 'factory/indoor', 'fairway', 'fastfood_restaurant', 'field/cultivated', 'field/wild', 'fire_escape', 'fire_station', 'firing_range/indoor', 'fishpond', 'florist_shop/indoor', 'food_court', 'forest/broadleaf', 'forest/needleleaf', 'forest_path', 'forest_road', 'formal_garden', 'fountain', 'galley', 'game_room', 'garage/indoor', 'garbage_dump', 'gas_station', 'gazebo/exterior', 'general_store/indoor', 'general_store/outdoor', 'gift_shop', 'golf_course', 'greenhouse/indoor', 'greenhouse/outdoor', 'gymnasium/indoor', 'hangar/indoor', 'hangar/outdoor', 'harbor', 'hayfield', 'heliport', 'herb_garden', 'highway', 'hill', 'home_office', 'hospital', 'hospital_room', 'hot_spring', 'hot_tub/outdoor', 'hotel/outdoor', 'hotel_room', 'house', 'hunting_lodge/outdoor', 'ice_cream_parlor', 'ice_floe', 'ice_shelf', 'ice_skating_rink/indoor', 'ice_skating_rink/outdoor', 'iceberg', 'igloo', 'industrial_area', 'inn/outdoor', 'islet', 'jacuzzi/indoor', 'jail/indoor', 'jail_cell', 'jewelry_shop', 'kasbah', 'kennel/indoor', 'kennel/outdoor', 'kindergarden_classroom', 'kitchen', 'kitchenette', 'labyrinth/outdoor', 'lake/natural', 'landfill', 'landing_deck', 'laundromat', 'lecture_room', 'library/indoor', 'library/outdoor', 'lido_deck/outdoor', 'lift_bridge', 'lighthouse', 'limousine_interior', 'living_room', 'lobby', 'lock_chamber', 'locker_room', 'mansion', 'manufactured_home', 'market/indoor', 'market/outdoor', 'marsh', 'martial_arts_gym', 'mausoleum', 'medina', 'moat/water', 'monastery/outdoor', 'mosque/indoor', 'mosque/outdoor', 'motel', 'mountain', 'mountain_snowy', 'movie_theater/indoor', 'museum/indoor', 'music_store', 'music_studio', 'nuclear_power_plant/outdoor', 'nursery', 'oast_house', 'observatory/outdoor', 'ocean', 'office', 'office_building', 'oil_refinery/outdoor', 'oilrig', 'operating_room', 'orchard', 'outhouse/outdoor', 'pagoda', 'palace', 'pantry', 'park', 'parking_garage/indoor', 'parking_garage/outdoor', 'parking_lot', 'parlor', 'pasture', 'patio', 'pavilion', 'pharmacy', 'phone_booth', 'physics_laboratory', 'picnic_area', 'pilothouse/indoor', 'planetarium/outdoor', 'playground', 'playroom', 'plaza', 'podium/indoor', 'podium/outdoor', 'pond', 'poolroom/establishment', 'poolroom/home', 'power_plant/outdoor', 'promenade_deck', 'pub/indoor', 'pulpit', 'putting_green', 'racecourse', 'raceway', 'raft', 'railroad_track', 'rainforest', 'reception', 'recreation_room', 'residential_neighborhood', 'restaurant', 'restaurant_kitchen', 'restaurant_patio', 'rice_paddy', 'riding_arena', 'river', 'rock_arch', 'rope_bridge', 'ruin', 'runway', 'sandbar', 'sandbox', 'sauna', 'schoolhouse', 'sea_cliff', 'server_room', 'shed', 'shoe_shop', 'shopfront', 'shopping_mall/indoor', 'shower', 'skatepark', 'ski_lodge', 'ski_resort', 'ski_slope', 'sky', 'skyscraper', 'slum', 'snowfield', 'squash_court', 'stable', 'stadium/baseball', 'stadium/football', 'stage/indoor', 'staircase', 'street', 'subway_interior', 'subway_station/platform', 'supermarket', 'sushi_bar', 'swamp', 'swimming_pool/indoor', 'swimming_pool/outdoor', 'synagogue/indoor', 'synagogue/outdoor', 'television_studio', 'temple/east_asia', 'temple/south_asia', 'tennis_court/indoor', 'tennis_court/outdoor', 'tent/outdoor', 'theater/indoor_procenium', 'theater/indoor_seats', 'thriftshop', 'throne_room', 'ticket_booth', 'toll_plaza', 'topiary_garden', 'tower', 'toyshop', 'track/outdoor', 'train_railway', 'train_station/platform', 'tree_farm', 'tree_house', 'trench', 'underwater/coral_reef', 'utility_room', 'valley', 'van_interior', 'vegetable_garden', 'veranda', 'veterinarians_office', 'viaduct', 'videostore', 'village', 'vineyard', 'volcano', 'volleyball_court/indoor', 'volleyball_court/outdoor', 'waiting_room', 'warehouse/indoor', 'water_tower', 'waterfall/block', 'waterfall/fan', 'waterfall/plunge', 'watering_hole', 'wave', 'wet_bar', 'wheat_field', 'wind_farm', 'windmill', 'wine_cellar/barrel_storage', 'wine_cellar/bottle_storage', 'wrestling_ring/indoor', 'yard', 'youth_hostel']
    elif dataset == 'voc2007':
        text = ['aeroplane','bicycle','bird','boat','bottle','bus',
                'car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep',
                'sofa','train','tvmonitor']
    else:
        print(' no dataset ')

    print(len(text), dataset)
    encoded_input = tokenizer(text, return_tensors='pt',padding=True, truncation=True)
    with torch.no_grad():
    # 使用模型进行编码
        outputs = model(**encoded_input)

    # 获取最后一层的hidden states
    last_hidden_states = outputs.last_hidden_state
    # print(last_hidden_states.shape,'last_hidden_states.shape')
    feature = last_hidden_states.mean(dim = 1)
    # 输出编码结果
    # print(feature[0])
    # print(feature.shape)
    np.save(dataset + '_nonorm_gpt2.npy',feature.cpu())