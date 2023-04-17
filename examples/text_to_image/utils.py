from diffusers import StableDiffusionPipeline
import torch
import os
import json
from math import ceil, sqrt
from PIL import Image
import random

class TaskVector():
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                pretrained_state_dict = torch.load(pretrained_checkpoint).state_dict()
                finetuned_state_dict = torch.load(finetuned_checkpoint).state_dict()
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
    
    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            pretrained_model = torch.load(pretrained_checkpoint)
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model

def save_image(pipeline, prompt, path):
    output = pipeline(prompt=prompt)
    image = output.images[0]
    nsfw = output.nsfw_content_detected
    image.save(path)
    return nsfw

def concat_images_in_square_grid(folder_path, prompt, output_path='output.png'):
    # Get a list of all .png files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png') and prompt in f]

    # Calculate the number of rows and columns for the grid
    total_images = len(image_files)
    grid_size = ceil(sqrt(total_images))
    rows, cols = grid_size, grid_size

    # Open the first image to get its dimensions
    first_image = Image.open(os.path.join(folder_path, image_files[0]))
    image_width, image_height = first_image.size

    # Create a blank image for the grid
    grid_width, grid_height = image_width * cols, image_height * rows
    grid = Image.new('RGB', (grid_width, grid_height))

    # Iterate through the images and add them to the grid
    for index, image_file in enumerate(image_files):
        image = Image.open(os.path.join(folder_path, image_file))
        x = (index % cols) * image_width
        y = (index // cols) * image_height
        grid.paste(image, (x, y))

    # Save the output image
    grid.save(output_path)

def get_random_prompt(artist_style):
     
    prompts = [
        "Whimsical twilight harbor",
        "Ethereal golden wheat field",
        "Serene underwater grotto",
        "Dreamy sun-kissed vineyard",
        "Majestic mountain pass",
        "Solitary autumn lane",
        "Luminous foggy forest",
        "Tranquil desert oasis",
        "Vibrant city skyline",
        "Enchanted garden path",
        "Radiant riverside village",
        "Celestial starry sky",
        "Melancholic rain-soaked street",
        "Timeless ancient ruins",
        "Nostalgic country farmhouse",
        "Opulent royal court",
        "Dazzling carnival procession",
        "Sublime cliffside retreat",
        "Icy winter wonderland",
        "Mysterious moonlit graveyard",
        "Arcadian countryside meadow",
        "Fiery volcanic eruption",
        "Graceful swan-filled lake",
        "Psychedelic jungle flora",
        "Idyllic beach sunset",
        "Baroque palace garden",
        "Energetic bustling market",
        "Rustic old mill",
        "Airy mountaintop vista",
        "Cozy snow-covered cabin",
        "Wildflower-speckled valley",
        "Elegant ballroom dance",
        "Surreal underwater city",
        "Daylit forest clearing",
        "Stormy shipwreck scene",
        "Neon-lit urban nightlife",
        "Romantic Venetian canal",
        "Warm alpine meadow",
        "Cool coastal cliff",
        "Pastoral sheep grazing",
        "Tangled ivy-covered tower",
        "Lush tropical waterfall",
        "Illuminated stained glass window",
        "Softly lit library",
        "Exquisite cherry blossom grove",
        "Nocturnal woodland creatures",
        "Monochrome moonlit beach",
        "Minimalistic geometric shapes",
        "Fleeting cloud formations",
        "Ornate medieval cathedral",
        "Vast rolling sand dunes",
        "Heavenly mountain monastery",
        "Bold geometric cityscape",
        "Dappled sunlight in a forest",
        "Gleaming crystalline cavern",
        "Shadowy abandoned mansion",
        "Quaint cobblestone street",
        "Colorful coral reef",
        "Silver moonlit desert",
        "Sunlit stained glass",
        "Golden harvest celebration",
        "Glistening frost-covered meadow",
        "Windswept ocean waves",
        "Gloomy underground catacombs",
        "Enchanted fairy glen",
        "Sparkling frozen waterfall",
        "Bustling train station",
        "Celestial planetary alignment",
        "Dynamic dancer's performance",
        "Charming countryside picnic",
        "Dreamlike floating islands",
        "Crisp morning dew",
        "Majestic ancient oak tree",
        "Spiraling cosmic nebula",
        "Serene serpentine river bend",
        "Dusky twilight marsh",
        "Vibrant kaleidoscopic patterns",
        "Delicate butterfly garden",
        "Ancient moss-covered bridge",
        "Towering redwood forest",
        "Sun-drenched citrus grove",
        "Enigmatic midnight masquerade",
        "Overgrown secret garden",
        "Warmly lit lantern festival",
        "Dramatic lighthouse storm",
        "Intricate spider web",
        "Swirling autumn leaves",
        "Roaring thunderstorm at sea",
        "Tranquil zen rock garden",
        "Shimmering peacock feathers",
        "Fanciful dragon's lair",
        "Misty mountain lake",
        "Regal peony bloom",
        "Glowing firefly dance",
        "Meditative bamboo grove",
        "Spellbinding comet's passage",
        "Soaring hot air balloons",
        "Blazing desert mirage",
        "Timeless ancient colosseum",
        "Starlit cosmic whirlpool",
        "Sunburst mountain range",
        "Frosty enchanted forest",
        "Dynamic ocean storm",
        "Lavender-scented hills",
        "Aurora-lit night sky",
        "Secret garden hideaway",
        "Gilded palace interior",
        "Candlelit monastery chamber",
        "Colorful bustling bazaar",
        "Serendipitous waterfall discovery",
        "Wind-swept seaside cliffs",
        "Moonlit cypress grove",
        "Sundrenched Tuscan villa",
        "Dreamy ancient library",
        "Majestic oceanic abyss",
        "Dramatic lightning-struck tree",
        "Mystical fog-enshrouded island",
        "Autumnal lakeside reflection",
        "Benevolent guardian angel",
        "Whispering willow-lined path",
        "Breathtaking alpine vista",
        "Ephemeral cherry tree blossoms",
        "Emerald rainforest canopy",
        "Rustic countryside windmill",
        "Charming cobblestone bridge",
        "Sleepy sunflower meadow",
        "Glimmering ice palace",
        "Ancient mossy stone circle",
        "Abandoned shipwreck cove",
        "Lost city of Atlantis",
        "Grandiose celestial observatory",
        "Cascading hidden waterfall",
        "Stoic lighthouse sentinel",
        "Vibrant underwater menagerie",
        "Majestic desert pyramids",
        "Windswept lavender fields",
        "Intimate garden sanctuary",
        "Flickering firelit hearth",
        "Ethereal northern lights",
        "Sun-kissed olive groves",
        "Sublime mountain temple",
        "Wistful weeping willow",
        "Pristine coral atoll",
        "Peaceful Japanese tea garden",
        "Fabled dragon mountain",
        "Glistening dew-kissed meadow",
        "Rain-soaked Parisian cafe",
        "Majestic Gothic cathedral",
        "Resplendent summer palace",
        "Serenading nightingale grove",
        "Twinkling bioluminescent bay",
        "Secluded moonlit beach",
        "Starry night carnival",
        "Spectral haunted forest",
        "Gleaming sunflower field",
        "Elusive forest nymph",
        "Flourishing koi pond",
        "Time-worn ancient battleground",
        "Exotic desert caravan",
        "Verdant rainforest temple",
        "Golden autumn forest",
        "Gentle babbling brook",
        "Crumbling castle ruins",
        "Luminous jellyfish bloom",
        "Bucolic vineyard hills",
        "Enigmatic monolith in the desert",
        "Lush oasis at twilight",
        "Astonishing celestial event",
        "Snow-capped mountain peak",
        "Dramatic cliffside city",
        "Awe-inspiring fjord",
        "Mythical phoenix rising",
        "Tapestry of wildflowers",
        "Fragrant rose garden",
        "Snowy village at dusk",
        "Rays of sunlight through clouds",
        "Sailing under a blood moon",
        "Rainbow after the storm",
        "Craggy coastal seascape",
        "Misty moorland heather",
        "Dancing fire spirits",
        "Harmonious orchestra of nature",
        "Enchanted forest glade",
        "Historic medieval market",
        "Golden wheat field at sunset",
        "Roaring campfire under the stars",
        "Sprawling Roman aqueduct",
        "Eerie ghost ship",
        "Bejeweled peacock throne",
        "Frolicking dolphins at dawn",
        "Lost city in the clouds",
        "Towering thunderhead clouds",
        "Mysterious labyrinth garden",
        "Azure mountain lake",
        "Glorious sunrise over the ocean",
        "Fluttering hummingbird frenzy",
        "Frost-covered spider web",
        "Secluded hilltop monastery",
        "Whispering sand dunes",
        "Otherworldly alien landscape",
        "Majestic eagle's nest",
        "Silent snowfall on pines",
        "Lunar landscape under starlight",
        "Frosty riverbank morning",
        "Soothing thermal springs",
        "Sunrise on a misty lake",
        "Playful otters in a stream",
        "Busy bee-filled apiary",
        "Sunset over a volcanic crater",
        "Crimson maple forest",
        "Dappled forest sunlight",
        "Astral dreamscape visions",
        "Wild mustang stampede",
        "Velvety moss-covered grotto",
        "Golden city of El Dorado",
        "Rugged highland moors",
        "Glowing lantern-lit procession",
        "Celestial angelic choir",
        "Iridescent mineral deposit",
        "Majestic white-tailed deer",
        "Ancient petroglyphs in a canyon",
        "Ivory tower in the clouds",
        "Dew-laden spiderwebs at dawn",
        "Crumbling Mayan ruins",
        "Cosmic planetary dance",
        "Solitary desert cactus",
        "Frothy ocean surf",
        "Opalescent tide pools",
        "Cavern of sparkling gems",
        "Radiant city at dusk",
        "Feathery snowflake patterns",
        "Vivid wildflower meadow",
        "Sundappled park promenade",
        "Quaint seaside cottages",
        "Cascading rainforest rivulets",
        "Celestial clockwork",
        "Moonlit reflection on the lake",
        "Dreamy cloud palace",
        "Ancient gnarled olive tree",
        "Fragrant lilac grove",
        "Majestic stag in the glen",
        "Blooming cactus garden",
        "Deserted island paradise",
        "Galloping horses in the surf",
        "Shadowy forest path",
        "Pristine mountain spring",
        "Time-lapse of a bustling city",
        "Glacial ice cave",
        "Ethereal morning mist",
        "Sleeping feline family",
        "Graceful ballet performance",
        "Swaying fields of barley",
        "Elegantly tangled vines",
        "Rippling desert mirage",
        "Chorus of birds at dawn",
        "Bubbling hot springs",
        "Flock of birds in flight",
        "Gnarled ancient tree",
        "Rays of moonlight through clouds",
        "Placid mountain tarn",
        "Enchanted will-o'-the-wisp",
        "Gilded Venetian masquerade",
        "Stunning arctic aurora",
        "Nautical shipyard at dusk",
        "Sunlit autumn orchard",
        "Lost treasure trove",
        "Emerald rice terraces",
        "Frenzied ant colony",
        "Dazzling fireworks display",
        "Migrating monarch butterflies",
        "Golden summer hayfield",
        "Gloaming over the bay",
        "Illuminated bioluminescent cave",
        "Sleepy woodland creatures",
        "Billowing storm clouds",
        "Rustic thatched-roof village",
        "Frozen waterfall cascade",
        "Scarlet macaws in flight",
        "Windswept tundra landscape",
        "Pristine white sand beach",
        "Festive holiday market",
        "Vibrant urban graffiti",
        "Jubilant spring festival",
        "Breathtaking mountain panorama",
        "Grazing highland cattle",
        "Undulating sandstone formations",
        "Ghostly ship graveyard",
        "Sun-dappled bamboo forest",
        "Coral-encrusted shipwreck",
        "Lunar eclipse in the night sky",
        "Ripening vineyard grapes",
        "Melodic wind chimes",
        "Mystical tree of life",
        "Cacophony of city life"]


    return "a painting of " + random.choice(prompts) + f" in the style of {artist_style}"