import random
import pandas as pd

# Original sets
set_1 = [
    "Spaceship", "Island", "Crown", "Dragon", "Mirror", "Circus", "Garden", "Library", "Time-travel", "Castle",
    "Detective", "Music", "Painting", "Festival", "Mountain", "Robot", "Ocean", "Ghost", "Forest", "Maze",
    "Witch", "Warrior", "Desert", "Secret", "Treasure", "Moon", "Dream", "Knight", "Snow", "Volcano",
    "River", "Spell", "King", "Pirate", "Storm", "Rainbow", "City", "Star", "Jungle", "Comet",
    "Wizard", "Quest", "Sword", "Angel", "Dinosaur", "Monster", "Light", "Darkness", "Portal", "Alien",
    "Fire", "Ice", "Sun", "Vampire", "Hero", "Scientist", "Time-machine", "Adventure", "Prophecy", "Thunder",
    "Lightning", "Phoenix", "Galaxy", "Labyrinth", "Mystery", "Unicorn", "Fountain", "Cliff", "Airship",
    "Elf", "Golem", "Curse", "Giant", "Jewel", "Magic", "Fairy", "Cave", "Shipwreck", "Waterfall",
    "Meteor", "Amulet", "Battlefield", "Throne", "Griffin", "Oasis", "Plague", "Ruins", "Tower",
    "Explorer", "Hybrid", "Legend", "Teleportation"
]

set_2 = [
    "Carnival", "Enchanted", "Mansion", "Riddle", "Heirloom", "Gypsy", "Lantern", "Festival", "Old Town", "Harvest",
    "Masquerade", "Locket", "Vineyard", "Ballet", "Monastery", "Fable", "Poet", "Lighthouse", "Coral Reef", "Scroll",
    "Painter", "Quill", "Tapestry", "Ballroom", "Alchemist", "Parchment", "Forge", "Chalice", "Minstrel", "Orchard",
    "Goblet", "Falcon", "Tapestry", "Carriage", "Knight", "Baroque", "Serenade", "Bazaar", "Steed", "Silhouette",
    "Ancestor", "Labyrinth", "Marionette", "Gauntlet", "Abbey", "Shire", "Peasant", "Scepter", "Cloister", "Aqueduct",
    "Heraldry", "Chariot", "Pigeon", "Aquamarine", "Brocade", "Cobblestone", "Puppeteer", "Caravan", "Pageant", "Sonnet",
    "Dagger", "Fountain", "Gondola", "Heir", "Jester", "Keepsake", "Lyre", "Mead", "Nectar", "Oracle",
    "Pilgrimage", "Quest", "Relic", "Scribe", "Tournament", "Vellum", "Willow", "Xylophone", "Yeoman", "Zodiac",
    "Banquet", "Codex", "Dowry", "Embroidery", "Flute", "Gargoyle", "Harp", "Icon", "Jubilee", "Knot",
    "Loom", "Minaret", "Nomad", "Obelisk", "Pavilion", "Quiver", "Rampart", "Scroll", "Tapestry"
]

set_3 = [
    "Morning", "Breakfast", "Commute", "Office", "Coffee", "Email", "Meeting", "Lunch", "Park", "Fitness",
    "Grocery", "Dinner", "Television", "Neighborhood", "Reading", "Laundry", "Cooking", "Garden", "Pet", "Smartphone",
    "Internet", "Social Media", "Shopping", "Rain", "Traffic", "Weekend", "Holiday", "Birthday", "Family",
    "Friends", "Cinema", "Music", "Concert", "Festival", "Beach", "Mountain", "Camping", "Hiking", "Sports",
    "Game", "Painting", "Art", "Museum", "Travel", "Adventure", "Night", "Stars", "Sunrise", "Sunset",
    "Winter", "Spring", "Summer", "Autumn", "Snow", "Rain", "Wind", "Storm", "Flower", "Tree",
    "River", "Lake", "Ocean", "Book", "Novel", "Poetry", "Diary", "Letter", "Phone Call", "Text Message",
    "Love", "Heartbreak", "Joy", "Sadness", "Anger", "Peace", "Hope", "Dream", "Aspiration", "Career",
    "Education", "School", "College", "University", "Teacher", "Student", "Learning", "Skill", "Hobby",
    "Craft", "Baking", "Recipe", "Health", "Exercise", "Yoga", "Meditation", "Relaxation", "Spa",
    "Massage", "Beauty", "Fashion"
]

# Create an empty DataFrame to store the results
df = pd.DataFrame(columns=["Keywords"])
# single_instruction_df = pd.DataFrame(columns=['Instruction', 'Constraints', 'BaseStory', 'SelectedConstraints', 'Number_of_Constraints', 'Final_Prompt', 'FinalGeneratedStory'])

# Repeat the word selection process 10 times
for i in range(9):
    # Randomly select 10 elements from each set
    selected_set_1 = random.sample(set_1, 10)
    selected_set_2 = random.sample(set_2, 10)
    selected_set_3 = random.sample(set_3, 10)

    # Remove selected words from original sets
    for word in selected_set_1:
        set_1.remove(word)
    for word in selected_set_2:
        set_2.remove(word)
    for word in selected_set_3:
        set_3.remove(word)

    # Combine the selected elements into one list
    selected_words = selected_set_1 + selected_set_2 + selected_set_3

    # Add the selected words to the DataFrame
    df.loc[i] = [', '.join(selected_words)]

# Export the DataFrame to a CSV file
df.to_csv("direction1/selected_keywords.csv", index=False)
print("Set 1:", set_1)
print("Set 2:", set_2)
print("Set 3:", set_3)