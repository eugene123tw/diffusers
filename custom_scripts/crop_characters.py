from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2


def crop_character_from_sprite(sprite, size):
    """
    Crop characters from a sprite sheet.

    Args:
        sprite (numpy.ndarray): The sprite sheet image.
        size (int): The size of each character in the sprite sheet.

    Returns:
        list: A list of cropped character images.
    """
    height, width, _ = sprite.shape
    characters = []
    for y in range(0, height, size):
        for x in range(0, width, size):
            character = sprite[y:y + size, x:x + size]
            if np.unique(character[..., -1]).size == 1:
                # Skip if the character is a single color (e.g., transparent)
                continue 
            characters.append(character)
    return characters


if __name__ == "__main__":
    sprites = [
        {
            "path": Path("/home/yuchunli/_DATASET/pixel_dataset/animals.png"),
            "pixel_size": 32,
        },
        {
            "path": Path("/home/yuchunli/_DATASET/pixel_dataset/monsters.png"),
            "pixel_size": 32,
        },
        {
            "path": Path("/home/yuchunli/_DATASET/pixel_dataset/rogues.png"),
            "pixel_size": 32,
        },
    ]

    sprite_np = "/home/yuchunli/_DATASET/pixel_dataset/sprites/sprites_1788_16x16.npy"
    save_path = Path("/home/yuchunli/_DATASET/pixel_dataset/crops")
    
    i = 0
    for sprite in sprites:
        sprite_path = sprite["path"]
        pixel_size = sprite["pixel_size"]
        sprite = cv2.imread(str(sprite_path))
        sprite = cv2.cvtColor(sprite, cv2.COLOR_BGR2RGB)
        characters = crop_character_from_sprite(sprite, pixel_size)
        for character in characters:
            # Save the character image
            save_name = f"character_{i}.png"
            cv2.imwrite(str(Path(save_path) / save_name), cv2.cvtColor(character, cv2.COLOR_RGB2BGR))
            print(f"Saved {save_path}")
            i += 1
    

    # plt.imshow(sprite[0])
    # plt.show()