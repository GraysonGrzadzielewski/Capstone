import math
from collections import namedtuple


# Class representing the screen space from SMB-nes
# Modeled after SethBling's approach to representing the screen in MarIO
class ScreenSpace(object):
    BoxRadius = 5

    def __init__(self, info):
        self.info = info
        self.marioX = info["player_xpos_high"] * 256 + info["player_xpos_low"]
        self.marioY = info["player_ypos"] + 16
        self.screenX = info["screen_x"]
        self.screenY = info["screen_y"]

    def get_tile(self, dx, dy):
        x = self.marioX + dx + 8
        y = self.marioY + dy - 16
        page = math.floor(x / 256) % 2
        subx = math.floor((x % 256) / 16)
        suby = math.floor((y - 32) / 16)

        # 0x500 = dec 1280 = map_tile_1
        map_addr_index = 1 + page * 13 * 16 + suby * 16 + subx

        if suby >= 13 or suby < 0:
            return 0
        if self.info[f"map_tile_{map_addr_index}"] is not 0:
            return 1
        else:
            return 0

    def get_sprites(self):
        sprites = []
        for slot in range(1, 6):
            enemy = self.info[f"e{slot}_spawned"]
            if enemy is not 0:
                ex = self.info[f"e{slot}_xpos_high"] * 256 + self.info[f"e{slot}_xpos_low"]
                ey = 24 + self.info[f"e{slot}_ypos_low"]

                sprites.append({"x": ex, "y": ey})

        return sprites

    def get_inputs(self):
        sprites = self.get_sprites()
        inputs = []

        for dy in range(-self.BoxRadius * 16, self.BoxRadius * 16, 16):
            for dx in range(-self.BoxRadius * 16, self.BoxRadius * 16, 16):
                inputs.append(0)

                tile = self.get_tile(dx, dy)
                if tile == 1 and self.marioY + dy < 0x1B0:
                    inputs[len(inputs) - 1] = 1

                for i in range(0, len(sprites)):
                    distx = abs(sprites[i]["x"] - (self.marioX + dx))
                    disty = abs(sprites[i]["y"] - (self.marioY + dy))
                    if distx <= 8 and disty <= 8:
                        inputs[len(inputs) - 1] = -1

        return inputs

    def print_inputs(self):
        sprites = self.get_sprites()
        inputs = ""

        for dy in range(-self.BoxRadius * 16, self.BoxRadius * 16, 16):
            inputs = inputs + "\n"
            for dx in range(-self.BoxRadius * 16, self.BoxRadius * 16, 16):
                value = 0

                tile = self.get_tile(dx, dy)
                if tile == 1 and self.marioY + dy < 0x1B0:
                    value = 1

                for i in range(0, len(sprites)):
                    distx = abs(sprites[i]["x"] - (self.marioX + dx))
                    disty = abs(sprites[i]["y"] - (self.marioY + dy))
                    if distx <= 8 and disty <= 8:
                        value = -1

                if value is -1:
                    inputs = inputs + "E"
                elif value is 0:
                    inputs = inputs + "_"
                elif value is 1:
                    inputs = inputs + "B"

        print(inputs)
