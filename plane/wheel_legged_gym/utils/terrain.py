# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from wheel_legged_gym.envs.base.legged_robot_config import LeggedRobotCfg


class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", "plane"]:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [
            np.sum(cfg.terrain_proportions[: i + 1])
            for i in range(len(cfg.terrain_proportions))
        ]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size / self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:
            self.randomized_terrain()

        self.heightsamples = self.height_field_raw
        if self.type == "trimesh":
            self.vertices, self.triangles = (
                terrain_utils.convert_heightfield_to_trimesh(
                    self.height_field_raw,
                    self.cfg.horizontal_scale,
                    self.cfg.vertical_scale,
                    self.cfg.slope_treshold,
                )
            )

    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)

    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop("type")
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain(
                "terrain",
                width=self.width_per_env_pixels,
                length=self.width_per_env_pixels,
                vertical_scale=self.vertical_scale,
                horizontal_scale=self.horizontal_scale,
            )

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)

    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=self.width_per_env_pixels,
            length=self.width_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale,
        )
        slope = difficulty * 0.5
        random_height = 0.05 + difficulty * 0.05
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.1
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty == 0 else 0.1
        gap_size = 1.0 * difficulty
        pit_depth = 1.0 * difficulty
        if self.cfg.custom_terrain_mode == "bidirectional_focus":
            # Use every column for the measured two-level obstacle: the first
            # half starts high and descends, the second half starts low and climbs.
            curb_double_drop_terrain(terrain, reverse=choice >= 0.5)
            return terrain
        if choice < self.proportions[0]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=0, platform_size=3.0)
        elif choice < self.proportions[1]:
            if (
                choice
                < self.proportions[0] + (self.proportions[1] - self.proportions[0]) / 2
            ):
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(
                terrain, slope=slope, platform_size=3.0
            )
        elif choice < self.proportions[2]:
            if (
                choice
                < self.proportions[1] + (self.proportions[2] - self.proportions[1]) / 2
            ):
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(
                terrain, slope=slope * 0.5, platform_size=3.0
            )
            terrain_utils.random_uniform_terrain(
                terrain,
                min_height=-random_height,
                max_height=random_height,
                step=0.005,
                downsampled_scale=0.2,
            )
        elif choice < self.proportions[4]:
            if choice < self.proportions[3]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(
                terrain, step_width=0.7, step_height=step_height, platform_size=4.0
            )
        elif choice < self.proportions[5]:
            # Split the final 10% advanced-obstacle category evenly. During
            # ordinary training this retains the original discrete obstacles;
            # during the bidirectional stage it supplies descending and
            # reverse-climb versions of the measured curb/drop course.
            custom_split = self.proportions[4] + 0.5 * (
                self.proportions[5] - self.proportions[4]
            )
            if self.cfg.custom_terrain_mode == "bidirectional":
                curb_double_drop_terrain(terrain, reverse=choice >= custom_split)
            elif choice < custom_split:
                num_rectangles = 20
                rectangle_min_size = 1.0
                rectangle_max_size = 2.0
                terrain_utils.discrete_obstacles_terrain(
                    terrain,
                    discrete_obstacles_height,
                    rectangle_min_size,
                    rectangle_max_size,
                    num_rectangles,
                    platform_size=3.0,
                )
            else:
                curb_double_drop_terrain(terrain)
        elif choice < self.proportions[6]:
            terrain_utils.stepping_stones_terrain(
                terrain,
                stone_size=stepping_stones_size,
                stone_distance=stone_distance,
                max_height=0.0,
                platform_size=4.0,
            )
        elif choice < self.proportions[7]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.0)
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.0)

        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length / 2.0 - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length / 2.0 + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width / 2.0 - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width / 2.0 + 1) / terrain.horizontal_scale)
        env_origin_z = (
            np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
        )
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]


def curb_double_drop_terrain(
    terrain,
    approach_length=1.5,
    curb_height=0.05,
    curb_top_length=0.20,
    first_drop=0.15,
    middle_platform_length=0.50,
    second_drop=0.20,
    reverse=False,
):
    """Create the measured curb/drop profile symmetrically about the spawn area.

    From the central platform, either travel direction encounters a 50 mm curb,
    a 200 mm curb top, a 150 mm drop, a 500 mm platform, and a final 200 mm
    drop. The mirrored layout lets both positive and negative velocity commands
    exercise the same obstacle sequence.
    """
    horizontal_scale = terrain.horizontal_scale
    vertical_scale = terrain.vertical_scale

    approach_cells = max(1, int(round(approach_length / horizontal_scale)))
    curb_cells = max(1, int(round(curb_top_length / horizontal_scale)))
    middle_cells = max(
        1, int(round(middle_platform_length / horizontal_scale))
    )

    center_x = terrain.length // 2
    positive_curb_start = center_x + approach_cells
    positive_curb_end = positive_curb_start + curb_cells
    positive_middle_end = positive_curb_end + middle_cells
    negative_curb_end = center_x - approach_cells
    negative_curb_start = negative_curb_end - curb_cells
    negative_middle_start = negative_curb_start - middle_cells

    if negative_middle_start < 0 or positive_middle_end > terrain.length:
        raise ValueError("curb/drop profile does not fit inside the terrain tile")

    curb_raw = int(round(curb_height / vertical_scale))
    middle_raw = int(round((curb_height - first_drop) / vertical_scale))
    final_raw = int(
        round((curb_height - first_drop - second_drop) / vertical_scale)
    )

    if reverse:
        # Spawn on the -0.30 m lower surface. Travelling in either direction
        # climbs 0.20 m, crosses 0.50 m, climbs 0.15 m, crosses the curb top,
        # then drops 0.05 m back to the surrounding zero-height surface.
        terrain.height_field_raw[:, :] = final_raw

        positive_middle_start = positive_curb_start
        positive_middle_end = positive_middle_start + middle_cells
        positive_curb_start = positive_middle_end
        positive_curb_end = positive_curb_start + curb_cells
        negative_middle_end = negative_curb_end
        negative_middle_start = negative_middle_end - middle_cells
        negative_curb_end = negative_middle_start
        negative_curb_start = negative_curb_end - curb_cells

        if negative_curb_start < 0 or positive_curb_end > terrain.length:
            raise ValueError("reverse curb/climb profile does not fit inside the tile")

        terrain.height_field_raw[
            positive_middle_start:positive_middle_end, :
        ] = middle_raw
        terrain.height_field_raw[positive_curb_start:positive_curb_end, :] = curb_raw
        terrain.height_field_raw[positive_curb_end:, :] = 0
        terrain.height_field_raw[
            negative_middle_start:negative_middle_end, :
        ] = middle_raw
        terrain.height_field_raw[negative_curb_start:negative_curb_end, :] = curb_raw
        terrain.height_field_raw[:negative_curb_start, :] = 0
    else:
        terrain.height_field_raw[:, :] = 0

        # Positive-x descending course.
        terrain.height_field_raw[positive_curb_start:positive_curb_end, :] = curb_raw
        terrain.height_field_raw[positive_curb_end:positive_middle_end, :] = middle_raw
        terrain.height_field_raw[positive_middle_end:, :] = final_raw

        # Mirrored negative-x descending course.
        terrain.height_field_raw[negative_curb_start:negative_curb_end, :] = curb_raw
        terrain.height_field_raw[negative_middle_start:negative_curb_start, :] = middle_raw
        terrain.height_field_raw[:negative_middle_start, :] = final_raw


def gap_terrain(terrain, gap_size, platform_size=1.0):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size

    terrain.height_field_raw[
        center_x - x2 : center_x + x2, center_y - y2 : center_y + y2
    ] = -1000
    terrain.height_field_raw[
        center_x - x1 : center_x + x1, center_y - y1 : center_y + y1
    ] = 0


def pit_terrain(terrain, depth, platform_size=1.0):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth
