from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Generator

import math
from enum import IntEnum, auto

import pygame
import pymunk
import pytmx

WIDTH, HEIGHT = 1000, 700

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
PURPLE = (187, 150, 255)

# Game settings
BLOCK_SIZE = 50
COLLISION_CHECK_MULTIPLIER = 0.5
FPS = 120

# Indicator settings
INDICATOR_ALPHA = 128  # Transparency value (0-255)
INDICATOR_RADIUS = BLOCK_SIZE * COLLISION_CHECK_MULTIPLIER / 2
INDICATOR_COLOR = BLACK
CAPTURE_INDICATOR_COLOR = RED

# Physics constants
GRAVITY = (0, 980)
FRICTION = 1
ELASTICITY = 0.5
DAMPING = 1
PHYSICS_ITERATIONS = int(6000 / FPS)


# Game states
class State(IntEnum):
    MENU = auto()
    LEVEL_SELECT = auto()
    PLAYING = auto()


class Collision(IntEnum):
    DEFAULT = auto()
    PIECE = auto()
    ENEMY = auto()
    SPIKE = auto()
    BOUNCY = auto()
    PASSTHROUGH = auto()
    DISAPPEARING = auto()
    TELEPORTER = auto()


class MovementIndicator(NamedTuple):
    x: int
    y: int
    obstacle: Part


class GameEngine:
    """Main game engine that handles physics, rendering and game state."""

    def __init__(self, screen: pygame.Surface) -> GameEngine:
        self.screen = screen
        self.current_level = 1
        self.game_state = State.MENU

        self.fade_surface = pygame.Surface((WIDTH, HEIGHT))
        self.fade_surface.fill(BLACK)
        self.fade_alpha = 0
        self.fade_step = 0

        pygame.mixer_music.load("./audio/game_music.mp3")
        pygame.mixer_music.set_volume(0)
        pygame.mixer_music.play(loops=-1, fade_ms=5000)
        pygame.mixer_music.set_volume(0.10)

        # Game elements
        self.backgrounds: list[pygame.Surface] = []
        self.statics: list[Part] = []
        self.kinematics: list[Part] = []
        self.dynamics: list[Part] = []
        self.selected: ChessPiece | None = None
        self.move_indicators: list[MovementIndicator] = []

        # Initialize physics
        self.space = pymunk.Space()
        self.space.gravity = GRAVITY
        self.space.damping = DAMPING
        self.space.iterations = PHYSICS_ITERATIONS
        self.collision_handler = self.space.add_wildcard_collision_handler(
            Collision.PIECE,
        )
        self.collision_handler.begin = self.piece_on_collision
        self.collision_handler.data["engine"] = self

        self.clock = pygame.time.Clock()

        # Menu elements
        self.start_bg = pygame.image.load("menu_assets/start_bg.png")
        self.level_select_bg = pygame.image.load("menu_assets/level_bg.png")
        self.start_button = pygame.image.load("menu_assets/play_btn.png")

        self.sidebar_icon1 = pygame.image.load(
            "menu_assets/sidebar_icon.png",
        ).convert_alpha()
        self.start_button_rect = self.start_button.get_rect(center=(750, 420))
        self.sidebar_rect = pygame.Rect((10, 10), self.sidebar_icon1.get_size())
        self.sidebar_icon2 = pygame.image.load(
            "menu_assets/x_sidebar.png",
        ).convert_alpha()
        self.sidebar_bg = pygame.image.load(
            "menu_assets/sidebar_bg.png",
        ).convert_alpha()

        # Menu options and state
        self.sidebar_options = ["Restart", "Level Selection", "Exit"]
        self.sidebar_font = pygame.font.SysFont("twcen", 25)
        self.sidebar_visible = False
        self.sidebar_width = 300
        self.sidebar_speed = 40
        self.sidebar_x = -self.sidebar_width
        self.selected_option = None

        # Level selection buttons
        self.level_buttons: list[tuple[pygame.Surface, pygame.Rect]] = []
        for i in range(1, 11):
            btn_img = pygame.image.load(f"menu_assets/lvl{i}_btn.png")
            btn_rect = btn_img.get_rect()
            self.level_buttons.append((btn_img, btn_rect))

        # Position level buttons
        start_x = 100
        start_y = 200
        gap_x = 150
        gap_y = 130

        for idx, (_, rect) in enumerate(self.level_buttons):
            row = idx // 5
            col = idx % 5
            rect.topleft = (start_x + col * gap_x, start_y + row * gap_y)

        self.hud_font = pygame.font.SysFont(None, 24)

        self.indicator_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

        self.tile_factory = {
            0: lambda x, y, p: King(x, y, self, properties=p),  # light king
            1: lambda x, y, p: Queen(x, y, self, properties=p),  # light queen
            2: lambda x, y, p: Bishop(x, y, self, properties=p),  # light bishop
            3: lambda x, y, p: Knight(x, y, self, properties=p),  # light knight
            4: lambda x, y, p: Rook(x, y, self, properties=p),  # light rook
            5: lambda x, y, p: Pawn(x, y, self, properties=p),  # light pawn
            6: lambda x, y, p: King(x, y, self, properties=p, enemy=True),  # dark king
            7: lambda x, y, p: King(
                x,
                y,
                self,
                properties=p,
                enemy=True,
            ),  # dark king (reversed)
            8: lambda x, y, p: Queen(
                x,
                y,
                self,
                properties=p,
                enemy=True,
            ),  # dark queen
            9: lambda x, y, p: Bishop(
                x,
                y,
                self,
                properties=p,
                enemy=True,
            ),  # dark bishop
            10: lambda x, y, p: Knight(
                x,
                y,
                self,
                properties=p,
                enemy=True,
            ),  # dark knight
            11: lambda x, y, p: Rook(x, y, self, properties=p, enemy=True),  # dark rook
            12: lambda x, y, p: Pawn(x, y, self, properties=p, enemy=True),  # dark pawn
            20: lambda x, y, p: ThroughFloor(x, y, self, properties=p),
            21: lambda x, y, p: IceBlock(x, y, self, properties=p),
            22: lambda x, y, p: SpikeLeft(x, y, self, properties=p),
            23: lambda x, y, p: SpikeUp(x, y, self, properties=p),
            24: lambda x, y, p: Bouncy(x, y, self, properties=p),
            25: lambda x, y, p: HorizontalPlatform(x, y, self, properties=p),
            26: lambda x, y, p: VerticalPlatform(x, y, self, properties=p),
            27: lambda x, y, p: DisappearingBlock(x, y, self, properties=p),
            28: lambda x, y, p: Teleporter(x, y, self, properties=p),
        }

    def add_part(self, part: Part | ChessPiece) -> None:
        """Add a part to the appropriate list and physics space."""
        match part.body.body_type:
            case pymunk.Body.STATIC:
                self.statics.append(part)
            case pymunk.Body.KINEMATIC:
                self.kinematics.append(part)
            case pymunk.Body.DYNAMIC:
                self.dynamics.append(part)

        self.space.add(part.body, part.shape)

    def remove_part(self, part: Part) -> None:
        """Remove a part from the game."""
        if self.selected == part:
            self.selected = None

        if part in self.statics:
            self.statics.remove(part)
        elif part in self.kinematics:
            self.kinematics.remove(part)
        elif part in self.dynamics:
            self.dynamics.remove(part)
        else:
            msg = "Part not found!"
            raise RuntimeError(msg)

        if part.body in self.space.bodies and part.shape in self.space.shapes:
            self.space.remove(part.body, part.shape)

    def clear_level(self) -> None:
        """Clear backgrounds, statics, and pieces."""
        self.backgrounds.clear()
        for part in self.statics[:]:
            self.remove_part(part)
        for part in self.kinematics[:]:
            self.remove_part(part)
        for part in self.dynamics[:]:
            self.remove_part(part)

    def is_win_condition(self) -> bool:
        return not any(
            piece.enemy if isinstance(piece, ChessPiece) else True
            for piece in self.dynamics
        )

    def find_part(self, part_id: pymunk.Shape) -> Part | None:
        search_list: list[Part] = []
        match part_id.body.body_type:
            case pymunk.Body.DYNAMIC:
                search_list = self.dynamics
            case pymunk.Body.KINEMATIC:
                search_list = self.kinematics
            case pymunk.Body.STATIC:
                search_list = self.statics

        for part in search_list:
            if part.shape == part_id:
                return part

        return None

    @staticmethod
    def piece_on_collision(
        arbiter: pymunk.Arbiter, _space: pymunk.Space, data: dict
    ) -> bool:
        piece_shape, other_shape = arbiter.shapes

        piece: ChessPiece = data["engine"].find_part(piece_shape)
        other: Part = data["engine"].find_part(other_shape)

        match other_shape.collision_type:
            case Collision.ENEMY:
                data["engine"].remove_part(other)
                return False
            case Collision.SPIKE:
                data["engine"].remove_part(piece)
                return False
            case Collision.BOUNCY:
                other: Bouncy
                bounce_height = other.bounce_height * BLOCK_SIZE
                bounce_velocity = math.sqrt(bounce_height * 2 * GRAVITY[1]) + 1
                piece_shape.body.velocity = (0, -bounce_velocity)
                return False
            case Collision.PASSTHROUGH:
                contacts: pymunk.ContactPointSet = arbiter.contact_point_set
                return contacts.normal.y >= 0
            case Collision.DISAPPEARING:
                other: DisappearingBlock
                other.fading = True
                return True
            case Collision.TELEPORTER:
                return False

        return True

    def generate_level(self, level_num: int) -> None:
        """Load and create level tiles from TMX file."""
        self.current_level = level_num
        level_path = f"./levels/level{level_num}.tmx"
        try:
            level_data = pytmx.load_pygame(level_path)
            for layer in level_data.layers:
                if isinstance(layer, pytmx.TiledImageLayer):
                    self.backgrounds.append(
                        level_data.images[level_data.tiledgidmap[(layer.gid)]],
                    )
                elif isinstance(layer, pytmx.TiledTileLayer):
                    self._process_tile_layer(layer, level_data)
                elif isinstance(layer, pytmx.TiledObjectGroup):
                    self._process_object_layer(layer, level_data)

            # Switch to playing state
            self.game_state = State.PLAYING

        except Exception as e:
            print(e)
            self.game_state = State.LEVEL_SELECT

    def _process_tile_layer(
        self,
        layer: pytmx.TiledTileLayer,
        level_data: pytmx.TiledMap,
    ) -> None:
        for x, y, gid in layer:
            properties = level_data.get_tile_properties_by_gid(gid)
            if properties is None:
                continue

            tile_id = properties["id"]
            tile = None

            if tile_id in (13, 14, 15, 16, 17, 18, 19, 28):
                tile = Part(
                    x * BLOCK_SIZE,
                    y * BLOCK_SIZE,
                    body_type=pymunk.Body.STATIC,
                    game_engine=self,
                    properties=properties,
                )

            elif tile_id in self.tile_factory:
                tile = self.tile_factory[tile_id](
                    x * BLOCK_SIZE,
                    y * BLOCK_SIZE,
                    properties,
                )

            if tile:
                tile.sprite = level_data.get_tile_image_by_gid(gid)

    def _process_object_layer(
        self,
        layer: pytmx.TiledObjectGroup,
        level_data: pytmx.TiledMap,
    ) -> None:
        for obj in layer:
            obj: pytmx.TiledObject
            x, y = obj.x, obj.y
            obj_id = obj.properties["id"]
            tile: Part | None = None

            if obj_id in self.tile_factory:
                tile = self.tile_factory[obj_id](x, y, obj.properties)
                tile.sprite = level_data.get_tile_image_by_gid(obj.gid)

    def handle_events(self) -> bool:
        """Process pygame events."""
        running = True
        for event in pygame.event.get():
            match event.type:
                case pygame.QUIT:
                    running = False
                case pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        running = self._handle_mouse_click(pygame.mouse.get_pos())
                case pygame.MOUSEMOTION:
                    running = self._handle_mouse_motion(event.pos)
                case pygame.KEYDOWN:
                    running = self._handle_keydown(event.key)

        return running

    def _handle_mouse_click(self, mouse_pos: tuple[int, int]) -> bool:
        """Process mouse click events."""
        match self.game_state:
            case State.MENU:
                return self._handle_menu_click(mouse_pos)
            case State.LEVEL_SELECT:
                return self._handle_level_select_click(mouse_pos)
            case State.PLAYING:
                return self._handle_playing_click(mouse_pos)

        return True

    def _handle_menu_click(self, mouse_pos: tuple[int, int]) -> bool:
        if self.start_button_rect.collidepoint(mouse_pos):
            self.fade_transition()
            self.game_state = State.LEVEL_SELECT
        return True

    def _handle_level_select_click(self, mouse_pos: tuple[int, int]) -> bool:
        for i, (_, rect) in enumerate(self.level_buttons):
            if rect.collidepoint(mouse_pos):
                self.fade_transition()
                self.clear_level()
                self.generate_level(i + 1)
                break
        return True

    def _handle_playing_click(self, mouse_pos: tuple[int, int]) -> bool:
        if self.sidebar_visible:
            if self.sidebar_rect.collidepoint(mouse_pos) or not pygame.Rect(
                -self.sidebar_x,
                0,
                self.sidebar_width,
                HEIGHT,
            ).collidepoint(mouse_pos):
                self.sidebar_visible = False
                return True

            for i, option in enumerate(self.sidebar_options):
                option_rect = pygame.Rect(
                    (self.sidebar_x + 30, 150 + i * 40),
                    self.sidebar_font.size(option),
                )
                if option_rect.collidepoint(mouse_pos):
                    return self.handle_option_click(option)

        elif self.sidebar_rect.collidepoint(mouse_pos):
            self.sidebar_visible = True
            return True

        moved = False
        if self.selected:
            for indicator_x, indicator_y, obstacle in self.move_indicators:
                # Create a rect for hit detection on indicators
                indicator_rect = pygame.Rect(
                    indicator_x - (BLOCK_SIZE / 2),
                    indicator_y - (BLOCK_SIZE / 2),
                    BLOCK_SIZE,
                    BLOCK_SIZE,
                )

                if indicator_rect.collidepoint(mouse_pos):
                    # Check if this is a capture move
                    if isinstance(obstacle, ChessPiece):
                        self.remove_part(obstacle)

                    if isinstance(obstacle, Teleporter):
                        self.selected.move_to(
                            obstacle.destination_x * BLOCK_SIZE,
                            obstacle.destination_y * BLOCK_SIZE,
                        )
                        moved = True
                        if isinstance(self.selected, Pawn):
                            self.selected.moved = True
                        self.selected = None
                        break

                    # Teleport selected piece to this indicator position
                    self.selected.move_to(indicator_x, indicator_y, velocity=True)
                    moved = True
                    if isinstance(self.selected, Pawn):
                        self.selected.moved = True
                    self.selected = None
                    break

            if not moved:
                self.selected = None

        # Check if clicked on a piece
        if self.selected is None and not moved:
            for piece in self.dynamics:
                if not piece.enemy and piece.rect.collidepoint(mouse_pos):
                    self.selected = piece
                    break
        return True

    def handle_option_click(self, option: str) -> bool:
        match option:
            case "Exit":
                return False
            case "Restart":
                self.clear_level()
                self.generate_level(self.current_level)
            case "Level Selection":
                self.fade_transition()
                self.clear_level()
                self.game_state = State.LEVEL_SELECT
                self.sidebar_x = -self.sidebar_width

        self.sidebar_visible = False

        return True

    def _handle_mouse_motion(self, pos: tuple[int, int]) -> None:
        if not self.sidebar_visible:
            return True

        self.selected_option = None
        for i, option in enumerate(self.sidebar_options):
            option_rect = pygame.Rect(
                (self.sidebar_x + 30, 150 + i * 40),
                self.sidebar_font.size(option),
            )
            if option_rect.collidepoint(pos):
                self.selected_option = i
                return True

        return True

    def _handle_keydown(self, _key: int) -> bool:
        return True

    def fade_transition(self, duration_ms: int = 400) -> None:
        """Create a fade transition effect."""

        frametime = 1000 / FPS
        steps = duration_ms / frametime / 2
        self.fade_step = int(255 / steps)

        while self.fade_alpha < 255:
            self.fade_alpha = min(255, self.fade_alpha + self.fade_step)
            self.fade_surface.set_alpha(self.fade_alpha)
            self.screen.blit(self.fade_surface, (0, 0))
            pygame.display.flip()
            self.clock.tick(FPS)

    def update(self) -> None:
        """Update game state and physics."""
        if self.game_state != State.PLAYING:
            return

        if self.sidebar_visible:
            self.sidebar_x = min(0, self.sidebar_x + self.sidebar_speed)
        else:
            self.sidebar_x = max(
                -self.sidebar_width,
                self.sidebar_x - self.sidebar_speed,
            )

        # Step physics
        dt = 1.0 / FPS
        steps = 100
        for _ in range(steps):
            self.space.step(dt / steps)

        for part in self.dynamics:
            part.on_physics(dt, self.space)
        for part in self.kinematics:
            part.on_physics(dt, self.space)
        for part in self.statics:
            part.on_physics(dt, self.space)

        # Update move indicators
        parts = []
        parts.extend(self.dynamics)
        parts.extend(self.kinematics)
        parts.extend(self.statics)
        if self.selected:
            self.move_indicators = self.selected.calculate_move_indicators(
                self.space, parts
            )
        else:
            self.move_indicators.clear()

        # Check win condition
        if self.is_win_condition():
            self.clear_level()
            next_level = self.current_level + 1
            if next_level <= 10:  # Assuming 10 levels max
                self.fade_transition()
                self.generate_level(next_level)
            else:
                self.fade_transition()
                self.game_state = State.MENU

    def render(self) -> None:
        """Render game objects to screen."""
        # Clear the screen
        self.screen.fill(WHITE)

        match self.game_state:
            case State.MENU:
                self._render_menu()
            case State.LEVEL_SELECT:
                self._render_level_select()
            case State.PLAYING:
                self._render_game()
                self._render_sidebar()

        self.screen.blit(self.fade_surface, (0, 0))
        self.fade_surface.set_alpha(self.fade_alpha)
        self.fade_alpha = max(0, self.fade_alpha - self.fade_step)

        pygame.display.flip()

    def _render_menu(self) -> None:
        self.screen.blit(self.start_bg, (0, 0))
        self.screen.blit(self.start_button, self.start_button_rect)

    def _render_level_select(self) -> None:
        self.screen.blit(self.level_select_bg, (0, 0))
        for img, rect in self.level_buttons:
            self.screen.blit(img, rect)

    def _render_game(self) -> None:
        # Create transparent surface for indicators
        self.indicator_surface.fill((0, 0, 0, 0))

        # Draw move indicators
        for indicator_x, indicator_y, obstacle in self.move_indicators:
            color = INDICATOR_COLOR
            if obstacle and isinstance(obstacle, ChessPiece):
                color = CAPTURE_INDICATOR_COLOR
            pygame.draw.circle(
                self.indicator_surface,
                (*color, INDICATOR_ALPHA),
                (int(indicator_x), int(indicator_y)),
                INDICATOR_RADIUS,
            )

        # Draw background
        for bg in self.backgrounds:
            self.screen.blit(bg, (0, 0))

        # Draw all parts
        for obj in self.kinematics:
            obj.draw(self.screen)
        for obj in self.statics:
            obj.draw(self.screen)
        for obj in self.dynamics:
            obj.draw(self.screen)

        # Blit the transparent indicator surface
        self.screen.blit(self.indicator_surface, (0, 0))

        # Display current level
        level_text = self.hud_font.render(
            f"Level {self.current_level}",
            True,
            BLACK,
        )
        self.screen.blit(level_text, (WIDTH - level_text.get_width() - 10, 10))

    def _render_sidebar(self) -> None:
        self.screen.blit(self.sidebar_bg, (self.sidebar_x, 0))

        for i, option in enumerate(self.sidebar_options):
            color = PURPLE if i == self.selected_option else BLACK
            option_text = self.sidebar_font.render(
                option,
                True,
                color,
            )
            self.screen.blit(option_text, (self.sidebar_x + 30, 150 + i * 40))

        icon = self.sidebar_icon2 if self.sidebar_visible else self.sidebar_icon1
        self.screen.blit(icon, self.sidebar_rect.topleft)

    def run_game_loop(self) -> None:
        """Run the main game loop."""
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.render()
            self.clock.tick(FPS)


class Part:
    """Base class for all physical objects in the game."""

    def __init__(
        self,
        x: int,
        y: int,
        width: int = BLOCK_SIZE,
        height: int = BLOCK_SIZE,
        block_offset: tuple[int, int] = (0, 0),
        body_type: int = pymunk.Body.DYNAMIC,
        game_engine: GameEngine | None = None,
        properties: dict | None = None,
    ) -> Part:
        if properties is None:
            self.properties: dict = {}
        else:
            self.properties = properties

        self.width: int = width
        self.height: int = height
        self.offset: tuple[int, int] = block_offset
        self._sprite: pygame.Surface | None = None

        # Create physics body
        self.body: pymunk.Body = pymunk.Body(body_type=body_type)
        self.shape: pymunk.Poly = pymunk.Poly.create_box(self.body, (width, height))
        self.shape.collision_type = Collision.DEFAULT
        self.shape.mass = 1

        self.body.position = (
            x + BLOCK_SIZE / 2 + self.offset[0],
            y + BLOCK_SIZE / 2 + self.offset[1],
        )
        self.shape.friction = FRICTION
        self.shape.elasticity = ELASTICITY

        # Add to game engine if provided
        if game_engine:
            game_engine.add_part(self)

    def on_physics(self, dt: float, space: pymunk.Space) -> None:
        pass

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the part to the given surface."""

        vertices = [self.body.local_to_world(v) for v in self.shape.get_vertices()]
        if self.sprite is not None:
            # Rotate sprite based on body angle
            rotated = pygame.transform.rotate(
                self.sprite,
                -math.degrees(self.body.angle),
            )

            x, y = self.body.position
            surface.blit(
                rotated,
                (
                    x - rotated.get_width() / 2 - self.offset[0],
                    y - rotated.get_height() / 2 - self.offset[1],
                ),
            )
        else:
            pygame_vertices = [(v.x, v.y) for v in vertices]
            pygame.draw.polygon(surface, (0, 0, 0), pygame_vertices)

    @property
    def rect(self) -> pygame.Rect:
        """Get pygame Rect for collision detection."""
        x, y = self.body.position
        return pygame.Rect(
            x - self.width / 2,
            y - self.height / 2,
            self.width,
            self.height,
        )

    @property
    def sprite(self) -> pygame.Surface | None:
        return self._sprite

    @sprite.setter
    def sprite(self, value: pygame.Surface) -> None:
        """Set the part's sprite."""
        if isinstance(value, pygame.Surface):
            self._sprite = value
        else:
            msg = f"Invalid sprite type: {type(value)}"
            raise TypeError(msg)


class ThroughFloor(Part):
    def __init__(
        self,
        x: int,
        y: int,
        game_engine: GameEngine | None = None,
        properties: dict | None = None,
    ) -> ThroughFloor:
        super().__init__(
            x,
            y,
            height=1,
            block_offset=(0, 25),
            body_type=pymunk.Body.STATIC,
            game_engine=game_engine,
        )
        self.shape.collision_type = Collision.PASSTHROUGH


class IceBlock(Part):
    def __init__(
        self,
        x: int,
        y: int,
        game_engine: GameEngine | None = None,
        properties: dict | None = None,
    ) -> IceBlock:
        super().__init__(
            x,
            y,
            body_type=pymunk.Body.STATIC,
            game_engine=game_engine,
            properties=properties,
        )
        self.shape.friction = 0.001


class SpikeLeft(Part):
    def __init__(
        self,
        x: int,
        y: int,
        game_engine: GameEngine | None = None,
        properties: dict | None = None,
    ) -> SpikeLeft:
        super().__init__(
            x,
            y,
            width=BLOCK_SIZE - 8,
            block_offset=(8, 0),
            body_type=pymunk.Body.STATIC,
            game_engine=game_engine,
        )
        self.shape.collision_type = Collision.SPIKE


class SpikeUp(Part):
    def __init__(
        self,
        x: int,
        y: int,
        game_engine: GameEngine | None = None,
        properties: dict | None = None,
    ) -> SpikeUp:
        super().__init__(
            x,
            y,
            height=BLOCK_SIZE - 8,
            block_offset=(0, 8),
            body_type=pymunk.Body.STATIC,
            game_engine=game_engine,
        )
        self.shape.collision_type = Collision.SPIKE


class Bouncy(Part):
    def __init__(
        self,
        x: int,
        y: int,
        game_engine: GameEngine | None = None,
        properties: dict | None = None,
    ) -> Bouncy:
        super().__init__(
            x,
            y,
            body_type=pymunk.Body.STATIC,
            game_engine=game_engine,
            properties=properties,
        )
        self.bounce_height: int = self.properties.get("bounce_height", 5)
        self.shape.collision_type = Collision.BOUNCY


class HorizontalPlatform(Part):
    def __init__(
        self,
        x: int,
        y: int,
        game_engine: GameEngine | None = None,
        properties: dict | None = None,
    ) -> HorizontalPlatform:
        super().__init__(
            x,
            y,
            body_type=pymunk.Body.KINEMATIC,
            game_engine=game_engine,
            properties=properties,
        )
        left = self.properties.get("left", 0)
        right = self.properties.get("right", 0)
        self.left_bound = (x - (left * BLOCK_SIZE)) + BLOCK_SIZE / 2
        self.right_bound = (x + (right * BLOCK_SIZE)) + BLOCK_SIZE / 2
        self.speed = self.properties.get("speed", 1)
        self.body.velocity = (
            self.properties.get("initial_speed", 1) * BLOCK_SIZE,
            0,
        )

    def on_physics(self, _dt: float, _space: pymunk.Space) -> None:
        x, y = self.body.position
        if x <= self.left_bound:
            self.body.velocity = (self.speed * BLOCK_SIZE, 0)
            self.body.position = self.left_bound, y
        elif x >= self.right_bound:
            self.body.velocity = (-self.speed * BLOCK_SIZE, 0)
            self.body.position = self.right_bound, y


class VerticalPlatform(Part):
    def __init__(
        self,
        x: int,
        y: int,
        game_engine: GameEngine | None = None,
        properties: dict | None = None,
    ) -> VerticalPlatform:
        super().__init__(
            x,
            y,
            body_type=pymunk.Body.KINEMATIC,
            game_engine=game_engine,
            properties=properties,
        )
        up = self.properties.get("up", 0)
        down = self.properties.get("down", 0)
        self.up_bound = (y - (up * BLOCK_SIZE)) + BLOCK_SIZE / 2
        self.down_bound = (y + (down * BLOCK_SIZE)) + BLOCK_SIZE / 2
        self.speed = self.properties.get("speed", 1)
        self.body.velocity = (
            0,
            self.properties.get("initial_speed", 1) * BLOCK_SIZE,
        )

    def on_physics(self, _dt: float, _space: pymunk.Space) -> None:
        x, y = self.body.position
        if y <= self.up_bound:
            self.body.velocity = (0, self.speed * BLOCK_SIZE)
            self.body.position = x, self.up_bound
        elif y >= self.down_bound:
            self.body.velocity = (0, -self.speed * BLOCK_SIZE)
            self.body.position = x, self.down_bound


class DisappearingBlock(Part):
    def __init__(
        self,
        x: int,
        y: int,
        game_engine: GameEngine | None = None,
        properties: dict | None = None,
    ) -> DisappearingBlock:
        super().__init__(
            x,
            y,
            body_type=pymunk.Body.STATIC,
            game_engine=game_engine,
            properties=properties,
        )

        self.active: bool = True
        self.fading: bool = False
        self.fade_time: int = self.properties.get("fade_time", 5)
        self.fade_timer: float = 0
        self.regen_time: int = self.properties.get("regen_time", 5)
        self.regen_timer: float = 0
        self.shape.collision_type = Collision.DISAPPEARING

    def on_physics(self, dt: float, space: pymunk.Space) -> None:
        if self.active:
            if self.fading:
                self.fade_timer += dt
                if self.fade_timer >= self.fade_time:
                    self.active = False
                    self.fading = False
                    self.fade_timer = 0
                    space.remove(self.shape, self.body)
        else:
            self.regen_timer += dt
            if self.regen_timer >= self.regen_time:
                self.active = True
                self.regen_timer = 0
                space.add(self.shape, self.body)

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the part to the given surface."""

        vertices = [self.body.local_to_world(v) for v in self.shape.get_vertices()]
        if self.sprite is not None:
            # Rotate sprite based on body angle
            rotated = pygame.transform.rotate(
                self.sprite,
                -math.degrees(self.body.angle),
            )

            if self.active:
                alpha_reduce = (255 * self.fade_timer / self.fade_time) // 51
                rotated.set_alpha(255 - alpha_reduce * 51)
            else:
                return

            x, y = self.body.position
            surface.blit(
                rotated,
                (
                    x - rotated.get_width() / 2 - self.offset[0],
                    y - rotated.get_height() / 2 - self.offset[1],
                ),
            )
        else:
            pygame_vertices = [(v.x, v.y) for v in vertices]
            pygame.draw.polygon(surface, (0, 0, 0), pygame_vertices)


class Teleporter(Part):
    def __init__(
        self,
        x: int,
        y: int,
        game_engine: GameEngine | None = None,
        properties: dict | None = None,
    ) -> Teleporter:
        super().__init__(
            x,
            y,
            body_type=pymunk.Body.STATIC,
            game_engine=game_engine,
            properties=properties,
        )
        self.destination_x: int = self.properties.get("destination_x")
        self.destination_y: int = self.properties.get("destination_y")
        self.shape.collision_type = Collision.TELEPORTER
        assert self.destination_x is not None, "set destination_x"
        assert self.destination_y is not None, "set destination_y"


class ChessPiece(Part):
    """Base class for all chess pieces."""

    temp_body = pymunk.Body()
    temp_shape = pymunk.Poly.create_box(
        temp_body,
        (
            BLOCK_SIZE * COLLISION_CHECK_MULTIPLIER,
            BLOCK_SIZE * COLLISION_CHECK_MULTIPLIER,
        ),
    )

    def __init__(
        self,
        x: int,
        y: int,
        game_engine: GameEngine | None = None,
        enemy: bool = False,
        properties: dict | None = None,
    ) -> ChessPiece:
        super().__init__(
            x,
            y,
            BLOCK_SIZE - 16,
            BLOCK_SIZE - 4,
            block_offset=(0, 4 / 2),
            game_engine=game_engine,
            properties=properties,
        )
        self.shape.collision_type = Collision.PIECE if not enemy else Collision.ENEMY
        self.enemy = enemy

    def get_possible_moves(self) -> list[Generator[tuple[int, int], None, None]]:
        """Get all possible move positions relative to current position."""
        # Each piece will override this
        return []

    def calculate_move_indicators(
        self,
        space: pymunk.Space,
        parts: list[Part],
    ) -> list:
        """Calculate absolute positions for move indicators."""
        moves = self.get_possible_moves()
        angle = self.body.angle
        sin_angle = math.sin(angle)
        cos_angle = math.cos(angle)
        x, y = self.body.position

        # Convert relative moves to absolute positions
        indicator_positions = []
        pawn_capture_moves = [(1, -1), (1, 1), (-1, -1), (-1, 1)]
        if isinstance(self, Pawn):
            moves.extend(self.get_capture_moves())

        for generator in moves:
            for dx, dy in generator:
                # Calculate the absolute position with rotation
                rotated_dx = dx * cos_angle - dy * sin_angle
                rotated_dy = dx * sin_angle + dy * cos_angle

                abs_x = x + rotated_dx * BLOCK_SIZE
                abs_y = y + rotated_dy * BLOCK_SIZE

                # Check if position is within screen bounds
                if not (0 < abs_x < WIDTH and 0 < abs_y < HEIGHT):
                    break

                # Create temporary shape for collision detection
                self.temp_body.position = abs_x, abs_y
                self.temp_body.angle = angle

                # Check for collisions with this shape
                collisions = space.shape_query(self.temp_shape)

                # Check if there's an enemy piece at the target position (for capture)
                enemy_piece = next(
                    (
                        p
                        for p in parts
                        if isinstance(p, ChessPiece)
                        and p.enemy
                        and any(c.shape == p.shape for c in collisions)
                    ),
                    None,
                )

                teleporter = next(
                    (
                        t
                        for t in parts
                        if isinstance(t, Teleporter)
                        and any(c.shape == t.shape for c in collisions)
                    ),
                    None,
                )

                obstacle = enemy_piece or teleporter

                if len(collisions) > 0 and obstacle is None:
                    break

                # can potentially break if two or more pieces/teleporters are close
                # to each other
                if obstacle and any(obstacle.shape != c.shape for c in collisions):
                    break

                if isinstance(self, Pawn):
                    is_capture_move = (dx, dy) in pawn_capture_moves
                    # Pawns can only move diagonally when capturing
                    if (is_capture_move and not enemy_piece) or (
                        not is_capture_move and len(collisions) > 0
                    ):
                        break

                indicator_positions.append(MovementIndicator(abs_x, abs_y, obstacle))

        return indicator_positions

    def move_to(self, x: int, y: int, velocity: bool = False) -> None:
        """Move the piece to a new position."""
        x0, y0 = self.body.position
        x0 = BLOCK_SIZE / 2 + (x0 // BLOCK_SIZE) * BLOCK_SIZE
        y0 = BLOCK_SIZE / 2 + (y0 // BLOCK_SIZE) * BLOCK_SIZE
        # Round off positions
        x1 = BLOCK_SIZE / 2 + (x // BLOCK_SIZE) * BLOCK_SIZE
        y1 = BLOCK_SIZE / 2 + (y // BLOCK_SIZE) * BLOCK_SIZE
        self.body.position = x1, y1
        if velocity:
            self.body.velocity = (x1 - x0) / 8, 0
        self.body.angular_velocity = 0
        self.body.angle = 0


class Pawn(ChessPiece):
    """Pawn chess piece."""

    def __init__(
        self,
        x: int,
        y: int,
        game_engine: GameEngine | None = None,
        enemy: bool = False,
        properties: dict | None = None,
    ) -> Pawn:
        super().__init__(x, y, game_engine, enemy=enemy, properties=properties)
        self.moved: bool = False

    def get_possible_moves(self) -> list[Generator[tuple[int, int]]]:
        if not self.moved:
            return [
                # 2 blocks North/South/East/West
                ((0, -i) for i in range(1, 3)),
                ((0, i) for i in range(1, 3)),
                ((i, 0) for i in range(1, 3)),
                ((-i, 0) for i in range(1, 3)),
            ]
        return [
            # 1 block North/South/East/West
            ((0, -i) for i in range(1, 2)),
            ((0, i) for i in range(1, 2)),
            ((i, 0) for i in range(1, 2)),
            ((-i, 0) for i in range(1, 2)),
        ]

    def get_capture_moves(self) -> list[Generator[tuple[int, int]]]:
        return [
            (i for i in [(1, -1)]),
            (i for i in [(1, 1)]),
            (i for i in [(-1, -1)]),
            (i for i in [(-1, 1)]),
        ]


class Knight(ChessPiece):
    """Knight chess piece."""

    def __init__(
        self,
        x: int,
        y: int,
        game_engine: GameEngine | None = None,
        enemy: bool = False,
        properties: dict | None = None,
    ) -> Knight:
        super().__init__(x, y, game_engine, enemy=enemy, properties=properties)

    def get_possible_moves(self) -> list[Generator[tuple[int, int]]]:
        # Knights move in L-shape
        return [
            (i for i in [(1, -2)]),
            (i for i in [(2, -1)]),
            (i for i in [(2, 1)]),
            (i for i in [(1, 2)]),
            (i for i in [(-1, 2)]),
            (i for i in [(-2, 1)]),
            (i for i in [(-2, -1)]),
            (i for i in [(-1, -2)]),
        ]


class Bishop(ChessPiece):
    """Bishop chess piece."""

    def __init__(
        self,
        x: int,
        y: int,
        game_engine: GameEngine | None = None,
        enemy: bool = False,
        properties: dict | None = None,
    ) -> Bishop:
        super().__init__(x, y, game_engine, enemy=enemy, properties=properties)

    def get_possible_moves(self) -> list[Generator[tuple[int, int]]]:
        # Bishops move diagonally
        return [
            # NE/SE/NW/SW
            ((i, -i) for i in range(1, 100)),
            ((i, i) for i in range(1, 100)),
            ((-i, -i) for i in range(1, 100)),
            ((-i, i) for i in range(1, 100)),
        ]


class Rook(ChessPiece):
    """Rook chess piece."""

    def __init__(
        self,
        x: int,
        y: int,
        game_engine: GameEngine | None = None,
        enemy: bool = False,
        properties: dict | None = None,
    ) -> Rook:
        super().__init__(x, y, game_engine, enemy=enemy, properties=properties)

    def get_possible_moves(self) -> list[Generator[tuple[int, int]]]:
        # Rooks move horizontally and vertically
        return [
            # North/South/East/West
            ((0, -i) for i in range(1, 100)),
            ((0, i) for i in range(1, 100)),
            ((i, 0) for i in range(1, 100)),
            ((-i, 0) for i in range(1, 100)),
        ]


class Queen(ChessPiece):
    """Queen chess piece."""

    def __init__(
        self,
        x: int,
        y: int,
        game_engine: GameEngine | None = None,
        enemy: bool = False,
        properties: dict | None = None,
    ) -> Queen:
        super().__init__(x, y, game_engine, enemy=enemy, properties=properties)

    def get_possible_moves(self) -> list[Generator[tuple[int, int]]]:
        # Queens move diagonally, horizontally, and vertically
        return [
            # North/South/East/West
            ((0, -i) for i in range(1, 100)),
            ((0, i) for i in range(1, 100)),
            ((i, 0) for i in range(1, 100)),
            ((-i, 0) for i in range(1, 100)),
            # NE/SE/NW/SW
            ((i, -i) for i in range(1, 100)),
            ((i, i) for i in range(1, 100)),
            ((-i, -i) for i in range(1, 100)),
            ((-i, i) for i in range(1, 100)),
        ]


class King(ChessPiece):
    """King chess piece."""

    def __init__(
        self,
        x: int,
        y: int,
        game_engine: GameEngine | None = None,
        enemy: bool = False,
        properties: dict | None = None,
    ) -> King:
        super().__init__(x, y, game_engine, enemy=enemy, properties=properties)

    def get_possible_moves(self) -> list[Generator[tuple[int, int]]]:
        # Kings move one square in any direction
        return [
            # North/South/East/West
            ((0, -i) for i in range(1, 2)),
            ((0, i) for i in range(1, 2)),
            ((i, 0) for i in range(1, 2)),
            ((-i, 0) for i in range(1, 2)),
            # NE/SE/NW/SW
            ((i, -i) for i in range(1, 2)),
            ((i, i) for i in range(1, 2)),
            ((-i, -i) for i in range(1, 2)),
            ((-i, i) for i in range(1, 2)),
        ]


def main() -> None:
    """Main function to run the game."""
    pygame.init()
    screen = pygame.display.set_mode(
        (WIDTH, HEIGHT),
        flags=pygame.RESIZABLE | pygame.SCALED,
    )
    pygame.display.set_caption("ChessFormer: Unbound")

    game = GameEngine(screen)

    # Run game loop
    game.run_game_loop()

    pygame.quit()


if __name__ == "__main__":
    main()
