import math
from enum import IntEnum, auto
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import pygame
import pymunk
import pytmx

WIDTH, HEIGHT = 1000, 700

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

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
PHYSICS_ITERATIONS = int(20000 / FPS)


# Game states
class State(IntEnum):
    MENU = auto()
    LEVEL_SELECT = auto()
    PLAYING = auto()


class Collision(IntEnum):
    DEFAULT = auto()
    PIECE = auto()
    SPIKE = auto()
    BOUNCY = auto()
    PASSTHROUGH = auto()
    DISAPPEARING = auto()


class GameEngine:
    """Main game engine that handles physics, rendering and game state"""

    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.current_level = 1
        self.game_state = State.MENU

        # Game elements
        self.backgrounds: List[pygame.Surface] = []
        self.statics: List[Part] = []
        self.kinematics: List[Part] = []
        self.dynamics: List[Part] = []
        self.selected: Optional[ChessPiece] = None
        self.move_indicators: List[Tuple[int, int, ChessPiece]] = []

        # Initialize physics
        self.space = pymunk.Space()
        self.space.gravity = GRAVITY
        self.space.damping = DAMPING
        self.space.iterations = PHYSICS_ITERATIONS
        self.collision_handler = self.space.add_wildcard_collision_handler(
            Collision.PIECE
        )
        self.collision_handler.begin = self.piece_on_collision
        self.collision_handler.pre_solve = self.piece_pre_solve
        self.collision_handler.data["engine"] = self

        self.clock = pygame.time.Clock()

        # Menu elements
        self.start_bg = pygame.image.load("menu_assets/start_bg.png")
        self.level_select_bg = pygame.image.load("menu_assets/level_bg.png")
        self.start_button = pygame.image.load("menu_assets/play_btn.png")
        self.start_button_rect = self.start_button.get_rect(center=(750, 420))

        # Level selection buttons
        self.level_buttons: List[Tuple[pygame.Surface, pygame.Rect]] = []
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

        # Pre-render game elements
        self.menu_font = pygame.font.SysFont(None, 28)
        self.hud_font = pygame.font.SysFont(None, 24)

        self.indicator_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

        self.instructions_text = self.hud_font.render(
            "Click pieces to select. Click indicators to move. Red indicators capture enemies. Press R to restart, ESC for menu",
            True,
            BLACK,
        )

        self.level_select_instructions = self.menu_font.render(
            "Select a level to play (ESC to return to menu)", True, BLACK
        )

        self.tile_factory = {
            0: lambda x, y, p: King(x, y, self, properties=p),  # light king
            1: lambda x, y, p: Queen(x, y, self, properties=p),  # light queen
            2: lambda x, y, p: Bishop(x, y, self, properties=p),  # light bishop
            3: lambda x, y, p: Knight(x, y, self, properties=p),  # light knight
            4: lambda x, y, p: Rook(x, y, self, properties=p),  # light rook
            5: lambda x, y, p: Pawn(x, y, self, properties=p),  # light pawn
            6: lambda x, y, p: King(x, y, self, properties=p, enemy=True),  # dark king
            7: lambda x, y, p: King(
                x, y, self, properties=p, enemy=True
            ),  # dark king (reversed)
            8: lambda x, y, p: Queen(
                x, y, self, properties=p, enemy=True
            ),  # dark queen
            9: lambda x, y, p: Bishop(
                x, y, self, properties=p, enemy=True
            ),  # dark bishop
            10: lambda x, y, p: Knight(
                x, y, self, properties=p, enemy=True
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
        }

    def add_part(self, part: Union["Part", "ChessPiece"]):
        """Add a part to the appropriate list and physics space"""
        match part.body.body_type:
            case pymunk.Body.STATIC:
                self.statics.append(part)
            case pymunk.Body.KINEMATIC:
                self.kinematics.append(part)
            case pymunk.Body.DYNAMIC:
                self.dynamics.append(part)

        self.space.add(part.body, part.shape)

    def remove_part(self, part: "Part"):
        """Remove a part from the game"""
        if self.selected == part:
            self.selected = None

        if part in self.statics:
            self.statics.remove(part)
        elif part in self.kinematics:
            self.kinematics.remove(part)
        elif part in self.dynamics:
            self.dynamics.remove(part)
        else:
            raise RuntimeError("Part not found!")

        self.space.remove(part.body, part.shape)

    def clear_level(self):
        """Clear backgrounds, statics, and pieces"""
        self.backgrounds = []
        for part in self.statics[:]:
            self.remove_part(part)
        for part in self.kinematics[:]:
            self.remove_part(part)
        for part in self.dynamics[:]:
            self.remove_part(part)

    def is_win_condition(self):
        return not any(
            piece.enemy if isinstance(piece, ChessPiece) else True
            for piece in self.dynamics
        )

    @staticmethod
    def piece_on_collision(
        arbiter: pymunk.Arbiter, space: pymunk.Space, data: Dict[str, Any]
    ):
        piece_shape, other_shape = arbiter.shapes

        match other_shape.collision_type:
            case Collision.SPIKE:
                for piece in data["engine"].dynamics:
                    piece: ChessPiece
                    if piece.shape == piece_shape:
                        data["engine"].remove_part(piece)
                        return False
            case Collision.BOUNCY:
                for bouncy in data["engine"].statics:
                    bouncy: Bouncy
                    if bouncy.shape == other_shape:
                        bounce_height = bouncy.bounce_height * BLOCK_SIZE
                        bounce_velocity = math.sqrt(bounce_height * 2 * GRAVITY[1]) + 1
                        piece_shape.body.velocity = (0, -bounce_velocity)
                        return False
            case Collision.PASSTHROUGH:
                contacts: pymunk.ContactPointSet = arbiter.contact_point_set
                return contacts.normal.y >= 0
            case Collision.DISAPPEARING:
                for disappearing in data["engine"].statics:
                    disappearing: DisappearingBlock
                    if disappearing.shape == other_shape:
                        if disappearing.active:
                            disappearing.fading = True
                            return True
                        else:
                            return False

        return True

    @staticmethod
    def piece_pre_solve(
        arbiter: pymunk.Arbiter, space: pymunk.Space, data: Dict[str, Any]
    ):
        piece_shape, other_shape = arbiter.shapes

        match other_shape.collision_type:
            case Collision.DISAPPEARING:
                for disappearing in data["engine"].statics:
                    disappearing: DisappearingBlock
                    if disappearing.shape == other_shape:
                        return disappearing.active

        return True

    def generate_level(self, level_num: int):
        """Load and create level tiles from TMX file"""
        self.current_level = level_num
        level_path = f"./levels/level{level_num}.tmx"
        try:
            level_data = pytmx.load_pygame(level_path)
            for layer in level_data.layers:
                if isinstance(layer, pytmx.TiledImageLayer):
                    self.backgrounds.append(
                        level_data.images[level_data.tiledgidmap[(layer.gid)]]
                    )
                elif isinstance(layer, pytmx.TiledTileLayer):
                    self._process_tile_layer(layer, level_data)
                elif isinstance(layer, pytmx.TiledObjectGroup):
                    self._process_object_layer(layer, level_data)

            # Switch to playing state
            self.game_state = State.PLAYING

        except Exception as e:
            print(f"Error loading level {level_num}: {e}")
            # Fall back to level selection if loading fails
            self.game_state = State.LEVEL_SELECT

    def _process_tile_layer(
        self, layer: pytmx.TiledTileLayer, level_data: pytmx.TiledMap
    ):
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
                    x * BLOCK_SIZE, y * BLOCK_SIZE, properties
                )

            if tile:
                tile.sprite = level_data.get_tile_image_by_gid(gid)

    def _process_object_layer(
        self, layer: pytmx.TiledObjectGroup, level_data: pytmx.TiledMap
    ):

        for obj in layer:
            obj: pytmx.TiledObject
            x, y = obj.x, obj.y
            obj_id = obj.properties["id"]
            tile: Optional[Part] = None

            if obj_id in self.tile_factory:
                tile = self.tile_factory[obj_id](x, y, obj.properties)
                tile.sprite = level_data.get_tile_image_by_gid(obj.gid)

    def handle_events(self):
        """Process pygame events"""
        for event in pygame.event.get():
            match event.type:
                case pygame.QUIT:
                    return False
                case pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.handle_mouse_click(pygame.mouse.get_pos())
                case pygame.KEYDOWN:
                    self.handle_keydown(event.key)

        return True

    def handle_keydown(self, key: int):
        match key:
            case pygame.K_r:
                if self.game_state == State.PLAYING:
                    self.clear_level()
                    self.generate_level(self.current_level)
            case pygame.K_ESCAPE:
                if self.game_state == State.PLAYING:
                    self.fade_transition()
                    self.clear_level()
                    self.game_state = State.LEVEL_SELECT
                elif self.game_state == State.LEVEL_SELECT:
                    self.fade_transition()
                    self.game_state = State.MENU

    def handle_mouse_click(self, mouse_pos: Tuple[int, int]):
        """Process mouse click events"""
        match self.game_state:
            case State.MENU:
                if self.start_button_rect.collidepoint(mouse_pos):
                    self.fade_transition()
                    self.game_state = State.LEVEL_SELECT

            case State.LEVEL_SELECT:
                for i, (_, rect) in enumerate(self.level_buttons):
                    if rect.collidepoint(mouse_pos):
                        self.fade_transition()
                        self.clear_level()
                        self.generate_level(i + 1)
                        break

            case State.PLAYING:
                # Check if clicked on a move indicator
                moved = False
                if self.selected:
                    for indicator_x, indicator_y, enemy_piece in self.move_indicators:
                        # Create a rect for hit detection on indicators
                        indicator_rect = pygame.Rect(
                            indicator_x - (BLOCK_SIZE / 2),
                            indicator_y - (BLOCK_SIZE / 2),
                            BLOCK_SIZE,
                            BLOCK_SIZE,
                        )

                        if indicator_rect.collidepoint(mouse_pos):
                            # Check if this is a capture move
                            if enemy_piece:
                                self.remove_part(enemy_piece)

                            # Teleport selected piece to this indicator position
                            self.selected.move_to(indicator_x, indicator_y)
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

    def fade_transition(self, duration=400):
        """Create a fade transition effect"""
        fade_surface = pygame.Surface((WIDTH, HEIGHT))
        fade_surface.fill(WHITE)

        steps = 256 // 15
        step_delay = duration // steps
        for alpha in range(0, 256, 15):
            fade_surface.set_alpha(alpha)
            self.screen.blit(fade_surface, (0, 0))
            pygame.display.flip()
            pygame.time.delay(step_delay)

    def update(self):
        """Update game state and physics"""
        if self.game_state != State.PLAYING:
            return

        # Step physics
        dt = 1.0 / FPS
        steps = 100
        for i in range(steps):
            self.space.step(dt / steps)

        for part in self.dynamics:
            part.on_physics(dt)
        for part in self.kinematics:
            part.on_physics(dt)
        for part in self.statics:
            part.on_physics(dt)

        # Update move indicators
        if self.selected:
            self.move_indicators = self.selected.calculate_move_indicators(
                self.space, self.dynamics
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

    def render(self):
        """Render game objects to screen"""
        # Clear the screen
        self.screen.fill(WHITE)

        match self.game_state:
            case State.MENU:
                self._render_menu()
            case State.LEVEL_SELECT:
                self._render_level_select()
            case State.PLAYING:
                self._render_game()

        pygame.display.flip()

    def _render_menu(self):
        self.screen.blit(self.start_bg, (0, 0))
        self.screen.blit(self.start_button, self.start_button_rect)

    def _render_level_select(self):
        self.screen.blit(self.level_select_bg, (0, 0))
        for img, rect in self.level_buttons:
            self.screen.blit(img, rect)
        self.screen.blit(
            self.level_select_instructions,
            (WIDTH // 2 - self.level_select_instructions.get_width() // 2, 50),
        )

    def _render_game(self):
        # Create transparent surface for indicators
        self.indicator_surface.fill((0, 0, 0, 0))

        # Draw move indicators
        for indicator_x, indicator_y, enemy_piece in self.move_indicators:
            color = CAPTURE_INDICATOR_COLOR if enemy_piece else INDICATOR_COLOR
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
        for obj in self.dynamics:
            obj.draw(self.screen)
        for obj in self.kinematics:
            obj.draw(self.screen)
        for obj in self.statics:
            obj.draw(self.screen)

        # Blit the transparent indicator surface
        self.screen.blit(self.indicator_surface, (0, 0))

        self.screen.blit(self.instructions_text, (10, 10))

        # Display current level
        level_text = self.hud_font.render(f"Level {self.current_level}", True, BLACK)
        self.screen.blit(level_text, (WIDTH - level_text.get_width() - 10, 10))

    def run_game_loop(self):
        """Main game loop"""
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.render()
            self.clock.tick(FPS)


class Part:
    """Base class for all physical objects in the game"""

    def __init__(
        self,
        x: int,
        y: int,
        width=BLOCK_SIZE,
        height=BLOCK_SIZE,
        block_offset=(0, 0),
        body_type=pymunk.Body.DYNAMIC,
        game_engine: GameEngine | None = None,
        properties: Dict[str, Any] = None,
    ):
        if properties == None:
            self.properties: Dict[str, Any] = {}
        else:
            self.properties = properties

        self.width: int = width
        self.height: int = height
        self.offset: Tuple[int, int] = block_offset
        self.visible: bool = True
        self._sprite: Optional[pygame.Surface] = None

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

    def on_physics(self, dt: float):
        pass

    def draw(self, surface: pygame.Surface):
        """Draw the part to the given surface"""
        if not self.visible:
            return

        vertices = [self.body.local_to_world(v) for v in self.shape.get_vertices()]
        if self.sprite is not None:
            # Rotate sprite based on body angle
            rotated = pygame.transform.rotate(
                self.sprite, -math.degrees(self.body.angle)
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
    def rect(self):
        """Get pygame Rect for collision detection"""
        x, y = self.body.position
        return pygame.Rect(
            x - self.width / 2,
            y - self.height / 2,
            self.width,
            self.height,
        )

    @property
    def sprite(self):
        return self._sprite

    @sprite.setter
    def sprite(self, value):
        """Set the sprite"""
        if isinstance(value, pygame.Surface):
            self._sprite = value
        else:
            raise ValueError(f"Invalid sprite type: {type(value)}")


class ThroughFloor(Part):
    def __init__(self, x, y, game_engine: GameEngine | None = None, properties=None):
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
    def __init__(self, x, y, game_engine: GameEngine | None = None, properties=None):
        super().__init__(
            x,
            y,
            body_type=pymunk.Body.STATIC,
            game_engine=game_engine,
            properties=properties,
        )
        self.shape.friction = 0.001


class SpikeLeft(Part):
    def __init__(self, x, y, game_engine: GameEngine | None = None, properties=None):
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
    def __init__(self, x, y, game_engine: GameEngine | None = None, properties=None):
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
    def __init__(self, x, y, game_engine: GameEngine | None = None, properties=None):
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
    def __init__(self, x, y, game_engine: GameEngine | None = None, properties=None):
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

    def on_physics(self, dt: float):
        x, y = self.body.position
        if x <= self.left_bound:
            self.body.velocity = (self.speed * BLOCK_SIZE, 0)
            self.body.position = self.left_bound, y
        elif x >= self.right_bound:
            self.body.velocity = (-self.speed * BLOCK_SIZE, 0)
            self.body.position = self.right_bound, y


class VerticalPlatform(Part):
    def __init__(self, x, y, game_engine: GameEngine | None = None, properties=None):
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

    def on_physics(self, dt: float):
        x, y = self.body.position
        if y <= self.up_bound:
            self.body.velocity = (0, self.speed * BLOCK_SIZE)
            self.body.position = x, self.up_bound
        elif y >= self.down_bound:
            self.body.velocity = (0, -self.speed * BLOCK_SIZE)
            self.body.position = x, self.down_bound


class DisappearingBlock(Part):
    def __init__(self, x, y, game_engine: GameEngine | None = None, properties=None):
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

    def on_physics(self, dt: float):
        if self.active:
            if self.fading:
                self.fade_timer += dt
                if self.fade_timer >= self.fade_time:
                    self.active = False
                    self.fading = False
                    self.fade_timer = 0
        else:
            self.regen_timer += dt
            if self.regen_timer >= self.regen_time:
                self.active = True
                self.regen_timer = 0

    def draw(self, surface: pygame.Surface):
        """Draw the part to the given surface"""
        if not self.visible:
            return

        vertices = [self.body.local_to_world(v) for v in self.shape.get_vertices()]
        if self.sprite is not None:
            # Rotate sprite based on body angle
            rotated = pygame.transform.rotate(
                self.sprite, -math.degrees(self.body.angle)
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


class ChessPiece(Part):
    """Base class for all chess pieces"""

    temp_body = pymunk.Body()
    temp_shape = pymunk.Poly.create_box(
        temp_body,
        (
            BLOCK_SIZE * COLLISION_CHECK_MULTIPLIER,
            BLOCK_SIZE * COLLISION_CHECK_MULTIPLIER,
        ),
    )

    def __init__(
        self, x, y, game_engine: GameEngine | None = None, enemy=False, properties=None
    ):
        super().__init__(
            x,
            y,
            BLOCK_SIZE - 15,
            BLOCK_SIZE - 4,
            block_offset=(0, 4 / 2),
            game_engine=game_engine,
            properties=properties,
        )
        self.shape.collision_type = Collision.PIECE
        self.enemy = enemy

    def get_possible_moves(self) -> List[Generator[Tuple[int, int], None, None]]:
        """Get all possible move positions relative to current position"""
        # Each piece will override this
        return []

    def calculate_move_indicators(
        self, space: pymunk.Space, pieces: List["ChessPiece"]
    ):
        """Calculate absolute positions for move indicators"""
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
                        for p in pieces
                        if p.enemy and any(c.shape == p.shape for c in collisions)
                    ),
                    None,
                )

                if len(collisions) > 0 and enemy_piece is None:
                    break

                if any(enemy_piece.shape != c.shape for c in collisions):
                    break

                if isinstance(self, Pawn):
                    is_capture_move = (dx, dy) in pawn_capture_moves
                    # Pawns can only move diagonally when capturing
                    if is_capture_move and not enemy_piece:
                        break
                    # Pawns can't move forward when blocked
                    elif not is_capture_move and len(collisions) > 0:
                        break

                indicator_positions.append((abs_x, abs_y, enemy_piece))

        return indicator_positions

    def move_to(self, x, y):
        """Move the piece to a new position"""
        x0, y0 = self.body.position
        x0 = BLOCK_SIZE / 2 + (x0 // BLOCK_SIZE) * BLOCK_SIZE
        # Round off positions
        x1 = BLOCK_SIZE / 2 + (x // BLOCK_SIZE) * BLOCK_SIZE
        y1 = BLOCK_SIZE / 2 + (y // BLOCK_SIZE) * BLOCK_SIZE
        self.body.position = x1, y1
        self.body.velocity = (x1 - x0) / 8, 0
        self.body.angle = 0


class Pawn(ChessPiece):
    """Pawn chess piece"""

    def __init__(self, x, y, game_engine=None, enemy=False, properties=None):
        super().__init__(x, y, game_engine, enemy=enemy, properties=properties)
        self.moved: bool = False

    def get_possible_moves(self):
        if not self.moved:
            return [
                # 2 blocks North/South/East/West
                ((0, -i) for i in range(1, 3)),
                ((0, i) for i in range(1, 3)),
                ((i, 0) for i in range(1, 3)),
                ((-i, 0) for i in range(1, 3)),
            ]
        else:
            return [
                # 1 block North/South/East/West
                ((0, -i) for i in range(1, 2)),
                ((0, i) for i in range(1, 2)),
                ((i, 0) for i in range(1, 2)),
                ((-i, 0) for i in range(1, 2)),
            ]

    def get_capture_moves(self):
        return [
            (i for i in [(1, -1)]),
            (i for i in [(1, 1)]),
            (i for i in [(-1, -1)]),
            (i for i in [(-1, 1)]),
        ]


class Knight(ChessPiece):
    """Knight chess piece"""

    def __init__(self, x, y, game_engine=None, enemy=False, properties=None):
        super().__init__(x, y, game_engine, enemy=enemy, properties=properties)

    def get_possible_moves(self):
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
    """Bishop chess piece"""

    def __init__(self, x, y, game_engine=None, enemy=False, properties=None):
        super().__init__(x, y, game_engine, enemy=enemy, properties=properties)

    def get_possible_moves(self):
        # Bishops move diagonally
        return [
            # NE/SE/NW/SW
            ((i, -i) for i in range(1, 100)),
            ((i, i) for i in range(1, 100)),
            ((-i, -i) for i in range(1, 100)),
            ((-i, i) for i in range(1, 100)),
        ]


class Rook(ChessPiece):
    """Rook chess piece"""

    def __init__(self, x, y, game_engine=None, enemy=False, properties=None):
        super().__init__(x, y, game_engine, enemy=enemy, properties=properties)

    def get_possible_moves(self):
        # Rooks move horizontally and vertically
        return [
            # North/South/East/West
            ((0, -i) for i in range(1, 100)),
            ((0, i) for i in range(1, 100)),
            ((i, 0) for i in range(1, 100)),
            ((-i, 0) for i in range(1, 100)),
        ]


class Queen(ChessPiece):
    """Queen chess piece"""

    def __init__(self, x, y, game_engine=None, enemy=False, properties=None):
        super().__init__(x, y, game_engine, enemy=enemy, properties=properties)

    def get_possible_moves(self):
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
    """King chess piece"""

    def __init__(self, x, y, game_engine=None, enemy=False, properties=None):
        super().__init__(x, y, game_engine, enemy=enemy, properties=properties)

    def get_possible_moves(self):
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


def main():
    """Main function to run the game"""
    pygame.init()
    screen = pygame.display.set_mode(
        (WIDTH, HEIGHT), flags=pygame.RESIZABLE | pygame.SCALED
    )
    pygame.display.set_caption("ChessFormer: Unbound")

    game = GameEngine(screen)
    # game.game_state = State.PLAYING
    # game.generate_level(1)

    # Run game loop
    game.run_game_loop()

    pygame.quit()


if __name__ == "__main__":
    main()
