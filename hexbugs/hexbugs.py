import numpy as np
import cv2
import colorsys
import time

from typing import List, Tuple, Dict


# Global parameters for simulation
MIN_X, MAX_X = 0, 1.92
MIN_Y, MAX_Y = 0, 1.08

STARTING_AGENTS = 10
NUM_GENES = 64
MUTATION_RATE = 0.005
MITOSIS_ENERGY_THRESHOLD = 1.0
ENERGY_LOSS_RATE = 0.0001
MIN_SPEED, MAX_SPEED = 0.0001, 0.001

NUM_FOOD = 256
FOOD_REPLENISH_RATE = 0.5

NUM_STIM = 32

NUM_ROCKS = 40

# Global parameters for display
DISPLAY_WIDTH, DISPLAY_HEIGHT = 1440, 810
WINDOW_NAME = 'Evolution Simulation'

# Global color palette (in BGR format for OpenCV)
COLORS = {
    'BACKGROUND': (230, 230, 230),    
    'ORGANIC_FOOD': (64, 190, 64),   # Green
    'STIM': (0, 192, 192),
    'ROCK': (128, 128, 128),       # Gray
}


GLOBAL_SCALE = np.abs((MAX_X - MIN_X) + (MAX_Y - MIN_Y)) / 2

class GraphBrain:
    def __init__(self, genome: str):
        self.input_nodes = ['stim_touch', 'rock_touch', 'food_touch', 'organic_food_in_sight', 'obstacle_ahead', 'energy_level', 'age']
        self.output_nodes = ['change_direction', 'speed_up', 'slow_down', 'eat', 'mitosis']
        self.operation_nodes = ['add', 'invert', 'random', 'osc', 'identity', 'gaussian']
        self.connections: Dict[str, List[Tuple[str, float]]] = self._decode_genome(genome)

    def _decode_genome(self, genome: str) -> Dict[str, List[Tuple[str, float]]]:
        connections = {node: [] for node in self.input_nodes + self.operation_nodes + self.output_nodes}
        for i in range(0, len(genome), 6):
            if i + 6 <= len(genome):
                gene = genome[i:i+6]
                source_index = int(gene[0], 16) % (len(self.input_nodes) + len(self.operation_nodes))
                target_index = int(gene[1], 16) % (len(self.operation_nodes) + len(self.output_nodes))
                weight = (int(gene[2:], 16) / 65535) * 2 - 1  # Map to [-1, 1]

                source = (self.input_nodes + self.operation_nodes)[source_index]
                target = (self.operation_nodes + self.output_nodes)[target_index]
                
                connections[source].append((target, weight))
        return connections

    def process(self, inputs: Dict[str, float]) -> Dict[str, float]:
        node_values = {**inputs}
        
        # Process operation nodes
        for op_node in self.operation_nodes:
            node_values[op_node] = 0

        # Propagate values through the network
        for _ in range(2):  # Number of propagation iterations
            new_values = node_values.copy()
            for source, targets in self.connections.items():
                for target, weight in targets:
                    if target in self.operation_nodes:
                        new_values[target] += self._apply_operation(target, node_values[source]) * weight
                    elif target in self.output_nodes:
                        new_values[target] = new_values.get(target, 0) + node_values[source] * weight
            node_values = new_values

        # Normalize output values
        outputs = {node: np.tanh(node_values.get(node, 0)) for node in self.output_nodes}
        return outputs

    def _apply_operation(self, operation: str, value: float) -> float:
        if operation == 'add':
            return value + np.random.normal()
        elif operation == 'invert':
            return value * -1
        elif operation == 'random':
            return (np.random.rand() * 2 - 1) * value
        elif operation == 'osc':
            return np.sin(time.time() / (2 * np.pi)) * value
        elif operation == 'identity':
            return value
        elif operation == 'gaussian':
            return np.exp(-value**2)
        else:
            return value
        


class Agent:
    def __init__(self, x: float, y: float, energy: float = 1.0):
        self.x = x
        self.y = y
        self.sight_range = 0.05 * GLOBAL_SCALE
        self.sight_angle = GLOBAL_SCALE * np.pi / 8
        self.eat_range = 0.0125 * GLOBAL_SCALE

        self.energy = energy
        self.age = 0
        self.direction = np.random.uniform(0, 2*np.pi)
        self.speed = 0.0
        
        self.genome: str = self.generate_random_genome(length=NUM_GENES)
        self.brain = GraphBrain(self.genome)

        self.set_params()

        self.color = np.random.randint(0, 256, 3)

    def generate_random_genome(self, length: int = 10) -> str:
        """Generate a random genome of hex digits."""
        length *= 6  # Each gene is 6 bits long
        return ''.join(np.random.choice(list('0123456789ABCDEF'), size=length))

    def print_genome(self):
        """Print the agent's genome in a readable format."""
        print(f'\nAgent genome: {self.genome}')
        for i in range(0, len(self.genome), 6):
            gene = self.genome[i:i+6]
            source_index = int(gene[0], 16) % (len(self.brain.input_nodes) + len(self.brain.operation_nodes))
            target_index = int(gene[1], 16) % (len(self.brain.operation_nodes) + len(self.brain.output_nodes))
            weight = (int(gene[2:], 16) / 65535) * 2 - 1  # Map to [-1, 1]
            
            source = (self.brain.input_nodes + self.brain.operation_nodes)[source_index]
            target = (self.brain.operation_nodes + self.brain.output_nodes)[target_index]
            
            print(f'{source} -> {target}: {weight:.2f}')

    def set_params(self):
        # We'll use specific components of the genome to set the parameters
        sight_range = 0.0
        sight_width = 0.0
        div = 0.0

        r = 0
        g = 0
        b = 0

        for i in range(0, len(self.genome), 6):
            gene = self.genome[i:i+6]
            sight_range += int(gene[0], 16) / 15
            sight_width += int(gene[3], 16) / 15
            
            div += 1.0
            
            r += int(gene[0:2], 16)
            g += int(gene[2:4], 16)
            b += int(gene[4:6], 16)

        r = int(r / div)
        g = int(g / div)
        b = int(b / div)
        self.color = (r, g, b)

        sight_width /= div
        sight_width = np.clip(sight_width, 0.0, 1.0)
        sight_width = np.power(sight_width, 4)
        self.sight_angle = sight_width * np.pi / 2

        sight_range /= div
        sight_range = np.clip(sight_range, 0.0, 1.0)
        sight_range = np.power(sight_range, 0.85)

        self.sight_range = (1.0 - sight_range) * 0.01 * GLOBAL_SCALE + sight_range * 0.125 * GLOBAL_SCALE
        self.sight_angle = sight_width * np.pi

    def move(self):
        """Move the agent based on its current direction and speed."""
        self.x += self.speed * np.cos(self.direction) * GLOBAL_SCALE
        self.y += self.speed * np.sin(self.direction) * GLOBAL_SCALE
        self.x = self.x % (MAX_X - MIN_X) + MIN_X
        self.y = self.y % (MAX_Y - MIN_Y) + MIN_Y
        energy_cost = (self.speed / (MAX_SPEED - MIN_SPEED))
        energy_cost = energy_cost * energy_cost
        self.energy -= energy_cost * 0.001

    def think(self, env_inputs: Dict[str, float]) -> Dict[str, float]:
        """Process environmental inputs through the brain and act on outputs."""
        outputs = self.brain.process(env_inputs)
        
        if np.abs(outputs['change_direction']) > 0.0:
            self.change_direction((self.speed / (MAX_SPEED - MIN_SPEED)) * outputs['change_direction'])
        
        self.speed += outputs['speed_up'] * 0.0005
        self.speed -= outputs['slow_down'] * 0.0005
        self.speed = np.clip(self.speed, MIN_SPEED, MAX_SPEED)

        return outputs

    def change_direction(self, angle_change: float):
        """Change the agent's direction by the given angle (in radians)."""
        self.direction += angle_change * 0.5
        self.direction %= 2*np.pi
        energy_cost = abs(angle_change) * 0.001
        self.energy -= energy_cost

    def eat(self, food_value: float):
        """Increase the agent's energy by consuming food."""
        self.energy += food_value
        self.energy = np.clip(self.energy, 0.0, 4.0)

    def is_alive(self) -> bool:
        """Check if the agent is still alive (has energy)."""
        return self.energy > 0
    
    def mitosis(self) -> 'Agent':
        """Create a new agent with a mutated genome."""
        new_energy = np.minimum(1.0, self.energy / 2)
        self.energy = new_energy
        new_genome = self.mutate_genome()
        random_pos = np.random.randn(2) * 0.01

        new_agent = Agent(self.x + random_pos[0], self.y + random_pos[1], energy=new_energy)
        new_agent.direction = np.random.uniform(0, 2*np.pi)
        new_agent.genome = new_genome
        new_agent.brain = GraphBrain(new_genome)
        new_agent.color = self.color

        new_agent.set_params()

        self.age = 0
        self.genome = self.mutate_genome()
        self.brain = GraphBrain(self.genome)
        self.direction = np.random.uniform(0, 2*np.pi)

        return new_agent
    
    def mutate_genome(self) -> str:
        """Mutate the agent's genome by flipping a few bits."""
        mutation_rate = MUTATION_RATE
        new_genome = ''
        for gene in self.genome:
            if np.random.rand() < mutation_rate:
                new_gene = np.random.choice(list('0123456789ABCDEF'))
            else:
                new_gene = gene
            new_genome += new_gene
        return new_genome
    
    def combine_genomes(self, other: 'Agent') -> str:
        """Combine the genomes of two agents by mixing their genes."""
        new_genome = ''
        for gene1, gene2 in zip(self.genome, other.genome):
            new_gene = np.random.choice([gene1, gene2])
            new_genome += new_gene
        return new_genome
    


class Environment:
    def __init__(self):
        self.organic_food = []
        self.stim = []
        self.rocks = []
        self.agents: List[Agent] = []
        
    def add_organic_food(self, num_food):
        for _ in range(num_food):
            x = np.random.uniform(MIN_X, MAX_X)
            y = np.random.uniform(MIN_Y, MAX_Y)

            # Ensure food is not placed on top of a rock
            '''
            for rock_x, rock_y, rock_radius in self.rocks:
                if np.sqrt((rock_x - x)**2 + (rock_y - y)**2) <= rock_radius + 0.1:
                    # we'll move the food to the edge of the rock
                    dx = rock_x - x
                    dy = rock_y - y

                    angle = np.arctan2(dy, dx)
                    new_x = rock_x - (rock_radius + 0.1) * np.cos(angle)
                    new_y = rock_y - (rock_radius + 0.1) * np.sin(angle)
                    x, y = new_x, new_y
            '''
            
            self.organic_food.append((x, y))
    
    def add_stim(self, num_stim):
        for _ in range(num_stim):
            x = np.random.uniform(MIN_X, MAX_X)
            y = np.random.uniform(MIN_Y, MAX_Y)

            # Ensure stim is not placed on top of a rock

            for rock_x, rock_y, rock_radius in self.rocks:
                if np.sqrt((rock_x - x)**2 + (rock_y - y)**2) <= rock_radius + 0.1:
                    # we'll move the stim to the edge of the rock
                    dx = rock_x - x
                    dy = rock_y - y

                    angle = np.arctan2(dy, dx)
                    new_x = rock_x - (rock_radius + 0.1) * np.cos(angle)
                    new_y = rock_y - (rock_radius + 0.1) * np.sin(angle)
                    x, y = new_x, new_y

            self.stim.append((x, y))
    
    def add_rocks(self, num_rocks):
        for _ in range(num_rocks):
            x = np.random.uniform(MIN_X, MAX_X)
            y = np.random.uniform(MIN_Y, MAX_Y)
            factor = np.abs((MAX_X - MIN_X) + (MAX_Y - MIN_Y)) / 2
            radius = np.random.uniform(factor * 0.01, factor * 0.0625)
            self.rocks.append((x, y, radius))

    def add_agents(self, num_agents: int):
        for _ in range(num_agents):
            x = np.random.uniform(MIN_X, MAX_X)
            y = np.random.uniform(MIN_Y, MAX_Y)
            self.agents.append(Agent(x, y))

    def update(self):
        '''Update the environment for one time step'''

        # We'll move the rocks across the screen
        for i in range(len(self.rocks)):
            x, y, r = self.rocks[i]
            x = x + 0.0005
            y = y + np.random.randn() * 0.00025
            x = x % (MAX_X - MIN_X) + MIN_X
            y = y % (MAX_Y - MIN_Y) + MIN_Y
            r = r * (1 + np.sin((time.time() + (np.pi * 2 * i / len(self.rocks)))/ 2) * 0.003)
            self.rocks[i] = (x, y, r)

        # Update agents
        for agent in self.agents:
            if agent.is_alive():
                
                # We'll perturb the agent's direction / position a little bit
                agent.direction += np.random.randn() * 0.01
                agent.x += np.random.randn() * 0.0001
                agent.y += np.random.randn() * 0.0001
                # Gather inputs for the agent's brain
                env_inputs = self.get_agent_inputs(agent)
                # Let the agent think and act
                outputs = agent.think(env_inputs)
                # Move the agent
                agent.move()
                # Handle collisions
                self.handle_collision(agent)
                # Handle eating
                if np.abs(outputs['eat']) > 0.5:
                    self.handle_eating(agent)
                # Increase agent's age
                agent.age += 1
                agent.energy -= ENERGY_LOSS_RATE
                # Handle mitosis
                if np.abs(outputs['mitosis']) > 0.5 \
                    and agent.energy > MITOSIS_ENERGY_THRESHOLD \
                    and agent.age > np.random.randint(250, 1000):
                    self.handle_mitosis(agent)


        # Replenish resources
        self.replenish_resources()

        # Remove dead agents
        self.agents = [agent for agent in self.agents if agent.is_alive()]

    
    def get_agent_inputs(self, agent: Agent) -> Dict[str, float]:
        """Gather environmental inputs for an agent."""
        inputs = {
            'stim_touch': self.stim_touch(agent),
            'rock_touch': self.rock_touch(agent),
            'food_touch': self.food_touch(agent),
            'organic_food_in_sight': self.organic_food_in_sight(agent),
            'obstacle_ahead': self.obstacle_ahead(agent),
            'agent_ahead': self.agent_ahead(agent),
            'energy_level': np.tanh(agent.energy),
            'age': np.tanh(agent.age / 1000)
        }
        return inputs

    def stim_touch(self, agent: Agent) -> float:
        # If the agent is touching a stim, return 1, else return 0
        for sx, sy in self.stim:
            # We'll find the distance between the agent and the stim
            dx = sx - agent.x
            dy = sy - agent.y

            # Keep in mind, we're on a toroidal surface
            dx = dx % (MAX_X - MIN_X)
            dy = dy % (MAX_Y - MIN_Y)

            distance = np.sqrt(dx**2 + dy**2)

            if distance <= agent.eat_range:
                return 1.0
            
        return 0.0
    
    def rock_touch(self, agent: Agent) -> float:
        # If the agent is touching a rock, return 1, else return 0
        for rx, ry, rr in self.rocks:
            dx = rx - agent.x
            dy = ry - agent.y
            distance = np.sqrt(dx**2 + dy**2)

            if distance <= rr + agent.eat_range:
                return 1.0
            
        return 0.0
    
    def food_touch(self, agent: Agent) -> float:
        # If the agent is touching food, return 1, else return 0
        for fx, fy in self.organic_food:
            dx = fx - agent.x
            dy = fy - agent.y
            distance = np.sqrt(dx**2 + dy**2)

            if distance <= agent.eat_range:
                return 1.0
            
        return 0.0
    
    def organic_food_in_sight(self, agent: Agent) -> float:
        """Check if food is in the agent's line of sight."""
        sight_range = agent.sight_range  
        sight_angle = agent.sight_angle
        
        food_sources = self.organic_food
        
        for food_x, food_y in food_sources:
            dx = food_x - agent.x
            dy = food_y - agent.y
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance <= sight_range:
                angle = np.arctan2(dy, dx) - agent.direction
                angle = (angle + np.pi) % (2 * np.pi) - np.pi  # Normalize angle
                
                if abs(angle) <= sight_angle:
                    # Food is in sight, return a value based on distance (closer = higher value)
                    return 1 - (distance / sight_range)
        
        return 0  # No food in sight

    def obstacle_ahead(self, agent: Agent) -> float:
        """Check if there's an obstacle ahead of the agent."""
        sight_range = agent.sight_range
        sight_angle = agent.sight_angle
        
        for rock_x, rock_y, rock_radius in self.rocks:
            dx = rock_x - agent.x
            dy = rock_y - agent.y
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance <= sight_range + rock_radius:
                angle = np.arctan2(dy, dx) - agent.direction
                angle = (angle + np.pi) % (2 * np.pi) - np.pi

                if abs(angle) <= sight_angle:

                    return_value = 1 - (distance - rock_radius) / sight_range
                    return return_value
        
        return 0  # No obstacle ahead
    
    def agent_ahead(self, agent: Agent) -> float:
        """Check if there's another agent ahead of the agent."""
        sight_range = agent.sight_range
        sight_angle = agent.sight_angle
        
        for other_agent in self.agents:
            if other_agent == agent:
                continue
            
            dx = other_agent.x - agent.x
            dy = other_agent.y - agent.y
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance <= sight_range:

                angle = np.arctan2(dy, dx) - agent.direction
                angle = (angle + np.pi) % (2 * np.pi) - np.pi

                if abs(angle) <= sight_angle:

                    return 1 - (distance / sight_range)
        
        return 0  # No agent ahead
        

    def handle_collision(self, agent: Agent):
        """Handle collisions between agents and rocks."""

        # If an agent collides with a still agent, we'll combine their genomes and create a new agent
        for other_agent in self.agents:
            if other_agent == agent:
                continue

            if other_agent.speed > MIN_SPEED * 2 or agent.speed < MAX_SPEED / 4:
                continue

            if other_agent.age < 500 or agent.age < 500:
                continue

            dx = other_agent.x - agent.x
            dy = other_agent.y - agent.y
            distance = np.sqrt(dx**2 + dy**2)

            if distance <= agent.eat_range + other_agent.eat_range:
                if np.random.rand() < 0.01:

                    new_genome1 = agent.combine_genomes(other_agent)
                    new_genome2 = agent.combine_genomes(other_agent)

                    # Replacing the other agent
                    new_agent1 = Agent(other_agent.x, other_agent.y, energy=other_agent.energy)
                    new_agent1.direction = other_agent.direction
                    # new_agent1.age = other_agent.age
                    new_agent1.genome = new_genome1
                    new_agent1.mutate_genome()
                    new_agent1.brain = GraphBrain(new_genome1)
                    new_agent1.color = other_agent.color
                    new_agent1.set_params()

                    # Mutating the current one
                    new_agent2 = Agent(agent.x, agent.y, energy=agent.energy)
                    new_agent2.direction = agent.direction
                    new_agent2.speed = agent.speed
                    # new_agent2.age = agent.age
                    new_agent2.genome = new_genome2
                    new_agent2.mutate_genome()
                    new_agent2.brain = GraphBrain(new_genome2)
                    new_agent2.color = agent.color
                    new_agent2.set_params()

                    self.agents.append(new_agent1)
                    self.agents.append(new_agent2)

                    # We'll remove both the previous agents from the environment
                    self.agents.remove(agent)
                    self.agents.remove(other_agent)

                    break



        for rock_x, rock_y, rock_radius in self.rocks:
            dx = rock_x - agent.x
            dy = rock_y - agent.y
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance <= rock_radius:
                # Collision with a rock
                # Find the closest point on the rock's edge to the agent
                # and move the agent to that point

                angle = np.arctan2(dy, dx)
                new_x = rock_x - rock_radius * np.cos(angle)
                new_y = rock_y - rock_radius * np.sin(angle)
                agent.x, agent.y = new_x, new_y
                agent.energy = agent.energy * 0.9

                

    def handle_eating(self, agent: Agent):
        """Handle the agent's attempt to eat nearby food."""
        eat_range = agent.eat_range
        
        # Check organic food
        for i, (food_x, food_y) in enumerate(self.organic_food):
            if np.sqrt((food_x - agent.x)**2 + (food_y - agent.y)**2) <= eat_range:
                agent.eat(1.0)
                self.organic_food.pop(i)
                return

            
    def handle_mitosis(self, agent: Agent):
        """Handle the agent's attempt to perform mitosis."""

        # Create a new agent with a mutated genome
        new_agent = agent.mitosis()
        self.agents.append(new_agent)

            
    def replenish_resources(self):
        """Replenish food and stim in the environment."""
        if len(self.organic_food) < NUM_FOOD:
            if np.random.rand() < FOOD_REPLENISH_RATE:
                self.add_organic_food(1)

    def visualize(self, verbose: bool = False):
        # Create a background
        display = np.full((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), COLORS['BACKGROUND'], dtype=np.uint8)
        
        # Function to convert simulation coordinates to display coordinates
        def sim_to_display(x, y):
            display_x = int((x - MIN_X) / (MAX_X - MIN_X) * DISPLAY_WIDTH)
            display_y = int((y - MIN_Y) / (MAX_Y - MIN_Y) * DISPLAY_HEIGHT)
            return (display_x, display_y)
        
        for food in self.organic_food:
            cv2.circle(display, sim_to_display(*food), 4, COLORS['ORGANIC_FOOD'], 3)
            cv2.circle(display, sim_to_display(*food), 4, (255, 255, 255), -1)
        
        for pos in self.stim:
            cv2.circle(display, sim_to_display(*pos), 4, COLORS['STIM'], 3)
        
        for rock in self.rocks:
            radius = rock[2]
            rx, ry = sim_to_display(radius, radius)
            r = (rx + ry) // 2
            cv2.circle(display, sim_to_display(rock[0], rock[1]), r, COLORS['ROCK'], -1)
        
        # Draw agents (red triangles)
        for agent in self.agents:
            if agent.is_alive():
                
                color = agent.color
                color_hls = colorsys.rgb_to_hls(*[c/255 for c in color])
                # We'll use the agent's energy level to change the brightness of the color
                color_hls = list(color_hls)
                color_hls[1] = np.clip(agent.energy, 0.0, 0.75)
                color_hls[2] = 1.0

                if agent.age < 16:
                    color_hls[1] = 1.0

                color = tuple(int(c * 255) for c in colorsys.hls_to_rgb(*color_hls))

                pos = sim_to_display(agent.x, agent.y)
                direction = agent.direction

                rx, ry = sim_to_display(agent.eat_range, agent.eat_range)
                cv2.ellipse(display, pos, (rx, ry), direction * 180 / np.pi, 0, 360, color, 5)
                # cv2.circle(display, pos, 5, color, -1)

                if verbose:
                    # We'll display the agent's look range
                    look_range = agent.sight_range
                    look_width = agent.sight_angle

                    look_pos = sim_to_display(agent.x + look_range * np.cos(direction), agent.y + look_range * np.sin(direction))
                    cv2.circle(display, look_pos, 2, color, -1)

                    peripherals_left = sim_to_display(agent.x + look_range * np.cos(direction - look_width), agent.y + look_range * np.sin(direction - look_width))
                    # cv2.line(display, pos, peripherals_left, color, 1)
                    cv2.circle(display, peripherals_left, 2, color, -1)

                    peripherals_right = sim_to_display(agent.x + look_range * np.cos(direction + look_width), agent.y + look_range * np.sin(direction + look_width))
                    # cv2.line(display, pos, peripherals_right, color, 1)
                    cv2.circle(display, peripherals_right, 2, color, -1)

                    # cv2.line(display, peripherals_left, peripherals_right, color, 1)

                    # we'll also give the agent a little arrow to show its direction
                    f0 = 0.025 * GLOBAL_SCALE
                    f1 = 0.03 * GLOBAL_SCALE
                    arrow_start = sim_to_display(agent.x + f0 * np.cos(direction), agent.y + f0 * np.sin(direction))
                    arrow_end = sim_to_display(agent.x + f1 * np.cos(direction), agent.y + f1 * np.sin(direction))
                    cv2.arrowedLine(display, arrow_start, arrow_end, color, 2, tipLength=2)

                
                t0 = 0.02 * GLOBAL_SCALE
                tx, ty = sim_to_display(agent.x - t0 * np.cos(direction), agent.y - t0 * np.sin(direction))
                cv2.circle(display, (tx, ty), 3, color, 2)
                



        display = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
        cv2.imshow(WINDOW_NAME, display)
        cv2.waitKey(1)


def main():
    env = Environment()
    env.add_rocks(NUM_ROCKS)
    env.add_organic_food(100)
    env.add_stim(NUM_STIM)
    env.add_agents(STARTING_AGENTS)

    for agent in env.agents:
        agent.print_genome()
    
    key = cv2.waitKey(1) & 0xFF

    while True:
        env.update()
        env.visualize()

        if key == ord('q') or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()