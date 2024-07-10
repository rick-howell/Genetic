# This will be a hexbug adjacent program with a few notable differences.
# 1. Instead of hex digits, we will use base 36 digits.
    # 0-9 will be the same as in base 10.
    # 10-35 will be represented by A-Z.
# 2. The brain will be feedforward, instead of allowing loops.
# 3. We will use vectorized operations to speed up the simulation and handle movement.
# 4. Everything should be modularized


# The genome will be a string of base 36 digits.
# Each gene in the genome will have the following structure:
    # 1:    Source node
    # 2:    Destination node
    # 3:    Layer
    # 4-6:  Weight


import numpy as np
import cv2
import networkx as nx
import matplotlib.pyplot as plt

import colorsys
import time
from typing import List, Tuple, Dict

BASE = 36

MIN_X, MAX_X = 0, 1.92
MIN_Y, MAX_Y = 0, 1.08

STARTING_POPULATION, MAX_POPULATION = 4, 64

DISPLAY_WIDTH, DISPLAY_HEIGHT = 1440, 810
WINDOW_NAME = 'Ev Sim'

NUM_GENES = 96
HIDDEN_LAYERS = 3
EAT_DISTANCE = 0.01
ENERGY_LIMIT = 4.0
AGE_STEP = 1
ENERGY_STEP = -0.00001
VELOCITY_LIMIT = 0.125
MUTATION_RATE = 0.1
MITOSIS_ENERGY = 0.5

FORCE_FACTOR = 0.05
FRICTION = 0.1

FOOD_ENERGY_STEP = -0.0001
STARTING_FOOD = 16
MAX_FOOD = 256
REPLENISH_RATE = 0.0001

NUM_POISON = 16
POISON_RADIUS = 0.05
POISON_SENSE = 0.05
POISON_STEP = -0.1

REPULSION_DISTANCE = 0.2
SAME_TYPE_REPULSION_DISTANCE = 0.0125
REPULSION_STRENGTH = 0.00025
SAME_TYPE_REPULSION_STRENGTH = 0.0005


class Brain:
    '''The main brain of the system
    
    It is setup as a feedforward neural network with a variable number of hidden layers.
    
    Input -> Hidden1 -> Hidden2 -> ... -> Output

    Example, if there are 2 hidden layers:
    Input -> Hidden1 -> Hidden2 -> Output

    The 'layer' determines the source of the connection. Since there are 3 sources here, i.e. one input and two hidden, the layer is modded by 3.
    We can think of the 'layer' as which arrow we are following in the above diagram.

    So a few example genes:
    AG0F3Z: Input[A] -> Hidden1[G], Weight F3Z
    1B7C0A: Hidden1[1] -> Hidden2[B], Weight C0A
    2D2E1B: Hidden2[2] -> Output[D], Weight E1B
    
    '''

    def __init__(self, genome: str, hidden_layers: int = 2):
        self.genome = genome
        self.hidden_layers = hidden_layers

        self.input_nodes = ['osc1', 'osc2', 'osc3', 'rng', 'age', 'energy', 'direction', 'velocity', 'dist_food', 'dist_poison', 'touch_food', 'touch_poison']
        self.output_nodes = ['move', 'turn', 'eat', 'mitosis']
        self._hidden_nodes = ['identity', 'sum', 'invert', 'tanh', 'indicator', 'gaussian', 'am', 'divide', 'multiply']
        self.hidden_nodes = []

        for i in range(hidden_layers):
            hn = self._hidden_nodes.copy()
            for item in hn:
                item.join(str(i))
            self.hidden_nodes.extend(hn)

        self.connections = self._decode_genome(genome)

    def _decode_genome(self, genome: str) -> Dict[str, List[Tuple[str, float]]]:
        '''Decodes the genome into a dictionary of nodes and weights'''

        connections = {node: [] for node in self.input_nodes + self.output_nodes + self.hidden_nodes}

        for i in range(0, len(genome), 6):
            if i + 6 <= len(genome):
                gene = genome[i:i+6]

                source_index = int(gene[0], BASE)
                dest_index = int(gene[1], BASE)
                layer = int(gene[2], BASE) % (self.hidden_layers + 1)
                weight = (int(gene[3:], BASE) / (BASE ** 3)) * 2.0 - 1.0

                if layer == 0:
                    source_index = source_index % len(self.input_nodes)
                    dest_index = dest_index % len(self.hidden_nodes)
                
                elif layer == self.hidden_layers:
                    source_index = source_index % len(self.hidden_nodes)
                    dest_index = dest_index % len(self.output_nodes)

                else:
                    source_index = source_index % len(self.hidden_nodes)
                    dest_index = dest_index % len(self.hidden_nodes)
                
                source_node = self.input_nodes[source_index] if layer == 0 else self.hidden_nodes[source_index]
                dest_node = self.hidden_nodes[dest_index] if layer < self.hidden_layers else self.output_nodes[dest_index]

                connections[source_node].append((dest_node, weight))

        return connections
    
    def process(self, inputs: Dict[str, float]) -> Dict[str, float]:
        node_values = {**inputs}

        # If the node values are not present, we'll initialize them to 0
        for node in self.hidden_nodes + self.output_nodes:
            if node not in node_values:
                node_values[node] = 0.0
        
        # Process hidden layers
        for _ in range(self.hidden_layers):
            new_values = {}
            for node in self.hidden_nodes + self.output_nodes:
                incoming = sum(node_values[src] * weight for src, connections in self.connections.items() for dest, weight in connections if dest == node)
                if 'identity' in node:
                    new_values[node] = incoming
                elif 'sum' in node:
                    new_values[node] = sum(node_values.values())
                elif 'invert' in node:
                    new_values[node] = -incoming
                elif 'tanh' in node:
                    new_values[node] = np.tanh(incoming)
                elif 'indicator' in node:
                    new_values[node] = 1 if incoming > 0 else 0
                elif 'gaussian' in node:
                    new_values[node] = np.exp(-incoming ** 2)
                elif 'am' in node:
                    new_values[node] = np.sin(time.time() * 5.0) * incoming
                elif 'divide' in node:
                    new_values[node] = incoming * 0.5
                elif 'multiply' in node:
                    new_values[node] = incoming * 2.0
            node_values.update(new_values)
        
        # Process output layer
        outputs = {}
        for node in self.output_nodes:
            incoming = sum(node_values[src] * weight for src, connections in self.connections.items() for dest, weight in connections if dest == node)
            outputs[node] = np.tanh(incoming)  # Using tanh activation for output nodes
        
        return outputs
    
    def plot(self):
        # We'll plot the neural network and the connections

         # Create a directed graph
        G = nx.DiGraph()

        # Add nodes
        input_nodes = self.input_nodes
        hidden_nodes = [node for i in range(self.hidden_layers) for node in self._hidden_nodes]
        output_nodes = self.output_nodes

        # Position nodes in layers
        pos = {}
        layers = [input_nodes] + [hidden_nodes[i:i+len(self._hidden_nodes)] for i in range(0, len(hidden_nodes), len(self._hidden_nodes))] + [output_nodes]
        
        for i, layer in enumerate(layers):
            layer_height = len(layer)
            for j, node in enumerate(layer):
                pos[node] = (i, j - (layer_height - 1) / 2)

        # Add edges (connections)
        for source, connections in self.connections.items():
            for dest, weight in connections:
                G.add_edge(source, dest, weight=weight)

        # Set up the plot
        plt.figure(figsize=(12, 8))
        
        # Draw the nodes
        nx.draw_networkx_nodes(G, pos, nodelist=input_nodes, node_color='lightblue', node_size=500, label='Input')
        for i in range(self.hidden_layers):
            start = i * len(self._hidden_nodes)
            end = (i + 1) * len(self._hidden_nodes)
            nx.draw_networkx_nodes(G, pos, nodelist=hidden_nodes[start:end], node_color=f'C{i+1}', node_size=500, label=f'Hidden {i+1}')
        nx.draw_networkx_nodes(G, pos, nodelist=output_nodes, node_color='salmon', node_size=500, label='Output')
        
        # Draw the edges
        edges = nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)
        
        # Add labels to nodes
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        # Color edges based on weight
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        vmin = min(weights)
        vmax = max(weights)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])

        for (u, v, d) in G.edges(data=True):
            edge_color = sm.to_rgba(d['weight'])
            edges = nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=[edge_color], arrows=True)

        # Add a legend
        plt.legend()

        # Remove axis
        plt.axis('off')
        
        # Show the plot
        plt.tight_layout()
        plt.show()



class Agent:

    def __init__(self, genome: str = ''):
        self.hls = (np.random.uniform(0.4, 0.8), 0.8, 0.8)
        self.num_genes = NUM_GENES
        self.hidden_layers = HIDDEN_LAYERS
        
        self.gene_length = 6
        self.genome = None

        if genome == '':
            self.genome = self.generate_random_genome()
        else:
            self.genome = genome

        self.brain = Brain(self.genome, self.hidden_layers)

    def generate_random_genome(self) -> str:
        return ''.join(np.random.choice(list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'), self.num_genes * self.gene_length))
    
    def think(self, env_inputs: Dict[str, float]) -> Dict[str, float]:
        return self.brain.process(env_inputs)



class Environment:

    def __init__(self):

        # We'll store the agent / world data in numpy arrays
        # Agent data
        self.agent = np.zeros(MAX_POPULATION, dtype=Agent)
        self.agent_age = np.zeros(MAX_POPULATION, dtype=np.float32)
        self.agent_energy = np.zeros(MAX_POPULATION, dtype=np.float32)
        self.agent_pos = np.random.uniform((MIN_X, MAX_Y), (MAX_X, MAX_Y), (MAX_POPULATION, 2))
        self.agent_vel = np.zeros((MAX_POPULATION, 2), dtype=np.float32)

        self.num_agents = STARTING_POPULATION

        # Food data
        self.food_pos = np.random.uniform((MIN_X, MIN_Y), (MAX_X, MAX_Y), (MAX_FOOD, 2))
        self.food_vel = np.zeros((MAX_FOOD, 2), dtype=np.float32)
        self.food_energy = np.zeros(MAX_FOOD, dtype=np.float32)

        # Poison data
        self.poison_pos = np.random.uniform((MIN_X, MIN_Y), (MAX_X, MAX_Y), (NUM_POISON, 2))
        self.poison_vel = np.zeros(((NUM_POISON, 2)), dtype=np.float32)


        # Distances between everything
        self.agent_food_distances = np.zeros((MAX_POPULATION, MAX_FOOD), dtype=np.float32)
        self.agent_poison_distances = np.zeros((MAX_POPULATION, NUM_POISON), dtype=np.float32)
        self.food_poison_distances = np.zeros((MAX_FOOD, NUM_POISON), dtype=np.float32)

        self.agent_agent_distances = np.zeros((MAX_POPULATION, MAX_POPULATION), dtype=np.float32)
        self.food_food_distances = np.zeros((MAX_FOOD, MAX_FOOD), dtype=np.float32)
        self.poison_poison_distances = np.zeros((NUM_POISON, NUM_POISON), dtype=np.float32)
    
    def add_agent(self, N: int = 1, genome: str = ''):

        for _ in range(N):
            new_position = np.random.uniform((MIN_X, MIN_Y), (MAX_X, MAX_Y))

            if self.num_agents < MAX_POPULATION:
                # Find the position where we can add the agent
                index = np.argmin(self.agent_energy)

                # We'll create a new agent
                if genome != '':
                    self.agent[index] = Agent(genome)
                else:
                    self.agent[index] = Agent()

                # We'll make the energy 1, and set the agent's position
                self.agent_energy[index] = 1.0
                self.agent_pos[index] = new_position

                self.num_agents += 1

    def add_food(self, N: int = 1):

        for _ in range(N):
            new_position = np.random.uniform((MIN_X, MIN_Y), (MAX_X, MAX_Y))

            # Let's count how much food we have
            num_food = np.sum(np.where(self.food_energy > 0, 1, 0))

            if num_food < MAX_FOOD:
                # Find the position where we can add the food
                index = np.argmin(self.food_energy)

                # We'll make the energy 1, and set the food's position
                self.food_energy[index] = 1.0
                self.food_pos[index] = new_position

    def update(self):

        # Agent Update
        indices = np.where(self.agent_energy > 0)[0]

        # distances between everything
        self.calculate_distances()
        
        for i in indices:
            agent = self.agent[i]
            inputs = self.get_agent_inputs(i)
            output: Dict[str, float] = agent.think(inputs)

            # We'll update the agent's velocity magnitude based on the output
            turn_direction = output['turn'] * 0.0625 + np.arctan2(self.agent_vel[i][1], self.agent_vel[i][0])
            velocity_magnitude = np.clip(output['move'], -1, 1)
            self.agent_vel[i] = np.array([np.cos(turn_direction), np.sin(turn_direction)]) * velocity_magnitude

            # We'll handle conditional outputs
            if output['eat'] > 0:
                self.handle_eating(i)
            
            if np.abs(output['mitosis']) > 0.5 and self.agent_energy[i] > MITOSIS_ENERGY and self.agent_age[i] > np.random.randn() * 20 + 150:
                self.handle_mitosis(i)


        # Finally update everyone's parameters
        self.agent_age += AGE_STEP
        self.agent_energy += ENERGY_STEP
        self.agent_energy = np.clip(self.agent_energy, 0.0, 1.0)
        # We'll count the number of agents that have energy > 0 now
        self.num_agents = np.sum(np.where(self.agent_energy > 0, 1, 0))

        velocity = np.tanh(self.agent_vel) * VELOCITY_LIMIT

        # Food Update
        self.food_energy += FOOD_ENERGY_STEP
        food_mask = np.random.rand(MAX_FOOD)
        food_mask = np.where(food_mask < REPLENISH_RATE, 1, 0)
        for i in range(MAX_FOOD):
            if food_mask[i] > 0:
                self.food_energy[i] = 1.0

        # Poison Update
        self.handle_poison()

        # Collision Handling
        self.handle_collisions()

        self.agent_pos += velocity * FORCE_FACTOR - velocity * FRICTION

        # We'll add some noise to the food and poison
        self.food_pos += np.random.randn(*self.food_pos.shape) * 0.000625
        self.poison_pos += np.random.randn(*self.poison_pos.shape) * 0.000625

        # Ensure everything stays within bounds
        self.agent_pos = np.mod(self.agent_pos, (MAX_X, MAX_Y))
        self.food_pos = np.mod(self.food_pos, (MAX_X, MAX_Y))
        self.poison_pos = np.mod(self.poison_pos, (MAX_X, MAX_Y))


    def get_agent_inputs(self, index: int) -> Dict[str, float]:
        '''Get the inputs for the agent at the given index'''
        inputs = {
            'osc1': np.sin(self.agent_age[index] / 100).item(),  # Oscillator 1, period 100
            'osc2': np.sin(time.time()).item(), 
            'osc3': 2 * np.floor(np.mod(time.time() / np.pi, 1)).item() - 1,  # Square wave oscillator
            'rng': np.random.uniform(-1, 1),
            'age': np.tanh(self.agent_age[index] / 1000).item(),
            'energy': self.agent_energy[index].item(),
            'direction': np.arctan2(self.agent_vel[index][1], self.agent_vel[index][0]).item() / np.pi,
            'velocity': np.linalg.norm(self.agent_vel[index]).item(),
            'dist_food': self.dist_to_closest_food(index).item(),
            'dist_poison': self.dist_to_closest_poison(index).item(),
            'touch_food': self.dist_to_closest_food(index).item() < EAT_DISTANCE,
            'touch_poison': self.dist_to_closest_poison(index).item() < POISON_RADIUS + POISON_SENSE
        }       
        return inputs 
    
    def _distance(self, A, B):
        # We'll use vectorized operations to calculate the distances between agents and food
        # Since we're on a toroidal world, we need to consider the wrap around      
        dx = np.fmod(A[:, 0][:, np.newaxis] - B[:, 0], MAX_X)
        dy = np.fmod(A[:, 1][:, np.newaxis] - B[:, 1], MAX_Y)
        return np.sqrt(dx ** 2 + dy ** 2)

    def calculate_distances(self):  
        agent_pos = self.agent_pos
        food_pos = self.food_pos
        poison_pos = self.poison_pos

        self.agent_food_distances = self._distance(agent_pos, food_pos)
        self.agent_poison_distances = self._distance(agent_pos, poison_pos)
        self.food_poison_distances = self._distance(food_pos, poison_pos)

        self.agent_agent_distances = self._distance(agent_pos, agent_pos)
        self.food_food_distances = self._distance(food_pos, food_pos)
        self.poison_poison_distances = self._distance(poison_pos, poison_pos)

    
    def dist_to_closest_food(self, agent_index: int) -> float:
        # We'll find the closest food to the agent
        distances = self.agent_food_distances[agent_index]
        closest_food_index = np.argmin(distances)
        return distances[closest_food_index]
    
    def dist_to_closest_poison(self, agent_index: int) -> float:
        distances = self.agent_poison_distances[agent_index]
        index = np.argmin(distances)
        if distances[index] <= POISON_RADIUS + POISON_SENSE:
            return np.array([1.0])
        else:
            return np.array([0.0])
        # return distances[index]
    
    def handle_eating(self, agent_index: int):
        # We'll find the closest food to the agent
        distances = self.agent_food_distances[agent_index]
        closest_food_index = np.argmin(distances)

        # If the agent is close enough to the food, we'll eat it
        if distances[closest_food_index] < EAT_DISTANCE:
            self.agent_energy[agent_index] += self.food_energy[closest_food_index]
            self.food_energy[closest_food_index] = 0.0


    def handle_mitosis(self, agent_index: int):
        # we'll take the agent, and create 2 new agents with mutated genomes
        old_genome = self.agent[agent_index].genome
        old_energy = self.agent_energy[agent_index]
        old_color = self.agent[agent_index].hls
        
        mutation_mask1 = np.random.rand(len(old_genome))
        mutation_mask1 = np.where(mutation_mask1 < MUTATION_RATE, 1, 0)

        mutation_mask2 = np.random.rand(len(old_genome))
        mutation_mask2 = np.where(mutation_mask2 < MUTATION_RATE, 1, 0)

        new_genome1 = ''.join([old_genome[i] if mutation_mask1[i] == 0 else np.random.choice(list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')) for i in range(len(old_genome))])
        new_genome2 = ''.join([old_genome[i] if mutation_mask2[i] == 0 else np.random.choice(list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')) for i in range(len(old_genome))])

        # We'll remove the old agent and add the new ones if there's room
        self.agent[agent_index] = Agent(new_genome1)
        self.agent_pos[agent_index] = self.agent_pos[agent_index]
        self.agent_energy[agent_index] = old_energy / 2
        self.agent_age[agent_index] = 0
        self.agent[agent_index].hls = old_color

        if self.num_agents < MAX_POPULATION:
            new_index = np.argmin(self.agent_energy)
            self.agent[new_index] = Agent(new_genome2)
            self.agent_energy[new_index] = 1.0
            self.agent_pos[new_index] = self.agent_pos[agent_index]
            self.agent_energy[new_index] = old_energy / 2
            self.agent_age[new_index] = 0
            self.agent[new_index].hls = old_color
            self.num_agents += 1

    def handle_poison(self):
        # Get indices where distances are within the poison radius
        p_idxs = np.where(self.agent_poison_distances < POISON_RADIUS, 1, 0)
        p_idxs = np.any(p_idxs == 1, axis=1)

        # We'll get the minimum distances
        p_distances = np.min(self.agent_poison_distances, axis=1)

        # Apply poison based on distance
        p_amount = POISON_STEP * (POISON_RADIUS - p_distances) / POISON_RADIUS
        p_amount *= p_idxs

        self.agent_energy += p_amount


    def handle_collisions(self):

        repulsion_distance = REPULSION_DISTANCE
        repulsion_strength = REPULSION_STRENGTH
        same_type_repulsion_distance = SAME_TYPE_REPULSION_DISTANCE  # New constant for same-type repulsion
        same_type_repulsion_strength = SAME_TYPE_REPULSION_STRENGTH  # New constant for same-type repulsion strength

        # Food-Poison repulsion
        distances = self.food_poison_distances
        close_pairs = np.argwhere(distances < repulsion_distance)

        for food_idx, poison_idx in close_pairs:
            direction = self.food_pos[food_idx] - self.poison_pos[poison_idx]
            distance = np.linalg.norm(direction)
            if distance > 0:
                direction /= distance
            force = direction * repulsion_strength * (1 - distance / repulsion_distance)
            self.food_vel[food_idx] += force
            self.poison_vel[poison_idx] -= force

        # Food-Food repulsion
        food_distances = self.food_food_distances
        close_food_pairs = np.argwhere((food_distances < same_type_repulsion_distance) & (food_distances > 0))

        for food1_idx, food2_idx in close_food_pairs:
            direction = self.food_pos[food1_idx] - self.food_pos[food2_idx]
            distance = np.linalg.norm(direction)
            if distance > 0:
                direction /= distance
            force = direction * same_type_repulsion_strength * (1 - distance / same_type_repulsion_distance)
            self.food_vel[food1_idx] += force
            self.food_vel[food2_idx] -= force

        # Poison-Poison repulsion
        poison_distances = self.poison_poison_distances
        close_poison_pairs = np.argwhere((poison_distances < same_type_repulsion_distance) & (poison_distances > 0))

        for poison1_idx, poison2_idx in close_poison_pairs:
            direction = self.poison_pos[poison1_idx] - self.poison_pos[poison2_idx]
            distance = np.linalg.norm(direction)
            if distance > 0:
                direction /= distance
            force = direction * same_type_repulsion_strength * (1 - distance / same_type_repulsion_distance)
            self.poison_vel[poison1_idx] += force
            self.poison_vel[poison2_idx] -= force

        # Update positions based on velocities
        self.food_pos += self.food_vel
        self.poison_pos += self.poison_vel

        # Ensure food and poison stay within bounds
        self.food_pos = np.mod(self.food_pos, (MAX_X, MAX_Y))
        self.poison_pos = np.mod(self.poison_pos, (MAX_X, MAX_Y))

        # Apply friction to gradually slow down food and poison
        friction = 1 - FRICTION
        self.food_vel *= friction
        self.poison_vel *= friction



    ############################################
    ############ Visualization #################
    ############################################

    def visualize(self):

        def local_to_global(local_pos: np.ndarray) -> np.ndarray:
            X = (local_pos[0] - MIN_X) / (MAX_X - MIN_X) * DISPLAY_WIDTH
            Y = (local_pos[1] - MIN_Y) / (MAX_Y - MIN_Y) * DISPLAY_HEIGHT

            return (int(X), int(Y))      

        # Create a blank image
        image = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)

        # Draw the agents
        live_agents = np.where(self.agent_energy > 0)[0]
        live_indices = live_agents.tolist()

        for i in live_indices:

            pos = local_to_global(self.agent_pos[i])
            hls = self.agent[i].hls

            light = hls[1] * np.tanh(self.agent_energy[i]) * 0.8
            if self.agent_age[i] < 32:
                light = 1.0

            color = colorsys.hls_to_rgb(hls[0], light, hls[2])
            color = tuple([int(255 * c) for c in color])
            image = cv2.circle(image, pos, 10, color, 5)

        # Draw the food
        live_food = np.where(self.food_energy > 0)[0]

        for i in live_food:
            pos = local_to_global(self.food_pos[i])
            h = 0.25
            l = np.clip(self.food_energy[i], 0, 1)
            s = 1.0
            color = colorsys.hls_to_rgb(h, l, s)
            color = tuple([int(255 * c) for c in color])
            image = cv2.circle(image, pos, 3, color, 1)

        # Draw the poison
        r = local_to_global((POISON_RADIUS, POISON_RADIUS))
        r = (r[0] + r[1]) / 2
        r = int(r)

        r2 = local_to_global((POISON_RADIUS + POISON_SENSE, POISON_RADIUS + POISON_SENSE))
        r2 = (r2[0] + r2[1]) / 2
        r2 = int(r2)

        for i in range(NUM_POISON):
            pos = local_to_global(self.poison_pos[i])

            h = 0.0
            l = 0.5
            s = 1.0
            color = colorsys.hls_to_rgb(h, l, s)
            color = tuple([int(255 * c) for c in color])
            # image = cv2.circle(image, pos, r, color, 2)
            # image = cv2.circle(image, pos, r2, color, 1)
            image = cv2.circle(image, pos, 5, color, -1)

        # Display the image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow(WINDOW_NAME, image)
        cv2.waitKey(1)



def main():

    env = Environment()

    env.add_agent(STARTING_POPULATION)
    env.add_food(STARTING_FOOD)

    # env.agent[0].brain.plot()

    key = cv2.waitKey(1) & 0xFF

    while True:

        env.visualize()
        env.update()
        
        if key == ord('q') or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()