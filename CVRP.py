import random
import csv
import time

def load_vrp_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    nodes = []
    demands = {}
    depot = capacity = None
    reading_nodes = reading_demands = reading_depot = False
    for line in lines:
        line = line.strip()
        if line.startswith("CAPACITY"):
            parts = line.split()
            last_part = parts[-1]
            capacity = int(last_part)
        elif line.startswith("NODE_COORD_SECTION"):
            reading_nodes = True
            continue
        elif line.startswith("DEMAND_SECTION"):
            reading_nodes = False
            reading_demands = True
            continue
        elif line.startswith("DEPOT_SECTION"):
            reading_demands = False
            reading_depot = True
            continue
        elif line.startswith("EOF"):
            break
        if reading_nodes:
            parts = line.split()
            nodes.append((int(parts[0]), int(parts[1]), int(parts[2])))
        elif reading_demands:
            parts = line.split()
            demands[int(parts[0])] = int(parts[1])
        elif reading_depot and int(line) > 0:
            depot = int(line)
    return nodes, demands, depot, capacity, filename

def save_to_csv(filename, data):
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerows(data)

def distance(a, b):
    return ((a[1] - b[1])**2 + (a[2] - b[2])**2) ** 0.5

def distances():
    dist_matrix = [[0] * len(nodes) for i in range(len(nodes))]
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            dist_matrix[i][j] = distance(nodes[i], nodes[j])
    return dist_matrix

def evaluate(solution):
    cost = sum(dist_matrix[solution[i] - 1][solution[i + 1] - 1] for i in range(len(solution) - 1))
    return cost

def back_to_depot(node, route, load, depot):
    new_route = list(route)
    if (load + demands[node] > capacity) and (new_route[-1] != depot):
        new_route.append(depot)  
        load = 0
    new_route.append(node)
    load += demands[node]
    return new_route, load

def cut_depot(route, depot):
    new_route = list(route)
    while depot in new_route:
        new_route.remove(depot)
    return new_route

def random_solution(load=0):
    sol = list(range(1, len(nodes) + 1))
    sol = cut_depot(sol, depot)
    random.shuffle(sol)
    route = [depot]
    for node in sol:
        route, load = back_to_depot(node, route, load, depot)
    if route[-1] != depot:
        route.append(depot)
    return route

#def greedy_algorithm(load=0):
    greedy_pop = []
    for n in range(1, len(nodes) + 1):
        unvisited = set(range(1, len(nodes) + 1))
        unvisited = cut_depot(unvisited, n)
        route = [n]
        while unvisited:
            last = route[-1]
            greed = min(unvisited, key = lambda x: dist_matrix[last - 1][x - 1])
            route, load = back_to_depot(greed, route, load, n)
            unvisited.remove(greed)
        if route[-1] != n:
            route.append(n)
        greedy_pop.append(route)
    return greedy_pop

def greedy_algorithm(load=0):
    greedy_pop = []
    n_init = set(range(1, len(nodes) + 1))
    n_init = cut_depot(n_init, depot)
    for n in n_init:
        unvisited = set(range(1, len(nodes) + 1))
        unvisited = cut_depot(unvisited, depot)
        unvisited.remove(n)
        route = [depot, n]
        while unvisited:
            last = route[-1]
            greed = min(unvisited, key = lambda x: dist_matrix[last - 1][x - 1])
            route, load = back_to_depot(greed, route, load, depot)
            unvisited.remove(greed)
        if route[-1] != depot:
            route.append(depot)
        greedy_pop.append(route)

    return greedy_pop

class TabuSearch:
    def __init__(self, max_iterations=10000, tabu_size=10, neighborhood=40):
        self.max_iterations = max_iterations
        self.tabu_size = tabu_size
        self.neighborhood = neighborhood

    def get_neighbors(self, solution, load=0):
        neighbors = []
        solution = cut_depot(solution, depot)
        for _ in range(self.neighborhood):
            neighbor = list(solution)
            i, j = random.sample(range(len(neighbor)), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            new_solution = [depot]
            for node in neighbor:
                new_solution, load = back_to_depot(node, new_solution, load, depot)
            if new_solution[-1] != depot:
                new_solution.append(depot)
            neighbors.append(new_solution)
        return neighbors

    def search(self, initial_solution):
        best_solution = list(initial_solution)
        best_candidate = list(initial_solution)
        tabu_list = [initial_solution]
        write_TS = []
        write_TS.append(['Iteracja TS', 'best'])
        for i in range(self.max_iterations):
            neighbors = self.get_neighbors(best_candidate)
            best_candidate_evaluation = float("inf")
            new_best_candidate = None
            for candidate in neighbors:
                if candidate not in tabu_list:
                    if evaluate(candidate) < best_candidate_evaluation:
                        new_best_candidate = candidate
                        best_candidate_evaluation = evaluate(candidate)
            if new_best_candidate is None:
                break
            best_candidate = new_best_candidate
            if evaluate(best_candidate) < evaluate(best_solution):
                best_solution = best_candidate
            tabu_list.append(best_candidate)
            if len(tabu_list) > self.tabu_size:
                tabu_list.pop(0)
            write_TS.append([i+1, evaluate(best_solution)])
        #save_to_csv("c:\\Users\\Michał\\Documents\\Studia\\Magisterskie\\1semestr\\Metody optymalizacji\\projekt0_met_opt.csv", write_TS)
        return best_solution

class EvolutionAlgorithm:
    def __init__(self, pop_size=1000, generations=250, cross_prob=0.9, mut_prob=0.4, Tour=2, elite_ratio=0.1):
        self.pop_size = pop_size
        self.generations = generations
        self.cross_prob = cross_prob
        self.mut_prob = mut_prob
        self.Tour = Tour
        self.elite_ratio = elite_ratio
        self.pop = []
        for i in range(pop_size):
            self.pop.append(random_solution())

    def evolve(self):
        write_EA = []
        write_EA.append([instance + ', Pop_size: ' + str(ea.pop_size) + ', Gens: ' + str(ea.generations) + ', Cross_prob: ' + str(ea.cross_prob) + ', Mut_prob: ' + str(ea.mut_prob) + ', Tour: ' + str(ea.Tour)+ ', elite_ratio: ' + str(ea.elite_ratio) + ', Mutacja: swap' + ', Krzyzowanie: OX'])
        write_EA.append(['Epoka EA', 'best', 'average', 'worst'])
        for i in range(self.generations):
            new_pop = []
            for individual in self.selection_elitism():
                new_pop.append(individual)
            write_EA.append([int(i), evaluate(ea.best_solution(self.pop)), ea.avg_solution(self.pop), evaluate(ea.worst_solution(self.pop))])
            while len(new_pop) != self.pop_size:
                p1, p2 = self.selection_tournament(), self.selection_tournament()
                if random.random() < self.cross_prob:
                    individual = self.crossover_OX(p1, p2)
                else: individual = list(p1)
                if random.random() < self.mut_prob:
                    individual = self.mutate_inverse(individual)
                new_pop.append(individual)
            self.pop = new_pop
        write_EA.append([int(self.generations), evaluate(ea.best_solution(self.pop)), ea.avg_solution(self.pop), evaluate(ea.worst_solution(self.pop))])
        #save_to_csv("c:\\Users\\Michał\\Documents\\Studia\\Magisterskie\\1semestr\\Metody optymalizacji\\projekt0_met_opt.csv", write_EA)

    def selection_elitism(self):
        sorted_pop = sorted(self.pop, key=lambda p: evaluate(p))
        elite_pop = sorted_pop[:int(len(self.pop)*self.elite_ratio)]
        return elite_pop

    def selection_tournament(self):
        tournament = random.sample(self.pop, self.Tour)
        best_individual = min(tournament, key=lambda p: evaluate(p))
        return best_individual
    
    def crossover_OX(self, p1, p2, load=0):
        p1 = cut_depot(p1, depot)
        p2 = cut_depot(p2, depot)
        cut1, cut2 = sorted(random.sample(range(len(p1)), 2))
        p1_part = []
        p2_part = []
        o1 = [None] * len(p1)
        for node in range(cut1, cut2):
            p1_part.append(p1[node])
            o1[node] = p1[node]
        for node in p2:
            if node not in p1_part:
                p2_part.append(node)
        p2_part_node = 0
        for node in range(len(o1)):
            if o1[node] is None:
                o1[node] = p2_part[p2_part_node]
                p2_part_node += 1
        individual = [depot]
        for node in o1:
            individual, load = back_to_depot(node, individual, load, depot)
        if individual[-1] != depot:
                individual.append(depot)
        return individual
    
    def crossover_CX(self, p1, p2, load=0):
        p1 = cut_depot(p1, depot)
        p2 = cut_depot(p2, depot)
        o1 = [None] * len(p1)
        visited = [False] * len(p1)
        cycle_step = 0
        cycle_positions = []
        while not visited[cycle_step]:
            visited[cycle_step] = True
            cycle_positions.append(cycle_step)
            cycle_step = p1.index(p2[cycle_step])
        for position in cycle_positions:
            o1[position] = p1[position]
        for i in range(len(p1)):
            if o1[i] is None:
                o1[i] = p2[i]
        individual = [depot]
        for node in o1:
            individual, load = back_to_depot(node, individual, load, depot)
        if individual[-1] != depot:
                individual.append(depot)
        return individual

    def mutate_swap_proportion(self, solution, load=0):
            solution = cut_depot(solution, depot)
            swapped=set()
            for node in range(len(solution)):
                if random.random() < self.mut_prob and node not in swapped:
                    sol_without_node = []
                    for i in range(len(solution)):
                        if i != node and i not in swapped:
                            sol_without_node.append(i)
                    if sol_without_node:
                        swap = random.choice(sol_without_node)
                        solution[swap], solution[node] = solution[node], solution[swap]
                        swapped.update({node, swap})
            new_solution = [depot]
            for node in solution:
                new_solution, load = back_to_depot(node, new_solution, load, depot)
            if new_solution[-1] != depot:
                new_solution.append(depot)
            return new_solution
    
    def mutate_inverse(self, solution, load=0):
            solution = cut_depot(solution, depot)
            inverse = [None] * len(solution)
            cut1, cut2 = sorted(random.sample(range(len(solution)), 2))
            node_inverse = cut2
            for node_solution in range(cut1, cut2+1):
                inverse[node_inverse] = solution[node_solution]
                node_inverse -= 1
            for node in range(len(inverse)):
                if inverse[node] is None:
                    inverse[node] = solution[node]
            new_solution = [depot]
            for node in inverse:
                new_solution, load = back_to_depot(node, new_solution, load, depot)
            if new_solution[-1] != depot:
                new_solution.append(depot)
            return new_solution
    
    def mutate_swap(self, solution, load=0):
            solution = cut_depot(solution, depot)
            i, j = random.sample(range(len(solution)), 2)
            solution[i], solution[j] = solution[j], solution[i]
            new_solution = [depot]
            for node in solution:
                new_solution, load = back_to_depot(node, new_solution, load, depot)
            if new_solution[-1] != depot:
                new_solution.append(depot)
            return new_solution
    
    def best_solution(self, pop):
        best = min(pop, key=lambda p: evaluate(p))
        return best
    
    def avg_solution(self, pop):
        avg = sum(evaluate(p) for p in pop) / len(pop)
        return avg
    
    def worst_solution(self, pop):
        worst = max(pop, key=lambda p: evaluate(p))
        return worst

if __name__ == "__main__":
    start_time = time.time()
    nodes, demands, depot, capacity, filename = load_vrp_file("c:\\Users\\Michał\\Documents\\Studia\\Magisterskie\\1semestr\\Metody optymalizacji\\A-n48-k7.txt")
    instance = filename.split("Metody optymalizacji\\")[1].split(".txt")[0]
    dist_matrix = distances()
    write = []
    random_pop = []
    ea = EvolutionAlgorithm()
    ts = TabuSearch()

    for i in range(10000):
        random_pop.append(random_solution())

    ea.evolve()

    best_tabu_solution = ts.search(random_solution())
    greedy_pop = greedy_algorithm()
    
    #write.append(['instancja', 'Optymalny wynik', 'Algorytm losowy[10k]', None, None, None, 'Algorytm zachlanny[n]', None, None, None, 'Algorytm Ewolucyjny[10x]', None, None, None, 'TS[10x]', None, None, None, 'Czas wykonania [s]'])
    end_time = time.time()
    write.append([instance, None, evaluate(ea.best_solution(random_pop)), None, None, None, evaluate(ea.best_solution(greedy_pop)), None, evaluate(ea.worst_solution(greedy_pop)), str(list(evaluate(x) for x in greedy_pop)), evaluate(ea.best_solution(ea.pop)), None, None, None, evaluate(best_tabu_solution), None, None, None, end_time-start_time])
    save_to_csv("c:\\Users\\Michał\\Documents\\Studia\\Magisterskie\\1semestr\\Metody optymalizacji\\projekt0_tab_met_opt.csv", write)
    