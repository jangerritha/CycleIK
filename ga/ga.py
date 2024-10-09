from abc import abstractmethod
import pytorch_kinematics as pk
import torch
from cycleik_pytorch import load_config, renormalize_pose, normalize_pose, slice_fk_pose, renormalize_joint_state, IKDataset
from tqdm import tqdm

class GA:

    def __init__(self, nbr_generations, population_size, mutation_factor, recombination_factor, config, robot, cuda=False, gpu=0):
        self.config = config

        self.nbr_generations = nbr_generations
        self.population_size = population_size
        self.mutation_factor = mutation_factor
        self.recombination_factor = recombination_factor

        self.train_data = self.config["train_data"]
        self.test_data = self.config["test_data"]
        self.robot_dof = self.config["robot_dof"]
        self.robot_urdf = self.config["robot_urdf"]
        self.robot_eef = self.config["robot_eef"]
        self.robot = robot

        device_name = f"cuda:{gpu}" if cuda else "cpu"
        self.device = torch.device(device_name)

        self.population = None
        self.target = None
        self.elite = None

        chain = pk.build_serial_chain_from_urdf(open(self.robot_urdf).read(), self.robot_eef)
        self.chain = chain.to(dtype=torch.float32, device=self.device)

        self.mutation_arrays = torch.zeros((self.nbr_generations, self.population_size, self.robot_dof + 1)).to(self.device)
        for i in range(self.nbr_generations):
            self.mutation_arrays[i, :, :self.robot_dof] = (
                    torch.randn((self.population_size, self.robot_dof)).to(self.device) * 0.3)

    def recombine(self):
        mating_pool = None
        new_population = None
        for individual in self.population.clone():
            recombination_probability = torch.rand(1,1)
            #print(individual[8])
            if recombination_probability[0,0] < 1 - individual[8]:
                if mating_pool is None:
                    mating_pool = individual
                else:
                    mating_pool = torch.vstack((mating_pool, individual))

        #print(mating_pool)
        while len(mating_pool) > 1:
            mating_index = torch.randint(size=(1, 2), high=len(mating_pool))
            individual_1 = mating_pool[mating_index[0,0], :8]
            individual_2 = mating_pool[mating_index[0,1], :8]

            new_individual_1 = torch.zeros(size=(1, self.robot_dof + 1)).to(self.device)
            new_individual_2 = torch.zeros(size=(1, self.robot_dof + 1)).to(self.device)

            crossover_points = torch.randint(size=(1,4), low=2,high=self.robot_dof)
            crossover_points = torch.unique(crossover_points, sorted=True)
            last_point=None
            for crossover_point in crossover_points:
                if last_point is None:
                    new_individual_1[0,:crossover_point] = individual_2[:crossover_point]
                    new_individual_2[0,:crossover_point] = individual_1[:crossover_point]
                else:
                    new_individual_1[0,last_point:crossover_point] = individual_2[last_point:crossover_point]
                    new_individual_2[0,last_point:crossover_point] = individual_1[last_point:crossover_point]
                last_point = crossover_point

            if new_population is None:
                #print(new_individual_1)
                new_population = new_individual_1
                #print(new_individual_2)
                #print(new_population)
                new_population = torch.concatenate((new_population, new_individual_2), dim=0)
                #print(len(new_population))
            else:
                new_population = torch.concatenate((new_population, new_individual_1, new_individual_2), dim=0)

            mating_pool = torch.concatenate((mating_pool[:mating_index[0,0]], mating_pool[mating_index[0,0] + 1:]), dim=0)
            mating_pool = torch.concatenate((mating_pool[:mating_index[0,1]], mating_pool[mating_index[0,1] + 1:]), dim=0)

        if len(new_population) > int(self.population_size * 0.8):
            new_population = new_population[:int(self.population_size * 0.8)]

        nbr_new_individuals = len(new_population)
        #print("\n\nNew Individuals: ", nbr_new_individuals)
        nbr_elites = self.population_size - nbr_new_individuals
        self.elite = self.population[:nbr_elites]

        return new_population, nbr_new_individuals

    def mutate(self, new_population, generation_idx):
        #print(new_population.shape)
        #print(self.mutation_arrays[generation_idx, :len(new_population)].shape)
        #print(self.mutation_arrays[generation_idx].shape)
        new_population = torch.add(new_population,
                                   self.mutation_arrays[generation_idx, :len(new_population)])
        #new_population[:, :8] = new_population[:, :8] + torch.randn((len(new_population), self.robot_dof)).to(self.device) * 0.3
        return new_population

    def evaluate_fitness(self, solution, target, sort=True):
        fk_tensor = self.chain.forward_kinematics(solution)
        forward_result = slice_fk_pose(fk_tensor, batch_size=len(solution))
        position_error = torch.sum(torch.abs(torch.subtract(forward_result[:, :3], target[:, :3])), dim=1) * 100
        orientation_error = torch.sum(torch.abs(torch.subtract(forward_result[:, 3:], target[:, 3:])), dim=1)
        full_error = position_error + orientation_error
        min_val = torch.min(full_error)
        max_val = torch.max(full_error)
        full_error -= min_val
        full_error /= max_val
        new_population = torch.zeros((self.population_size, self.robot_dof)).to(self.device)
        if sort:
            sorted, indices = torch.sort(full_error)
            #print(sorted)
            for e, index in enumerate(indices):
                new_population[e] = solution[index, :self.robot_dof]
            self.population = torch.concat((new_population, torch.reshape(sorted, shape=(len(sorted), 1))), dim=1)
            return sorted
        return full_error


    def init_population(self, initial_seeds):
        nbr_seeds = len(initial_seeds)
        nbr_random_individuals = self.population_size - nbr_seeds
        random_individuals = (torch.rand(nbr_random_individuals, self.robot_dof).to(self.device) * 2) - 1
        population = torch.concat((initial_seeds, random_individuals), dim=0)
        idx = torch.randperm(population.shape[0])
        population = population[idx].view(population.size())
        fitness_value = torch.zeros((self.population_size, 1)).to(self.device)
        torch.concat((population, fitness_value), dim=1)
        return population

    def run(self, target, initial_seeds=torch.Tensor([])):
        self.population = self.init_population(initial_seeds)
        #print(len(self.population))
        self.target = target.repeat((self.population_size, 1))

        for i in tqdm(range(self.nbr_generations)):
            self.evaluate_fitness(self.population, self.target)
            if i == self.nbr_generations - 1:
                return self.population[0]
            new_population, nbr_new_individuals = self.recombine()
            new_population = self.mutate(new_population, i)
            new_population = torch.concatenate((new_population, self.elite), dim=0)
            self.population = new_population[:self.population_size]
