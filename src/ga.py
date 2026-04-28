"""
Genetic Algorithm for black-box adversarial attack generation.
No access to model weights or gradients — only input/output queries.
"""

import numpy as np

# CIFAR-10 нормализованные границы пикселей [0,1] → нормализованное пространство
_MEAN = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
_STD  = np.array([0.2023, 0.1994, 0.2010]).reshape(3, 1, 1)
NORM_MIN = (0 - _MEAN) / _STD
NORM_MAX = (1 - _MEAN) / _STD


class GeneticAttack:
    def __init__(
        self,
        model,
        population_size=50,
        generations=30,
        mutation_rate=0.1,
        mutation_strength=0.05,
        epsilon=0.1,
        device="cpu",
    ):
        """
        Args:
            model:             target neural network (black-box)
            population_size:   number of individuals per generation
            generations:       number of evolution steps
            mutation_rate:     probability of mutating each gene
            mutation_strength: magnitude of mutation noise
            epsilon:           max allowed perturbation (L-inf norm)
            device:            'cpu' or 'cuda'
        """
        self.model = model
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.epsilon = epsilon
        self.device = device

        self.model.eval()

    def _fitness(self, image, perturbations, true_label):
        """
        Fitness = prob(best_wrong_class) - prob(true_class).
        Непрерывный сигнал: GA учится даже когда атака ещё не удалась.
        """
        import torch

        scores = []
        for p in perturbations:
            adv = np.clip(image + p, NORM_MIN, NORM_MAX)
            adv_tensor = torch.FloatTensor(adv).unsqueeze(0).to(self.device)

            with torch.no_grad():
                probs = torch.softmax(self.model(adv_tensor), dim=1)[0]

            prob_true = probs[true_label].item()
            probs[true_label] = 0.0
            prob_best_wrong = probs.max().item()

            scores.append(prob_best_wrong - prob_true)

        return np.array(scores)

    def _crossover(self, parent1, parent2):
        mask = np.random.rand(*parent1.shape) > 0.5
        child = np.where(mask, parent1, parent2)
        return child

    def _mutate(self, individual):
        mutation_mask = np.random.rand(*individual.shape) < self.mutation_rate
        noise = np.random.randn(*individual.shape) * self.mutation_strength
        individual = individual + mutation_mask * noise
        individual = np.clip(individual, -self.epsilon, self.epsilon)
        return individual

    def attack(self, image, true_label):
        """
        Run GA to find adversarial perturbation for a single image.

        Args:
            image:      numpy array (C, H, W) in [0, 1]
            true_label: int, correct class

        Returns:
            best_perturbation: numpy array same shape as image
            history:           list of best fitness per generation
        """
        # Initialize random population within epsilon ball
        population = [
            np.random.uniform(-self.epsilon, self.epsilon, image.shape)
            for _ in range(self.population_size)
        ]

        history = []
        best_perturbation = population[0]
        best_fitness = -1

        for gen in range(self.generations):
            fitness_scores = self._fitness(image, population, true_label)

            gen_best_idx = np.argmax(fitness_scores)
            if fitness_scores[gen_best_idx] > best_fitness:
                best_fitness = fitness_scores[gen_best_idx]
                best_perturbation = population[gen_best_idx].copy()

            history.append(best_fitness)

            # Selection: keep top 50%
            n_survivors = self.population_size // 2
            survivor_idx = np.argsort(fitness_scores)[-n_survivors:]
            survivors = [population[i] for i in survivor_idx]

            # Create next generation
            next_population = survivors.copy()
            while len(next_population) < self.population_size:
                p1, p2 = np.random.choice(len(survivors), 2, replace=False)
                child = self._crossover(survivors[p1], survivors[p2])
                child = self._mutate(child)
                next_population.append(child)

            population = next_population

        return best_perturbation, history
