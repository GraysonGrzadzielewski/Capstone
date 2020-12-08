import neat

class MarioStatisticsReporter(neat.StatisticsReporter):

    def __init__(self, out_dir='./'):
        neat.StatisticsReporter.__init__(self)
        self.out_dir = out_dir

    # Statistics reporter that saves after every run
    def post_evaluate(self, config, population, species, best_genome):
        neat.StatisticsReporter.post_evaluate(self, config, population, species, best_genome)
        self.save()

    def save(self):
        self.save_genome_fitness(delimiter=',', filename=f'{self.out_dir}fitness_history.csv')
        self.save_species_count(delimiter=",", filename=f'{self.out_dir}speciation.csv')
        self.save_species_fitness(delimiter=",", filename=f'{self.out_dir}species_fitness.csv')
