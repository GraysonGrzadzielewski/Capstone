import torch
from pathlib import Path
import time
import json
import gc

class Common():
    def __init__(self, savedir, name=None):
        self.savedir = Path(savedir)
        if name is None:
            self.name = self.savedir / (str(time.time()) + '.pt')
        else:
            self.name= self.savedir / name

    def set_save_dir(self, directory):
        self.savedir = directory

    def backup(self):
        if self.name.exists():
            if self.name.with_suffix('.bak').exists():
                self.name.with_suffix('.bak').unlink()
            self.name.rename(self.name.with_suffix('.bak'))

    def save_torch_agent(self, agent, cnn, per, aopt, popt):
        self.backup()  # Backup model before overwriting
        torch.save({
            'agent': agent.state_dict(),
            'cnn': cnn.state_dict(),
            'per': per.state_dict(),
            'aopt': aopt.state_dict(),
            'popt': popt.state_dict()
        }, self.name)


class StatTrack():
    """ Track execution statistics for curiosity
    """

    def __init__(self, savedir, name=None, load=False):

        self.savedir = Path(savedir)
        if name is None:
            self.name = self.savedir / (str(time.time()) + '.json')
        else:
            self.name= self.savedir / name
        if load:
            f = open(self.name, 'r')
            jstring = f.read()
            f.close()
            self.board = json.loads(jstring)
        else:
            self.board = {
                "Checkpoints": 0,
                "Iterations": 0,
                "CheckpointScores": [],
                "CheckpointAverage": [],
                "IterationScores": [],
                "IterationAverage": []
            }
        self.check = self.board["CheckpointScores"]
        self.check_av = self.board["CheckpointAverage"]
        self.iter = self.board["IterationScores"]
        self.iter_av = self.board["IterationAverage"]

    def update_checkpoint(self, tup):
        self.check.append(tup[0])
        self.check_av.append(tup[1])
        self.board["Checkpoints"] += 1

    def update_iter(self, tup):
        self.iter.append(tup[0])
        self.iter_av.append(tup[1])
        self.board["Iterations"] += 1

    def save_stats(self):
        jstring = json.dumps(self.board)
        with open(self.name, 'w') as f:
            f.write(jstring)
            f.close()
