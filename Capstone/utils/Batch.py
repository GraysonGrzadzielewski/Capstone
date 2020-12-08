import random


class Batch():
    def __init__(self, size, minibatch_size):
        self.size = size
        self.minibatch = minibatch_size
        self.buffer_dict = {}

    def append(self, value_t):
        """
        Append value to the buffer, popping off the oldest value if it's full
        :param value_t:
            tuple of total_reward, value_batch
        :return: None
        """
        value, r = value_t
        if len(self.buffer_dict) == 0 or r > min(self.buffer_dict.keys()):
            if len(self.buffer_dict) > self.size:
                self._pop()
            self.buffer_dict[r] = value

    def iter_minibatch(self):
        """
        Iterator to go through and get random minibatches
        :return: iterator for minibatches
        """

        buffer = []
        for k in self.buffer_dict.keys():
            buffer += [v for v in self.buffer_dict[k]]
        random.shuffle(buffer)  # shuffle batch to get minbatches out ofs
        # for batch, flag in lookahead(range(int((self.size / self.minibatch)*.25))):
        for batch in range(int(self.size / self.minibatch)):
            yield buffer[(batch*self.minibatch):(batch+1)*self.minibatch]  # yield a random minibatch
        #del self.buffer_dic
        self.buffer_dict = {}

    def _pop(self):
        _min = min(self.buffer_dict.keys())
        self.buffer_dict.pop(_min)


class FileBuffer():
    def __init__(self, size, minibatch_size):
        import tempfile
        self.size = size
        self.minibatch = minibatch_size
        self.file = tempfile.TemporaryFile()