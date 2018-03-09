from torch.utils.data.dataset import Dataset

class TextDataset(Dataset):

    def __init__(self, source, bptt):
        super(TextDataset, self).__init__()
        self.data, self.target = self.cut_to_sen(source, bptt)


    def __len__(self):
        return self.data.shape[0] - 1

    def __getitem__(self, idx):
        sen = self.data[idx]
        target = self.target[idx]
        return sen, target

    def cut_to_sen(self, source, bptt):
        num_sentence = (source.size(0) - 1) // bptt

        data = source.narrow(0, 0, num_sentence * bptt)
        target = source.narrow(0, 1, num_sentence * bptt)

        data = data.view(-1, bptt)
        target = target.view(-1, bptt)

        return data, target

